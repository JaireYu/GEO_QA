import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from JsonParse import GetLabels
from utils import MODEL_CLASSES,set_seed
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)
class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.domain_label_lst, self.slot_label_lst = GetLabels(os.path.join(args.data_dir, args.input_field_file))
        self.intent_label_lst = self.slot_label_lst
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index
        """model_class选择我们自己的model， Config_Class以config基类配置加入我们自己finetuning-task"""
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        self.model = self.model_class(self.bert_config, args, self.domain_label_lst, self.intent_label_lst, self.slot_label_lst)
        """根据do_pred来选择load的model|^"""
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        # n是参数的name: BERT_NAME: embeddings.word_embeddings.weight encoder.layer.5.output.LayerNorm.bias等
        # 下面这段代码的意思是，如果no_decay中的任何一个字段都不在name中则对para使用L2正则项, 否则默认设为0, 即bias相关的不带偏置项,
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # 调度学习率在初期上升，后期下降(warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0             #总步数
        tr_loss = 0.0
        self.model.zero_grad()      # 清空梯度

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        self.evaluate("test")
        for _ in train_iterator:    # 一次遍历数据集
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):   # 取出一个batch: 原数据集是tuple(5 * Tensor(4478)) 所以一个batch是tuple(5 * Tensor(16))
                self.model.train()  # 告诉pytorch正在训练 而不是预测
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'domain_ids': batch[0],
                          'intent_ids': batch[1],
                          'pos_ids': batch[2],
                          'neg_ids0': batch[3],
                          'neg_ids1': batch[4],
                          'neg_ids2': batch[5],
                          'neg_ids3': batch[6],
                          'all_slot_ids': batch[7],
                          'all_slot_length' : batch[8],
                          'all_pos_length' : batch[9],
                          'all_neg_length' : batch[10]
                          }
                loss = self.model(**inputs)[0]      #该语句自动执行forward, 与显式调用forward不同的是这个过程还会调用一些hooks

                if self.args.gradient_accumulation_steps > 1:       #取一个step的平均loss
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0: # 一个step结束, 需要更新参数
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()    #一个loss的积累过程结束，更新参数
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()  #清空梯度
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:  #logging_step:200步之后进行dev
                        self.evaluate("test")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:    # 200步save model
                        self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        self.model.eval()
        answers_pred = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'domain_ids': batch[0],
                          'intent_ids': batch[1],
                          'pos_ids': batch[2],
                          'neg_ids0': batch[3],
                          'neg_ids1': batch[4],
                          'neg_ids2': batch[5],
                          'neg_ids3': batch[6],
                          'all_slot_ids': batch[7],
                          'all_slot_length' : batch[8],
                          'all_pos_length' : batch[9],
                          'all_neg_length' : batch[10]
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, answer = outputs

                eval_loss += tmp_eval_loss.mean().item()    # 对batch内的
            nb_eval_steps += 1
            if answers_pred is None:
                answers_pred = answer.detach().cpu().numpy()  # intent输出转化成numpy()
            else:
                answers_pred = np.append(answers_pred, answer.detach().cpu().numpy(),
                                         axis=0)  # np.append()是拼接两个nparray的操作
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        labels = np.zeros(answers_pred.size)
        total_result = {
            "acc": (labels == answers_pred).mean(),
            "intent_acc": np.array([1 if answer_pred == 1 or answer_pred==2 else 0 for answer_pred in answers_pred]).mean()
        }
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model #意思是只加载model本身
        model_to_save.save_pretrained(output_dir)       # save模型
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))  #save_trainingconfig
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)   # 在文件夹中自动加载bert的配置文件
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config,
                                                          args=self.args, intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst, domain_label_lst = self.domain_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")