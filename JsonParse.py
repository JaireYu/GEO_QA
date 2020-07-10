import json
import copy
import MySQLdb
import re
import numpy as np
import random
from itertools import chain
import os
import torch
import logging
from torch.utils.data import TensorDataset
logger = logging.getLogger(__name__)
class InputExample(object):
    """
    文件中的每一个样例
    query - intent - variable_label - answer
    """
    def __init__(self, query, domain_label, intent_label, slot_labels, answer):
        self.domain_label = domain_label
        self.query = query
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.answer = answer

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Inputfeatures(object):
    def __init__(self, domain_label, positive_token, negative_tokens, slot_labels_ids, intents_label, slot_length, pos_length, neg_length):
        self.domains_label = domain_label   #number_seq
        self.positive_token = positive_token            #正例，id序列
        self.negative_tokens = negative_tokens                          #反例，id序列的列
        self.intents_label = intents_label                #number_Seq
        self.slot_labels_ids = slot_labels_ids
        self.slot_length = slot_length
        self.pos_length = pos_length
        self.neg_length = neg_length

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def GetLabels(input_file):
    domain = []
    intent = []
    with open(input_file, "r", encoding='utf-8') as reader:
        for line in reader.readlines():
            domain.append(line.split()[0])
            intent.append(line.split()[1])
    domain = sorted(set(domain),key=domain.index)
    intent = sorted(set(intent),key=intent.index)
    return domain, intent

def ParseJsonFile(input_file, input_field_file):
    examples = {'train':[], 'dev':[], 'test':[]}
    entities = {}
    domain_field, intent_field = GetLabels(input_field_file)
    SQLpattern = re.compile(r'SELECT\s*([^\s]*\.[^\s]*(?:\s*,\s*[^\s]*\.[^\s]*)*)\s*FROM\s*(\w*)')
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='yjr',
        passwd='1234',
        db='GEO',
    )
    cur = conn.cursor()
    cur.execute('show tables')  # 执行查询
    tables = cur.fetchall()  # 拿到返回结果
    for table in tables:
        cur.execute("SELECT * FROM {}".format(table[0]))
        col_name_list = [tuple[0] for tuple in cur.description]
        results = cur.fetchall()
        for result in results:
            for index, col in enumerate(col_name_list):
                if col in entities.keys():
                    entities[col].append(result[index])
                else:
                    entities[col] = []
                    entities[col].append(result[index])
    for entity in entities:
        entities[entity] = list(set(entities[entity]))
        print(entity)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    for index, data in enumerate(input_data):
        if len(data['sql']) != 1:
            continue
        SQL = data['sql'][0]
        SQLTokens = SQL.split()
        #intents = [0 for i in intent_field]
        intents_token = []
        #domain = [0 for i in domain_field]
        try:
            """
            for intent in SQLpattern.search(SQL).group(1).split(','):
                intent = intent.split()[0].split('.')[1]
                intents_token.append(intent)
                assert intent in intent_field, "fail1"
                intents[intent_field.index(intent)] = 1
            label_num = sum(intents)
            assert label_num != 0, "fail2"
            """
            intent = SQLpattern.search(SQL).group(1).split(',')[0].split()[0].split('.')[1]
            intents_token.append(intent)
            intents = intent_field.index(intent)
            """
            暂未考虑多domain(form)
            """
            _domain = SQLpattern.search(SQL).group(2)
            assert _domain in domain_field, "fail3"
            domain = domain_field.index(_domain)
        except:
            continue  ### 有待修改
        for sentence in data['sentences']:
            uniqueSQL = SQL
            query = sentence['text']
            query = query.split()
            slot = [0 for i in range(len(query))]
            for var in sentence['variables']:
                for i, word in enumerate(query):
                    if query[i] == var:
                        assert var[0:-1].upper() in intent_field
                        slot[i] = intent_field.index(var[0:-1].upper()) + 1 #将18个slot映射到1-19
                        query[i] = sentence['variables'][var]
                uniqueSQL = uniqueSQL.replace(var, sentence['variables'][var])
            cur = conn.cursor()
            cur.execute(uniqueSQL)  # 执行查询
            results = cur.fetchall()  # 拿到返回结果
            if len(results) != 1:
                continue
            if sentence['question-split'] is 'test':
                """
                test集合的负样本要更多一些
                """
                pass
            else:
                answer = list(chain(*results))
                wrong_answers = []
                for intent in intents_token:
                    wrong_answers = np.append(wrong_answers, np.random.choice(entities[intent.lower()], 2))
                random_index = random.sample(entities.keys(), 2)
                for random_i in random_index:
                    wrong_answers = np.append(wrong_answers, np.random.choice(entities[random_i], 1))
                np.random.shuffle(wrong_answers)
                wrong_answers = list(wrong_answers[:4])
                assert  len(wrong_answers) == 4
                for index, wrong_answer in enumerate(wrong_answers):
                    wrong_answers[index] = str(wrong_answer).split()
                answers = wrong_answers
                answers.insert(0, answer)
            example = InputExample(
                query= query,
                intent_label= intents,
                domain_label= domain,
                slot_labels= slot,
                answer= answers
            )
            examples[sentence['question-split']].append(example)
    return examples


def convert_examples_to_features(examples, max_query_len, max_answer_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    features = []
    for ex_index, example in enumerate(examples):
        query_tokens = []
        _slot_labels_ids = []
        assert len(example.query) == len(example.slot_labels)
        for word, slot_label in zip(example.query, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            query_tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            _slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
        slot_length = len(_slot_labels_ids)
        #处理CLS+query，得到input，seg，slot，mask
        if len(query_tokens) > max_query_len - 2: #超过最大的截断
            query_tokens = query_tokens[:(max_query_len - 2)]
            _slot_labels_ids = _slot_labels_ids[:(max_query_len - 2)]
        query_tokens += [sep_token]
        _slot_labels_ids += [pad_token_label_id]
        _seg_ids = [sequence_a_segment_id] * len(query_tokens)
        _slot_labels_ids = [pad_token_label_id] + _slot_labels_ids
        _query_tokens = [cls_token] + query_tokens
        _seg_ids = [cls_token_segment_id] + _seg_ids
        _input_ids = tokenizer.convert_tokens_to_ids(_query_tokens)
        slot_labels_ids = _slot_labels_ids + [pad_token_label_id] * (max_query_len+max_answer_len-len(_slot_labels_ids))
        #加入answer:
        neg_matadatas = []
        neg_length = []
        pos_matadata = None
        pos_length = None
        for index,answer in enumerate(example.answer):
            answer_tokens = []
            for word in answer:
                word_tokens = tokenizer.tokenize(str(word))
                if not word_tokens:
                    word_tokens = [unk_token]  # For handling the bad-encoded word
                answer_tokens.extend(word_tokens)
            length = len(answer_tokens)
            # 处理answer，得到input，seg，slot，mask
            if len(answer_tokens) > max_answer_len:  # 超过最大的截断
                tokens = query_tokens + answer_tokens[:(max_answer_len)]
            seg_ids = _seg_ids + [sequence_b_segment_id] * len(answer_tokens)
            input_ids = _input_ids + tokenizer.convert_tokens_to_ids(answer_tokens)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_query_len + max_answer_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            seg_ids = seg_ids + ([pad_token_segment_id] * padding_length)
            matadata = [input_ids, attention_mask, seg_ids]

            if index == 0:
                pos_matadata = matadata
                pos_length = length
            else:
                neg_matadatas.append(matadata)
                neg_length.append(length)
        feature = Inputfeatures(
            domain_label= example.domain_label,
            intents_label=example.intent_label,
            negative_tokens=neg_matadatas,
            neg_length = neg_length,
            positive_token=pos_matadata,
            pos_length = pos_length,
            slot_labels_ids = slot_labels_ids,
            slot_length = slot_length,
        )
        features.append(feature)
    return features

def load_and_cache_examples(args, tokenizer, mode):
    # Load data features from cache or dataset file
    # 首先判断路径下是否存在cache
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,               #mode = train/dev/test
            list(filter(None, args.model_name_or_path.split("/"))).pop(),   #选择尾项
            args.max_query_len,    #50
            args.max_answer_len
        )
    )
    input_file = os.path.join(args.data_dir, args.input_file)
    input_field_file = os.path.join(args.data_dir, args.input_field_file)
    #如果cache存在，从cache中load
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset")
        examples = ParseJsonFile(input_file, input_field_file)
        if mode == "train":
            examples = examples['train']
        elif mode == "dev":
            examples = examples['dev']
        elif mode == "test":
            examples = examples['test']
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_query_len, args.max_answer_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_domain_ids = torch.tensor([f.domains_label for f in features], dtype=torch.long)
    all_intent_ids = torch.tensor([f.intents_label for f in features], dtype=torch.long)
    all_neg_ids0 = torch.tensor([f.negative_tokens[0] for f in features], dtype=torch.long)
    all_neg_ids1 = torch.tensor([f.negative_tokens[1] for f in features], dtype=torch.long)
    all_neg_ids2 = torch.tensor([f.negative_tokens[2] for f in features], dtype=torch.long)
    all_neg_ids3 = torch.tensor([f.negative_tokens[3] for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.positive_token for f in features], dtype=torch.long)
    all_slot_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    all_slot_length = torch.tensor([f.slot_length for f in features], dtype=torch.long)
    all_pos_length = torch.tensor([f.pos_length for f in features], dtype=torch.long)
    all_neg_length = torch.tensor([f.neg_length for f in features], dtype=torch.long)
    dataset = TensorDataset(all_domain_ids, all_intent_ids, all_neg_ids0, all_neg_ids1, all_neg_ids2, all_neg_ids3,
                            all_pos_ids, all_slot_ids, all_slot_length, all_pos_length, all_neg_length)
    return dataset