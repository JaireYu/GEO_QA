import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, MODEL_CLASSES
from JsonParse import load_and_cache_examples


def main(args):
    init_logger()
    tokenizer = load_tokenizer(args) # 选择模型BertTokenizer执行from_pretrained

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="geography.json", type=str, help="data to load")
    parser.add_argument("--model_dir", default="./naive_joint_model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--input_field_file", default="geography-fields.txt", type=str, help="data to load")
    parser.add_argument("--slot_labels", default=20, type=int, help="slot label num")
    parser.add_argument("--domain_labels", default=8, type=int, help="domain label num")
    parser.add_argument("--alpha", default=0.1, type=int, help="bias weight for None slot")
    parser.add_argument("--model_type", default="naive_joint", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_query_len", default=30, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_answer_len", default=15, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--lstm_embedding_size", default=256, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--lstm_deep", default=2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=8, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=75, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    args = parser.parse_args()
    main(args)