import os
import random
import logging
import torch.nn.functional as F
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
from models import GEO_QA_Naive, GEO_QA_LSTM, GEO_QA_Sematic, GEO_QA_NaiveJoint

MODEL_CLASSES = {
    'naive': (BertConfig, GEO_QA_Naive, BertTokenizer),
'slot_joint': (BertConfig, GEO_QA_LSTM, BertTokenizer),
'all_joint': (BertConfig, GEO_QA_Sematic, BertTokenizer),
'naive_joint': (BertConfig, GEO_QA_NaiveJoint, BertTokenizer)
}
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def load_tokenizer(args):
    """arg.model_name_or_path 在main中定义，默认为bert-base-uncased, 选择模型BertTokenizer执行from_pretrained"""
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

def set_seed(args):         # 对所有可能出现随机数的部分设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

def pointwise_loss(score_matrix, labels):
    """
    this is a function to calculate the pointwise loss, give a question Q and a answer list [A1, A2, ... An]
    :param score_matrix: is a batch*samples matrix
    :param labels: is a boolean(0/1) labels batch*samples matrix
    :return: loss = -\sum_{n=1}^N{y_n*log(score_n)}
    """
    softmax_qp = F.softmax(score_matrix, dim=1)
    loss = 0
    for score, label in zip(softmax_qp, labels):
        for s,l in zip(score, label):
            loss += - l * torch.log(s)
    return loss/softmax_qp.shape[0]

def maxmargin_loss(score_matrix, labels, M):
    """
    this is a function to calculate the hinge_loss given a question Q and a answer list [A1, A2, ... An], with margin M
    :param score_matrix:
    :param labels:
    :param M:
    :return: average of hinge_loss
    """
    softmax_qp = F.softmax(score_matrix, dim=1)
    loss = 0
    for score, label in zip(softmax_qp, labels):
        pos_indices = []
        neg_indices = []
        MM_loss = 0
        for l_index, l in enumerate(label):
            print(l)
            if bool(l == 1):
                pos_indices.append(l_index)
            else:
                neg_indices.append(l_index)
        for pos_index in pos_indices:
            for neg_index in neg_indices:
                MM_loss += M + score[pos_index] - score[neg_index]
        MM_loss = MM_loss / (len(pos_indices)*len(neg_indices))
        loss += MM_loss
    return loss/softmax_qp.shape[0]

def entropy_loss(score_matrix, labels):
    """
    this is cross-entropy loss function given a question Q and a answer list [A1, A2, ... An]
    :param score_matrix:
    :param labels:
    :return:loss
    """
    softmax_qp = F.softmax(score_matrix, dim=1)
    lossfunc = torch.nn.BCELoss()
    return lossfunc(softmax_qp, labels)

def NLLloss(score_matrix, labels):
    """
    NLLloss consider only
    :param score_matrix:
    :param labels:
    :return:
    """
    softmax_qp = F.softmax(score_matrix, dim=1)
    softmax_qp = torch.log(softmax_qp)
    loss = 0
    for score, label in zip(softmax_qp, labels):
        mask = torch.eq(label, 1)
        pos_score = -score[mask].sum()
        loss += pos_score
    return loss/softmax_qp.shape[0]
