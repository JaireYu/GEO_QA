import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
import numpy as np

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class LSTM_slot_filler(torch.nn.Module):
    def __init__(self,embedding_dim,target_size, layer_num, dropout_rate):
        super(LSTM_slot_filler,self).__init__()
        self.lstm = nn.LSTM(embedding_dim, target_size, layer_num, batch_first=True, dropout=dropout_rate)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                size = param.shape[0]
                for id in range(4):
                    torch.nn.init.orthogonal_(param.data[id*size//4: (id+1)*size//4])
            elif 'weight_ih' in name:
                size = param.shape[0]
                for id in range(4):
                    torch.nn.init.xavier_uniform_(param.data[id*size//4: (id+1)*size//4])
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self,inputs, lengths, ids):
        input = inputs[0][:, 1:]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=input, lengths=lengths, batch_first=True)
        packed_out,self.hidden=self.lstm(packed_input)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out

class LSTM(torch.nn.Module):
    def __init__(self,embedding_dim,target_size, layer_num, dropout_rate):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(embedding_dim, target_size, layer_num, batch_first=True, dropout=dropout_rate)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                size = param.shape[0]
                for id in range(4):
                    torch.nn.init.orthogonal_(param.data[id * size // 4: (id + 1) * size // 4])
            elif 'weight_ih' in name:
                size = param.shape[0]
                for id in range(4):
                    torch.nn.init.xavier_uniform_(param.data[id * size // 4: (id + 1) * size // 4])
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self,inputs, lengths):
        """
        :param inputs: 乱序的输入
        :param lengths: 输入对应的长度
        :return: 返回的是对应输入顺序的输出
        """
        lengths, index = lengths.sort(0, descending=True)
        _, ids = torch.sort(index, dim=0)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=inputs, lengths=lengths, batch_first=True)
        packed_out,self.hidden=self.lstm(packed_input)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out[ids]

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(DomainClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class SimilarityScore(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.):
        super(SimilarityScore, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 256),  # ->64
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64), #->64
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 16), #16
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 4),  # 4
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4, 1),  # 4
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class GEO_QA_Naive(BertPreTrainedModel):
    def __init__(self, bert_config, args, domain_label_lst, intent_label_lst, slot_label_lst):
        super(GEO_QA_Naive, self).__init__(bert_config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        self.SimilarityScore = SimilarityScore(bert_config.hidden_size, args.dropout_rate)

    def net(self, input_ids, attention_mask, segment_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        return self.SimilarityScore(outputs[1])

    def forward(self, domain_ids, intent_ids, pos_ids, neg_ids0, neg_ids1, neg_ids2, neg_ids3, all_slot_ids, all_slot_length, all_pos_length, all_neg_length):
        score_pos = self.net(pos_ids[:,0], pos_ids[:,1], pos_ids[:,2])
        score_neg0 = self.net(neg_ids0[:,0], neg_ids0[:,1], neg_ids0[:,2])
        score_neg1 = self.net(neg_ids1[:,0], neg_ids1[:,1], neg_ids1[:,2])
        score_neg2 = self.net(neg_ids2[:,0], neg_ids2[:,1], neg_ids2[:,2])
        score_neg3 = self.net(neg_ids3[:,0], neg_ids3[:,1], neg_ids3[:,2])
        score = torch.cat((score_pos, score_neg0, score_neg1, score_neg2, score_neg3), 1) #batch*5
        softmax_qp = F.softmax(score, dim=1)[:, 0]  #(batch_size)
        results =  torch.max(score, dim=1)[1]
        loss = torch.sum(-torch.log(softmax_qp))
        return loss, results

class GEO_QA_NaiveJoint(BertPreTrainedModel):
    def __init__(self, bert_config, args, domain_label_lst, intent_label_lst, slot_label_lst):
        super(GEO_QA_NaiveJoint, self).__init__(bert_config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        self.SimilarityScore = SimilarityScore(bert_config.hidden_size, args.dropout_rate)
        self.slot_classifier = SlotClassifier(bert_config.hidden_size, self.num_slot_labels, self.args.dropout_rate)

    def net(self, input_ids, attention_mask, segment_ids, all_slot_lengths, all_slot_ids, pos=True):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        if pos is True:
            loss = 0
            slot_pred = self.slot_classifier(outputs[0])
            for length, slot_score, slot_id in zip(all_slot_lengths, slot_pred, all_slot_ids):
                prob = F.softmax(slot_score, dim=1)
                for i in range(int(length)):
                    if(int(slot_id[i]) == 0): # tag None
                        loss += -1/self.args.batch_size * self.args.alpha * torch.log(prob[1+i][slot_id[1+i]])
                    else:
                        loss += -1/self.args.batch_size * torch.log(prob[1+i][slot_id[1+i]])
            return self.SimilarityScore(outputs[1]), loss
        else:
            return self.SimilarityScore(outputs[1])

    def forward(self, domain_ids, intent_ids, pos_ids, neg_ids0, neg_ids1, neg_ids2, neg_ids3, all_slot_ids, all_slot_length, all_pos_length, all_neg_length):
        score_pos,slot_loss = self.net(pos_ids[:,0], pos_ids[:,1], pos_ids[:,2],all_slot_length, all_slot_ids, pos=True)
        score_neg0 = self.net(neg_ids0[:,0], neg_ids0[:,1], neg_ids0[:,2], all_slot_length, all_slot_ids, pos=False)
        score_neg1 = self.net(neg_ids1[:,0], neg_ids1[:,1], neg_ids1[:,2], all_slot_length, all_slot_ids, pos=False)
        score_neg2 = self.net(neg_ids2[:,0], neg_ids2[:,1], neg_ids2[:,2], all_slot_length, all_slot_ids, pos=False)
        score_neg3 = self.net(neg_ids3[:,0], neg_ids3[:,1], neg_ids3[:,2], all_slot_length, all_slot_ids, pos=False)
        score = torch.cat((score_pos, score_neg0, score_neg1, score_neg2, score_neg3), 1) #batch*5
        softmax_qp = F.softmax(score, dim=1)[:, 0]  #(batch_size)
        results =  torch.max(score, dim=1)[1]
        loss = torch.sum(-torch.log(softmax_qp)) + slot_loss
        return loss, results

class GEO_QA_LSTM(BertPreTrainedModel):
    def __init__(self, bert_config, args, domain_label_lst, intent_label_lst, slot_label_lst):
        super(GEO_QA_LSTM, self).__init__(bert_config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        self.SimilarityScore = SimilarityScore(bert_config.hidden_size, args.dropout_rate)
        self.query_lstm = LSTM_slot_filler(bert_config.hidden_size,args.slot_labels,2, self.args.dropout_rate)

    def net_slot(self, input_ids, attention_mask, segment_ids, lengths, ids, all_slot_ids, alpha):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        slot_filler = self.query_lstm(outputs, lengths, ids)
        loss = 0
        for length, slot_score, slot_id in zip(lengths, slot_filler, all_slot_ids):
            prob = F.softmax(slot_score, dim=1)
            for i in range(int(length)):
                if(int(slot_id[i]) == 0): # tag None
                    loss += -alpha * torch.log(prob[i][slot_id[i]])
                else:
                    loss += -torch.log(prob[i][slot_id[i]])
        return self.SimilarityScore(outputs[1]), loss

    def net(self, input_ids, attention_mask, segment_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        return self.SimilarityScore(outputs[1])

    def forward(self, domain_ids, intent_ids, pos_ids, neg_ids0, neg_ids1, neg_ids2, neg_ids3, all_slot_ids, all_slot_length, all_pos_length, all_neg_length):
        lengths, index = all_slot_length.sort(0, descending=True)
        _, ids = torch.sort(index, dim=0)
        domain_ids = domain_ids[index]
        intent_ids = intent_ids[index]
        pos_ids = pos_ids[index]
        neg_ids0 = neg_ids0[index]
        neg_ids1 = neg_ids1[index]
        neg_ids2 = neg_ids2[index]
        neg_ids3 = neg_ids3[index]
        all_slot_ids = all_slot_ids[index]
        score_pos, slot_loss = self.net_slot(pos_ids[:,0], pos_ids[:,1], pos_ids[:,2], lengths, ids, all_slot_ids, self.args.alpha)
        score_neg0 = self.net(neg_ids0[:,0], neg_ids0[:,1], neg_ids0[:,2])
        score_neg1 = self.net(neg_ids1[:,0], neg_ids1[:,1], neg_ids1[:,2])
        score_neg2 = self.net(neg_ids2[:,0], neg_ids2[:,1], neg_ids2[:,2])
        score_neg3 = self.net(neg_ids3[:,0], neg_ids3[:,1], neg_ids3[:,2])
        score = torch.cat((score_pos, score_neg0, score_neg1, score_neg2, score_neg3), 1) #batch*5
        softmax_qp = F.softmax(score, dim=1)[:, 0]  #(batch_size)
        results =  torch.max(score, dim=1)[1]
        loss = torch.sum(-torch.log(softmax_qp))
        loss += slot_loss
        return loss, results

class GEO_QA_Sematic(BertPreTrainedModel):
    def __init__(self, bert_config, args, domain_label_lst, intent_label_lst, slot_label_lst):
        super(GEO_QA_Sematic, self).__init__(bert_config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        self.lstm = LSTM(bert_config.hidden_size, args.lstm_embedding_size, args.lstm_deep, self.args.dropout_rate)
        self.MLP = MLP(2*args.lstm_embedding_size, args.dropout_rate)
        self.intent_classifier = IntentClassifier(bert_config.hidden_size, self.num_intent_labels, self.args.dropout_rate)
        self.domain_classifier = DomainClassifier(bert_config.hidden_size, self.num_domain_labels, self.args.dropout_rate)
        self.slot_classifier = SlotClassifier(bert_config.hidden_size, self.num_slot_labels, self.args.dropout_rate)

    def net(self, input_ids, attention_mask, segment_ids, answer_length, slot_lengths, pos=True
            , all_slot_ids = None, intent_label = None, domain_label = None):
        """
        bert+double lstm model
        :param input_ids:
        :param attention_mask:
        :param segment_ids:
        :param answer_length:
        :param slot_lengths:
        :return:
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        query_embeddings = []
        answer_embeddings = []
        for index, embedding in enumerate(outputs[0]):
            query_embeddings.append(embedding[1:max(slot_lengths)+1].detach().numpy())
            answer_embeddings.append(embedding[slot_lengths[index]+2:slot_lengths[index]+2+max(answer_length)].detach().numpy())
        query_embeddings = torch.tensor(query_embeddings)
        answer_embeddings = torch.tensor(answer_embeddings)
        query_output = self.lstm(query_embeddings, slot_lengths)[:,-1,:]
        answer_output = self.lstm(answer_embeddings, answer_length)[:,-1,:]
        match_result = self.MLP(torch.cat((query_output, answer_output), 1))
        total_loss = 0
        if pos is True:
            intent = self.intent_classifier(outputs[1])
            domain = self.domain_classifier(outputs[1])
            slot_pred = self.slot_classifier(outputs[0])
            slot_loss = 0
            for length, slot_score, slot_id in zip(slot_lengths, slot_pred, all_slot_ids):
                prob = F.softmax(slot_score, dim=1)
                for i in range(int(length)):
                    if (int(slot_id[i]) == 0):  # tag None
                        slot_loss += -1/self.args.batch_size * self.args.alpha * torch.log(prob[1+i][slot_id[1+i]])
                    else:
                        slot_loss += -1/self.args.batch_size * torch.log(prob[1+i][slot_id[1+i]])
            loss_func = nn.CrossEntropyLoss()
            intent_loss = loss_func(intent.view(-1, self.num_intent_labels),
                                              intent_label.view(-1))
            domain_loss = loss_func(domain.view(-1, self.num_domain_labels),
                                              domain_label.view(-1))
            total_loss += intent_loss + domain_loss + slot_loss
            return total_loss, match_result
        else:
            return match_result

    def forward(self, domain_ids, intent_ids, pos_ids, neg_ids0, neg_ids1, neg_ids2, neg_ids3, all_slot_ids, all_slot_length, all_pos_length, all_neg_length):
        other_loss, score_pos = self.net(pos_ids[:,0], pos_ids[:,1], pos_ids[:,2],
                                        all_pos_length, all_slot_length, pos=True,
                                        all_slot_ids=all_slot_ids,
                                        intent_label=intent_ids, domain_label=domain_ids)
        score_neg0 = self.net(neg_ids0[:,0], neg_ids0[:,1], neg_ids0[:,2], all_pos_length, all_slot_length, pos=False)
        score_neg1 = self.net(neg_ids1[:,0], neg_ids1[:,1], neg_ids1[:,2], all_pos_length, all_slot_length, pos=False)
        score_neg2 = self.net(neg_ids2[:,0], neg_ids2[:,1], neg_ids2[:,2], all_pos_length, all_slot_length, pos=False)
        score_neg3 = self.net(neg_ids3[:,0], neg_ids3[:,1], neg_ids3[:,2], all_pos_length, all_slot_length, pos=False)
        score = torch.cat((score_pos, score_neg0, score_neg1, score_neg2, score_neg3), 1) #batch*5
        softmax_qp = F.softmax(score, dim=1)[:, 0]  #(batch_size)
        results =  torch.max(score, dim=1)[1]
        loss = torch.sum(-torch.log(softmax_qp))
        loss += other_loss
        return loss, results