import torch
import torch.nn as nn
import math
from pytorch_pretrained_bert.modeling import (BertModel,
                                              BertPreTrainedModel)

'''
只有 NetX2 NetX3 在用
X 表示拼接 768D 的 CLS 向量到每个字上
NetX2 不做 mask B(title)
NetX3 做 mask 只保留 A 句
'''


class Net_look(BertPreTrainedModel):
    '''look bert'''

    def __init__(self, config):
        super(Net_look, self).__init__(config)
        self.bert = BertModel(config)
        # print all layers
        print(self.bert)
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     print(idx, n, m)


class Net_fz(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(Net_fz, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        for idx, (n, m) in enumerate(self.bert.named_modules()):
            if idx < 179:
                for p in m.parameters():
                    p.requires_grad = False
                    print(idx, n)
            else:
                print("not freeze: ", idx, n)
        print("model freeze over!")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                active_input_ids = input_ids.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                active_input_ids = input_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits, active_labels, active_input_ids
        else:
            return logits


class Net02(BertPreTrainedModel):
    def __init__(self, config, num_labels_ent, num_labels_emo):
        super(Net02, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ent = nn.Linear(config.hidden_size, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels_ent=None, labels_emo=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)
        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]
                active_labels_ent = labels_ent.view(-1)[active_loss]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]
                active_labels_emo = labels_emo.view(-1)[active_loss]
                active_input_ids = input_ids.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)
                active_logits_emo = logits_ent.view(-1, self.num_labels_emo)
                active_labels_ent = labels_ent.view(-1)
                active_labels_emo = labels_emo.view(-1)
                active_input_ids = input_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_input_ids
        else:
            return logits_ent, logits_emo


class NetEnd2End(BertPreTrainedModel):
    def __init__(self, config, num_labels_ent, num_labels_emo):
        super(NetEnd2End, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ent = nn.Linear(config.hidden_size, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_loss]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_loss]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                active_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                active_labels_emo = active_labels_emo[ent_mask]
                assert active_logits_emo.size(0) == active_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_loss]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[ent_mask]  # [L', ]
                active_labels_emo = labels_emo.view(-1)[ent_mask]
                assert active_logits_emo.size(0) == active_labels_emo.size(0)
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetEnd2EndX(BertPreTrainedModel):
    def __init__(self, config, num_labels_ent, num_labels_emo):
        super(NetEnd2EndX, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_loss]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_loss]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                active_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                active_labels_emo = active_labels_emo[ent_mask]
                assert active_logits_emo.size(0) == active_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_loss]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[ent_mask]  # [L', ]
                active_labels_emo = labels_emo.view(-1)[ent_mask]
                assert active_logits_emo.size(0) == active_labels_emo.size(0)
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetX(BertPreTrainedModel):
    def __init__(self, config, num_labels_ent, num_labels_emo):
        super(NetX, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = nn.Dropout(0.2)(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_loss]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_loss]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                mask_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                mask_labels_emo = active_labels_emo[ent_mask]
                assert mask_logits_emo.size(0) == mask_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_loss]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                mask_logits_emo = active_logits_emo[ent_mask]
                active_labels_emo = labels_emo.view(-1)
                mask_labels_emo = active_logits_emo[ent_mask]
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, mask_logits_emo, mask_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetX2(BertPreTrainedModel):
    '''不 mask 掉 B 句'''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetX2, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_loss]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_loss]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                mask_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                mask_labels_emo = active_labels_emo[ent_mask]
                assert mask_logits_emo.size(0) == mask_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_loss]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                mask_logits_emo = active_logits_emo[ent_mask]
                active_labels_emo = labels_emo.view(-1)
                mask_labels_emo = active_logits_emo[ent_mask]
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, mask_logits_emo, mask_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetX3(BertPreTrainedModel):
    '''
    只对A句进行loss计算
    '''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetX3, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_seg = token_type_ids.view(-1)[active_mask]
                active_seg = active_seg == 0

                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_mask][active_seg]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_mask][active_seg]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_mask][active_seg]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_mask][active_seg]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                mask_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                mask_labels_emo = active_labels_emo[ent_mask]
                assert mask_logits_emo.size(0) == mask_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 2  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                mask_logits_emo = active_logits_emo[ent_mask]
                active_labels_emo = labels_emo.view(-1)
                mask_labels_emo = active_logits_emo[ent_mask]
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, mask_logits_emo, mask_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetX3_fz(BertPreTrainedModel):
    '''
    NeX3 的 freeze 版本
    '''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetX3_fz, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def freeze(self):
        for idx, (n, m) in enumerate(self.bert.named_modules()):
            if idx < 179:
                for p in m.parameters():
                    p.requires_grad = False
                    print(idx, n)
            else:
                print("not freeze: ", idx, n)
        print("model freeze over!")

    def unfreeze(self):
        freeze_paras = []
        for idx, (n, m) in enumerate(self.bert.named_modules()):
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    freeze_paras.append(p)
        print("unfreeze over")
        return freeze_paras

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = nn.Dropout(self.dp)(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_seg = token_type_ids.view(-1)[active_mask]
                active_seg = active_seg == 0

                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_mask][active_seg]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_mask][active_seg]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_mask][active_seg]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_mask][active_seg]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                mask_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                mask_labels_emo = active_labels_emo[ent_mask]
                assert mask_logits_emo.size(0) == mask_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 2  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                mask_logits_emo = active_logits_emo[ent_mask]
                active_labels_emo = labels_emo.view(-1)
                mask_labels_emo = active_logits_emo[ent_mask]
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, mask_logits_emo, mask_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetX4(BertPreTrainedModel):
    '''
    基于NetX3 fc 为两层 或者可以 768*2 -> 768 -> 3/4
    '''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetX4, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent_1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier_ent_2 = nn.Linear(config.hidden_size, num_labels_ent)
        self.classifier_emo_1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier_emo_2 = nn.Linear(config.hidden_size, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = self.dp(sequence_output)
        sequence_output_ent = self.classifier_ent_1(sequence_output)
        logits_ent = self.classifier_ent_2(sequence_output_ent)
        sequence_output_emo = self.classifier_emo_1(sequence_output)
        logits_emo = self.classifier_emo_1(sequence_output_emo)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_seg = token_type_ids.view(-1)[active_mask]
                active_seg = active_seg == 0

                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_mask][active_seg]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_mask][active_seg]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_mask][active_seg]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_mask][active_seg]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                mask_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                mask_labels_emo = active_labels_emo[ent_mask]
                assert mask_logits_emo.size(0) == mask_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 2  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                mask_logits_emo = active_logits_emo[ent_mask]
                active_labels_emo = labels_emo.view(-1)
                mask_labels_emo = active_logits_emo[ent_mask]
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, mask_logits_emo, mask_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetX5(BertPreTrainedModel):
    '''
    基于NetX3 明确使用 xvaier_normal_ 初始化 添加 BN 代替 dropout
    '''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp=0.2):
        super(NetX5, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        # self.dropout = nn.Dropout(self.dp)
        self.bn = nn.BatchNorm1d(config.hidden_size * 2, eps=2e-1)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        # weight init
        nn.init.xavier_uniform_(self.classifier_ent.weight)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        nn.init.xavier_uniform_(self.classifier_emo.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        logits_ent = self.bn(self.classifier_ent(sequence_output))
        logits_emo = self.bn(self.classifier_emo(sequence_output))

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_seg = token_type_ids.view(-1)[active_mask]
                active_seg = active_seg == 0

                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_mask][active_seg]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_mask][active_seg]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_mask][active_seg]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_mask][active_seg]  # [L, ]
                # 在实体基础上再做情感 通过 argmax 的 mask 实现    可能让 ent 预训练一会儿再做情感可能会更好....
                # 目前的方法可以有:
                #     1. 只对  1,即  B 做情感  [Y]  注意这里的情感只有 POS NEG NORM ([错误划掉]没有 O 要不然取出实体可能不能得到其情感(预测为O))
                #     2. 122 做情感 可投票
                #     3. 122 222 111 都做情感
                # 似乎把metric的逻辑和训练逻辑弄在一起了
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 1  # [L', ]
                mask_logits_emo = active_logits_emo[ent_mask]  # [L', ]
                mask_labels_emo = active_labels_emo[ent_mask]
                assert mask_logits_emo.size(0) == mask_labels_emo.size(0)
                # active_input_ids = input_ids.view(-1)[active_loss] #[L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                ent_mask = torch.argmax(torch.softmax(active_logits_ent, dim=-1), dim=-1)  # [L,]
                ent_mask = ent_mask == 2  # [L', ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                mask_logits_emo = active_logits_emo[ent_mask]
                active_labels_emo = labels_emo.view(-1)
                mask_labels_emo = active_logits_emo[ent_mask]
                active_labels_ent = labels_ent.view(-1)
                # active_input_ids = input_ids.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, mask_logits_emo, mask_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetXLast(BertPreTrainedModel):
    '''
    只对A句进行loss计算
    '''

    def __init__(self, config, num_labels, dp):
        super(NetXLast, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dp = dp
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     if idx < 179:
        #         for p in m.parameters():
        #             p.requires_grad = False
        #             print(idx, n)
        #     else:
        #         print("not freeze: ", idx, n)
        # print("model freeze over!")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = nn.Dropout(self.dp)(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            active_mask = attention_mask.view(-1) == 1
            active_seg = token_type_ids.view(-1)[active_mask]
            active_seg = active_seg == 0

            active_logits = logits.view(-1, self.num_labels)[active_mask][active_seg]  # [L, C_ent]
            active_labels = labels.view(-1)[active_mask][active_seg]  # [L, ]
            active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
            return active_logits, active_labels, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits


class Net03(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(Net03, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # freeze bert!
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # only unfreeze 10 11
        for idx, (n, m) in enumerate(self.bert.named_modules()):
            if idx < 179:
                for p in m.parameters():
                    p.requires_grad = False
                    print(idx, n)
            else:
                print("not freeze: ", idx, n)
        print("model freeze over!")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 200)
        self.classifier2 = nn.Linear(200, 100)
        self.classifier3 = nn.Linear(100, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier1(sequence_output)
        logits = self.dropout(logits)
        logits = self.classifier2(logits)
        logits = self.dropout(logits)
        logits = self.classifier3(logits)

        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                active_input_ids = input_ids.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                active_input_ids = input_ids.view(-1)
            # return active_logits, active_labels,logits, labels
            return active_logits, active_labels, active_input_ids
        else:
            return logits


############################################################## 精简版 ============================================
class NetY1(BertPreTrainedModel):
    '''不加 CLS'''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetY1, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent = nn.Linear(config.hidden_size, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_loss]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_loss]  # [L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_loss]  # [L, ]
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                active_labels_emo = labels_emo.view(-1)
                active_labels_ent = labels_ent.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetY2(BertPreTrainedModel):
    '''CLS 但不 mask 掉 B 句'''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetY2, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_loss]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_loss]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_loss]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_loss]  # [L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_loss]  # [L, ]
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                active_labels_emo = labels_emo.view(-1)
                active_labels_ent = labels_ent.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetY3(BertPreTrainedModel):
    '''
    只对A句进行loss计算
    '''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetY3, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_seg = token_type_ids.view(-1)[active_mask]
                active_seg = active_seg == 0

                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_mask][active_seg]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_mask][active_seg]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_mask][active_seg]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_mask][active_seg]  # [L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                active_labels_emo = labels_emo.view(-1)
                active_labels_ent = labels_ent.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo


class NetY3_fz(BertPreTrainedModel):
    '''
    NeX3 的 freeze 版本
    '''

    def __init__(self, config, num_labels_ent, num_labels_emo, dp):
        super(NetY3_fz, self).__init__(config)
        self.num_labels_ent = num_labels_ent
        self.num_labels_emo = num_labels_emo
        self.bert = BertModel(config)
        self.dp = dp
        self.dropout = nn.Dropout(self.dp)
        self.classifier_ent = nn.Linear(config.hidden_size * 2, num_labels_ent)
        self.classifier_emo = nn.Linear(config.hidden_size * 2, num_labels_emo)
        self.apply(self.init_bert_weights)

    def freeze(self):
        for idx, (n, m) in enumerate(self.bert.named_modules()):
            if idx < 179:
                for p in m.parameters():
                    p.requires_grad = False
                    print(idx, n)
            else:
                print("not freeze: ", idx, n)
        print("model freeze over!")

    def unfreeze(self):
        freeze_paras = []
        for idx, (n, m) in enumerate(self.bert.named_modules()):
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    freeze_paras.append(p)
        print("unfreeze over")
        return freeze_paras

    def forward(self, input_ids, myinput_ids=None, token_type_ids=None, attention_mask=None, labels_ent=None,
                labels_emo=None):
        # _ is [CLS] 可以试试拼接在序列每个字上
        sequence_output, CLS = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # bs, 128 ,768   bs,768
        # use repeat
        CLS = CLS.repeat(1, sequence_output.size(1)).view(sequence_output.size())
        sequence_output = torch.cat([sequence_output, CLS], dim=-1)
        # sequence_output = self.dropout(sequence_output)
        sequence_output = nn.Dropout(self.dp)(sequence_output)
        logits_ent = self.classifier_ent(sequence_output)
        logits_emo = self.classifier_emo(sequence_output)

        if labels_ent is not None and labels_emo is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss, use 10 class !!!
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_seg = token_type_ids.view(-1)[active_mask]
                active_seg = active_seg == 0

                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)[active_mask][active_seg]  # [L, C_ent]
                active_labels_ent = labels_ent.view(-1)[active_mask][active_seg]  # [L, ]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)[active_mask][active_seg]  # [L, C_emo]
                active_labels_emo = labels_emo.view(-1)[active_mask][active_seg]  # [L, ]
                active_myinput_ids = myinput_ids.view(-1)[active_mask][active_seg]  # [L, ]
            else:
                active_logits_ent = logits_ent.view(-1, self.num_labels_ent)  # [L, C_ent]
                active_logits_emo = logits_emo.view(-1, self.num_labels_emo)  # [ent_mask] #[L', ]
                active_labels_emo = labels_emo.view(-1)
                active_labels_ent = labels_ent.view(-1)
                active_myinput_ids = myinput_ids.view(-1)
            return active_logits_ent, active_labels_ent, active_logits_emo, active_labels_emo, active_myinput_ids
        else:
            # 可以加ent_mask 也可以不加
            return logits_ent, logits_emo

###################################################################################################################
