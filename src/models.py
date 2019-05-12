import torch
import torch.nn as nn
import math
from pytorch_pretrained_bert.modeling import (BertModel,
                                              BertPreTrainedModel)


class Net00(BertPreTrainedModel):
    def __init__(self, config):
        super(Net00, self).__init__(config)
        self.bert = BertModel(config)
        # print all layers
        print(self.bert)
        # for idx, (n,m) in enumerate(self.bert.named_modules()):
        #     print(idx, n, m)


class Net0(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(Net0, self).__init__(config)
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
        sequence_output = nn.Dropout(self.dp)(sequence_output)
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
                ent_mask = ent_mask == 2  # [L', ]
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


############################
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(768, 400)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 11)

    def forward(self, x):
        '''
        :param x: [bs, L, C]
        :return:
        '''
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


############################ attention  #########################
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bia


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        pass
