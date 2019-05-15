#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 11:50
# @Author  : 邵明岩
# @File    : data_process_v1.py
# @Software: PyCharm

import re
import pickle
from zhon import hanzi


def data_dump(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)
    print("store data successfully!")


def get_label(sent, entity):
    def get_entity_len(d):
        return len(d['entity'])

    ner = ['O' for _ in range(len(sent))]

    entity = sorted(entity, key=get_entity_len, reverse=True)
    for d in entity:
        emotion = d['emotion'].strip()
        e_l = d['entity']
        name = ''.join(e_l)
        e_n = len(e_l)
        for i in range(len(sent) - e_n + 1):
            a = ''.join(sent[i:i + e_n])
            if a == name:
                flag = False
                for r in range(e_n):
                    if ner[r + i].startswith('B') or ner[r + i].startswith('I'):
                        flag = True
                        break

                if flag is True:
                    pass
                else:
                    for r in range(e_n):
                        if r == 0:
                            ner[r + i] = 'B-{}'.format(emotion)
                        else:
                            ner[r + i] = 'I-{}'.format(emotion)
    return ner


def get_label_no_emotion(sent, entity):
    def get_entity_len(d):
        return len(d['entity'])

    ner = ['O' for _ in range(len(sent))]

    entity = sorted(entity, key=get_entity_len, reverse=True)
    for d in entity:
        e_l = d['entity']
        name = ''.join(e_l)
        e_n = len(e_l)
        for i in range(len(sent) - e_n + 1):
            a = ''.join(sent[i:i + e_n])
            if a == name:
                flag = False
                for r in range(e_n):
                    if ner[r + i].startswith('B') or ner[r + i].startswith('I'):
                        flag = True
                        break

                if flag is True:
                    pass
                else:
                    for r in range(e_n):
                        if r == 0:
                            ner[r + i] = 'B'
                        else:
                            ner[r + i] = 'I'

    return ner


def get_sentences(content):
    # 基本分句，。|｡|！|\!|？|\?，用这6个符号分
    sentences = re.split(r'(。|｡|！|\!|？|\?)', content)  # 保留分割符
    new_sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        sent = sent.strip()
        new_sents.append(sent)
    res = []
    sentence = ''
    for sent in new_sents:
        temp_sents = []
        if len(sent) > 100:  # 大于100长度继续分，；|、|,|，|﹔|､，6个次级分句
            sents = re.split(r'(；|、|,|，|﹔|､)', sent)  # 保留分割符
            for s in sents:
                if len(s) > 100:  # 子分句也大于100，采用截断式分句
                    ss = []
                    j = 100
                    while j < len(s):
                        ss.append(s[j - 100:j])
                        j = j + 100
                    if len(s[j - 100:len(s)]) < 20:
                        temp = ss[-1] + s[j - 10:len(s)]
                        ss[-1] = temp[0:int(len(temp) / 2)]
                        ss.append(temp[int(len(temp) / 2):])
                    else:
                        ss.append(s[j - 10:len(s)])
                    temp_sents.extend(ss)

                else:
                    temp_sents.append(s)
        else:
            temp_sents.append(sent)

        # temp_sents获得了所有子句，将子句尽可能组成100长度得长句，减少训练时间
        for temp in temp_sents:
            if len(sentence + temp) <= 100:
                sentence = sentence + temp
            else:
                res.append(sentence)
                sentence = temp

    if sentence != '':
        res.append(sentence)

    return res



def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    sent = sent.replace('\n', '')
    sent = sent.replace('\r', '')
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w.strip() for w in chars if len(w.strip()) > 0]
    new_chars = []
    for c in chars:
        if len(c) > 1:
            punctuation = hanzi.punctuation
            punctuation = '|'.join([p for p in punctuation])
            pattern = re.compile(' |({})'.format(punctuation))
            cs = pattern.split(c)
            for w in cs:
                if w and len(w.strip()) > 0:
                    new_chars.append(w.strip())
        else:
            new_chars.append(c)
    return new_chars


def seg_char_sents(sentences):
    results = [seg_char(sent) for sent in sentences]
    return results


def get_core_entityemotions(entityemotions):
    results = []
    for ee in entityemotions:
        result = {}
        result['entity'] = seg_char(ee['entity'])
        result['emotion'] = ee['emotion']
        results.append(result)

    return results


if __name__ == '__main__':

    datas_process = pickle.load(open('new_data/coreEntityEmotion_example.pkl', 'rb'))
    datas = []
    datas_em = []
    all_index = len(datas_process)
    for index, data in enumerate(datas_process):
        print('{}/{}'.format(index, all_index))
        new_data = {}
        new_data_em = {}
        new_data['newsId'] = data['newsId']
        new_data_em['newsId'] = data['newsId']

        new_data['coreEntityEmotions'] = get_core_entityemotions(data['coreEntityEmotions'])
        new_data_em['coreEntityEmotions'] = new_data['coreEntityEmotions']

        if len(data['title']) > 125:
            data['title'] = data['title'][:125]
            print('warning:标题被截断!!')
        title = seg_char(data['title'])
        title_labels = get_label_no_emotion(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data['title'] = (title, title_labels)
        title_labels = get_label(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data_em['title'] = (title, title_labels)

        if len(data['content']) == 0:
            new_data['content'] = []
            new_data_em['content'] = []
            pass
        else:
            data['content'] = data['content'].strip()
            if data['content'][-1] not in '。｡！!？?':
                data['content'] = data['content'] + '。'
            sentences = get_sentences(data['content'])
            sentences = seg_char_sents(sentences)
            content = []
            content_em = []
            for sent in sentences:
                sent_labels = get_label_no_emotion(sent, new_data['coreEntityEmotions'])
                assert len(sent) == len(sent_labels)
                content.append((sent, sent_labels))
                sent_labels = get_label(sent, new_data['coreEntityEmotions'])
                assert len(sent) == len(sent_labels)
                content_em.append((sent, sent_labels))
            new_data['content'] = content
            new_data_em['content'] = content_em

        datas.append(new_data)
        datas_em.append(new_data_em)

    data_dump(datas, 'new_data2/example_ner_no_emotion.pkl')
    data_dump(datas_em, 'new_data2/example_ner_has_emotion.pkl')

    datas_process = pickle.load(open('new_data/coreEntityEmotion_train.pkl', 'rb'))
    datas = []
    datas_em = []
    all_index = len(datas_process)
    for index, data in enumerate(datas_process):
        print('{}/{}'.format(index, all_index))
        new_data = {}
        new_data_em = {}
        new_data['newsId'] = data['newsId']
        new_data_em['newsId'] = data['newsId']

        new_data['coreEntityEmotions'] = get_core_entityemotions(data['coreEntityEmotions'])
        new_data_em['coreEntityEmotions'] = new_data['coreEntityEmotions']

        if len(data['title']) > 125:
            data['title'] = data['title'][:125]
            print('warning:标题被截断!!')
        title = seg_char(data['title'])
        title_labels = get_label_no_emotion(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data['title'] = (title, title_labels)
        title_labels = get_label(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data_em['title'] = (title, title_labels)

        if len(data['content']) == 0:
            new_data['content'] = []
            new_data_em['content'] = []
            pass
        else:
            data['content'] = data['content'].strip()
            if data['content'][-1] not in '。｡！!？?':
                data['content'] = data['content'] + '。'
            sentences = get_sentences(data['content'])
            sentences = seg_char_sents(sentences)
            content = []
            content_em = []
            for sent in sentences:
                sent_labels = get_label_no_emotion(sent, new_data['coreEntityEmotions'])
                assert len(sent) == len(sent_labels)
                content.append((sent, sent_labels))
                sent_labels = get_label(sent, new_data['coreEntityEmotions'])
                assert len(sent) == len(sent_labels)
                content_em.append((sent, sent_labels))
            new_data['content'] = content
            new_data_em['content'] = content_em

        datas.append(new_data)
        datas_em.append(new_data_em)

    data_dump(datas, 'new_data2/train_ner_no_emotion.pkl')
    data_dump(datas_em, 'new_data2/train_ner_has_emotion.pkl')


    datas_process = pickle.load(open('new_data/coreEntityEmotion_test.pkl', 'rb'))
    datas = []
    all_index = len(datas_process)
    for index, data in enumerate(datas_process):
        print('{}/{}'.format(index, all_index))
        new_data = {}
        new_data['newsId'] = data['newsId']
        if len(data['title']) > 125:
            data['title'] = data['title'][:125]
            print('warning:标题被截断!!')
        title = seg_char(data['title'])
        new_data['title'] = title

        if len(data['content']) == 0:
            new_data['content'] = []
        else:
            data['content'] = data['content'].strip()
            if data['content'][-1] not in '。｡！!？?':
                data['content'] = data['content'] + '。'
            sentences = get_sentences(data['content'])
            sentences = seg_char_sents(sentences)
            content = []
            for sent in sentences:
                content.append(sent)
            new_data['content'] = content

        datas.append(new_data)

    data_dump(datas, 'new_data2/test_ner.pkl')

