#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 19:20
# @Author  : 邵明岩
# @File    : get_sents_fix.py
# @Software: PyCharm

import re
import pickle
from zhon import hanzi
import string
import json
import html


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
        result['entity'] = seg_char(clean_text(ee['entity']))
        result['emotion'] = ee['emotion']
        results.append(result)

    return results


def ishan(char):
    # for python 3.x
    # sample: ishan('一') == True, ishan('我&&你') == False
    return '\u4e00' <= char <= '\u9fff'


def clean_text(text):
    new_text = []
    for char in text:
        if ishan(char) or char in string.digits or char in string.ascii_letters or char in (hanzi.punctuation + string.punctuation):
            new_text.append(char)
        elif char == '\t' or char == ' ':
            new_text.append(' ')
        else:
            continue

    new_text = ''.join(new_text)
    new_text = re.sub(r' +', ' ', new_text)
    new_text = html.unescape(new_text)
    new_text = re.sub(
        r'(http|ftp)s?://([^\u4e00-\u9fa5＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。])*',
        '', new_text)
    new_text = re.sub(r'[!"#%&\'()*+-./:;<=>?@[\]^_`{|}~]{2,}', '', new_text)

    return new_text


if __name__ == '__main__':

    f = open('../data/coreEntityEmotion_example.txt', 'r')
    datas = []
    datas_em = []
    all_index = len(f.readlines())
    f.seek(0)
    for index, line in enumerate(f.readlines()):
        data = json.loads(line)
        print('{}/{}'.format(index, all_index))
        new_data = {}
        new_data_em = {}
        new_data['newsId'] = data['newsId']
        new_data_em['newsId'] = data['newsId']

        new_data['coreEntityEmotions'] = get_core_entityemotions(data['coreEntityEmotions'])
        new_data_em['coreEntityEmotions'] = new_data['coreEntityEmotions']

        title = clean_text(data['title'].strip())

        if len(title) > 125:
            title = title[:125]
            print('warning:标题被截断!!')
        title = seg_char(title)
        title_labels = get_label_no_emotion(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data['title'] = (title, title_labels)
        title_labels = get_label(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data_em['title'] = (title, title_labels)
        data['content'] = clean_text(data['content'].strip())
        if len(data['content'].strip()) == 0:
            new_data['content'] = []
            new_data_em['content'] = []
            pass
        else:
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

    f.close()
    data_dump(datas, '../datasets/example_ner_no_emotion.pkl')
    data_dump(datas_em, '../datasets/example_ner_has_emotion.pkl')

    f = open('../data/coreEntityEmotion_train.txt', 'r')
    datas = []
    datas_em = []
    all_index = len(f.readlines())
    f.seek(0)
    for index, line in enumerate(f.readlines()):
        data = json.loads(line)
        print('{}/{}'.format(index, all_index))
        new_data = {}
        new_data_em = {}
        new_data['newsId'] = data['newsId']
        new_data_em['newsId'] = data['newsId']

        new_data['coreEntityEmotions'] = get_core_entityemotions(data['coreEntityEmotions'])
        new_data_em['coreEntityEmotions'] = new_data['coreEntityEmotions']

        title = clean_text(data['title'].strip())

        if len(title) > 125:
            title = title[:125]
            print('warning:标题被截断!!')
        title = seg_char(title)
        title_labels = get_label_no_emotion(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data['title'] = (title, title_labels)
        title_labels = get_label(title, new_data['coreEntityEmotions'])
        assert len(title) == len(title_labels)
        new_data_em['title'] = (title, title_labels)
        data['content'] = clean_text(data['content'].strip())
        if len(data['content'].strip()) == 0:
            new_data['content'] = []
            new_data_em['content'] = []
            pass
        else:
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

    f.close()
    data_dump(datas, '../datasets/train_ner_no_emotion.pkl')
    data_dump(datas_em, '../datasets/train_ner_has_emotion.pkl')

    f = open('../data/coreEntityEmotion_test_stage2.txt', 'r')
    datas = []
    all_index = len(f.readlines())
    f.seek(0)
    for index, line in enumerate(f.readlines()):
        data = json.loads(line)
        print('{}/{}'.format(index, all_index))
        new_data = {}
        new_data['newsId'] = data['newsId']

        data['title'] = clean_text(data['title'].strip())
        if len(data['title']) > 125:
            data['title'] = data['title'][:125]
            print('warning:标题被截断!!')
        title = seg_char(data['title'])
        new_data['title'] = title
        data['content'] = clean_text(data['content'].strip())
        if len(data['content'].strip()) == 0:
            new_data['content'] = []
        else:
            if data['content'][-1] not in '。｡！!？?':
                data['content'] = data['content'] + '。'
            sentences = get_sentences(data['content'])
            sentences = seg_char_sents(sentences)
            content = []
            for sent in sentences:
                content.append(sent)
            new_data['content'] = content

        datas.append(new_data)

    f.close()
    data_dump(datas, '../datasets/test_ner2.pkl')