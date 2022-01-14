#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leeyuwen
"""
import sys
import os
import xml.etree.ElementTree as ET
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from nltk.corpus import stopwords, wordnet
import nltk
import argparse
import math
import itertools
from gensim import corpora
from gensim.corpora import Dictionary

def read_xmlfile(path):
    tree = ET.parse(path)
    root = tree.getroot()
    Article = root.findall("PubmedArticle")
    text_list = []
    doc = {}
    for elem in Article:
        article_text = {}
        title = elem.find("MedlineCitation").find("Article").find("ArticleTitle").text
        for article in elem.find("MedlineCitation").find("Article").findall("Abstract"): #內文位置 
            if (article.find('AbstractText').text):
                abs = article.find('AbstractText')
                if 'Label' in abs.attrib:
                    article_text[abs.attrib['Label']] =  abs.text
                else:
                    article_text['Abstract'] =  abs.text
            else:
                continue
        doc[title] = article_text
    '''將空內文刪掉'''
    del_list = []
    for item in doc.items():
        if item[1] == {}:
            del_list.append(item[0])
    for item in del_list:
        del doc[item]
    doc = dict(itertools.islice(doc.items(), 1000)) #只取前一千篇文章
    for title, article in doc.items():
        for label, text in article.items():
            text_list.append(title + text)
    
    return doc, text_list

def word_preprocess(text_list):
    words = []
    for idx, articles in enumerate(text_list):
        word = text2word(articles) # sent2word
        word = clean_word(word) #stop word
        word = lemma(word) # lemma
        words.append(word)
        # text = [str(w) for word in words for w in word]
    return words

'''字串變成單字'''
def text2word(text): 
    words = []
    split_word = text.split(']')
    text = ' '.join(str(x).lower() for x in split_word)
    split_word = text.split("'s")
    for word in split_word:
        words = re.split(r'[!(\')#$"&…%^*+,-./{}[;:<=>?@~ \　]+', word)
    return words[:-1]

'''stop word'''
def clean_word(text):
    sentences = [re.sub(r'[^a-z0-9]', ' ', sent.lower()) for sent in text]
    clean_words = []
    for sent in sentences:
        words = [word for word in sent.split() if not word.replace('-', '').isnumeric()]
        words = stop_word(words)
        if(words):
            clean_words.append(' '.join(words))
    return clean_words

'''將sentences 切成 words'''
def splitsent2words(text):
    tokens = [x.split() for x in text]
    return tokens

def porter(words):
    ps = PorterStemmer()
    porter_list = []
    for word in words:
        porter_list.append(ps.stem(word))

    return porter_list

def stop_word(words):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in words if not w.lower() in stop_words]
    return filtered_sentence

def lemma(sentences):
    # nltk.download('omw-1.4')
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()
    lemma_word = [lemmatizer.lemmatize(sentence, get_wordnet_pos(sentence)) for sentence in sentences]
    return lemma_word



'''字數count'''
def word_count(text):
    article_voc = {}
    for idx, sent in enumerate(text.items()):
        vocabulary = {}
        for words in sent[1].items():
            for word in words[1]:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
        article_voc[idx] = vocabulary
    return article_voc

def tf_idf(word, voc_list, idx_word):
    count_sum = 0
    for article in voc_list.items():
        count_sum += article[1] # 字總數
    return tf(word, voc_list, count_sum) * idf(word, idx_word)

def tf(word, voc_list, count_sum): # word count / 總count
    word_count = 0
    if word in voc_list:
        word_count = voc_list[word]
    return word_count / count_sum

'''這個字出現在幾篇文章中'''
def idf(word, idx_word):
    article_count = len(idx_word)
    word_count = 0
    for words in idx_word:
        if word in words:
            word_count += 1
    return math.log(article_count / word_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True, help="Filename")
    args = parser.parse_args()
    Filename = args.filename
    path = f'./xml/{Filename}_1000.xml'
    save_path = f'./data/{Filename}.pickle'

    doc, text_list = read_xmlfile(path) #doc: {title :{label, text}}, text_list : [title + label + text]
    words = word_preprocess(text_list)
    # id2word = corpora.Dictionary(words)
    
    with open(save_path, 'wb')as fpick:
        pickle.dump(words, fpick)

if __name__ == '__main__':
    main()