import sys
import os
import xml.etree.ElementTree as ET
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from nltk.corpus import stopwords, wordnet
import nltk
import argparse
import math
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly
import plotly.graph_objs as go
from gensim import corpora
from gensim.models import Word2Vec
from gensim.corpora import Dictionary

def combine_xml():
    data = []
    with open('./data/covid.pickle', 'rb') as file:
        covid = pickle.load(file)
    with open('./data/heartdisease.pickle', 'rb') as file:
        heartdisease = pickle.load(file)
    with open('./data/pneumonia.pickle', 'rb') as file:
        pneumonia = pickle.load(file)
    data = covid + heartdisease + pneumonia
    return data

def buile_vocabulary(data):
    vocabulary = []
    for text in data:
        for word in text:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary

def cosine_distance(model, word, target_list, num):
    cosine_dict = {}
    word_list = []
    a = model[word]
    for item in target_list:
        if item != word:
            b = model[item]
            cos_sim = dot(a, b)/(norm(a) * norm(b))
            cosine_dict[item] = cos_sim
    dist_sort = sorted(cosine_dict.items(), key = lambda dist: dist[1], reverse = True) # Decedning order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

def word_process(model, user_input, n):

    result_word = []
    for words in user_input:
        sim_words = model.most_similar(words, topn = n)
        sim_words = append_list(sim_words, words)
        result_word.extend(sim_words)
    print(result_word)
    return result_word

def append_list(sim_words, words):

    list_of_words = []
    for i in range(len(sim_words)):  
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

def display_closestwords(model, word, size):
    arr = np.empty((0, size), dtype = 'f')
    word_labels = [word]
    close_words = model.similar_by_word(word)
    # print(close_words)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for word_score in close_words:
        word_vector = model[word_score[0]]
        word_labels.append(word_score[0])
        arr = np.append(arr, np.array([word_vector]), axis = 0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

def display_tsne_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=None, perplexity = 0, learning_rate = 0, iteration = 0, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    word_vectors = np.array([model[w] for w in words])
    three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]
    
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]
    data = []
    count = 0
    for i in range (len(user_input)):
        trace = go.Scatter3d(
            x = three_dim[count:count+topn,0], 
            y = three_dim[count:count+topn,1],  
            z = three_dim[count:count+topn,2],
            text = words[count:count+topn],
            name = user_input[i],
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )
                
        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
        data.append(trace)
        count = count+topn
    
    trace_input = go.Scatter3d(
        x = three_dim[count:,0], 
        y = three_dim[count:,1],  
        z = three_dim[count:,2],
        text = words[count:],
        name = 'input words',
        textposition = "top center",
        textfont_size = 20,
        mode = 'markers+text',
        marker = {
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
    # Configure the layout
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()

def display_tsne_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, topn=None, perplexity = 0, learning_rate = 0, iteration = 0, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    word_vectors = np.array([model[w] for w in words])
    # For 2D, change the three_dim variable into something like two_dim like the following:
    two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]
    data = []
    count = 0
    for i in range (len(user_input)):
        trace = go.Scatter(
            x = two_dim[count:count+topn,0], 
            y = two_dim[count:count+topn,1], 
            text = words[count:count+topn],
            name = user_input[i],
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }
        )
        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
        data.append(trace)
        count = count+topn
    trace_input = go.Scatter(
        x = two_dim[count:,0], 
        y = two_dim[count:,1],  
        text = words[count:],
        name = 'input words',
        textposition = "top center",
        textfont_size = 20,
        mode = 'markers+text',
        marker = {
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
    # Configure the layout
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()

def main():
    data = combine_xml()
    # voc = buile_vocabulary(data)
    model = Word2Vec(data, min_count=5,vector_size= 100,workers=3, window =5)
    model.save("word2vec_cbow.model")
    model = Word2Vec.load("word2vec_cbow.model")
    # val = cosine_distance(word_vectors_model, 'diagnosis', voc, 5)
    # print(val)
    # print(word_vectors_model.most_similar('heartdisease', topn=10))
    word_vectors_model = model.wv
    user_input = ['heart', 'coronavirus', 'pneumonia', 'fever', 'cough']
    topn = 15
    words = word_process(word_vectors_model, user_input, topn)
    similar_word = [word[0] for word in words]
    
    similarity = [word[1] for word in words]
    similar_word.extend(user_input)
    labels = [word[2] for word in words]
    label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    # display_closestwords(word_vectors_model, 'covid-19', 100)
    display_tsne_scatterplot_2D(word_vectors_model, user_input, similar_word, labels, color_map, topn, 5, 500, 10000)


if __name__ == '__main__':
    main()