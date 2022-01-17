from flask import Flask, render_template, request
import pickle
from gensim.models import Word2Vec
import numpy as np
import spacy
import re
import difflib
import plotly
from plotly import io
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import json

app = Flask(__name__)
#-----------------Article Search --------------------------#
def ranking(KeyWord):
    model = Word2Vec.load('../script/word2vec.model')
    KeyWords = preprocess(KeyWord)
    with open('../data/Doc2Vec.pkl', 'rb') as fpick:
        docVec = pickle.load(fpick)
    with open('../data/FullTitle.pkl', 'rb') as fpick:
        titles = pickle.load(fpick)
    find = []
    rankDoc = []
    if len(KeyWords) == 1 and KeyWords[0] not in model.wv.index_to_key:
       find = difflib.get_close_matches(KeyWords[0], model.wv.index_to_key, cutoff=0.7)
    else:
        wordVec = []
        for word in KeyWords:
            if word in model.wv.index_to_key:
                wordVec.append(model.wv.word_vec(word))
            else:
                wordVec.append(np.zeros(300))
        wordVec = np.mean(wordVec, axis=0)
        rank = []
        for doc in docVec:
            similar = np.dot(wordVec, doc)/(np.linalg.norm(wordVec)*np.linalg.norm(doc))
            rank.append(similar)
        rank_idx = np.argsort(-1*np.array(rank))
        rankDoc = [titles[idx] for idx in rank_idx]
    return rankDoc, find

def preprocess(KeyWord):
    words = [re.sub(r'[^a-z0-9|^-|^\']', '', word.lower()) for word in KeyWord.split()]
    nlp = spacy.load('en_core_web_sm')
    word_lemma = [word.lemma_ for word in nlp(' '.join(words))]
    return word_lemma

def cbow_process(model, user_input, n):
    result_word = []
    for words in user_input:
        sim_words = model.most_similar(words, topn = n)
        sim_words = append_list(sim_words, words)
        result_word.extend(sim_words)

    return result_word

def append_list(sim_words, words):
    list_of_words = []
    for i in range(len(sim_words)):  
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

@app.route("/", methods=['POST','GET'])
def index():
    with open('../data/full.pkl', 'rb')as fpick:
        full = pickle.load(fpick)
    with open('../data/Category.pkl', 'rb')as fpick:
        cate = pickle.load(fpick)
    if request.method == "POST" and request.form["Search"]!="":
        KeyWord = request.form["Search"]
        rankDoc, find = ranking(KeyWord)
        return render_template("index.html", full = full, KeyWord = KeyWord, rankDoc = rankDoc, find = find, cate = cate)
    else:
        KeyWord = ""
        return render_template("index.html", full = full, KeyWord = KeyWord, cate = cate)

@app.route("/show", methods=['POST','GET'])
def show():
    title = ""
    with open('../data/full.pkl', 'rb')as fpick:
        full = pickle.load(fpick)
    with open('../data/Category.pkl', 'rb')as fpick:
        cate = pickle.load(fpick)
    if request.method == "POST":
        title = request.form["Article"]
    return render_template("show.html", full = full, cate = cate, title = title, KeyWord = "")

@app.route("/overview", methods=['POST','GET'])
def overview():
    return render_template("overview.html", KeyWord = "")

#-----------------Article Search --------------------------#

#-----------------   Analysis    --------------------------#
@app.route("/about", methods=['POST','GET'])
def about():
    with open('../data/kmeans_result.pkl', 'rb')as fpick:
        km = pickle.load(fpick)
    model = Word2Vec.load("../script/word2vec_cbow.model")
    word_vectors_model = model.wv
    user_input = ['heart', 'coronavirus', 'pneumonia', 'fever', 'cough']
    topn = 15
    words = cbow_process(word_vectors_model, user_input, topn)
    similar_word = [word[0] for word in words]
    similar_word.extend(user_input)
    labels = [word[2] for word in words]
    label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    data_json= display_tsne_scatterplot_2D(word_vectors_model, user_input, similar_word, labels, color_map, topn, 5, 500, 10000)
        # 將圖表存成json
    dataJSON = json.dumps(data_json, cls=plotly.utils.PlotlyJSONEncoder)
    km_json= display_kmeans(km)
        # 將圖表存成json
    kmJSON = json.dumps(km_json, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("about.html", **locals(), KeyWord = "")

def display_kmeans(km):
    data = []
    cluster = ["Cluster 1: heart, ventricular, echocardiography, coronary, myocardial",
    "Cluster 2: cov, test, assay, sample, rt, detection",
    "Cluster 3: covid19, sars, pcr, coronavirus, infection",
    "Cluster 4: pneumoniae, mycoplasma, infection, pathogen, case",
    "Cluster 5: diagnosis, lung, clinical, chest, pulmonary, bacterial"]
    for i in range(len(km)):
        trace = go.Scatter(
            x = [item[0] for item in km[i]], 
            y = [item[1] for item in km[i]],
            text = [item[2] for item in km[i]],
            name = cluster[i],
            mode = 'markers',
            marker = {
                'size': 10,
                'opacity': 0.8,
                'color' : i
            },    
        )
        data.append(trace)

    layout = go.Layout(
        showlegend=True,
        legend=dict(
        x=0,
        y=0),
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        width = 1000,
        height = 1000
    )
    plot_figure = go.Figure(data = data, layout = layout)
    show_data = [plot_figure]

    return show_data

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
        width = 1000,
        height = 1000
    )

    plot_figure = go.Figure(data = data, layout = layout)
    # print(plot_figure)
    show_data = [plot_figure]
    # plot_figure.show()
    return show_data

#-----------------   Analysis    --------------------------#
if __name__ == "__main__":
    app.run(debug=False, host = '0.0.0.0', port='8080')