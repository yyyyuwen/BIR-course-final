from flask import Flask, render_template, request
import pickle
from gensim.models import Word2Vec
import numpy as np
import spacy
import re
import difflib

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
        return render_template("index.html", full = full, KeyWord = KeyWord)        
#-----------------Article Search --------------------------#

#-----------------   Analysis    --------------------------#
@app.route("/about", methods=['POST','GET'])
def about():
    return render_template("about.html")
#-----------------   Analysis    --------------------------#
if __name__ == "__main__":
    app.run(debug=True)