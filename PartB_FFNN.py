# -*- coding: utf-8 -*-

""" Neural network approach of part B"""

import re
import nltk

import numpy as np
import itertools


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from string import maketrans
#from sklearn.externals import joblib
import pickle


#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers.embeddings import Embedding
#from keras.layers.recurrent import LSTM
#from keras.preprocessing.text import Tokenizer
#from keras.layers import SpatialDropout1D
import keras.preprocessing.text
#from keras import backend
from keras.layers import Activation
#from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf


xml_fname = 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml'
xml_fname2 = 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml'
xml_test = 'SemEval2016-Task3-CQA-QL-dev-subtaskA.xml'


def transform_pairwise(X, y):
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    print(y.shape)
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i,0] - X[j,0])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        
    print('Done transform --------------------------')
    return np.asarray(X_new), np.asarray(y_new).ravel()
    
# Had to write this work around because of an error in keras which gave an error 
# when running tokenizer.texts_to_sequences() -- source: https://github.com/fchollet/keras/issues/1072

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]
    
keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

#Preprocess text 
def clean_text(a):
    nr = len(a)
    text = []
    for i in xrange(0, nr ):
        letters = re.sub("[^a-zA-Z]", " ", a[i].get_text()) 
        words = letters.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        text.append(" ".join(map(lambda x: lemmatizer.lemmatize(x), meaningful_words)))

    return(text)

#Obtain labels either a boolean value, when we want relevancy, or the actual albel            
def get_label(test_c,label,expected):
    gr = []
    if (expected == 'value'):
        for i in range(0,len(test_c)):
            lab = test_c[i].get(label,[])
            gr.append(lab)
    elif (expected == 'bool'):
        for i in range(0,len(test_c)):
            lab = test_c[i].get(label,[])
            if (lab  == 'Good'):
                gr.append(2)
            elif (lab == 'Bad'):
                gr.append(0)
            else:
                gr.append(1)        
    return gr
    
#Get cosine similarity
def cos_sim(q,a):
    return cosine_similarity(q, a)    

#Get similarity between the questions and answers and create an array
def similarity(q,a):
    sim = []
    n=len(q)
    for i in range(0,n):
        for j in range (0,10):
            sim.append(cos_sim(q[i].reshape(1, -1),a[i*10+j].reshape(1,-1)))    
    return sim

def create_line(q,ans,qid,a_id,res,simil):
    # ans - which parameter to print from
    # q - index of the question
    
    sort = sorted(range(ans,ans+10), key=lambda k: res[k])
    #print(sort)
    
    for i in range(ans,ans+10):
        data = []
        #'"<Question_ID>   <Answer_ID>   <RANK>   <SCORE>   <LABEL>" '
        data.append(q_id[q])
        data.append(a_id[i])
        data.append(sort[i%10])
        data.append(res[i][0])
        c = simil[i]
        if(c[0][0] == 0):
            data.append('false')
        else: 
            data.append('true')
        f.write('\t'.join(map(str,data)))
        f.write('\n')

#Tried to have Questions ans input as well, but for now couldn't figure out
#how to merge the three layers, question vector, asnwer vector and similarity
#in a way to be accepted by the model
def mergeInfo(ans,quest,dist):
    new = []
    
    for i in range(len(quest)):
       for j in range (0,10):
           new.append(keras.layers.merge.Concatenate([quest[i],ans[i*10+j]],dist[i*10+j][0][0]))  
    return new        
                     
                                        
        
#-----------MAIN--------------#
if __name__ == '__main__':
    openit = open(xml_fname,'r')
    soup = BeautifulSoup(openit, "xml")
    
    opentest = open(xml_test,'r')
    test_soup = BeautifulSoup(opentest, "xml")
    
    lemmatizer = WordNetLemmatizer()
    
    train_c = soup.find_all('RelComment') # Find all comments in the XML file
    train_q = soup.find_all('RelQuestion')
    train_array = clean_text(train_c)
    train_quest = clean_text(train_q)
    
    test_c = test_soup.find_all('RelComment')
    test_q = test_soup.find_all('RelQuestion')
    
    test_array = clean_text(test_c)
    test_quest = clean_text(test_q)
    vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 2000, ngram_range = ( 1, 4 ),
                                sublinear_tf = True )
    
    label_train = get_label(train_c,'RELC_RELEVANCE2RELQ','bool')   
    label_test = get_label(test_c,'RELC_RELEVANCE2RELQ','bool')
    q_id = get_label(test_q,'RELQ_ID','value')
    a_id = get_label(test_c,'RELC_ID','value')


    vectorizer = vectorizer.fit(train_array)
    train_features = vectorizer.transform(train_array)
    test_features = vectorizer.transform(test_array)
    
    train_quest_features = vectorizer.transform(train_quest)
    test_quest_features = vectorizer.transform(test_quest)
    
    
    cos_sim_train = similarity(train_quest_features.toarray(),train_features.toarray())
    cos_sim_test = similarity(test_quest_features.toarray(),test_features.toarray())
    
    ############################################################
    ##train = pickle.load(open('/Users/lukacs_orsi/Desktop/Text_Based_information_retrival/Part B/transformed_train', 'rb'))
    #x_train, y_train = transform_pairwise(np.asarray(cos_sim_train), label_train)
    #x_train = tolist(x_train)
    #y_train = tolist(y_train)
    #model = Sequential()
    #model.add(Dense(12, input_dim=62304083, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    #
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ## Fit the model
    #model.fit(x_train, y_train, epochs=150, batch_size=10)
    #x_test,y_test = transform_pairwise(np.asarray(cos_sim_test), label_test)
    #scores = model.evaluate(x_test,y_test)
    ############################################################
    
    x_traino, y_traino = transform_pairwise(np.asarray(cos_sim_train), label_train)
    x_testo, y_testo = transform_pairwise(np.asarray(cos_sim_test), label_test)
    INPUT_DIM = x_traino.shape
    
    doc_train = tf.convert_to_tensor(x_traino)
    #doc_train = tf.cast(doc_train, tf.float32)
    #doc_test = tf.convert_to_tensor(x_test)
    #doc_test = tf.cast(doc_test, tf.float32)
    #
    y_train = tf.convert_to_tensor(y_traino)
    #y_train = tf.cast(y_train, tf.float32)
    #y_test = tf.convert_to_tensor(y_test)
    #y_test = tf.cast(y_test, tf.float32)
    
    model = Sequential() 
    ######model.add(Embedding(max_features, 128))
    model.add(Dense(150, input_shape=(1,), init="uniform", activation="relu"))   
    model.add(Dense(50, init="uniform", activation = "relu"))
    #model.add(Dense(25, init="uniform",activation = "relu"))
    model.add(Dense(10, init="uniform",activation = "relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    # train the model using SGD
    print("[INFO] compiling model...")
    #sgd = SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])
                  
    #model.fit(doc_train, y_train, nb_epoch=10, batch_size=100,verbose = 1)
    model.fit(x_traino, y_traino, nb_epoch=1, batch_size=10,verbose = 1)
    
    #print "training over"
    
    filename = 'rank_ffnn.sav'
    
    #pickle.dump(rank_svm, open(filename, 'wb'))
    
    #print "outputed to file the trained model"
    
    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(x_testo, y_testo, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    # Train model.
    #NUM_EPOCHS = 10
    #BATCH_SIZE = 10
    
    #history = model.fit(doc_test, y_test, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, verbose = 1)
    
    # Generate scores from document/query features.
    #get_score = backend.function(doc_train)
    #get_score(x_test)
    ############################################################
    
