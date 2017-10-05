# -*- coding: utf-8 -*-
import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from string import maketrans

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.layers import SpatialDropout1D
import keras.preprocessing.text
from sklearn.metrics.pairwise import cosine_similarity


xml_fname = 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml'
xml_fname2 = 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml'
#xml_test = 'SemEval2016-Task3-CQA-QL-dev-subtaskA.xml'
xml_test = 'test_input.xml'



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

#Return only text, no changes, just a string
def only_text(a):
    nr = len(a)
    text = []
    for i in xrange(0, nr ):
        t= re.sub("[^a-zA-Z]", " ", a[i].get_text())
        #t = t.encode('ascii')
        words = t.split()
        
        text.append(" ".join(words))
    return text

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
                gr.append(1)
            else:
                gr.append(0)        
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
vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),
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

train_cos = similarity(train_quest_features.toarray(),train_features.toarray())
test_cos = similarity(test_quest_features.toarray(),test_features.toarray())
#---------LSTM---------
max_features = 2000   # limit the total number of words that we are interested 
                       # in modeling to the X most frequent words
EMBEDDING_DIM = 100
maxlen = 80            # Max answer || question length
batch_size = 32        # Number of comments used to space out weight updates
nb_classes = 2

train_l = np.array(label_train)
test_l = np.array(label_test)

tokenizer = Tokenizer(num_words=max_features)
text_only = only_text(train_c)

tokenizer.fit_on_texts(text_only)
sq_train = tokenizer.texts_to_sequences(train_array)
sq_test = tokenizer.texts_to_sequences(test_array)

qutrain_sq = tokenizer.texts_to_sequences(train_quest)
qutest_sq = tokenizer.texts_to_sequences(test_quest)

# -------- Trying to merge two layers and make the thrid

               
#new_train = mergeInfo(sq_train,qutrain_sq,train_cos)
#new_test = mergeInfo(sq_test,qutest_sq,test_cos)

#new_train = sequence(new_train)
#new_test = sequence(new_test)

# We need this because to train the embedding layer the length of the sequence has to be ct
# Embedding layers map from indices to vectors

print('We need to pad the comments to have the same length')
train_d = sequence.pad_sequences(sq_train, maxlen=maxlen)
test_d = sequence.pad_sequences(sq_test, maxlen=maxlen)

print('train_d shape:', train_d.shape)
print('test_d shape:', test_d.shape)

qutrain_sq = sequence.pad_sequences(qutrain_sq, maxlen=maxlen)
qutest_sq = sequence.pad_sequences(qutest_sq, maxlen=maxlen)

# Skip-gram, CBOW, and GloVe (or any other word2vec variant) are pre-trained 
# word embeddings which can be set as the weight of an embedding layer.
# If the weight of this layer (generally the first layer of the network) is not 
# initialized by these pre-trained vectors, the model/network itself would assign 
# random weights and will learn the embeddings (i.e. weights) on the fly.

print('--- Building model ---')


model = Sequential()
#model.add(Merge([sq_train,qutrain_sq,train_cos], mode='concat'))
model.add(Embedding(max_features, 128)) # 128 is the embedding_vecor_length 
                                        # It is the legth of the first layers' vectors
model.add(SpatialDropout1D(0.2))                     # SpatialDropout1D will help promote independence between feature maps
model.add(LSTM(128))                                 # 128 here is the number of 'neurons' used
model.add(Dense(1, activation='sigmoid'))            # because this is a classification problem 
                                                     # we use a Dense output layer with a single neuron 
                                                     # and a sigmoid activation function to make 0 or 1 
                                                     # predictions for the two classes

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

print('Train.........')


#new_train = mergInfo(train_d,qutrain_sq,train_cos)
#new_test = mergInfo(test_d,qutest_sq,test_cos)
#
#new_train = sequence(new_train)
#new_test = sequence(new_test)

model.fit(train_d, train_l ,batch_size=batch_size, epochs=1)
#model.fit(new_train, train_l, validation_data=(new_test,test_l) ,batch_size=batch_size, epochs=1)
          
#score, acc = model.evaluate(test_d, test_l,
                            #batch_size=batch_size)

#print('Test score:', score)
#print('Test accuracy:', acc)

print("Generating test predictions...")
preds = model.predict_classes(test_d, verbose=0)

f = open('/Users/lukacs_orsi/Desktop/Text_Based_information_retrival/Semieval/Assignment/TBIR_task/scorer/part2_out_test', 'w')

#print('prediction accuracy: ', accuracy_score(test_l, preds))

print('Printing to file...')

for i in range(0,len(q_id)):      
    create_line(i,i*10,q_id,a_id,preds,test_cos) 
f.close()
