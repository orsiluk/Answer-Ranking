""" Try II """

# -*- coding: utf-8 -*-
import re
import nltk

import numpy as np
import itertools
#from scipy import stats
#import pylab as pl
from sklearn import svm #, linear_model, cross_validation

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from string import maketrans
#from sklearn.externals import joblib
import pickle

import keras.preprocessing.text
from sklearn.metrics.pairwise import cosine_similarity


xml_fname = 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml'
xml_fname2 = 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml'
#xml_test = 'SemEval2016-Task3-CQA-QL-dev-subtaskA.xml'
xml_test = 'test_input.xml'



def toPairs(X,simans, y):
    X_trans = []
    y_trans = []
    if y !=0:
        y = np.asarray(y) # Turn input labels into array
        if y.ndim == 1:
            y = np.c_[y, np.ones(y.shape[0])] #concatonate the two arrays y and an array of ones
        #print('y.shape',y.shape)
        comb = itertools.combinations(range(X.shape[0]), 2) # Use iterator for efficiency and 
                                                            # create combinations of 2 of all the elements
        #print y
        for k, (i, j) in enumerate(comb):                   # For each element in the combination
                                                            # k is the number of combinations
            if y[i,0] == y[j,0] or y[i,1] != y[j,1]:   
                continue
            val = X[i,0] - X[j,0] + cos_sim(simans[i].reshape(1, -1),simans[j].reshape(1,-1))
            X_trans.append(val[0])
    
            if y[i, 0] == y[j, 0]:
                if cos_sim(simans[i].reshape(1, -1),simans[j].reshape(1,-1))>0: y_trans.append(y[i, 0])
                elif X[i,0] - X[j,0] > 0 : y_trans.append(1)
                else: y_trans.append(-1)
            
            else: y_trans.append(np.sign(y[i, 0] - y[j, 0]))
    
        return np.asarray(X_trans), np.asarray(y_trans).ravel() #, np.asarray(m)
    else:
        
        comb = itertools.combinations(range(X.shape[0]), 2) # Use iterator for efficiency and 
                                                            # create combinations of 2 of all the elements
        
        for k, (i, j) in enumerate(comb):                   # For each element in the combination
                                                            # k is the number of combinations
            val = X[i,0] - X[j,0] + cos_sim(simans[i].reshape(1, -1),simans[j].reshape(1,-1))
            X_trans.append(val[0])
    
        return np.asarray(X_trans),0
   
class RankSVM(svm.LinearSVC): # We use LinearSVC as a base and we override it
    """Pairwise ranking with an underlying LinearSVC model created by overriding 
    the svm.LinearSVC method
    """

    def fit(self, X,Xo, y):

        print('fit....')
        #print(len(X), len(y))
        i=0
        X_trans = []
        y_trans =[]
        while (i<X.shape[0]):
            #simans = (similarity(Xo[i:i+10],Xo[i:i+10]))
            xx,yy = toPairs(X[i:i+10],Xo[i:i+10],y[i:i+10])
            X_trans.extend(xx)
            y_trans.extend(yy)
            i +=10
        #print(len(X_trans),X_trans)
        X_trans = np.asarray(X_trans)
        y_trans = np.asarray(y_trans)
        #print(X_trans.shape,y_trans.shape)
        #simans = np.asarray(simans)
        
        #X_trans, y_trans = toPairs(X,simans, y)
        
        super(RankSVM, self).fit(X_trans, y_trans) # Override self -- Update
        return self, X_trans, y_trans

    #def decision_function(self, X):
    #    if np.count_nonzero(X) == 0 : return X
    #    else : return np.dot(X, self.coef_.ravel()) # Using the hinge loss function re-weight the array

    def predict(self, X):
        print('predicting .....')

        return np.dot(X, self.coef_.ravel()) # Use loss function to sort the array

    def score(self, X,Xo):

        print('scores........')
        X_trans,y_trans = toPairs(X,Xo,0)

        #X_trans = np.asarray(X_trans)
        #y_trans = np.asarray(y_trans)
        #X_trans, y_trans = toPairs(X, y)
        if sum(X_trans) == 0:
            #score = 0
            result = np.zeros(10)
        else: 
            #score =  np.mean(super(RankSVM, self).predict(X_trans) == y_trans)
            result = super(RankSVM, self).predict(X_trans)
        return  result 
             
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

def create_line(q,ans,qid,a_id,rank,res):
    # ans - which parameter to print from
    # q - index of the question
    for i in range(ans,ans+10):
        data = []
        #'"<Question_ID>   <Answer_ID>   <RANK>   <SCORE>   <LABEL>" '
        data.append(q_id[q])
        data.append(a_id[i])
        data.append(rank[i%10])
        c = res[i]
        data.append(c[0][0])
        #if(c[0][0]> 0):
        #    data.append('true')
        #else: 
        #    data.append('false')
        if(rank[i%10]> 4):
            data.append('true')
        else: 
            data.append('false')
        f.write('\t'.join(map(str,data)))
        f.write('\n')      
                     
                                        
        
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
    
    #print the performance of ranking

    rank_svm, x ,y = RankSVM().fit(np.asarray(cos_sim_train), train_features.todense(), label_train)

    print "training over"
    
    #print(rank_svm.score(np.asarray(cos_sim_test[0:10]),test_features.todense(), label_test[0:10]))
    
    #filename = '/Users/lukacs_orsi/Desktop/Text_Based_information_retrival/Part B/rank_svm_comp.sav'
    
    #pickle.dump(rank_svm, open(filename, 'wb'))
    #
    #print "outputed to file the trained model"
    
    #loaded_model = pickle.load(open(filename, 'rb'))
    #print 'loaded model'

    #fin = []
    listing = []
    i = 0

    while i<=len(cos_sim_test)-10:
        pairs = rank_svm.score(np.asarray(cos_sim_test[i:i+10]),test_features.todense()[i:i+10].reshape(-1, 1))
        listing.append(pairs)           

        i =i+10
    comb = itertools.combinations(range(10), 2)
    combinations=[]
    for k, (i, j) in enumerate(comb):
        combinations.append([i,j])
        
    listorder = []
    for i in range(len(listing)):
        line  = listing[i]
        order=[]
        for j in range(10):
            pos = 0
            for k in range(line.shape[0]):
                if combinations[k][0] == j and line[k]<0: pos +=1
                elif combinations[k][1] == j and line[k]>0: pos +=1
            order.append(pos)
        listorder.append(order)
    
    #print "Printing to file"
    #f = open('learningto_rank', 'w')
    #
    #for i in range(0,len(q_id)):      
    #    create_line(i,i*10,q_id,a_id,listorder[i],cos_sim_test) 
    #f.close()
