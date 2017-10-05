from __future__ import division, unicode_literals
#from textblob import TextBlob
#import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords # Import the stop word list
import re
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

#import sys
#import os
#
#class tfidf:
#  def __init__(self):
#    self.weighted = False
#    self.documents = []
#    self.corpus_dict = {}
#
#  def addDocument(self, doc_name, list_of_words):
#    # building a dictionary
#    doc_dict = {}
#    for w in list_of_words:
#      doc_dict[w] = doc_dict.get(w, 0.) + 1.0
#      self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0
#
#    # normalizing the dictionary
#    length = float(len(list_of_words))
#    for k in doc_dict:
#      doc_dict[k] = doc_dict[k] / length
#
#    # add the normalized document to the corpus
#    self.documents.append([doc_name, doc_dict])
#
#  def similarities(self, list_of_words):
#    """Returns a list of all the [docname, similarity_score] pairs relative to a list of words."""
#
#    # building the query dictionary
#    query_dict = {}
#    for w in list_of_words:
#      query_dict[w] = query_dict.get(w, 0.0) + 1.0
#
#    # normalizing the query
#    length = float(len(list_of_words))
#    for k in query_dict:
#      query_dict[k] = query_dict[k] / length
#
#    # computing the list of similarities
#    sims = []
#    for doc in self.documents:
#      score = 0.0
#      doc_dict = doc[1]
#      for k in query_dict:
#        if k in doc_dict:
#          score += (query_dict[k] / self.corpus_dict[k]) + (doc_dict[k] / self.corpus_dict[k])
#      sims.append([doc[0], score])
#
#    return sims
    
    
xml_fname = 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml'
xml_fname2 = 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml'
xml_test = 'SemEval2016-Task3-CQA-QL-dev-subtaskA.xml'
#xml_test = 'test_input.xml'
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(analyzer = "word") 
 
#/////////////////////////////---------                                                         
def split_into_lemmas(message):
    #message = unicode(message, 'utf8').lower()
    words = tb(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

                 
#/////////////////////////////---------                 
def clean_text(a):
    nr = len(a)
    text = []
    for i in xrange(0, nr ):
        letters = re.sub("[^a-zA-Z]", " ", a[i].get_text()) 
        words = letters.lower().split()
        stops = set(stopwords.words("english"))
        useful_w = [w for w in words if not w in stops]
        text.append(" ".join(map(lambda x: lemmatizer.lemmatize(x), useful_w)))
    return(text)


#/////////////////////////////---------
def only_text(a):
    nr = len(a)
    text = []
    for i in xrange(0, nr ):
        t= re.sub("[^a-zA-Z]", " ", a[i].get_text())
        #t = t.encode('ascii')
        words = t.split()
        
        #text.append(" ".join(words))
        text.append(words)
    return text


#/////////////////////////////----------
def get_label(test_c,label):
    gr = []
    for i in range(0,len(test_c)):
        #print(nq[i].get('RELC_RELEVANCE2RELQ',[]))
        lab = test_c[i].get(label,[])
        if (lab  == 'PotentiallyUseful'):
            gr.append('Bad')
        else:
            gr.append(lab)
    return gr
  
    
#/////////////////////////////----------  
def create_line(q,ans):
    # ans - which parameter to print from
    # q - index of the question
    res = []
    
    for i in range (ans,ans+10):
        res.append(cosine_similarity(tfidf_questions[q], tfidf_answers[i]))
    sort = sorted(range(len(res)), key=lambda k: res[k])
    for i in range(ans,ans+10):
        data = []
        #'"<Question_ID>   <Answer_ID>   <RANK>   <SCORE>   <LABEL>" '
        data.append(q_id[q])
        data.append(a_id[i])
        data.append(sort[i % 10])
        r=res[i % 10]
        data.append(r[0][0])
        data.append('true')
        f.write('\t'.join(map(str,data)))
        f.write('\n')
            
#----------////////////////////////MAIN//////////////////////////----------            
openit = open(xml_fname,'r')
soup = BeautifulSoup(openit, "xml")
a = soup.find_all('RelComment') # Find all comments in the XML file

opentest = open(xml_test,'r')
test_soup = BeautifulSoup(opentest, "xml")
test_c = test_soup.find_all('RelComment')
test_q = test_soup.find_all('RelQuestion')

start_time = datetime.datetime.now()

train_array = clean_text(a)
print('Time elapsed:', datetime.datetime.now() - start_time)

bag_of_words = vectorizer.fit(train_array) 

grade = get_label(test_c,'RELC_RELEVANCE2RELQ')
q_id = get_label(test_q,'RELQ_ID')
a_id = get_label(test_c,'RELC_ID')

test_array = clean_text(test_c)
train_bag = bag_of_words.transform(train_array)
tesing_bag = bag_of_words.transform(test_array)

tfidf_transformer = TfidfTransformer().fit(tesing_bag)

quest_array = clean_text(test_q)
quest_bag = bag_of_words.transform(quest_array)

tfidf_transformer = TfidfTransformer().fit(train_bag)
tfidf_answers = tfidf_transformer.transform(tesing_bag)   #Tf-idf representation of answers
tfidf_questions = tfidf_transformer.transform(quest_bag)  #Tf-idf representation of questions
#
#
print('Time elapsed:', datetime.datetime.now() - start_time)
#
#------- Print results to a file ------

f = open('/Users/lukacs_orsi/Desktop/Text_Based_information_retrival/Semieval/Assignment/TBIR_task/scorer/out', 'a')

for i in range(0,len(q_id)):      
    create_line(i,i*10) 
     
print('Time elapsed:', datetime.datetime.now() - start_time)
f.close()
