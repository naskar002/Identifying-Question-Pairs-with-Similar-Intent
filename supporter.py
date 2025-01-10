import re
from bs4 import BeautifulSoup
import numpy as np
import pickle
from nltk.stem import PorterStemmer
import nltk
import string
# tfidf = pickle.load(open('vectorizer_1.pkl','rb'))/
ps = PorterStemmer()
tf = pickle.load(open('tfidf.pkl','rb'))
new_sen = []

def text_preprocess(text):
  #lower casing
  text =  text.lower()
  #tokenize
  text = nltk.word_tokenize(text)

  new_text = []
  for i in text:
    if i.isalnum():
      new_text.append(i)

  text = new_text[:]
  new_text.clear()

  for word in text:
    if word not in nltk.corpus.stopwords.words('english') and word not in string.punctuation:
      new_text.append(word)

  text = new_text[:]
  new_text.clear()
  for word in text:
    new_text.append(ps.stem(word))

  return " ".join(new_text)

def common_words(q1,q2):
  q1_words = set(map(lambda x: x.lower().strip(),q1.split(" ")))
  q2_words = set(map(lambda x: x.lower().strip(),q2.split(" ")))
  return len(q1_words & q2_words)

def total_words(q1,q2):
  q1_words = list(map(lambda x: x.lower().strip(),q1.split(" ")))
  q2_words = list(map(lambda x: x.lower().strip(),q2.split(" ")))
  total_words = q1_words + q2_words
  return len(total_words)

def query_pt_creator(q1,q2):
  q1 = text_preprocess(q1)
  q2 = text_preprocess(q2)
  vec1 = tf.transform([q1]).toarray()
  vec2 = tf.transform([q2]).toarray()
  new_sen.append(vec1)
  new_sen.append(vec2)
  char_len_q1 = q1.str.len()
  char_len_q2 = q2.str.len()

  num_words_q1 = q1.apply(lambda x: len(x.split(" ")))
  num_words_q2 = q2.apply(lambda x: len(x.split(" ")))
  new_sen.append([char_len_q1,char_len_q2,num_words_q1,num_words_q2])
  cmn_word = common_words(q1,q2)
  final_words = total_words(q1,q2)
  word_share = cmn_word/final_words
  new_sen.append([cmn_word,final_words,word_share])
  return new_sen

  


