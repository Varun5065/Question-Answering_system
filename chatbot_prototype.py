# -*- coding: utf-8 -*-

import numpy as np
import nltk
import string
import random
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import drive
drive.mount('/content/drive')

f = open('/content/data.txt', 'r', errors = 'ignore')
doc = f.read()

doc = doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
sentence_token = nltk.sent_tokenize(doc)
word_token = nltk.word_tokenize(doc)

def greeting_response(text):
  text = text.lower()
  bot_greetings = ['hi', 'hello']
  user_greetings = ['hi', 'hello']
  for word in text.split():
    if word in user_greetings:
      return random.choice(bot_greetings)

def index_sort(list_v):
  length = len(list_v)
  list_index = list(range(0, length))
  x = list_v
  for i in range(length):
    for j in range(length):
      if x[list_index[i]] > x[list_index[j]]:
        list_index[i], list_index[j] = list_index[j], list_index[i]
  return list_index

def bot_response(user_input):
  user_input = user_input.lower()
  sentence_token.append(user_input)
  bot_response = ''
  cm = CountVectorizer().fit_transform(sentence_token)
  print(cm)
  similarity_scores = cosine_similarity(cm[-1], cm)
  similarity_scores_list = similarity_scores.flatten()
  index = index_sort(similarity_scores_list)
  index = index[1:]
  response_flag = 0

  j = 0
  for i in range(len(index)):
    if similarity_scores_list[index[i]] > 0.0:
      bot_response = bot_response+' '+sentence_token[index[i]]
      response_flag = 1
      j += 1
    if j > 2:
      break
  if response_flag == 0:
    bot_response = bot_response+' '+'Sorry, I cannot understand what you are saying'
  sentence_token.remove(user_input)
  return bot_response

print('Hey I am the doc bot. How may I assist you today?')
print('If you wish to end this convrsation, type bye')
exit_list = ['exit', 'bye']
while True:
  user_input = input('User:')
  if user_input.lower() in exit_list:
    print('Ok. See you later')
    break
  else:
    if greeting_response(user_input) != None:
      print('Doc bot: '+greeting_response(user_input))
    else:
      print('Doc bot: '+bot_response(user_input))



