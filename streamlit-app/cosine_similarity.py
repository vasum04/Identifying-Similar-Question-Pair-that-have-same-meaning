
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
print(stopwords.words('english'))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
nltk.download('wordnet')
from nltk import *

stop_words = set(stopwords.words('english'))
 

def fn_cos_sim(a,b):


  a = a.lower()
  b = b.lower()


# tokenization
  X_list = word_tokenize(a)
  Y_list = word_tokenize(b)

  # sw contains the list of stopwords
  sw = stopwords.words('english')
  l1 = []
  l2 = []

  # remove stop words from the string
  X_set = {w for w in X_list if not w in sw}
  Y_set = {w for w in Y_list if not w in sw}

  # form a set containing keywords of both strings
  rvector = X_set.union(Y_set)
  for w in rvector:
      if w in X_set:
        l1.append(1)  # create a vector
      else:
        l1.append(0)
      if w in Y_set:
        l2.append(1)
      else:
        l2.append(0)
  c = 0

  # cosine formula
  for i in range(len(rvector)):
      c += l1[i]*l2[i]
  cosine = c / float((sum(l1)*sum(l2))**0.5)
  return cosine
