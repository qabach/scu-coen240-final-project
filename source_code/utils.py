"""
anta Clara University 

COEN 240 - Machine Learning 

Final Project

Quan Bach
Anh Truong

Utilities 

"""


stopword_filename = './utils/stopwords.txt'

try:
    stopwords = open(stopword_filename,'r').read().split('\n')
except FileExistsError():
    print('file ' + stopword_filename + 'not found in pwd')




