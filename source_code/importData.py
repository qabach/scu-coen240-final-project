"""
Santa Clara University 

COEN 240 - Machine Learning 

Final Project

Quan Bach
Anh Truong

By running this file, it will process the dataset folder all subdirectories. 
Then it will create the vocabularu set and vocabulary dictionary to store in numpy files.
These files will be in the folder npy.


"""
import os 
from utils import stopwords
import re  #for regex to remove punctuations
import numpy as np


def procressData(directory):
    # initializing stats  
    count_of_documents          = 0 
    count_of_sentences          = 0
    count_of_words              = 0 
    count_of_unique_words       = 0


    # initialize return dict and set 
    vocab_dictionary    = {}   
    vocab_set           = set()

    # walk through the entire directory 
    print('Loading data...')
    for subdir, dirs, files in os.walk(directory):
        count_of_documents += len(files)
        for filename in files:
            print(os.path.join(subdir,filename))
            with open(os.path.join(subdir,filename),'rb') as input_file:
                database = input_file.readlines() # all lines of each file 
                # Process each line 
                for line in database:
                    # to skip files or words that cannot decode 
                    # i.e. to avoid unicode decode error below
                    # ** UnicodeDecodeError: 'utf-8' codec can't decode byte 0xFF in position n: invalid start byte **
                    try:
                        line_list = line.decode().split()
                    except UnicodeDecodeError:
                        continue
                    for word in line_list:
                        word = str(word).lower()
                        word = re.sub(r'[^\w\s]', '', word) # remove puntuations with regex
                        if word not in stopwords and word != '': #check if is a stop word 
                            count_of_words += 1     
                            vocab_set.add(word)
                            if word not in vocab_dictionary.keys():
                                vocab_dictionary[word] = 1
                            else:
                                vocab_dictionary[word] = vocab_dictionary.get(word) + 1 #increment count if word is already in dictionary 
            input_file.close()
    
            # read file again to count sentences 
            with open(os.path.join(subdir,filename),'rb') as input_file:
                entire_file = input_file.read() # read the whole file as once 
                # sentence tends to end with . ! ? 
                # so we approx. the count by the count of string that ends with . or ! or ? 
                count_of_sentences += len(set(re.split(r'[.!?]+', str(entire_file)))) # increment the count of sentence by splitting entire file i.e. the whole text file, then count the length
            input_file.close()

    count_of_unique_words = len(vocab_set)
    stats_set = (count_of_documents, count_of_sentences, count_of_words ,count_of_unique_words)
    
    print('\n' + 'Loading data...Completed! ' + str(count_of_documents) + ' files loaded')
    return stats_set, vocab_dictionary, vocab_set
                        

if __name__ == "__main__": 
    if not os.path.exists('./npy/'):
            os.makedirs('./npy/')
            
    input_directory = './dataset'
    stat_set, v_dict, v_set = procressData(input_directory)
    stat_dict = {'number of documents'      : list(stat_set)[0],
                 'number of sentences'      : list(stat_set)[1],
                 'number of words'          : list(stat_set)[2],
                 'number of unique words'   : list(stat_set)[3]}
    
    np.save('./npy/vocabulary_dict.npy',v_dict)
    np.save('./npy/stats_dict.npy', stat_dict)
    np.save('./npy/vocabulary_set.npy',v_set)
    print ('Saved vocabulary set, vocabulary dictionary, statistics set to npy.')
    
    

    
    
    