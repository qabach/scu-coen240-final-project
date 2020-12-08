"""
Santa Clara University

COEN 240 - Machine Learning

Final Project

Quan Bach
Anh Truong



By running this file, it will process all the dataset folder's  subdirectories.
Then it will create the:
-vocabulary set
-vocabulary dictionary
-statistics set
-document names list
-bag of words list

in the npy format

These files will be in the folder npy.


"""
import os
import re  #for regex to remove punctuations
import numpy as np
from utils import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize # for word and sentence tokennizing
from nltk.stem import WordNetLemmatizer # for lemmatization 



def procressData(directory):
    
    # initializing return stats
    count_of_documents          = 0
    count_of_sentences          = 0
    count_of_words              = 0
    count_of_unique_words       = 0


    # initialize return objects
    vocab_dictionary    = {}
    vocab_set           = set() # vocabulary
    docu_names          = [] # list of all document names
    bow_list            = [] # bag of words; this will be used to construct the incidence matrix afterward
    content_list        = [] # list that contains the files content for TF-IDF computation afterward
    
    # initialize stem
    word_lem =  WordNetLemmatizer()

    # walk through the entire directory
    print('Loading data...')
    for subdir, dirs, files in os.walk(directory):
        count_of_documents += len(files)
        for filename in files:
            if filename == '.DS_Store':
                continue
            docu_names.append(filename)
            print(os.path.join(subdir,filename))
            tmp_dict = {} # temp dict to contain the count of words of each file
            with open(os.path.join(subdir,filename),'rb') as input_file:
                database = input_file.readlines() # all lines of each file
                # Process each line
                for line in database:
                    # to skip files or words that cannot decode
                    # i.e. to avoid unicode decode error below
                    # ** UnicodeDecodeError: 'utf-8' codec can't decode byte 0xFF in position n: invalid start byte **
                    try:
                        line_list = word_tokenize(line.decode())
                    except UnicodeDecodeError:
                        continue
                    for word in line_list:
                        word = str(word).lower()
                        word = re.sub(r'[^\w\s]', '', word)         # remove puntuations with regex
                        word = word_lem.lemmatize(word)             # lemmatizing the word
                        if word not in stopwords and word != '':    # check if is a stop word or empty
                            count_of_words += 1
                            vocab_set.add(word)
                            if word not in vocab_dictionary.keys():
                                vocab_dictionary[word] = 1
                            else:
                                vocab_dictionary[word] = vocab_dictionary.get(word) + 1 #increment count if word is already in dictionary
                            
                            # same logic as above
                            if word not in tmp_dict.keys():
                                tmp_dict[word] = 1
                            else:
                                tmp_dict[word] = tmp_dict.get(word) + 1
            
            input_file.close()
            bow_list.append(tmp_dict)
            # read file again to count sentences
            with open(os.path.join(subdir,filename),'rb') as input_file:
                # to skip files  that cannot decode
                # i.e. to avoid unicode decode error below
                # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xFF in position n: invalid start byte
                try:
                    entire_file = input_file.read().decode() # read the whole file as once
                except UnicodeDecodeError:
                    continue
                # some clean up
                entire_file = entire_file.lower()
                entire_file = entire_file.replace("\n",'')
                content_list.append(entire_file)
           
                count_of_sentences += len(sent_tokenize(entire_file))
                
            input_file.close()

    count_of_unique_words = len(vocab_set)
    stats_set = (count_of_documents, count_of_sentences, count_of_words ,count_of_unique_words)
    
    print('\n' + 'Completed! ' + str(count_of_documents) + ' files loaded')
    return stats_set, vocab_dictionary, vocab_set, docu_names, bow_list, content_list 
                        

if __name__ == "__main__":
    if not os.path.exists('./npy/'):
            os.makedirs('./npy/')
            
    input_directory = './20news-18828' # input dataset folder
    stat_set, v_dict, v_set,docu_names, bow_list, content_list  = procressData(input_directory)
    stat_dict = {'number of documents'      : list(stat_set)[0],
                 'number of sentences'      : list(stat_set)[1],
                 'number of words'          : list(stat_set)[2],
                 'number of unique words'   : list(stat_set)[3]}
    
    np.save('./npy/vocabulary_dict.npy',v_dict)
    np.save('./npy/stats_dict.npy', stat_dict)
    np.save('./npy/vocabulary_set.npy',list(v_set))
    np.save('./npy/docu_names.npy',docu_names)
    np.save('./npy/bow_list.npy',bow_list)
    print ('Saved: '                + '\n' +
           '-vocabulary set'        + '\n' +
           '-vocabulary dictionary' + '\n' +
           '-statistics set'        + '\n' +
           '-document names list'   + '\n' +
           '-bag of words list'     + '\n' +
           '-content list'          + '\n' +
           'to npy')
    
    

    
    
    
