import glob
import os
from bs4 import BeautifulSoup
import nltk
from itertools import chain
import json
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('popular')  
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import time
import pickle

OUTPUT_FOLDER_PATH = os.path.join("output") if (os.path.exists("output")) else os.mkdir("output")
count = 0
FORWARD_INDEX_FILE = "forward_index.json"
INVERTED_INDEX_FILE = "inverted_index.json"
DATASET_FOLDER = os.path.join("ft911")

def write_to_json(file_name,data):
    with open(os.path.join(OUTPUT_FOLDER_PATH,file_name), "w") as filePointer:
        json.dump(data,filePointer)

def convert_lower_case(doc_text):
    return doc_text.lower()

#forward indexing
def get_frequency_in_document(tokens):
    forward_word_frequency = nltk.FreqDist(tokens)
    return dict([(m, n) for m, n in forward_word_frequency.items()])

# function for inverted index
def get_inverted_index(clean_token_list,document_forward_index_dict):
    print("\nInverted index process\n\n")
    inverted_index_dict = {}
    for token in clean_token_list:
        print("Work in progress for : "+ token)
        document_word_dict = {}
        for key,value_dict in document_forward_index_dict.items():
            try:
                document_word_dict[key] = value_dict[token]
            except:
                pass
        inverted_index_dict[token] = document_word_dict
    return inverted_index_dict

def tokenize_and_cleaning(document_text):
    try:
        # remove all punctuations
        document_text_without_punctuation =  re.sub(r'[^\w\s]', ' ', document_text) 
        # Convert in tokens (split by white space)
        document_token_list = document_text_without_punctuation.split()
        # remove numbers and alphanumeric strings
        removed_numbers_tokens = [item for item in document_token_list if not any(data.isdigit() for data in item)]
        # get stop words from nltk
        stop_words_nltk = set(stopwords.words("english"))  
        # remove stop words
        removed_stopword_tokens = [word for word in removed_numbers_tokens if not word in stop_words_nltk]   
        # stem words using PorterStemmer
        ps = PorterStemmer() 
        stemmed_tokens = [ps.stem(word) for word in removed_stopword_tokens]
        return get_frequency_in_document(stemmed_tokens),stemmed_tokens
    except:
        return []

def serch_data(inverted_index_dict):
    search_flag = 'y'
    while(search_flag.lower() == 'y'):
        query = input("\nPlease enter the word to search : ")
        _,query_tokens = tokenize_and_cleaning(query)
        if(query_tokens):
            for token in query_tokens:
                if(token in inverted_index_dict.keys()):
                    print(token + " : " + str(inverted_index_dict[token]) + "\n")
                else: 
                    print(token + " : This word not found in any document\n")
        search_flag = input("\nDo you wanna continue search for other data ? enter y or n : ")

if __name__ == "__main__":
    document_dict = {}
    token_list = []
    token_dictionary = {}
    document_forward_index_dict = {}
    try:
        if(os.path.exists('result_object.pkl')):
            result_obj = pickle.load( open( "result_object.pkl", "rb" ) )
            print("---Processing time : {} seconds ---\n".format(result_obj["processing_time"]))
            serch_data(result_obj["inverted_index_dict"])
        else:
            start_time = time.time()
            # get file path and iterate over DATASET_FOLDER("ft911") to get file
            for infile in glob.glob(os.path.join(DATASET_FOLDER, '*')):
                # read files which contain document
                review_file = open(infile,'r').read()
                # Using Beautifulsoup library extract details
                soup = BeautifulSoup(review_file)
                # From file this will extract all data betwenn <DOC> and </DOC>
                document_list = soup.find_all('doc')
                if(len(document_list)>0):
                    # Iterate through all docment tag in single file
                    for idx,data in enumerate(document_list):
                        print("processing {} document of {}".format(idx,data.find('docno')))
                        # append processed tokens in token list
                        forward_index_data,tokens = tokenize_and_cleaning(data.find("text").get_text().lower())
                        document_forward_index_dict[data.find('docno').get_text()]= forward_index_data
                        token_list.append(tokens)
            token_list = list(chain.from_iterable(token_list))
            clean_token_list = sorted(token_list)
            clean_token_list = list(dict.fromkeys(clean_token_list))
            inverted_index_dict =get_inverted_index(clean_token_list,document_forward_index_dict)

            # Write data in json file
            write_to_json(FORWARD_INDEX_FILE,document_forward_index_dict)
            write_to_json(INVERTED_INDEX_FILE,inverted_index_dict)
            print("\n---Processing time : {} seconds ---\n".format(time.time() - start_time))
            result_obj = {
                "inverted_index_dict" : inverted_index_dict,
                "processing_time" : time.time() - start_time
            }
            pickle.dump( result_obj, open( "result_object.pkl", "wb" ) )
            # For user input and search the word
            serch_data(inverted_index_dict)
    except FileExistsError as fnf:
        print("please check file path")
    except Exception as e :
        print(e)
