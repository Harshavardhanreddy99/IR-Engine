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

OUTPUT_FOLDER_PATH = os.path.join("output") if (os.path.exists("output")) else os.mkdir("output")
count = 0
DOCUMENT_OUTPUT_FILE = "document.json"
TOKEN_OUTPUT_FILE = "token.json"
DATASET_FOLDER = os.path.join("ft911")

def write_to_json(file_name,data):
    with open(os.path.join(OUTPUT_FOLDER_PATH,file_name), "w") as filePointer:
        json.dump(data,filePointer)

def convert_lower_case(doc_text):
    return doc_text.lower()

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
        # remove duplicate tokens 
        clean_tokens = list(dict.fromkeys(stemmed_tokens))
        return clean_tokens
    except:
        return []

if __name__ == "__main__":
    document_dict = {}
    token_list = []
    token_dictionary = {}
    try:
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
                    token_list.append(tokenize_and_cleaning(data.find("text").get_text().lower()))
                    # add data in document dictionary
                    document_dict[data.find('docno').get_text()] = count
                    count = count + 1
        token_list = list(chain.from_iterable(token_list))
        clean_token_list = sorted(token_list)
        clean_token_list = list(dict.fromkeys(clean_token_list))
        token_dict = {key: value for value, key in enumerate(clean_token_list)}

        write_to_json(DOCUMENT_OUTPUT_FILE,document_dict)
        write_to_json(TOKEN_OUTPUT_FILE,token_dict)
        with open("parse_output.txt","w") as fp:
            for key,value in token_dict.items():
                fp.write(str(key) + "               " + str(value) +"\n")
            fp.write("..........................................\n")
            for key,value in document_dict.items():
                fp.write(str(key) + "               " + str(value) +"\n")
    except FileExistsError as fnf:
        print("please check file path")
    except Exception as e :
        print(e)
