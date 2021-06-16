import glob
import os
from bs4 import BeautifulSoup
import nltk
from itertools import chain
import json
import nltk
import ssl
from collections import Counter
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
import pandas as pd
import numpy as np
import math

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
    data_frequency_dict = {}
    for token in clean_token_list:
        print("Work in progress for : "+ token)
        document_word_dict = {}
        for key,value_dict in document_forward_index_dict.items():
            try:
                document_word_dict[key] = value_dict[token]
            except:
                pass
        inverted_index_dict[token] = document_word_dict
        data_frequency_dict[token] = len(document_word_dict)
    return inverted_index_dict,data_frequency_dict

def tokenize_and_cleaning(document_text):
    try:
        # remove all punctuations
        document_text_without_punctuation =  re.sub(r'[^\w\s]', ' ', document_text) 
        # Convert in tokens (split by white space)
        document_token_list = document_text_without_punctuation.split()
        # remove numbers and alphanumeric strings
        removed_numbers_tokens = [item for item in document_token_list if not any(data.isdigit() for data in item)]
        # get stop words from nltk
        fp = open("stopwordlist.txt", "r")
        stop_words =[word.strip() for word in fp.read().split("\n")]
        fp.close()
        stop_words_nltk = set(stop_words)  
        # remove stop words
        removed_stopword_tokens = [word for word in removed_numbers_tokens if not word in stop_words_nltk]   
        # stem words using PorterStemmer
        ps = PorterStemmer() 
        stemmed_tokens = [ps.stem(word) for word in removed_stopword_tokens]
        return get_frequency_in_document(stemmed_tokens),stemmed_tokens
    except:
        return []

def clean_query(list_part):
    result = []
    for data in list_part:
        text = data.getText()
        if(":" in text):
            result.append(text.split(':', 1)[1].strip())
    return result

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def gen_vector(N,tokens,total_vocab,data_freq):
    Q = np.zeros((len(total_vocab)))
    counter = Counter(tokens)
    words_count = len(tokens)
    query_weights = {}
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        if(token in data_freq):
            df = data_freq[token] or 0
        else:
            df = 0
        idf = math.log((N+1)/(df+1))
        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q

def cosine_similarity(N,document_list, query, D, total_vocab, data_freq,query_id):
    print("Cosine Similarity")
    # preprocessed_query = preprocess(query)
    # tokens = word_tokenize(str(preprocessed_query))
    _,tokens = tokenize_and_cleaning(str(query))
    print("\nQuery:", query)
    print("")
    print(tokens)
    d_cosines = []
    query_vector = gen_vector(N,tokens,total_vocab,data_freq) 
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))   
    out = np.array(d_cosines)
    doc_number = [x for x in range(len(out))]
    df = pd.DataFrame()
    df["document_number"] = [document_list[x] for x in doc_number]
    df["query_id"] = query_id
    df["similarity_score"] = out
    print("")
    df = df.sort_values(by='similarity_score', ascending=False).reset_index()
    df["rank"] = [i+1 for i in range(len(df["document_number"]))]
    return (df)

def tf_idf_score(N,processed_text,data_freq):
    doc = 0
    tf_idf = {}
    keys_list = list(processed_text.keys())
    for key,value in processed_text.items():
        counter = Counter(value)
        words_count = len(value)
        for token in np.unique(value):
            tf = counter[token]/words_count
            df = data_freq[token]
            idf = np.log((N+1)/(df+1))
            tf_idf[keys_list.index(key), token] = tf*idf
    return tf_idf

def extract_query_to_df():
    query_df = pd.DataFrame()
    data =''
    with open('topics.txt', 'rb') as file: 
        data = file.read()
    soup = BeautifulSoup(data,features="html.parser")
    document_num = clean_query(soup.find_all('num'))
    document_title = [x.get_text().strip() for x in soup.find_all('title')]
    document_desc = clean_query(soup.find_all('desc'))
    document_narr = clean_query(soup.find_all('narr'))
    query_df["num"] = document_num
    query_df["title"] = document_title
    query_df["desc"] = document_desc
    query_df["narr"] = document_narr
    return query_df

# def serch_data(inverted_index_dict):
#     search_flag = 'y'
#     while(search_flag.lower() == 'y'):
#         query = input("\nPlease enter the word to search : ")
#         _,query_tokens = tokenize_and_cleaning(query)
#         if(query_tokens):
#             for token in query_tokens:
#                 if(token in inverted_index_dict.keys()):
#                     print(token + " : " + str(inverted_index_dict[token]) + "\n")
#                 else: 
#                     print(token + " : This word not found in any document\n")
#         search_flag = input("\nDo you wanna continue search for other data ? enter y or n : ")

def rank_document(result_obj):
    tf_idf = tf_idf_score(len(result_obj["token_list_2d"]),result_obj["document_token_dict"],result_obj["data_frequency_dict"])
    D = np.zeros((len(result_obj["token_list_2d"]),len(result_obj["data_frequency_dict"])))      
    total_vocab = [x for x in result_obj["data_frequency_dict"]]
    if(os.path.exists('D.pkl')):
        D = pickle.load( open( "D.pkl", "rb" ) )["D"]
    else:
        for i in tf_idf:
            try:
                ind = total_vocab.index(i[1])
                D[i[0]][ind] = tf_idf[i]
            except Exception as e:
                pass
        pickle.dump( {"D":D}, open( "D.pkl", "wb" ))

    # print(result_obj["token_list_2d"])
    query_df = extract_query_to_df()
    query_df["title_desc"] = query_df["title"] + " "+ query_df["desc"]
    query_df["title_narrative"] = query_df["title"] + " "+ query_df["narr"]
    document_name_list = [x for x in result_obj["document_token_dict"]]
    # Score for titles
    title_frame = pd.DataFrame()
    top_title_frame = pd.DataFrame()
    title_desc_frame = pd.DataFrame()
    top_title_desc_frame = pd.DataFrame()
    title_narr_frame = pd.DataFrame()
    top_title_narr_frame = pd.DataFrame()
    for i in range(len(query_df)):
        title_frame = title_frame.append(cosine_similarity(len(result_obj["token_list_2d"]),document_name_list,query_df["title"][i],D,total_vocab,result_obj["data_frequency_dict"],query_df["num"][i]))
        top_title_frame = top_title_frame.append(cosine_similarity(len(result_obj["token_list_2d"]),document_name_list,query_df["title"][i],D,total_vocab,result_obj["data_frequency_dict"],query_df["num"][i]).head(10))
    for i in range(len(query_df)):
        title_desc_frame = title_desc_frame.append(cosine_similarity(len(result_obj["token_list_2d"]),document_name_list,query_df["title_desc"][i],D,total_vocab,result_obj["data_frequency_dict"],query_df["num"][i]))
        top_title_desc_frame = top_title_desc_frame.append(cosine_similarity(len(result_obj["token_list_2d"]),document_name_list,query_df["title_desc"][i],D,total_vocab,result_obj["data_frequency_dict"],query_df["num"][i]).head(10))
    for i in range(len(query_df)):
        title_narr_frame = title_narr_frame.append(cosine_similarity(len(result_obj["token_list_2d"]),document_name_list,query_df["title_narrative"][i],D,total_vocab,result_obj["data_frequency_dict"],query_df["num"][i]))
        top_title_narr_frame = top_title_narr_frame.append(cosine_similarity(len(result_obj["token_list_2d"]),document_name_list,query_df["title_narrative"][i],D,total_vocab,result_obj["data_frequency_dict"],query_df["num"][i]).head(10))
    title_frame = title_frame.reset_index()[['query_id','document_number','rank','similarity_score']]
    title_desc_frame = title_desc_frame.reset_index()[['query_id','document_number','rank','similarity_score']]
    title_narr_frame = title_narr_frame.reset_index()[['query_id','document_number','rank','similarity_score']]
    top_title_frame.reset_index()[['query_id','document_number','rank','similarity_score']].to_csv(OUTPUT_FOLDER_PATH+'/title_similarity_output.txt', header=None, index=None, sep='\t', mode='w')
    top_title_desc_frame.reset_index()[['query_id','document_number','rank','similarity_score']].to_csv(OUTPUT_FOLDER_PATH+'/title_desc_similarity_output.txt', header=None, index=None, sep='\t', mode='w')
    top_title_narr_frame.reset_index()[['query_id','document_number','rank','similarity_score']].to_csv(OUTPUT_FOLDER_PATH+'/title_narrative_similarity_output.txt', header=None, index=None, sep='\t', mode='w')
    qrels_df = qrels_to_df()
    print("For Titles")
    precision_and_recall(qrels_df,title_frame.loc[title_frame["similarity_score"]>0])
    print("\nFor Titles + Description")
    precision_and_recall(qrels_df,title_desc_frame.loc[title_desc_frame["similarity_score"]>0])
    print("\nFor Titles + Narrative")
    precision_and_recall(qrels_df,title_narr_frame.loc[title_narr_frame["similarity_score"]>0])

def qrels_to_df():
    qrels_data = []
    with open("main.qrels", "r") as filePointer:
        lines = filePointer.readlines()
        for line in lines:
            if((line.startswith('352')) 
            or line.startswith('353') 
            or line.startswith('354') 
            or line.startswith('359')):
                qrels_data.append(line.split())
    qrels_df = pd.DataFrame(qrels_data,columns=['q_id','num','document_id','relevance'])
    qrels_df = qrels_df.loc[qrels_df['relevance'] == '1'] 
    return qrels_df

def precision_and_recall(qrels_df,retrived_docs_df):
    query_id_list = ["352","353","354","359"]
    precision = []
    recall = []
    for q_id in query_id_list:
        relevent_retrived_documents = 0
        relevent_documents_qrels = qrels_df.loc[qrels_df["q_id"]==q_id]["document_id"]
        retrived_documents = retrived_docs_df.loc[retrived_docs_df["query_id"]==q_id]["document_number"].tolist()
        for doc in relevent_documents_qrels:
            if(doc in retrived_documents):
                relevent_retrived_documents = relevent_retrived_documents+1
        print("relevent retrived documents for query"+ q_id +" :"+ str(relevent_retrived_documents))
        precision.append(relevent_retrived_documents/len(retrived_documents))
        recall.append(relevent_retrived_documents/len(relevent_documents_qrels))
    precision_recall_df = pd.DataFrame()
    precision_recall_df["query_id"] = query_id_list
    precision_recall_df["precision"] = precision
    precision_recall_df["recall"] = recall
    print(precision_recall_df)

if __name__ == "__main__":
    document_dict = {}
    document_tokens_dict = {}
    token_dictionary = {}
    document_forward_index_dict = {}
    token_list_2d = []
    token_list = []
    try:
        if(os.path.exists('result_object.pkl')):
            result_obj = pickle.load( open( "result_object.pkl", "rb" ) )
            print("---Processing time : {} seconds ---\n".format(result_obj["processing_time"]))
            rank_document(result_obj)
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
                        #token_list_2d.append([x for x in forward_index_data])
                        document_tokens_dict[data.find('docno').get_text()] = tokens
                        document_forward_index_dict[data.find('docno').get_text()]= forward_index_data
                        token_list.append(tokens)
            token_list_2d = token_list[:]
            token_list = list(chain.from_iterable(token_list))
            clean_token_list = sorted(token_list)
            clean_token_list = list(dict.fromkeys(clean_token_list))
            inverted_index_dict,data_frequency_dict =get_inverted_index(clean_token_list,document_forward_index_dict)

            # Write data in json file
            write_to_json(FORWARD_INDEX_FILE,document_forward_index_dict)
            write_to_json(INVERTED_INDEX_FILE,inverted_index_dict)
            write_to_json("DF.json",data_frequency_dict)
            print("\n---Processing time : {} seconds ---\n".format(time.time() - start_time))
            result_obj = {
                "document_token_dict" : document_tokens_dict,
                "token_list_2d" : token_list_2d,
                "inverted_index_dict" : inverted_index_dict,
                "data_frequency_dict" : data_frequency_dict,
                "processing_time" : time.time() - start_time
            }
            pickle.dump( result_obj, open( "result_object.pkl", "wb" ) )
            rank_document(result_obj)
    except FileExistsError as fnf:
        print("please check file path")
    # except Exception as e :
    #     print(e)