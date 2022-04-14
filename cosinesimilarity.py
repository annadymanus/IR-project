import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
import numpy as np

def create_data_dict(): #create dict {queryid: querytfidf, docid, doctfidf...}

    test_data = pickle.load(open('test_data.pickle', 'rb'))
    dict_data = defaultdict(list)
    for data in test_data:
        dict_value = []
        for i in range(len(data)):
            dict_value.append(data[i])
        dict_data[data[0]].append(dict_value[1:])
    return dict_data

def compute_cosine_similarity(dict_data):
    dict_cosine_val = {}
    for key in dict_data.keys():
        cosine_results = []
        for document in dict_data[key]: #compute cosine similarity for the query with every document of the query
            query_tfidf = document[1].reshape(1, -1) #reshape to "fake" 2Dvector [] --> [[]]
            docu_tfidf = document[3].reshape(1, -1) #reshape to "fake" 2Dvector [] --> [[]]
            cos_val = cosine_similarity(query_tfidf, docu_tfidf)
            cosine_results.append([document[0], cos_val, document[-1]]) #add actual label from the input data (important for evaluation metrics)
        dict_cosine_val[key] = sorted(cosine_results, key=lambda x: x[1]) #sort doctuments by their cosine similarity for each key

        print("################## QUEST " + key + " SUCCESFULL ##################")
    with open('cosine_similarity_TEST_results', 'wb') as handle: #save cosinesimilarity results to a file using pickle
        pickle.dump(dict_cosine_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compute_jaccard(dict_data):
    dict_jaccard_val = {}
    for key in dict_data.keys():
        jaccard_results = []
        for document in dict_data[key]:  # compute cosine similarity for the query with every document of the query
            query_tfidf = document[1].reshape(1, -1)  # reshape to "fake" 2Dvector [] --> [[]]
            docu_tfidf = document[3].reshape(1, -1)  # reshape to "fake" 2Dvector [] --> [[]]
            query_tokens = set(query_tfidf.lower().split())
            doc_tokens = set(docu_tfidf.lower().split())
            jaccard_val = len(query_tokens.intersection(doc_tokens))/len(query_tokens.union(doc_tokens))
            jaccard_results.append([document[0], jaccard_val, document[-1]])  # add actual label from the input data (important for evaluation metrics)
        dict_jaccard_val[key] = sorted(jaccard_results, key=lambda x: x[1])  # sort doctuments by their cosine similarity for each key

        print("################## QUEST " + key + " SUCCESFULL ##################")
    with open('jaccard_TEST_results', 'wb') as handle:  # save cosinesimilarity results to a file using pickle
        pickle.dump(dict_jaccard_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_avg_precision(filename):
    cosine_dict = pickle.load(open(filename, 'rb'))
    avg_prec = 0
    for key, cosine_tuples in cosine_dict.items():        
        labels = np.array([x[-1] for x in cosine_tuples]).squeeze()
        sims = np.array([x[1] for x in cosine_tuples]).squeeze()        
        prec = average_precision_score(labels, sims)
        print(key, prec)
        avg_prec += prec/len(cosine_dict)
    print("AVERAGE PREC: ", avg_prec)

if __name__ == "__main__":
    dict_data = create_data_dict()
    compute_cosine_similarity(dict_data)
    compute_jaccard(dict_data)
    get_avg_precision('cosine_similarity_TEST_results')