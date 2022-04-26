import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
from rank_bm25 import BM25Okapi
import Data_Iterator


def create_data_dict(): #create dict {queryid: querytfidf, docid, doctfidf...}

    test_data = pickle.load(open('test_data.pickle', 'rb'))
    dict_data = defaultdict(list)
    for data in test_data:
        dict_value = []
        for i in range(len(data)):
            dict_value.append(data[i])
        dict_data[data[0]].append(dict_value[1:])
    return dict_data

def cosine_similarity(dict_data):
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

def jaccard_similarity(tuples):

    # Implementing Jaccard algorithm for calculating similarity scores between two texts
    jaccard_similarity_val = {}

    for tuple in tuples:
        query = tuple[2] #get the query
        doc = tuple[3] #get the document

        query_intersection = query.intersection(doc)
        query_union = query.union(doc)
        similarity_score = len(query_intersection) / len(query_union)
        jaccard_similarity_val[tuple[0]] = similarity_score #use the query ID as the dictionary key

    return jaccard_similarity_val

if __name__ == "__main__":
    blank_data = Data_Iterator.create_blank_dataset('dev')
    print(type(blank_data))
    """
    dict_data = create_data_dict()
    cosine_similarity(dict_data)
    get_avg_precision('cosine_similarity_TEST_results')
    """