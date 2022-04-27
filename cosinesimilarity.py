import Preprocessing
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
import Data_Iterator


def create_data_dict(): #create dict {queryid: querytfidf, docid, doctfidf...}
    test_data = pickle.load(open('train_tf_idf.pickle', 'rb'))
    return test_data

def compute_cosine_similarity(data):
    cosine_results = {}
    for task in data:
        cosine_results[task[0]] = {}
    for task in data:
        query_tfidf = task[2].reshape(1, -1) #reshape to "fake" 2Dvector [] --> [[]]
        docu_tfidf = task[3].reshape(1, -1) #reshape to "fake" 2Dvector [] --> [[]]
        cos_val = cosine_similarity(query_tfidf, docu_tfidf)
        cosine_results[task[0]][task[1]] = cos_val #add actual label from the input data (important for evaluation metrics)
        print("################## COSINE QUEST " + task[0] + " SUCCESFULL ##################")
    with open('cosine_similarity_TEST_results', 'wb') as handle:  # save cosinesimilarity results to a file using pickle
        pickle.dump(cosine_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("COSINE SIMILARITY FINISHED")

    """
    for result in cosine_results:
        dict_cosine_val[result[0]] = sorted(cosine_results, key=lambda x: x[1]) #sort doctuments by their cosine similarity for each key

       
    with open('cosine_similarity_TEST_results', 'wb') as handle: #save cosinesimilarity results to a file using pickle
        pickle.dump(dict_cosine_val, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

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

def compute_jaccard_similarity(dict_data):
    jaccard_result = {}
    for task in data:
        jaccard_result[task[0]] = {}
    for task in data:
        query_tfidf = task[2].reshape(1, -1)  # reshape to "fake" 2Dvector [] --> [[]]
        docu_tfidf = task[3].reshape(1, -1)  # reshape to "fake" 2Dvector [] --> [[]]
        query_intersection = np.intersect1d(query_tfidf,docu_tfidf)
        query_union = np.union1d(query_tfidf,docu_tfidf)
        if len(query_intersection) > 0:
            print(len(query_intersection), len(query_union))
            print("################## JACCARD QUEST " + task[0] + " SUCCESFULL ##################")
        jaccard_score = len(query_intersection) / len(query_union)
        jaccard_result[task[0]][task[1]] = jaccard_score


    with open('RESULTS/jaccard_similarity_TEST_results', 'wb') as handle:  # save cosinesimilarity results to a file using pickle
        pickle.dump(jaccard_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("JACCARD SIMILARITY FINISHED")



if __name__ == "__main__":
    data = create_data_dict()
    compute_jaccard_similarity(data)
    """cosine_similarity(dict_data)
    jaccard_similarity(dict_data)
    get_avg_precision('cosine_similarity_TEST_results')"""