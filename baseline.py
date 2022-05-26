import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def create_data_dict(): #create dict {queryid: querytfidf, docid, doctfidf...}
    test_data = pickle.load(open('./data/test_tf_idf.pickle', 'rb'))
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
    with open('./model_predictions/cosine_similarity.pickle', 'wb') as handle:  # save cosinesimilarity results to a file using pickle
        pickle.dump(cosine_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    with open('./model_predictions/jaccard', 'wb') as handle:  # save cosinesimilarity results to a file using pickle
        pickle.dump(dict_jaccard_val, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_jaccard_similarity(dict_data):
    jaccard_result = {}
    for task in data:
        jaccard_result[task[0]] = {}
    for task in data:
        query_tfidf = task[2].reshape(1, -1)  # reshape to "fake" 2Dvector [] --> [[]]
        docu_tfidf = task[3].reshape(1, -1)  # reshape to "fake" 2Dvector [] --> [[]]
        query_intersection = np.intersect1d(query_tfidf,docu_tfidf)
        query_union = np.union1d(query_tfidf,docu_tfidf)
        jaccard_score = len(query_intersection) / len(query_union)
        jaccard_result[task[0]][task[1]] = jaccard_score

    with open('./model_predictions/jaccard_similarity', 'wb') as handle:  # save cosinesimilarity results to a file using pickle
        pickle.dump(jaccard_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data = create_data_dict()
    compute_cosine_similarity(data)