from fileinput import filename
import spacy
import Data_Iterator
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from tqdm import tqdm

def get_avg_embedding(spacy_doc):
    """Returns the average of all word embeddings of the document"""
    return sum([token.vector for token in spacy_doc])/len(spacy_doc)
   
def preprocess(samples: List[Tuple[str,str,bool]]):
    """ 
    Create Tf-Idf and embedding vectors for queries and documents.
    Returns List of Tuples of shape (Query Tf-Idf, Query Embedding, Document Tf-Idf, Document Embedding, Label)
    """ 

    nlp = spacy.load("en_core_web_md")
    nlp.disable_pipes("parser") #remove pipe we do not need

    def remove_urls(text: str) -> str:
        """Removes url strings"""
        return re.sub(r'http\S+', '', text)

    #Transform from [[query0, doc0], [query1, doc1],...] to [query0, doc0, query1, doc1, ...]
    raw_texts = []
    for sample in samples:
        query = remove_urls(sample[0])
        doc = remove_urls(sample[1])
        raw_texts.extend([query, doc])
    
    #Spacy Pipeline: Tokenize, Lemmatize, Vectorize (Embeddings), Extract Entities (not yet working, missing knowledge base)
    q_and_docs = []
    for item in tqdm(nlp.pipe(raw_texts, n_process=16), total=len(raw_texts), desc="NLP Pipe"):
        q_and_docs.append(item)

    #Get TFidf Vectors
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, stop_words='english', lowercase=False)
    tf_idfs = vectorizer.fit_transform([[token.lemma_ for token in doc] for doc in q_and_docs]) #returns Tfidf Sparse Matrix. Each row is a document/query.
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    tf_idfs = svd.fit_transform(tf_idfs) #reduce dimensionality to 100 elements

    #Output Tuples
    processed_samples = []
    for i in range(len(samples)):        
        query = q_and_docs[2*i]
        doc = q_and_docs[2*i+1]
        label = np.array([samples[i][-1]])
        query_tf_idf = tf_idfs[2*i] #extract tfidf vector of query
        doc_tf_idf = tf_idfs[2*i+1] #extract tfidf vector of document
        avg_query_emb = get_avg_embedding(query)
        avg_doc_emb = get_avg_embedding(doc)
        #query_entities = [entity.kb_id_ for entity in set(query.ents)] #Not yet working, needs knowledge base!
        #doc_entities = [entity.kb_id_ for entity in set(doc.ents)]
        processed_samples.append((query_tf_idf, avg_query_emb, doc_tf_idf, avg_doc_emb, label))
    return processed_samples

def write_dataset(samples: List[Tuple[str,str,bool]], filename: str):
    """Saves processed_samples to .pickles file"""
    samples = preprocess(samples)
    with open('test.pickle', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

#TESTING CODE
if __name__ == "__main__":
    positive_samples = list(Data_Iterator.sample_generator("train", True))
    generated_samples = []    
    for sample in tqdm(positive_samples, desc="Load Data"):
        generated_samples.append(sample)    
    write_dataset(generated_samples, "positive_train_data.csv")