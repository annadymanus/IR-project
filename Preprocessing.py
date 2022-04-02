from fileinput import filename
import os.path
from turtle import pos
import spacy
import Data_Iterator
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from typing import Tuple, List
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from tqdm import tqdm

def get_avg_embedding(spacy_doc):
    """Returns the average of all word embeddings of the document"""
    return sum([token.vector for token in spacy_doc])/len(spacy_doc)
   
def preprocess(samples: List[Tuple[str,str,str,str,bool]], num_chunks=20):
    """ 
    Create Tf-Idf and embedding vectors for queries and documents.
    Returns List of Tuples of shape (Query ID, Doc ID, Query Tf-Idf, Query Embedding, Document Tf-Idf, Document Embedding, Label)
    """ 
    def remove_urls(text: str) -> str:
        """Removes url strings"""
        return re.sub(r'http\S+', '', text)

    #Transform from [[query0, doc0], [query1, doc1],...] to [query0, doc0, query1, doc1, ...]
    raw_texts = []
    for sample in samples:
        query = remove_urls(sample[2])
        doc = remove_urls(sample[3])        
        raw_texts.extend([query, doc])

    #Spacy Pipeline: Tokenize, Lemmatize, Vectorize (Embeddings), Extract Entities (not yet working, missing knowledge base)  
    #All this shit is necessary because of out of memory issues
    def split(a, n):
        """Splits list into n equal sized chunks"""
        k, m = divmod(len(a), n)
        return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    raw_texts_list = split(raw_texts, num_chunks)
    for i, chunk in enumerate(raw_texts_list):
        if os.path.isfile(f'temp_{i}.pickle'): continue #Just in case stuff crashes.
        q_and_docs_chunk = []
        nlp = spacy.load("en_core_web_md")
        nlp.max_lenth = 2000000
        nlp.disable_pipes("parser", "ner") #remove pipe we do not need            
        for item in tqdm(nlp.pipe(chunk, batch_size=32, n_process=16), total=len(chunk), desc="NLP Pipe"):
            #Only save the relevant stuff to prevent out of memory issues. Spacy.doc and Spacy.token are too large and contain stuff we dont need
            q_and_docs_chunk.append(([token.lemma for token in item if token.is_alpha and token.text.lower() not in ENGLISH_STOP_WORDS], get_avg_embedding(item)))
        del nlp #Free up space asap
        with open(f'temp_{i}.pickle', 'wb') as handle:
            pickle.dump(q_and_docs_chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    q_and_docs = []
    for i in tqdm(range(len(raw_texts_list)), desc="Load temp files"):
        q_and_docs.extend(pickle.load(open(f"temp_{i}.pickle", "rb" )))

    q_and_docs_lemmas = [item[0] for item in q_and_docs]
    q_and_docs_embeddings = [item[1] for item in q_and_docs]    

    #Get TFidf Vectors
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tf_idfs = vectorizer.fit_transform([lemmas for lemmas in q_and_docs_lemmas]) #returns Tfidf Sparse Matrix. Each row is a document/query.
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    tf_idfs = svd.fit_transform(tf_idfs) #reduce dimensionality to 100 elements

    #Output Tuples
    processed_samples = []
    for i in range(len(samples)): 
        qid = samples[i][0]
        docid = samples[i][1]
        label = np.array([samples[i][-1]])
        query_tf_idf = tf_idfs[2*i] #extract tfidf vector of query
        doc_tf_idf = tf_idfs[2*i+1] #extract tfidf vector of document
        avg_query_emb = q_and_docs_embeddings[2*i]
        avg_doc_emb = q_and_docs_embeddings[2*i+1]
        processed_samples.append((qid, docid, query_tf_idf, avg_query_emb, doc_tf_idf, avg_doc_emb, label))
    return processed_samples

def write_dataset(processeed_samples: List[Tuple], filename: str):
    """Saves processed_samples to .pickle file"""
    with open(filename, 'wb') as handle:
        pickle.dump(processeed_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    samples = Data_Iterator.sample_generator("test", positive=True)
    generated_samples = []    
    for sample in tqdm(samples, desc="Load Data"):
        generated_samples.append(sample)
    preprocessed_samples = preprocess(generated_samples, num_chunks=5)
    write_dataset(preprocessed_samples, "test_data.pickle")