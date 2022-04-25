import Data_Iterator
import numpy as np
import torch
import os.path
import os
import pickle
import re
import spacy
from fileinput import filename
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from turtle import pos
from transformers import BartTokenizer, BartModel
from tqdm import tqdm
from typing import Tuple, List
import glob


def split(a, n):
    """Splits list into n equal sized chunks"""
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def get_output_tuples(representation, samples: List[Tuple[str,str,str,str,bool]]):
    output_tuples = []
    for i in range(len(samples)): 
        qid = samples[i][0]
        docid = samples[i][1]
        label = np.array([samples[i][-1]])
        query_repr = representation[2*i] #extract representarion vector of query
        doc_repr= representation[2*i+1] #extract representarion vector of document
        output_tuples.append((qid, docid, query_repr, doc_repr, label))
    
def first_preprocess(samples: List[Tuple[str,str,str,str,bool]], num_chunks=20):
    """ 
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

    return raw_texts

def spacy_tokenize(raw_texts: List[str], num_chunks: int = 20, remove_cache = True, mode="lemmatize"):
    '''uses spacy to tokenize and lemmatize'''
    if remove_cache:
        for f in glob.glob("temp_*.pickle"):
            os.remove(f)
    raw_texts_list = split(raw_texts, num_chunks)
    for i, chunk in enumerate(raw_texts_list):
        if os.path.isfile(f'temp_{i}.pickle'): continue #Just in case stuff crashes.
        q_and_docs_chunk = []
        nlp = spacy.load("en_core_web_md")
        nlp.max_length = 3000000
        nlp.disable_pipes("parser", "ner", "tok2vec") #remove pipe we do not need
        if mode != "lemmatize": nlp.disable_pipes("lemmatizer")
        for item in tqdm(nlp.pipe(chunk, batch_size=32, n_process=16), total=len(chunk), desc="NLP Pipe"):
            if mode == "lemmatize":
                #Only save the relevant stuff to prevent out of memory issues. Spacy.doc and Spacy.token are too large and contain stuff we dont need
                q_and_docs_chunk.append([token.lemma for token in item if token.is_alpha and token.text.lower() not in ENGLISH_STOP_WORDS])
            elif mode == "tokenize":
                q_and_docs_chunk.append([token.text for token in item])
            elif mode == "sentencize":
                q_and_docs_chunk.append([sent.text for sent in item.sents])
        del nlp #Free up space asap
        with open(f'temp_{i}.pickle', 'wb') as handle:
            pickle.dump(q_and_docs_chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del raw_texts
    q_and_docs = []
    for i in tqdm(range(len(raw_texts_list)), desc="Load temp files"):
        q_and_docs.extend(pickle.load(open(f"temp_{i}.pickle", "rb" )))
    return q_and_docs

def count_vector(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    '''
    Create Count Vectors for queries and documents.
    '''
    reshaped_clean_samples = first_preprocess(samples)
    samples = [item[:2] for item in samples] #Clear up space
    q_and_docs = spacy_tokenize(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache)    

    #Get Count Vectors
    if d_set == "train":
        vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)    
        vectorizer.fit(q_and_docs) #returns Sparse Matrix. Each row is a document/query.
        Data_Iterator.write_dataset(vectorizer, "count_vectorizer.pickle")
    
    vectorizer = pickle.load(open(f"count_vectorizer.pickle", "rb" ))
    counts = vectorizer.transform(q_and_docs)

    #Output Tuples
    count_vector_samples = get_output_tuples(counts, samples)
    Data_Iterator.write_dataset(count_vector_samples, f"{d_set}_count_vector.pickle")
    return f"{d_set}_count_vector"

def tf_idf(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    '''
    Create Tf-Idf for queries and documents.
    '''
    reshaped_clean_samples = first_preprocess(samples)
    samples = [item[:2] for item in samples] #Clear up space
    q_and_docs = spacy_tokenize(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache)

    #Get Count Vectors
    if d_set == "train":
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)    
        vectorizer.fit(q_and_docs) #returns Sparse Matrix. Each row is a document/query.
        Data_Iterator.write_dataset(vectorizer, "tfidf_vectorizer.pickle")
        
        tf_idfs = vectorizer.fit_transform(q_and_docs)
        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        tf_idfs = svd.fit(tf_idfs)
        Data_Iterator.write_dataset(svd, "svd.pickle")
    
    vectorizer = pickle.load(open(f"tfidf_vectorizer.pickle", "rb" ))
    tf_idfs = vectorizer.transform(q_and_docs)
    svd = pickle.load(open(f"svd.pickle", "rb" ))
    tf_idfs = svd.fit_transform(tf_idfs) #reduce dimensionality to 100 elements
    
    #Output Tuples
    tf_idf_samples = get_output_tuples(tf_idfs, samples)
    Data_Iterator.write_dataset(tf_idf_samples, f"{d_set}_tf_idf.pickle")
    return f"{d_set}_tf_idf"
    
def context_word_embedding(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):

    if remove_cache:
        os.remove("temp_token*.pickle")

    reshaped_clean_samples = first_preprocess(samples)

    raw_texts_list = split(reshaped_clean_samples, num_chunks)
    for i, chunk in enumerate(raw_texts_list):
        if os.path.isfile(f'temp_{i}.pickle'): continue #Just in case stuff crashes.
        q_and_docs_chunk = []
        nlp = spacy.load("en_core_web_md")
        nlp.max_length = 2000000
        nlp.disable_pipes("parser", "ner") #remove pipe we do not need            
        for item in tqdm(nlp.pipe(chunk, batch_size=32, n_process=16), total=len(chunk), desc="NLP Pipe"):
            #Only save the relevant stuff to prevent out of memory issues. Spacy.doc and Spacy.token are too large and contain stuff we dont need
            q_and_docs_chunk.append(sum([token.vector for token in item])/len(item))
        del nlp #Free up space asap
        with open(f'temp_cont_word_{i}.pickle', 'wb') as handle:
            pickle.dump(q_and_docs_chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    cont_w_emb = []
    for i in tqdm(range(len(raw_texts_list)), desc="Load temp files"):
        cont_w_emb.extend(pickle.load(open(f"temp_cont_word_{i}.pickle", "rb" )))

    #Output Tuples
    cont_word_emb_samples = get_output_tuples(cont_w_emb, samples)
    Data_Iterator.write_dataset(cont_word_emb_samples, f"{d_set}_cont_word_emb.pickle")
    return f"{d_set}_cont_word_emb"
    

def non_context_word_embedding(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    """Description..."""
    import gensim.downloader    

    reshaped_clean_samples = first_preprocess(samples)
    q_and_docs = spacy_tokenize(reshaped_clean_samples, mode="tokenize", remove_cache=remove_cache)
    
    glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    non_cont_w_emb = [sum([glove_vectors[token] for token in text])/len(text) for text in q_and_docs]
    
    #Output Tuples
    cont_word_emb_samples = get_output_tuples(non_cont_w_emb, samples)
    Data_Iterator.write_dataset(cont_word_emb_samples, f"{d_set}_non_cont_word_emb.pickle")
    return f"{d_set}_non_cont_word_emb"    


def sentence_embedding(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    reshaped_clean_samples = first_preprocess(samples)
    q_and_docs = spacy_tokenize(reshaped_clean_samples, mode="sentencize", remove_cache=remove_cache)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base").to(device)
    model = BartModel.from_pretrained("facebook/bart-base").to(device)
    
    sent_embeddings = []
    for text in q_and_docs:
        embeds = []
        for sent in text:
            tokens = tokenizer(sent, return_tensors="pt", truncation=True)
            tokens = tokens[:1024]
            embeds.append(model(**tokens).last_hidden_state[0][-1].cpu().numpy())
        sent_embeddings.append(sum(embeds)/len(embeds))
    
    #Output Tuples
    sent_emb_samples = get_output_tuples(sent_embeddings, samples)
    Data_Iterator.write_dataset(sent_emb_samples, f"{d_set}_sent_emb.pickle")
    return f"{d_set}_sent_emb"    

    
def document_embedding(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q_and_docs = first_preprocess(samples)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base").to(device)
    model = BartModel.from_pretrained("facebook/bart-base").to(device)
    
    doc_embeddings = []
    for text in q_and_docs:
        tokens = tokenizer(text, return_tensors="pt", truncation=True)
        doc_emb = model(**tokens).last_hidden_state[0][-1].cpu().numpy()
        doc_embeddings.append(doc_emb)
    
    #Output Tuples
    doc_emb_samples = get_output_tuples(doc_embeddings, samples)
    Data_Iterator.write_dataset(doc_emb_samples, f"{d_set}_doc_emb.pickle")
    return f"{d_set}_doc_emb"

if __name__ == "__main__":
    #Data_Iterator.create_blank_dataset("train")
    
    blank_data = list(Data_Iterator.get_sample_texts('train_data.pickle'))
    tf_idf(blank_data, "train", remove_cache=False)
    count_vector(blank_data, "train", remove_cache=False)
    #non_context_word_embedding(blank_data, "train", remove_cache=False)
    #context_word_embedding(blank_data, "train")    
    