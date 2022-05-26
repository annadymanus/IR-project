import Data_Iterator
import numpy as np
import os.path
import os
import pickle
import re
import spacy
import scipy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, ENGLISH_STOP_WORDS, CountVectorizer
from transformers import BartTokenizerFast
from tqdm import tqdm
from typing import Tuple, List
import glob
from datasets import Dataset


def identity(x):
    return x

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
        try:
            query_repr = representation[2*i] #extract representarion vector of query
            doc_repr= representation[2*i+1] #extract representarion vector of document
            output_tuples.append((qid, docid, query_repr, doc_repr, label))
        except IndexError:
            break
    return output_tuples
    
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
    q_and_docs = []
    for partial_output in spacy_tokenize_generator(raw_texts=raw_texts, num_chunks=num_chunks, remove_cache = remove_cache, mode=mode):
         q_and_docs.extend(partial_output)
    return q_and_docs

def spacy_tokenize_generator(raw_texts: List[str], num_chunks: int = 20, remove_cache = True, mode="lemmatize"):
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

        chunk = [text[:1000000] for text in chunk]

        nlp.disable_pipes("parser", "ner") #remove pipe we do not need
        if mode != "lemmatize": nlp.disable_pipes("lemmatizer", "tagger")
        if mode != "vectorize": nlp.disable_pipes("tok2vec")
        for item in tqdm(nlp.pipe(chunk, batch_size=32, n_process=16), total=len(chunk), desc="NLP Pipe"):
            if mode == "lemmatize":
                #Only save the relevant stuff to prevent out of memory issues. Spacy.doc and Spacy.token are too large and contain stuff we dont need
                q_and_docs_chunk.append([token.lemma for token in item if token.is_alpha and token.text.lower() not in ENGLISH_STOP_WORDS])
            elif mode == "tokenize":
                q_and_docs_chunk.append([token.text for token in item])
            elif mode == "sentencize":
                q_and_docs_chunk.append([sent.text for sent in item.sents])
            elif mode == "vectorize":
                q_and_docs_chunk.append([token.vector for token in item])
        del nlp #Free up space asap
        with open(f'temp_{i}.pickle', 'wb') as handle:
            pickle.dump(q_and_docs_chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del raw_texts
    q_and_docs = []
    for i in tqdm(range(len(raw_texts_list)), desc="Load temp files"):
        yield pickle.load(open(f"temp_{i}.pickle", "rb" ))
    

def count_vector(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    '''
    Create Count Vectors for queries and documents.
    '''
    reshaped_clean_samples = first_preprocess(samples)
    samples = [[item[0], item[1], item[-1]] for item in samples] #Clear up space    

    #Get Count Vectors
    if d_set == "train":
        q_and_docs = spacy_tokenize(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache, num_chunks=num_chunks)
        del reshaped_clean_samples
        vectorizer = CountVectorizer(tokenizer=identity, lowercase=False, max_features=6000, preprocessor=identity)
        vectorizer.fit(q_and_docs) #returns Sparse Matrix. Each row is a document/query.
        Data_Iterator.write_dataset(vectorizer, 'count_vectorizer.pickle')        
    
    vectorizer = pickle.load(open("count_vectorizer.pickle", "rb" ))
    counts = None
    for batch in spacy_tokenize_generator(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache, num_chunks=num_chunks):
        if counts is None:
            counts = vectorizer.transform(batch)
        else:
            counts = scipy.sparse.vstack([counts, vectorizer.transform(batch)])
    
    #Output Tuples
    count_vector_samples = get_output_tuples(counts, samples)
    Data_Iterator.write_dataset(count_vector_samples, f"{d_set}_count_vector.pickle")
    return f"{d_set}_count_vector.pickle"

def tf_idf(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    '''
    Create Tf-Idf for queries and documents.
    '''
    reshaped_clean_samples = first_preprocess(samples)
    samples = [[item[0], item[1], item[-1]] for item in samples] #Clear up space    

    #Get Count Vectors
    if d_set == "train":
        if os.path.isfile('count_vectorizer.pickle'):
            vectorizer = pickle.load(open("count_vectorizer.pickle", "rb"))
        else:
            q_and_docs = spacy_tokenize(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache, num_chunks=num_chunks)            
            vectorizer = CountVectorizer(tokenizer=identity, lowercase=False)    
            vectorizer.fit(q_and_docs)
            Data_Iterator.write_dataset(vectorizer, 'count_vectorizer.pickle')
        
        counts = None
        for batch in spacy_tokenize_generator(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache, num_chunks=num_chunks):
            if counts is None:
                counts = vectorizer.transform(batch)
            else:
                counts = scipy.sparse.vstack([counts, vectorizer.transform(batch)])
        del vectorizer        

        transformer = TfidfTransformer()
        transformer.fit(counts)
        Data_Iterator.write_dataset(transformer, 'tfidf_transformer.pickle')
        tf_idfs = transformer.transform(counts)
        del transformer        

        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        svd.fit(tf_idfs)
        Data_Iterator.write_dataset(svd, "tfidf_svd.pickle")
        del svd        

    vectorizer = pickle.load(open("count_vectorizer.pickle", "rb" ))
    transformer = pickle.load(open("tfidf_transformer.pickle", "rb" ))
    svd = pickle.load(open("tfidf_svd.pickle", "rb" ))

    tf_idfs = None
    for batch in spacy_tokenize_generator(reshaped_clean_samples, mode="lemmatize", remove_cache=remove_cache, num_chunks=num_chunks):
        if tf_idfs is None:
            tf_idfs = svd.transform(transformer.transform(vectorizer.transform(batch)))
        else:
            batch_tf_idfs = svd.transform(transformer.transform(vectorizer.transform(batch)))
            tf_idfs = np.vstack([tf_idfs, batch_tf_idfs])
    
    #Output Tuples
    tf_idf_samples = get_output_tuples(tf_idfs, samples)
    Data_Iterator.write_dataset(tf_idf_samples, f"{d_set}_tf_idf.pickle")
    return f"{d_set}_tf_idf"
    

def non_context_word_embedding(samples: List[Tuple[str,str,str,str,bool]], d_set: str, num_chunks: int = 20, remove_cache=True):
    """Description..."""
    import gensim.downloader
    glove_vectors = gensim.downloader.load('word2vec-google-news-300')

    reshaped_clean_samples = first_preprocess(samples)
    samples = [[item[0], item[1], item[-1]] for item in samples] #Clear up space    

    non_cont_w_emb = []
    for batch in spacy_tokenize_generator(reshaped_clean_samples, mode="tokenize", remove_cache=remove_cache, num_chunks=num_chunks):
        batch_emb = [sum([glove_vectors[token] for token in text if token in glove_vectors])/len(text) for text in batch]
        non_cont_w_emb.extend(batch_emb)
    
    #Output Tuples
    cont_word_emb_samples = get_output_tuples(non_cont_w_emb, samples)
    Data_Iterator.write_dataset(cont_word_emb_samples, f"{d_set}_non_cont_word_emb.pickle")
    return f"{d_set}_non_cont_word_emb"


def bart_tokenize(samples: List[Tuple[str,str,str,str,bool]], d_set: str):
    q_and_docs = first_preprocess(samples)
    samples = [[item[0], item[1], item[-1]] for item in samples] #Clear up space    

    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    dataset = Dataset.from_dict({"text": q_and_docs})
    del q_and_docs

    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)  
    dataset = dataset.map(encode, batched=True, remove_columns=["text"])

    dataset = [(item["input_ids"], item["attention_mask"]) for item in dataset]
    tokenized_samples = get_output_tuples(dataset, samples)
    Data_Iterator.write_dataset(tokenized_samples, f"{d_set}_bart_tokenized.pickle")
    return f"{d_set}_bart_tokenized.pickle"

if __name__ == "__main__":
    #Data_Iterator.create_blank_dataset("test")    
    blank_data = Data_Iterator.get_sample_texts('test_data.pickle')
    #tf_idf(blank_data, "test", remove_cache=False, num_chunks=3)
    count_vector(blank_data, "test", remove_cache=False, num_chunks=3)
    #non_context_word_embedding(blank_data, "test", remove_cache=False, num_chunks=3)    
    #context_word_embedding(blank_data, "train")    
    #bart_tokenize(blank_data, "test")
    