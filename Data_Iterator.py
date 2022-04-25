import csv
from collections import defaultdict
import random
from typing import Tuple, List
import time
import pickle
from tqdm import tqdm



def write_dataset(data, filename: str):
    """Saves processed_samples to .pickle file"""
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_sample_texts(filename: str):
    """Load dataset with according texts. Yields tuples of shape (qid, docid, query_text, doc_text, label)"""
    docs_lookup_filename = "msmarco-docs-lookup.tsv"
    full_docs_filename = "fulldocs-new.trec"

    if "train" in filename:
        qtext_filename = "queries.doctrain.tsv"
    elif "dev" in filename:
        qtext_filename = "queries.docdev.tsv"
    elif "test" in filename:
        qtext_filename = "msmarco-test2019-queries.tsv"

    # Load Document offsets into dictionary
    lookups = {}
    with open(docs_lookup_filename, "r", encoding="utf-8") as lookup_file:
        lookup_reader = csv.reader(lookup_file, delimiter="\t")
        for row in lookup_reader:
            lookups[row[0]] = int(
                row[1]
            )  # trec format row (row[2] would be for tsv format row)

    # Load Query Relations into dictionary
    qrels = defaultdict(list)    
    pairs = pickle.load(open(filename, 'rb'))
    for pair in pairs:
        qrels[pair[0]].append((pair[1], pair[2]))

    qtexts = open(qtext_filename, "r", encoding="utf-8")
    qtexts_len = len(qtexts.readlines())
    qtexts.seek(0)

    docs = open(full_docs_filename, "r", encoding="utf-8")
    
    query_reader = csv.reader(qtexts, delimiter="\t")    
    for row in tqdm(query_reader, total=qtexts_len, desc="Read Texts"):
        qid = row[0]
        qtext = row[1]

        relations = qrels.get(qid, None)
        if relations == None: continue #Nôt sure if necessary, but just in case        
        
        for relation in relations: #In test set each query has multiple docs
            docid = relation[0]
            label = relation[1]            
            offset = lookups[docid]
            docs.seek(offset)            
            
            doc_text = ""
            is_text = False
            while True:
                line = docs.readline()            
                if "</TEXT>" in line:
                    break
                if is_text:
                    doc_text += line
                if "<TEXT>" in line:
                    is_text = True
            yield (qid, docid, qtext, doc_text, label)            
    qtexts.close()
    docs.close()
        
def create_blank_dataset(d_set: str):
    """Creates balanced dataset and saves List of (query id, document id, label) tuples to .pickle file. Parameter "set" can be either "train", "dev" or "test"."""    
    random.seed(42)
    if set == "test":
        samples = list(sample_generator(d_set))
    else:
        negative_samples = list(sample_generator(d_set, False))
        positive_samples = list(sample_generator(d_set, True))
        samples = negative_samples + positive_samples
        random.shuffle(samples)
    write_dataset(samples, f"{d_set}_data.pickle")

def sample_generator(d_set: str, positive: bool = True) -> Tuple[str, str, bool]:
    """
    A generator function yielding positive or negative data samples from the training, development or test set. Returns a Tuple containing the query id, document id, and relevance.
    Samples are generated in order of the queries listed in the queries.tsv files. For positive samples, the matching document is chosen. For negative samples, a random other
    document is selected. If d_set is set to "test", parameter "positive" is irrelevant.
    """    
    docs_lookup_filename = "msmarco-docs-lookup.tsv"

    if d_set == "train":
        qrels_filename = "msmarco-doctrain-qrels.tsv"

    elif d_set == "dev":
        qrels_filename = "msmarco-docdev-qrels.tsv"

    elif d_set == "test":
        qrels_filename = "qrels-docs.tsv"
    
    #Get all Doc Ids
    docids = []
    with open(docs_lookup_filename, "r", encoding="utf-8") as lookup_file:
        lookup_reader = csv.reader(lookup_file, delimiter="\t")
        for row in lookup_reader:
            docids.append(row[0])
    
    qrels = open(qrels_filename, "r", encoding="utf-8")  
    qrels_reader = csv.reader(qrels, delimiter=" ")    
    for row in qrels_reader:
        qid = row[0]
        docid = row[2]
        label = int(row[3])
        if d_set != "test" and not positive:
            random_docid = random.choice(docids)
            while random_docid == docid:
                random_docid = random.choice(docids)
            docid = random_docid
            label = False                    
        yield (qid, docid, label)        
    qrels.close()

    