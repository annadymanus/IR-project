import csv
from collections import defaultdict
import random
from typing import Tuple
import time

def sample_generator(set: str, positive: bool = True) -> Tuple[str, str, bool]:
    """
    A generator function yielding positive or negative data samples from the training or development set. Returns a Tuple containing the query text, document text, and relevance.
    Samples are generated in order of the queries listed in queries.doctrain.tsv or queries.doctrain.tsv. For positive samples, the matching document is chosen. For negative samples, a random other
    document is selected. If set is set to "test", parameter "positive" is irrelevant.
    """    
    docs_lookup_filename = "msmarco-docs-lookup.tsv"
    full_docs_filename = "fulldocs-new.trec"

    if set == "train":
        qtext_filename = "queries.doctrain.tsv"
        qrels_filename = "msmarco-doctrain-qrels.tsv"

    elif set == "dev":
        qtext_filename = "queries.docdev.tsv"
        qrels_filename = "msmarco-docdev-qrels.tsv"

    elif set == "test":
        qtext_filename = "msmarco-test2019-queries.tsv"
        qrels_filename = "qrels-docs.tsv"

    # Load Query Relations into dictionary
    qrels = defaultdict(list)    
    with open(qrels_filename, "r", encoding="utf-8") as qrels_file:
        qrels_reader = csv.reader(qrels_file, delimiter=" ")
        for row in qrels_reader:
            qrels[row[0]].append((row[2], row[3])) #Document ID and Label (label only relevant in testset)

    # Load Document offsets into dictionary
    lookups = {}
    with open(docs_lookup_filename, "r", encoding="utf-8") as lookup_file:
        lookup_reader = csv.reader(lookup_file, delimiter="\t")
        for row in lookup_reader:
            lookups[row[0]] = int(
                row[1]
            )  # trec format row (row[2] would be for tsv format row)

    qtexts = open(qtext_filename, "r", encoding="utf-8")    
    docs = open(full_docs_filename, "r", encoding="utf-8")
    
    query_reader = csv.reader(qtexts, delimiter="\t")    
    for row in query_reader:
        qid = row[0]
        qtext = row[1]

        relations = qrels.get(qid, None)
        if relations == None: continue #Nôt sure if necessary, but just in case        
        
        for relation in relations: #In test set each query has multiple docs
            docid = relation[0]
            if set == "test":
                label = False if relation[1] == 0 else True #According to article, human rating 2 is considered equivalent to 1
            else:
                label = positive
            if label == True or set == "test":
                offset = lookups[docid]
                docs.seek(offset)
            else:
                # Select some random other document
                random_entry = random.choice(list(lookups.items()))
                while random_entry[0] == docid:
                    random_entry = random.choice(list(lookups.items()))
                offset = random_entry[1]
                docid = random_entry[0]
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
            yield (qtext, doc_text, label)
    qtexts.close()
    docs.close()

#TESTING CODE
if __name__ == "__main__":
    start = time.time()
    # Example for positive samples
    positive_samples = iter(sample_generator("train", True))
    for i in range(10):
        next(positive_samples)
    # Example for positive samples
    negative_samples = iter(sample_generator("train", False))
    for i in range(10):
        next(negative_samples)
    end = time.time()
    print("TIME Uncompressed: ", end-start)