import csv
from encodings import utf_8
import random
from typing import Tuple

def sample_generator(set: str, positive: bool) -> Tuple(str, str, bool):
    """
    A generator function yielding positive or negative data samples from the training or development set. Returns a Tuple containing the query text, document text, and relevance.
    Samples are generated in order of the queries listed in queries.doctrain.tsv or queries.doctrain.tsv. For positive samples, the matching document is chosen. For negative samples, a random other
    document is selected.
    """
    docs_lookup_filename = "msmarco-docs-lookup.tsv"
    full_docs_filename = "fulldocs-new.trec"

    if set == "train":
        qtext_filename = "queries.doctrain.tsv"
        qrels_filename = "msmarco-doctrain-qrels.tsv"

    elif set == "dev":
        qtext_filename = "queries.docdev.tsv"
        qrels_filename = "msmarco-docdev-qrels.tsv"

    # Load Query Relations into dictionary
    qrels = {}
    with open(qrels_filename, "r") as qrels_file:
        qrels_reader = csv.reader(qrels_file, delimiter=" ")
        for row in qrels_reader:
            qrels[row[0]] = row[2]

    # Load Document offsets into dictionary
    lookups = {}
    with open(docs_lookup_filename, "r") as lookup_file:
        lookup_reader = csv.reader(lookup_file, delimiter="\t")
        for row in lookup_reader:
            lookups[row[0]] = int(
                row[1]
            )  # trec format row (row[2] would be for tsv format row)

    with open(qtext_filename, "r") as qtexts, open(
        full_docs_filename, "r", encoding="utf_8"
    ) as docs:
        query_reader = csv.reader(qtexts, delimiter="\t")
        for row in query_reader:
            qid = row[0]
            qtext = row[1]
            docid = qrels[qid]
            if positive:
                offset = lookups[docid]
                docs.seek(offset)
            else:
                # Select some random other document
                random_entry = random.choice(list(lookups.items()))
                while random_entry[0] == docid:
                    random_entry = random.choice(list(lookups.items()))
                offset = random_entry[1]
                docid = random_entry[0]
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
            yield (qtext, doc_text, positive)


# Example for positive samples
positive_samples = iter(sample_generator("train", True))
for i in range(10):
    print(next(positive_samples))

# Example for positive samples
negative_samples = iter(sample_generator("train", False))
for i in range(10):
    print(next(negative_samples))