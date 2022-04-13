# Data
Go sure following files are in the directory:    
- fulldocs-new.trec    
- msmarco-docdev-qrels.tsv
- msmarco-docs-lookup.tsv
- msmarco-doctrain-qrels.tsv
- msmarco-test2019-queries.tsv
- msmarco-test2020-queries.tsv
- queries.docdev.tsv
- queries.doctrain.tsv

Github does not allow large files, so I added them to .gitignore. Go sure you have them on your local device.

# How to get your DataSet
I simplified the data generation process to allow the generation of specific preprocessed data. 

## 1. Generate Blank Dataset
At first, create a blank dataset consisting of (queryID, docID, label) tuples by running:
```python
create_blank_dataset("train")
```
Replace "train" by "dev" or "test" to get your desired dataset. The function will then generate a balanced dataset containing positive and negative samples. The function saves this dataset to a file named "train_data.pickle" (or "dev_data.pickle" or "test_data.pickle" respectively). The randomness in negative labeled data generation is seeded, so don't worry about not ending up with the same data set as the others.

## 2. Expand Dataset with Features
You can use the generated data set as basis to expand it with the feature you want to use in your model. For instance, in order to get a dataset with Tf-Idf vectors, one simply needs to load the blank dataset into ```python Preprocessing.tf_idf()```:
```python
blank_data = pickle.load(open('train_data.pickle', 'rb'))
Preprocessing.tf_idf(blank_data, "train")
```
The function will then save a file named "train_tf_idf.pickle" which contains a list of (queryID, docID, query_tf_idf, doc_tf_idf, label) tuples.
If during execution the function causes an error (typically an "out of memory" error or "too long document" error), you can simply execute the function again with the ``` remove_cache = True ``` flag set and it will continue where it stopped last.

If you e.g. wish to get a data set with sentence_embeddings, you just need to run
```python
blank_data = pickle.load(open('train_data.pickle', 'rb'))
Preprocessing.sentence_embedding(blank_data, "train")
```
instead. The filename will then be accordingly 'train_sent_emb.pickle'.

