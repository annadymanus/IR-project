# IR-project
## Data
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

## How to get your DataSet
I simplified the data generation process to allow the generation of specific preprocessed data. 

1. At first, create a blank dataset consisting of (queryID, docID, label) tuples by running:```python create_blank_dataset("train")``` Replace "train" by "dev" or "test" to get your desired dataset. The function will then generate a balanced dataset containing positive and negative samples. The function saves this dataset to a file named "train_data.pickle" (or "dev_data.pickle" or "test_data.pickle" respectively)


For instance, in order to get a dataset with Tf-Idf vectors, one simply needs to call

