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

## Data_Iterator.py
Contains generator function yielding positive or negative samples for desired dataset. The original training sets only contain positive samples. To be able to effectively train our model we also need negative samples. Current policy of generating negative samples is to simply choose a random document (not the true one) for a query.
