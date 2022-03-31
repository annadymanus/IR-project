import gzip


def import_dataset():
    file_name = 'msmarco-docs.trec.gz'
    f = gzip.open(file_name, 'r') #read from a buffer so we never load the 8 GB in memory
    for line in f:
        print(line)
    f.close()

if __name__ == "__main__":
    import_dataset()