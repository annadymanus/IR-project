import pickle



def load_datasets():

    datasets_loaded = []

    with open("./RESULTS/cosine_similarity_TEST_results", "rb") as f:
        imported_cosine_similarity_dataset = f.read()

    cosine_similarity_dataset = pickle.loads(imported_cosine_similarity_dataset)
    datasets_loaded.append(cosine_similarity_dataset)

    with open("./RESULTS/jaccard_similarity_TEST_results", "rb") as f:
        imported_jaccard_similarity_dataset = f.read()

    jaccard_similarity_dataset = pickle.loads(imported_jaccard_similarity_dataset)
    datasets_loaded.append(jaccard_similarity_dataset)

    return datasets_loaded

def merge_datasets(datasets):

    merged_dataset = {}

    for query in datasets[0].keys():
        merged_dataset[query] = {}
        for doc in datasets[0][query].keys():
            merged_dataset[query][doc] = [int(datasets[0][query][doc]),datasets[1][query][doc]]

    with open('RESULTS/merged_results','wb') as handle:  # save merged dataset to pickle
        pickle.dump(merged_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return merged_dataset


if __name__ == "__main__":
    datasets = load_datasets()
    merged_dataset = merge_datasets(datasets)
    print(merged_dataset)