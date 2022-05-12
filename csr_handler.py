from scipy.sparse import csr_matrix
import pickle

def unpack_csrs(data):
    unpacked_data = []
    for dp in data:
        unpacked_data.append([
            dp[0], 
            dp[1], 
            ((dp[2].data, dp[2].indices, dp[2].indptr), dp[2].shape),
            ((dp[3].data, dp[3].indices, dp[3].indptr), dp[3].shape),
            dp[-1]])
    return unpacked_data

def pack_csrs(unpacked_data):
    data = []
    for dp in unpacked_data:
        data.append([
            dp[0], 
            dp[1], 
            csr_matrix(dp[2]),
            csr_matrix(dp[3]),
            dp[-1]])
    return data

if __name__ == "__main__":
    for set in ["test", "dev", "train"]:
        data = pickle.load(open(f"{set}_count_vector.pickle", "rb"))
        data = unpack_csrs(data)
        pickle.dump(data, open(f"{set}_count_vector_unpacked.pickle", "wb"))
