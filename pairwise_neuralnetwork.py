from importlib.metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
from torch.optim import AdamW
import itertools
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, y):
        """Gets query-doc vector concatenation of document x and of document y"""
        
        #Send both query-doc concatenations through same NN
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))

        #Substract the two output vectors
        z = x - y

        #Send through final layer and through sigmoid to scale between 0 and 1
        z = self.fc4(z)
        z = torch.sigmoid(z)

        return z

def visualize_net():
    """get pretty picture for presentation (not important)"""
    from torchviz import make_dot
    x=torch.ones(10, requires_grad=True)
    y=torch.ones(10, requires_grad=True)
    net = Net(10)
    pred = net(x,y)
    make_dot(pred, params=dict(list(net.named_parameters()))).render("nn", format="png")

class PairwiseDataset(Dataset):
    def __init__(self, filename, bows_filename=None):
        raw_data = pickle.load(open(filename, "rb" ))

        #TODO: Add option to include bag-of-words (bows) to the feature vector. This is needed when run in scoring mode
        # so that the naive-bayes probabilities can be calculated. @Alvaro take a look at pointise_neuralnetwork.py

        #Create dictionary with {qid: [[docid, query_vector, doc_vector, label], [docid, query_vector, doc_vector, label], ...]}
        #For Training and Dev set, it will be always two entries per qid, because we always have only one positive and one negative sample
        sorted_data = defaultdict(list)
        for item in raw_data:
            sorted_data[item[0]].append(item[1:])
        
        self.data = []      
        for key, value in sorted_data.items():
            assert len(value) == 2 #Go sure we have really just two docs
            assert np.array_equal(value[0][1], value[1][1]) # Go both documents really belong to the same query (same query_vector). Just to double check...
            assert value[0][-1] != value[1][-1] #Go sure they have different label 
            
            #Append: [qid, doc1_id, doc2_id, query_vector, doc1_vector, doc2_vector, label] where label is 0 when doc1 is the positive one and 1 otherwise
            self.data.append([key, value[0][0], value[1][0], torch.tensor(value[0][1]), torch.tensor(value[0][2]), torch.tensor(value[1][2]), not bool(value[0][-1])])                    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx][0:3]
        representations = self.data[idx][3:-1]
        label = self.data[idx][-1]
        return ids, representations, label

class SinglewiseDataset(Dataset):
    """For Testing..."""

    #@Alvaro: This is gonna be a bit complicated. In the test datasheet we have for each query 100 or more documents. 
    #So we cant possibly look at every combination (we would be speaking of 10 000 combinations for each query).
    #We still need to be able to rank them though, so I guess we could assum transitivity:
    #If docA > docB and docB > docC, then we simply assume docA > docC as well.
    #So you kinda need to write a dataloader which kinda serves data like this:
    
    #docA vs docB
    #docC vs docA
    #docC vs docB
    #docD vs docA
    #docD vs docB
    #docD vs docC
    #...

    #This would at half the amount of combinations we need to do...Which would still be a lot...
    #Maybe you can come up with a better idea.

    #The important thing is that at the end of the day we need to have sufficient comparisons to bring all docs into an order.

    pass


def train(representation, scoring, epochs=2, batch_size=16, bart_emb_type=None):
    """Gets as input the representation name (e.g. "tf_idf")"""
    if scoring:
        train_dataset = PairwiseDataset(filename = f"preprocessed_data/train_{representation}.pickle", bows_filename=f"preprocessed_data/train_count_vector.pickle")
        dev_dataset = PairwiseDataset(f"preprocessed_data/dev_{representation}.pickle", bows_filename=f"preprocessed_data/train_count_vector.pickle")
        vector_size = 3 * 2 #Number of scoring functions is the size of input vector. Times 2, bc always of two documents 

    else:
        train_dataset = PairwiseDataset(filename = f"preprocessed_data/train_{representation}.pickle")
        dev_dataset = PairwiseDataset(f"preprocessed_data/dev_{representation}.pickle")
        vector_size = train_dataset.data[0][3].shape[-1] * 2 #Get size of representations. Multiplied by two, bc document and query vector are always concatenated

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    net = Net(vector_size).to(device)

    criterion = nn.BCELoss() #Binary Cross Entropy Loss
    optimizer = AdamW(net.parameters())

    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            ids, reps, labels = data

            if bart_emb_type:
                    reps[0] = get_bart_embeddings(reps[0], bart_emb_type)
                    reps[1] = get_bart_embeddings(reps[1], bart_emb_type)
                    reps[2] = get_bart_embeddings(reps[2], bart_emb_type)

            if scoring:
                cosine1 = get_batch_cosines(reps[0], reps[1])
                cosine2 = get_batch_cosines(reps[0], reps[2])
                jacc1 = get_batch_jaccard(reps[0], reps[1])
                jacc2 = get_batch_jaccard(reps[0], reps[2])
                #TODO: Naive bayes
                #nb1 = get_naive_bayes(...)
                #nb2 = get_naive_bayes(...)
                inputs = torch.tensor([cosine1, jacc1, nb1, cosine2, jacc2, nb2]).to(device)
            
            else:
                query_vec = torch.tensor(reps[0]).to(device)
                doc1_vec = torch.tensor(reps[1]).to(device)
                doc2_vec = torch.tensor(reps[2]).to(device)

                #concatenate query and doc representation
                inputs1 = torch.concat((query_vec, doc1_vec), dim=-1)
                inputs2 = torch.concat((query_vec, doc2_vec), dim=-1)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs1.float(), inputs2.float())
            loss = criterion(outputs.squeeze(), labels.float().to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        with torch.no_grad():
            net.eval()
            eval_loss = 0.0
            for i, data in enumerate(dev_dataloader):
                ids, reps, labels = data

                if bart_emb_type:
                    reps[0] = get_bart_embeddings(reps[0], bart_emb_type)
                    reps[1] = get_bart_embeddings(reps[1], bart_emb_type)
                    reps[2] = get_bart_embeddings(reps[2], bart_emb_type)

                if scoring:
                    cosine1 = get_batch_cosines(reps[0], reps[1])
                    cosine2 = get_batch_cosines(reps[0], reps[2])
                    jacc1 = get_batch_jaccard(reps[0], reps[1])
                    jacc2 = get_batch_jaccard(reps[0], reps[2])
                    #TODO: Naive bayes
                    #nb1 = get_naive_bayes(...)
                    #nb2 = get_naive_bayes(...)
                    inputs = torch.tensor([cosine1, jacc1, nb1, cosine2, jacc2, nb2]).to(device)
                
                else:
                    query_vec = torch.tensor(reps[0]).to(device)
                    doc1_vec = torch.tensor(reps[1]).to(device)
                    doc2_vec = torch.tensor(reps[2]).to(device)

                    #concatenate query and doc representation
                    inputs1 = torch.concat((query_vec, doc1_vec), dim=-1)
                    inputs2 = torch.concat((query_vec, doc2_vec), dim=-1)


                outputs = net(inputs1.float(), inputs2.float())
                loss = criterion(outputs.squeeze(), labels)

                eval_loss += loss.item()        
            print(f'Epoch {epoch + 1} dev loss: {eval_loss / len(dev_dataloader)}')            
        
    if scoring:
        torch.save(net.state_dict(), f=f'./models/{representation}_pairwise_scoring.model')
    else:
        torch.save(net.state_dict(), f=f'./models/{representation}_pairwise.model')

def test(representation, modelfile):
    """Gets as input the representation name (e.g. "tf_idf")"""
    #net = Net(vector_size*2)
    #net.load_state_dict(torch.load(modelfile))


a = PairwiseDataset("dev_tf_idf.pickle")