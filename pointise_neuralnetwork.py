import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle5 as pickle
from collections import defaultdict
from torch.optim import AdamW
import itertools
import numpy as np
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, input_size, scoring):
        super(Net, self).__init__()
        if scoring:
            self.fc1 = nn.Linear(input_size, 16) 
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 8)
            self.fc4 = nn.Linear(8, 1)
        else:
            self.fc1 = nn.Linear(input_size, 256) 
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

class PointwiseDataset(Dataset):
    #for the clear representation based 
    def __init__(self, filename, bows_filename=None):
        self.data = pickle.load(open(filename, "rb" ))
        self.data = [list(elem) for elem in self.data]

        self.bows = None
        if bows_filename:
            self.bows = pickle.load(open(bows_filename, "rb" ))

        #Fix empty entries resulting form empty queries/docs
        for i in range(len(self.data)):
            if type(self.data[i][2]) is float:               
                self.data[i][2] = np.zeros(self.data[0][2].size)                
            if type(self.data[i][3]) is float:               
                self.data[i][3] = np.zeros(self.data[0][3].size)                


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx][0:2]
        label = self.data[idx][-1]
        representations = self.data[idx][2:4]

        if self.bows:            
            #Add BOW for naive bayes
            representations = np.concatenate(representations, self.bows[idx])
            return ids, representations, label
        else:
            return ids, representations, label

def train(representation, scoring, epochs=2, batch_size=16, bart_emb_type=None):
    """Gets as input the representation name (e.g. "tf_idf")"""
    if scoring:
        train_dataset = PointwiseDataset(filename = f"preprocessed_data/train_{representation}.pickle", bows_filename=f"preprocessed_data/train_count_vector.pickle")
        dev_dataset = PointwiseDataset(f"preprocessed_data/dev_{representation}.pickle", bows_filename=f"preprocessed_data/train_count_vector.pickle")
        vector_size = 3 #Number of scoring functions

    else:
        train_dataset = PointwiseDataset(filename = f"preprocessed_data/train_{representation}.pickle")
        dev_dataset = PointwiseDataset(f"preprocessed_data/dev_{representation}.pickle")
        vector_size = train_dataset.data[0][3].shape[-1] * 2 #Get size of representations    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    
    net = Net(vector_size, scoring).to(device)
    criterion = nn.BCEWithLogitsLoss() #Binary Cross Entropy Loss
    optimizer = AdamW(net.parameters())

    for epoch in range(epochs): 
        net.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            ids, reps, labels = data

            if bart_emb_type:
                    reps[0] = get_bart_embeddings(reps[0], bart_emb_type)
                    reps[1] = get_bart_embeddings(reps[1], bart_emb_type)

            if scoring:
                cosine = get_batch_cosines(reps[0], reps[1])
                jacc = get_batch_jaccard(reps[0], reps[1])                
                nb = get_naive_bayes(reps[2])
                inputs = torch.tensor(cosine, jacc, nb).to(device)
            else:
                repr1 = torch.tensor(reps[0]).to(device)
                repr2 = torch.tensor(reps[1]).to(device)
                inputs = torch.concat((repr1, repr2), dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float().to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 1 and i > 1000:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        
        #Evaluate
        with torch.no_grad():
            net.eval()
            eval_loss = 0.0
            for i, data in enumerate(dev_dataloader):
                ids, reps, labels = data

                if scoring:
                    cosine = get_batch_cosines(reps[0], reps[1])
                    jacc = get_batch_jaccard(reps[0], reps[1])                
                    nb = get_naive_bayes(reps[2])
                    inputs = torch.tensor(cosine, jacc, nb).to(device)
                else:
                    repr1 = torch.tensor(reps[0]).to(device)
                    repr2 = torch.tensor(reps[1]).to(device)
                    inputs = torch.concat((repr1, repr2), dim=1)              

                outputs = net(inputs.float())
                loss = criterion(outputs, labels.float().to(device))
                eval_loss += loss.item()
            
            print(f'Epoch {epoch + 1} dev loss: {eval_loss / len(dev_dataloader)}')            
            
    if scoring:
        torch.save(net.state_dict(), f=f'./models/{representation}_pointwise_scoring.model')
    else:
        torch.save(net.state_dict(), f=f'./models/{representation}_pointwise.model')



def test(representation, scoring, batch_size):
    """Gets as input the representation name (e.g. "tf_idf")"""
    
    
    if scoring:
        test_dataset = PointwiseDataset(filename = f"preprocessed_data/test_{representation}.pickle", bows_filename=f"test_count_vector.pickle")
        vector_size = 3 #Number of scoring functions
        model_file=f'./models/{representation}_pointwise_scoring.model'

    else:
        test_dataset = PointwiseDataset(filename = f"preprocessed_data/test_{representation}.pickle")        
        vector_size = test_dataset.data[0][3].shape[-1]  * 2  #Get size of representations
        model_file=f'./models/{representation}_pointwise.model'


    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    net = Net(vector_size, scoring).to(device)
    net.load_state_dict(torch.load(model_file))

    #Evaluate

    #Save here the (queryID, docID), labels and model_score for evaluation
    query_ids = []
    doc_ids = []
    scores = []

    with torch.no_grad():
        net.eval()
        eval_loss = 0.0
        for i, data in enumerate(test_dataloader):
            ids, reps, labels = data

            if scoring:
                cosine = get_batch_cosines(reps[0], reps[1])
                jacc = get_batch_jaccard(reps[0], reps[1])                
                nb = get_naive_bayes(reps[2])
                inputs = torch.tensor(cosine, jacc, nb).to(device)
            else:
                repr1 = torch.tensor(reps[0]).to(device)
                repr2 = torch.tensor(reps[1]).to(device)
                inputs = torch.concat((repr1, repr2), dim=1)            

            outputs = net(inputs.float())
            outputs = outputs.cpu().numpy()
            
            scores.extend(outputs.squeeze().tolist())
            query_ids.extend(list(ids[:][0]))
            doc_ids.extend(list(ids[:][1]))

    test_outputs = defaultdict(list)
    for i in range(len(query_ids)):
        test_outputs[query_ids[i]].append((doc_ids[i], scores[i]))

    if scoring:
        filename = f'model_predictions/{representation}_pointwise_scoring_preds.pickle'
    else:
        filename = f'model_predictions/{representation}_pointwise_preds.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)



