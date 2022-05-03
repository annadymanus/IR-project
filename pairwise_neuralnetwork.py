from importlib.metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
from torch.optim import AdamW
import itertools

def evaluate(preds):
    """TODO: Some function which brings predictions [(qid, doc1id, doc2id, relevant_doc),...] into order and compares with true labels"""
    pass

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, y):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))

        z = x - y
        z = self.fc4(z)

        return z

def visualize_net():
    """get pretty picture"""
    from torchviz import make_dot
    x=torch.ones(10, requires_grad=True)
    y=torch.ones(10, requires_grad=True)
    net = Net(10)
    pred = net(x,y)
    make_dot(pred, params=dict(list(net.named_parameters()))).render("nn", format="png")

class PairwiseDataset(Dataset):
    def __init__(self, filename):
        raw_data = pickle.load(open(filename, "rb" ))
        sorted_data = defaultdict(list)
        for item in raw_data:
            sorted_data[item[0]] = item[1:]
        
        self.data = []      
        for key, value in sorted_data.items:   
            combinations = list(itertools.combinations(value, 2))
            for comb in combinations:
                assert comb[0][1] == comb[1][1]
                assert comb[0][-1] != comb[1][-1]
                self.data.append([key, comb[0][0], comb[1][0], torch.tensor(comb[0][1]), torch.tensor(comb[0][2]), torch.tensor(comb[1][2]), not bool(comb[0][-1])])                    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx][0:3]
        representations = self.data[idx][3:-1]
        label = self.data[idx][-1]
        return ids, representations, label

class SinglewiseDataset(Dataset):
    """For Testing"""
    pass


def train(representation, epochs=2):
    """Gets as input the representation name (e.g. "tf_idf")"""
    train_dataset = PairwiseDataset(f"train_{representation}.pickle")
    dev_dataset = PairwiseDataset(f"dev_{representation}.pickle")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64)

    vector_size = train_dataset.data[0][3].size()[-1] #Get size of representations
    net = Net(vector_size*2)

    criterion = nn.BCEWithLogitsLoss() #Binary Cross Entropy Loss
    optimizer = AdamW(net.parameters())

    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            ids, reprs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            #concatenate query and doc representation
            input1 = torch.concat((reprs[0],reprs[1]), dim=-1)
            input2 = torch.concat((reprs[0],reprs[2]), dim=-1)

            outputs = net(input1, input2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        net.eval()
        eval_loss = 0.0
        for i, data in enumerate(dev_dataloader):
            ids, reprs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            #concatenate query and doc representation
            input1 = torch.concat((reprs[0],reprs[1]), dim=-1)
            input2 = torch.concat((reprs[0],reprs[2]), dim=-1)

            outputs = net(input1, input2)
            loss = criterion(outputs, labels)

            eval_loss += loss.item()
        
        print(f'Epoch {epoch + 1} dev loss: {eval_loss / len(dev_dataloader)}')            

        
    
    torch.save(net.state_dict(), PATH=f'./{representation}_pairwise.model')

def test(representation, modelfile):
    """Gets as input the representation name (e.g. "tf_idf")"""
    #net = Net(vector_size*2)
    #net.load_state_dict(torch.load(modelfile))
    pass