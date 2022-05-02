from tabnanny import verbose
import numpy as np
import pickle
import csv
import xgboost as xgb
import time
import pandas as pd
from collections import defaultdict

def read_pickle(filename: str): #Read pickle to get the info
    list_pickle =  pickle.load(open(filename,"rb"))
    return list_pickle

def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['qid'])])

def main():
    #Reading data, can be changed when we have the real data
    start = time.time()
    print("Reading pos_pickle")
    pos_pickle=read_pickle("pos_train_data.pickle")
    print("Reading neg_pickle")
    neg_pickle=read_pickle("neg_train_data.pickle")
    print("Reading training data")
    train_doc = defaultdict(list)    
    with open("msmarco-doctrain-top100", "r") as file:
        qrels_reader = csv.reader(file, delimiter=" ")
        for row in qrels_reader:
            train_doc[row[0],row[2]].append(float(row[4]))
    end = time.time()
    print("Time reading:", end-start)
    #Here we need to make the training data
    #For the test, we will just concanate the query_tfidf with the doc_tfidf
    x_val = []
    y_val = []
    qid = []
    for tuple in sorted(pos_pickle, key = lambda x: int(x[0])):
        aux = train_doc[tuple[0],tuple[1]]
        if(len(aux)!=0):
            y_val.append(float(aux.pop()))
        else:
            y_val.append(-100.0)
        qid.append(int(tuple[0]))
        x = np.append(tuple[2],tuple[4])
        x_val.append(x)
    
    #We will put the data in a Dataframe for easier split and data manipulation
    df = pd.DataFrame([x_val,y_val,qid]).transpose()
    df.columns=['tuple', 'rank', 'qid']

    p_train = 0.90 # Train porcentaje split.
    train = df[:int((len(df))*p_train)]
    test = df[int((len(df))*p_train):]
    x_train = train.loc[:, ~train.columns.isin(['qid','rank'])]
    y_train = train.loc[:, train.columns.isin(['rank'])]
    groups = train.groupby('qid').size().to_frame('size')['size'].to_numpy()
    x_predict = test.loc[:, ~test.columns.isin(['rank'])]
    #Model creation, we can give more restriction and variables in the future
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        booster="gbtree",
        tree_method='gpu_hist',
        learning_rate=0.1,
        max_depth=15
    )
    
    #The fit phase doesn't work becasuse we have different lengh vector, so we can't put in an array.
    model.fit(x_train.to_numpy(),y_train.to_numpy(), group=groups, verbose=True)

    predictions = (x_predict.groupby('qid').apply(lambda x: predict(model, x))) 
    model.save_model("PairwiseNN_1.0.json")
    print("Finish")

#I left this here for later predictions, there is for a NumpyDataframe, but we can use it as a baseline


if __name__ == "__main__":
    main()