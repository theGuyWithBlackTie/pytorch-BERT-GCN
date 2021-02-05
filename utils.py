import os
import datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.stats as ss

import Dataset
import config


def cut_off_dataset(df, frequency):
    source_cut_data = df[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
    source_cut      = source_cut_data.source_id.value_counts()[(source_cut_data.source_id.value_counts() >= frequency)]
    source_id       = np.sort(source_cut.keys())
    df              = df.loc[df['source_id'].isin(source_id)]
    return df


def slicing_citation_text(df, number):
    df['leftSTRING'] = df['left_citated_text'].str[-number:]
    df['rightSTRING']= df['right_citated_text'].str[:number]
    return df


def split_dataset(df, year):
    ''' Splitting te dataset based on publishing year of publication '''
    trainIdx = df['target_year'][df['target_year'] < year].index
    testIdx  = df['target_year'][df['target_year'] >= year].index
    trainDF  = df.loc[trainIdx]
    testDF   = df.loc[testIdx]
    return trainDF, testDF


def get_label(df, trainDF, testDF):
    label = preprocessing.LabelBinarizer()
    label.fit_transform(df['source_id'].values)
    trainDF = convert_argmax(trainDF, label)
    testDF  = convert_argmax(testDF, label)
    #print('Shape of trainDF is: ',trainDF.shape,' and of testDF is: ',testDF.shape)
    return trainDF, testDF, label


def convert_argmax(df,label):
    y = df['source_id'].values
    y = label.transform(y)
    y = np.argmax(y, axis=1) # This returns the index of the label (or of 1) e.g. 0 0 0 0 1 0 0 0 -> 4

    df['LabelIndex'] = y
    return df


def loadDataset():
    dataFile = pd.read_csv("data/full_context_PeerRead.csv")
    column   = ['left_citated_text', 'right_citated_text', 'target_id', 'source_id', 'target_year', 'target_author']
    df       = dataFile[column]
    df       = cut_off_dataset(df, config.FREQUENCY)
    df       = slicing_citation_text(df, config.SEQ_LENGTH)


    trainDF, testDF                 = split_dataset(df, config.YEAR)
    trainDF, testDF, labelGenerator = get_label(df, trainDF, testDF)

    trainDF = trainDF.reset_index(drop=True)
    testDF  = testDF.reset_index(drop=True)


    trainDatatset = Dataset.BertBaseDataset(
        contextLeft=trainDF["leftSTRING"].values,  
        targetIndex = trainDF["LabelIndex"].values,
        contextRight=trainDF["rightSTRING"].values,
        isRight = config.isRight
        )

    testDatatset  = Dataset.BertBaseDataset(
        contextLeft=testDF["leftSTRING"].values,
        targetIndex = trainDF["LabelIndex"].values,
        contextRight=testDF["rightSTRING"].values,
        isRight = config.isRight
        )

    return trainDatatset, testDatatset, labelGenerator

def writeMetrics(text, path):
    print(text)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a+') as metricFile:
        metricFile.write(text)



def recall(eachLabelRank, real, topK):
    totalTestNos = len(real)
    for eachElem in topK:
        count = 0
        for index in range(0, totalTestNos):    # traversing through all rows
            labelIndex = real[index]
            #print('labelIndex: ',labelIndex,' ----- eachLabelRank[index][labelIndex]', eachLabelRank[index][labelIndex])
            if eachLabelRank[index][labelIndex] <= eachElem:
                count += 1
        result = count / totalTestNos
        text   = 'Recall@'+str(eachElem)+' : '+str(result)+'\n'
        writeMetrics(text, config.METRICS_PATH.format(config.modelName))



def mrr(eachLabelRank, real):
    totalTestNos = len(real)
    rrSum    = 0
    for index in range(0, totalTestNos):
        rank   = eachLabelRank[index][real[index]]
        rrSum += 1/rank
    
    mrrSum = rrSum / totalTestNos
    text   = 'mrr: '+str(mrrSum)+'\n'
    writeMetrics(text, config.METRICS_PATH.format(config.modelName))

'''
def map(eachLabelRank, real): # I am not sure whether this metric is correctly calculated or not
    realRankMatrix = np.zeros(eachLabelRank.shape)
    for index in range(0, len(real)):
        realRankMatrix[index][real[index]] = 1
'''

def metric(predicted, real):
    ranks = []
    for eachElem in predicted:
        ranks.append(len(eachElem) + 1 - ss.rankdata(eachElem))
    
    writeMetrics(str(datetime.datetime.now()), config.METRICS_PATH.format(config.modelName))
    topK = [5, 10, 30, 50, 80]
    print('Calculating Recalls Now...')
    recall(ranks, real, topK)

    print('Calculating Mean Reciprocal Rank (MRR) now...')
    mrr(ranks, real)

    #print('Calculating Mean Average Precision (MAP) now...')
    #map(ranks, real)


