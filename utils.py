import pandas as pd
import numpy as np
from sklearn import preprocessing

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
    df       = slicing_citation_text(df, config.MAX_LEN)


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

