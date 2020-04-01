import os
import sys

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split



def main():
    df = setup_df('train.csv')
    X_train, X_test, y_train, y_test = setup_df('train.csv',  'outClass')
    X_train, X_test, y_train, y_test = setup_df('train.csv',  'outClass', 'test.csv')



def setup_df(trainFileName, outClass=None, testFileName=None, testPortion=0.3):
    
    if outClass:
        trainFilePath = find_file(trainFileName)
        df_train = pd.read_csv(trainFilePath)
        
        attributes = [col for col in df_train.columns if col != outClass]

        if testFileName:
            testFilePath = find_file(testFileName)
            df_test = pd.read_csv(testFilePath)
            
            X_train, X_test = df_train[attributes], df_test[attributes]
            y_train, y_test = df_train[outClass], df_test[outClass]

        else:
            X = df_train[attributes]
            y = df_train[outClass]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testPortion, random_state=100, stratify=y)

        return X_train, X_test, y_train, y_test

    else:
        filePath = find_file(trainFileName)
        df = pd.read_csv(filePath)

        return df




def export_results(X_test, y_test, clf_or_reg, exoprt_path, outFileName='pred_results'):

    final_testSet = X_test

    y_test_df = pd.DataFrame(np.array(y_test), index=final_testSet.index, columns=['y_test'])

    final_testSet = final_testSet.join(y_test_df)

    y_pred = clf_or_reg.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index=final_testSet.index, columns=['y_pred'])

    final_testSet = final_testSet.join(y_pred_df)

    final_testSet.to_csv(path_or_buf = exoprt_path + '\\' + outFileName + '.csv', sep=',', encoding='utf-8', index=False)


"""
https://stackoverflow.com/questions/1724693/find-a-file-in-python
https://www.geeksforgeeks.org/os-walk-python/
"""


def find_file(fileName):
    for root, dirs, files in os.walk("../../", topdown=True):
        if fileName in files:
            return os.path.join(root, fileName)
    # If file is not found:
    raise FileNotFoundError("File  \'{}\'  Not Found".format(fileName))




if __name__ == "__main__":
    main()
