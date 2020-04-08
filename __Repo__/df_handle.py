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



def setup_df(trainFileName, y_name=None, testFileName=None, testPortion=0.3, searchIn="../../"):
    
    if y_name:
        df_train = pd.read_csv(find_file(trainFileName, search_inside_folder = searchIn))
        
        X_names = [col for col in df_train.columns if col != y_name]

        if testFileName:
            df_test = pd.read_csv(find_file(testFileName, search_inside_folder = searchIn))
            
            X_train, X_test = df_train[X_names], df_test[X_names]
            y_train, y_test = df_train[y_name], df_test[y_name]

        else:
            X = df_train[X_names]
            y = df_train[y_name]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testPortion, random_state=100, stratify=y)

        return X_train, X_test, y_train, y_test

    else:
        filePath = find_file(trainFileName, search_inside_folder = searchIn)
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


def find_file(fileName, search_inside_folder="../../"):
    for root, dirs, files in os.walk(search_inside_folder, topdown=True):
        if fileName in files:
            return os.path.join(root, fileName)
    # If file is not found:
    raise FileNotFoundError("File  \'{}\'  Not Found".format(fileName))




if __name__ == "__main__":
    main()
