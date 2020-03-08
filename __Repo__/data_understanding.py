import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools

from collections import defaultdict


def main():
    df = pd.read_csv('file')

    dfHead, dfTail, *other = dataset_state(df)
    dfHead

    statistics, pearson_corr, spearman_corr, kendall_corr = numeric_columns_stats(df)
    spearman_corr


def dataset_state(df):
    head = df.head()
    tail = df.tail()
    objectsN = df.shape[0]
    attributesN = df.shape[1]
    types = df.dtypes
    missValues = df.isnull().sum()
    nullValues = (df == 0).sum(axis=0)

    return head, tail, objectsN, attributesN, types, missValues, nullValues
    

    

""" ATTRIBUTI NUMERICI"""

def numeric_columns_stats(df):
    numeric_columns = get_numeric_columns(df)
    
    statistics = df[numeric_columns].describe()
    pearson_corr = df[numeric_columns].corr('pearson')
    spearman_corr = df[numeric_columns].corr('spearman')
    kendall_corr = df[numeric_columns].corr('kendall')

    return statistics, pearson_corr, spearman_corr, kendall_corr


def numeric_columns_distributions(df, density=0):

    binsN = round(math.log(len(df), 2) + 1)  # numero di Bins dalla formula di Sturges

    for col in get_numeric_columns(df):
        if density:
            df[col].plot(kind='kde')
            plt.ylabel(col + ' Densities', size=10)
        else:
            df[col].hist(bins=binsN, alpha=0.9)
            plt.ylabel(col + ' Frequencies', size=10)

        plt.title(col, pad=15, size=15)
        plt.xlabel(col + ' Values', size=10)
        plt.show()


def numeric_columns_boxplots(df, grouped_by=0):
    
    for col in get_numeric_columns(df):
        if grouped_by:
            df.boxplot(column=[col], by=grouped_by)
        else:
            df.boxplot(column=[col])
        
        plt.title(col, pad=30)
        plt.show()


def numeric_columns_scatterplot(df, grouped_by, transparence=1, gauss_kde=0):

    numeric_columns = get_numeric_columns(df)
    combs = itertools.combinations(numeric_columns, 2)

    for col_comb in combs:
        plt.figure(figsize=(6, 4))

        if gauss_kde:
            x = df[col_comb[0]].values
            y = df[col_comb[1]].values
            xy = np.vstack([x, y])
            z = stats.gaussian_kde(xy)(xy)
            
            plt.scatter(x, y, c=z, s=10)
        
        else:
            outClass = sorted(list(df[grouped_by].unique()))
            
            for i in outClass:
                x = df[col_comb[0]].loc[df[grouped_by] == i].values
                y = df[col_comb[1]].loc[df[grouped_by] == i].values
                
                plt.scatter(x, y,  label='Class = ' + str(i), s=10, alpha=transparence)
                plt.legend()
 
        plt.title('Scatter Plot {}  VS  {}'.format(col_comb[0], col_comb[1]), pad=20, fontsize=15)
        plt.xlabel(col_comb[0], fontsize=10)
        plt.ylabel(col_comb[1], fontsize=10)
        plt.show()
    
    pd.plotting.scatter_matrix(df[numeric_columns], figsize=(15, 10))




def get_numeric_columns(df):
    numeric_columns = df._get_numeric_data().columns
    numeric_columns = [
        col for col in numeric_columns if df[col].nunique() > 20]
    return numeric_columns



""" ATTRIBUTI CATEGORICI"""

def categoric_columns_distributions(df):
  
    for col in get_categoric_columns(df):
            
        df[col].value_counts().plot(kind="bar")
        plt.title(col, pad=15, size=15)
        plt.ylabel(col + ' Frequencies', size=10)
        plt.xlabel(col + ' Values', size=10)
        plt.show()


def categoric_columns_crosstab(df, grouped_by, probabilities=0):

    for col in get_categoric_columns(df):

        col_crosstab = pd.crosstab(df[col], df[grouped_by])

        if probabilities:
            col_crosstab = col_crosstab.div(
                col_crosstab.sum(1).astype(float), axis=0)
            col_crosstab.plot(kind='bar', stacked=True)
            plt.title(col + ' Crosstab by ' + grouped_by, pad=20, size=15)
            plt.ylabel('Cross Probabilities', size=10)
            plt.xlabel(col + ' Values', size=10)
            plt.show()

            print(col_crosstab)
        else:
            col_crosstab.plot(kind='bar', stacked=True)
            plt.title(col + ' Crosstab by ' + grouped_by, pad=20, size=15)
            plt.ylabel('Cross Frequencies', size=10)
            plt.xlabel(col + ' Values', size=10)
            plt.show()

            print(col_crosstab)


def get_categoric_columns(df):
    categoric_columns = df.select_dtypes(include=['object']).columns
    categoric_columns = [
        col for col in categoric_columns if df[col].nunique() < 50]
    return categoric_columns




if __name__ == "__main__":
    main()
