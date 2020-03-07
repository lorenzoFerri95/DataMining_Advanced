import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools

from collections import defaultdict


def main():
    df = pd.read_csv('file')

    print('Primi Valori del Dataset:')
    df.head()
    print('Ultimi Valori del Dataset:')
    df.tail()
    print('Dimensioni del Dataset:\n',
            'Numero di Oggetti: {}'.format(df.shape[0]), '\n',
            'Numero di Attributi: {}'.format(df.shape[1]))
    print('Data Types degli Attributi:')
    df.dtypes
    print('Numero di Valori Nulli degli Attributi:')
    df.isnull().sum()
    print('Numero di Valori pari a 0 degli Attributi:')
    (df == 0).sum(axis=0)



""" ATTRIBUTI NUMERICI"""

def numeric_columns_distributions(df, dens=0):

    binsN = round(math.log(len(df), 2) + 1)  # numero di Bins dalla formula di Sturges
    numeric_columns = get_numeric_columns(df)

    for col in numeric_columns:
        if dens:
            df[col].plot(kind='kde')
            plt.ylabel(col + ' Densities', size=10)
        else:
            df[col].hist(bins=binsN, alpha=0.9)
            plt.ylabel(col + ' Frequencies', size=10)

        plt.title(col, pad=15, size=15)
        plt.xlabel(col + ' Values', size=10)
        plt.show()


def numeric_columns_boxplots(df, grouped_by=0):
    
    numeric_columns = get_numeric_columns(df)

    for col in numeric_columns:
        if grouped_by:
            df.boxplot(column=[col], by=grouped_by)
        else:
            df.boxplot(column=[col])
        
        plt.title(col, pad=30)
        plt.show()


def numeric_columns_scatterplot(df, grouped_by, transparence=1, gauss_kde=0):

    combs = itertools.combinations(get_numeric_columns(df), 2)

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
 
        plt.title('Scatter Plot {}  VS  {}'.format(col_comb[0], col_comb[1]), pad=20, fontsize=15)
        plt.xlabel(col_comb[0], fontsize=10)
        plt.ylabel(col_comb[1], fontsize=10)
        plt.legend()
        plt.show()




def get_numeric_columns(df):
    numeric_columns = df._get_numeric_data().columns
    numeric_columns = [
        col for col in numeric_columns if df[col].nunique() > 20]
    return numeric_columns



""" ATTRIBUTI CATEGORICI"""

def categoric_columns_distributions(df, grouped_by=0):

    categoric_columns = get_categoric_columns(df)

    for col in categoric_columns:

        if grouped_by:
            col_crosstab = pd.crosstab(df[col], df[grouped_by])
            col_crosstab_prob = col_crosstab.div(
                col_crosstab.sum(1).astype(float), axis=0)
            col_crosstab_prob.plot(kind='bar', stacked=True)
            plt.title(col + ' Crosstab by ' + grouped_by, pad=20, size=15)
            plt.ylabel('Cross Probabilities', size=10)
            
        else:
            df[col].value_counts().plot(kind="bar")
            plt.title(col, pad=15, size=15)
            plt.ylabel(col + ' Frequencies', size=10)
        
        plt.xlabel(col + ' Values', size=10)
        plt.show()


def get_categoric_columns(df):
    categoric_columns = df.select_dtypes(include=['object']).columns
    categoric_columns = [
        col for col in categoric_columns if df[col].nunique() < 50]
    return categoric_columns




if __name__ == "__main__":
    main()
