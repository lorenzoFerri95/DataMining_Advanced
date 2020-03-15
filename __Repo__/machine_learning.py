import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


import data_understanding
import data_preparation
import file_handling

"""
https://stats.stackexchange.com/questions/46368/cross-validation-and-parameter-tuning

https://scikit-learn.org/stable/modules/model_evaluation.html

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

https://scikit-learn.org/stable/modules/cross_validation.html
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

https://scikit-learn.org/stable/modules/grid_search.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html   #### OTTIMO
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
"""




def main():

    X_train, X_test, y_train, y_test = file_handling.setup_df('train.csv', 'outClass', 'test.csv')

    model = DecisionTreeClassifier()
    model.get_params(deep=False)
    params_domain_dic = {'min_samples_leaf': range(1,200)}

    clf = RandomizedSearchCV(model, param_distributions=params_domain_dic,
    n_iter=100, scoring='f1_macro', random_state=0)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_

    y_pred = clf.predict(X_test)

    print('Accuracy: %s' % accuracy_score(y_test, y_pred))
    print()
    print('F1-score: %s' % f1_score(y_test, y_pred, average=None))
    print()
    print('Average F1-score: %s' % clf.score(X_test, y_test))
    print()
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()