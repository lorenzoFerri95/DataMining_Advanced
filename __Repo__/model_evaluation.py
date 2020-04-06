import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import permutation_importance
import data_understanding

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


"""
https://scikit-learn.org/stable/modules/model_evaluation.html
https://stats.stackexchange.com/questions/46368/cross-validation-and-parameter-tuning
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://scikit-learn.org/stable/modules/cross_validation.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
"""



#############################################  CLASSIFICATION  ###############################################


def validate_clf(clf, X_train, y_train):

    '''valadazione del classificatore sul training set con una Corss-Validation che divide il dataset in 5 parti e fa 5 ierazioni:
    in ogni iterazione 4/5 del dataset vengono usati per istruire il modello (training) e il restante 1/5 viene usato per testarlo (test).
    in ogni iterazione viene calcolata la performance (prima con l'accuracy poi con l'f1 score) sul test set (del modello istruito sul training)
    infine vengono riportati i valori medi di accuracy ed f1 calcolati nelle 5 iterazioni.
    '''
    
    best_model_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print('Accuracy at a 95 percent confidence interval: %0.2f (+/- %0.2f)' % (
        best_model_scores.mean(), best_model_scores.std() * 2), end='\n\n')

    best_model_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
    print("F1 Score at a 95 percent confidence interval: %0.2f (+/- %0.2f)" % (
        best_model_scores.mean(), best_model_scores.std() * 2))



def feature_importance(clf, X_train, y_train):

    nbr_features = X_train.shape[1]

    try:
        tree_feature_importances = clf.feature_importances_
        sorted_idx = tree_feature_importances.argsort()[-nbr_features:]

        plt.barh(range(nbr_features), tree_feature_importances[sorted_idx])
        plt.yticks(range(nbr_features), X_train.columns[sorted_idx])
        plt.title("Feature Importances on Training Set")
        plt.show()

    except:
        perm_importance = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=100, n_jobs=2)

        sorted_idx = perm_importance.importances_mean.argsort()[-nbr_features:]

        plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx])
        plt.title("Permutation Importances on Training Set")
        plt.tight_layout()
        plt.show()



def decision_boundary_scatterplots(clf, X_train, y_train):

    numeric_columns = data_understanding.get_numeric_columns(X_train)
    combs = itertools.combinations(numeric_columns, 2)

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

    h = .02  # step size in the mesh

    for col_comb in combs:

        X=np.array(X_train[list(col_comb)])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        clf.fit(X, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 4))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y_train, edgecolors='k', s=20, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Decision Boundary {} vs {}'.format(col_comb[0], col_comb[1]), fontsize=20, pad=20)
        plt.xlabel(col_comb[0], fontsize=15)
        plt.ylabel(col_comb[1], fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=10)

        plt.show()




def test_clf(clf, X_test, y_test):
    
    y_pred = clf.predict(X_test)
    
    print('Accuracy: %s' % accuracy_score(y_test, y_pred), end='\n\n')
    print('F1-score: %s' % f1_score(y_test, y_pred, average=None), end='\n\n')
    print('Weighted Average F1-score: %s' % f1_score(y_test, y_pred, average='weighted'), end='\n\n')
    print(classification_report(y_test, y_pred), end='\n\n')

    
    y_score = clf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=3, label='$AUC$ = %.3f' % (roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()





#################################################  REGRESSION  #################################################


def test_reg(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))