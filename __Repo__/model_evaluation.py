
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


"""
https://scikit-learn.org/stable/modules/model_evaluation.html
https://stats.stackexchange.com/questions/46368/cross-validation-and-parameter-tuning
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://scikit-learn.org/stable/modules/cross_validation.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
"""



def classifier_validate(clf, X_train, y_train):
    
    best_model_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print('Accuracy at a 95 percent confidence interval: %0.2f (+/- %0.2f)' % (
        best_model_scores.mean(), best_model_scores.std() * 2), end='\n\n')

    best_model_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
    print("F1 Score at a 95 percent confidence interval: %0.2f (+/- %0.2f)" % (
        best_model_scores.mean(), best_model_scores.std() * 2))




def classifier_test(clf, X_test, y_test):
    
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

