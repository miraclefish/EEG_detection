import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV
from test import plot_2D, plot_3D

class RFC:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_data_set(self, num_train, seed=0):
        X_data, y_data = shuffle(self.X, self.y, random_state=seed)
        X_train, y_train = X_data[:num_train], y_data[:num_train]
        X_test, y_test = X_data[num_train:], y_data[num_train:]
        return X_train, y_train, X_test, y_test

    def search_best_params(self, X_train, y_train, params):
        params_defult = {'n_estimators': [20],
                         'min_samples_split': [100],
                         'min_samples_leaf': [20],
                         'max_depth': [10],
                         'max_features': ['sqrt']}
        for item, value in params_defult.items():
            if not (item in params.keys()):
                params[item] = value
        gsearch = GridSearchCV(estimator=RandomForestClassifier(random_state=10),
                               param_grid=params,scoring='accuracy', cv=5)
        gsearch.fit(X_train, y_train)
        best_params = gsearch.best_params_
        print(best_params)
        rf = gsearch.best_estimator_
        return best_params, rf

    def rf_train(self, rf_best, X_train, y_train, X_test, y_test):

        rf_best.fit(X_train, y_train)
        y_pred_train = rf_best.predict(X_train)
        y_pred_test = rf_best.predict(X_test)
        train_accuracy = accuracy_score(y_pred_train, y_train)
        test_accuracy = accuracy_score(y_pred_test, y_test)
        # print("oob_score: ", rf_best.oob_score_)
        print("train accuracy: ", train_accuracy)
        print("test accuracy: ", test_accuracy)
        return y_pred_train, y_pred_test



    def plot_confusion_matrix(self,y_pred, y_true, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def report(self, y_pred, y_true, labels, target_names, digits = 3):
        print(classification_report(y_pred, y_true, labels=labels, target_names=target_names, digits=digits))

    def feature_selection(self, rf_best, x, y, select_num, plot2D = False, plot3D = False):

        n = x.shape[0]
        importance = rf_best.feature_importances_
        index = np.argsort(importance)[::-1]
        x_import = x[:, index[:, select_num]]
        assert (x_import.shape[0] == n & x_import.shape[1] == select_num)
        if plot2D:
            plot_2D(x_import, y, [0,1,2])
        if plot3D:
            plot_3D(x_import, y, [0,1,2])
        return x_import, index[:, select_num]