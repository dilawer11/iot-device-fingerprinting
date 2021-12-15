import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from iotpackage.Utils import findOptimalThreshold, plotAUCCurve, plotCM, getDevicesDataset, getDeviceNameAndNumber, getDatasetAndDevice, getCategoryMapping, addToListMapping, findCategory, remapLabel
from iotpackage.FeatureSelection import FeatureSelector
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from progressbar import progressbar, ProgressBar
from psutil import cpu_count
from datetime import datetime
from iotpackage.__vars import companyCategories, generalCategories
import os

class NonIotFilter():
    fs = None #Feature Selector
    clf = None #Main Classifier
    filtering_method = None # The method you want to select to filter data when predicting
    printDetails = True

    def __init__(self, fs, n_estimators=100, n_jobs=-1, filtering_method='threshold'):
        self.fs = fs
        self.clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
        self.filtering_method = filtering_method
    def fit(self, data, y, positive_label='IoT'): # Fit Function Call Made Visisble
        y = y.apply(lambda x: 1 if x == positive_label else 0)
        self.fs.fit(data)
        X = self.fs.transform(data)
        self.clf.fit(X,y)
        return

    def predict(self, data, y=None, positive_label='IoT', save_path=None):   # Predict Function Call Made Visible
        y = y.apply(lambda x: 1 if x == positive_label else 0)
        y_pred_auto, y_pred_prob = self.__predict_clf(data)
        auc_score, opt_threshold = self.__compute_threshold(y, y_pred_prob)
        y_pred_threshold = self.__compute_pred_threshold(opt_threshold, y_pred_prob)
        
        # Saving the results in the csv file
        if save_path is not None:
            result_file = save_path + '-iot-non-iot.csv'
            result_columns = ['T-Accuracy', 'T-Precision', 'T-Recall', 'A-Accuracy', 'A-Precision', 'A-Recall']
            if os.path.exists(result_file):
                result_df = pd.read_csv(result_file)
            else:
                result_df = pd.DataFrame(columns=result_columns)
            idx = result_df.shape[0]
            t_acc = accuracy_score(y, y_pred_threshold)
            t_prs, t_rcl, _, _ = precision_recall_fscore_support(y, y_pred_threshold, average='binary')
            a_acc = accuracy_score(y, y_pred_auto)
            a_prs, a_rcl, _, _ = precision_recall_fscore_support(y, y_pred_auto, average='binary')
            result_df.loc[idx, 'T-Accuracy'] = t_acc
            result_df.loc[idx, 'T-Precision'] = t_prs
            result_df.loc[idx, 'T-Recall'] = t_rcl
            result_df.loc[idx, 'A-Accuracy'] = a_acc
            result_df.loc[idx, 'A-Precision'] = a_prs
            result_df.loc[idx, 'A-Recall'] = a_rcl
            result_df.to_csv(result_file, index=False)
        if self.filtering_method == 'auto':
            return data[y_pred_auto == 1].reset_index(drop=True)
        elif self.filtering_method == 'threshold':
            return data[y_pred_threshold == 1].reset_index(drop=True)
        else:
            raise Exception(f'filtering_method {self.filtering_method} not correct')
        return None

    def __compute_pred_threshold(self, threshold, y_pred_prob):
        y_pred_threshold = y_pred_prob.apply(lambda x: 1 if x >= threshold else 0)
        return y_pred_threshold

    def __compute_threshold(self, y_true, y_pred_prob):
        # Compute the ROC Curve and optThreshold
        fpr, tpr, threshold = roc_curve(y_true, y_pred_prob)
        auc_score = roc_auc_score(y_true, y_pred_prob)
        optFPR, optTPR, optThreshold = findOptimalThreshold(fpr, tpr, threshold)

        # Print Details
        if self.printDetails: print('Optimal Threshold:', optThreshold, 'AUC', auc_score)
        return auc_score, optThreshold

    def __predict_clf(self, data, prob=True):
        X = self.fs.transform(data)
        y_pred = self.clf.predict(X)
        if prob:
            y_pred_prob = pd.DataFrame(self.clf.predict_proba(X), columns=self.clf.classes_).loc[:,1]
            if self.printDetails: print('NonIotFilter: __predict_clf: y_pred_prob.shape', y_pred_prob.shape, self.clf.classes_)
            return y_pred, y_pred_prob
        else:
            return y_pred

