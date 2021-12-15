import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from iotpackage.Utils import perLabelSample, NON_IOT, loadFeatureData, getCommonLabelData, renameLabels, findOptimalThreshold, plotAUCCurve, plotCM, getDevicesDataset, getDeviceNameAndNumber, getDatasetAndDevice, getCategoryMapping, addToListMapping, findCategory, remapLabel, normalizeFeatureData
from iotpackage.FeatureSelection import FeatureSelector
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from progressbar import progressbar, ProgressBar
from psutil import cpu_count
from datetime import datetime
import math
from iotpackage.__vars import companyCategories, generalCategories, renameDevices, companyDevices
from copy import deepcopy
from iotpackage.Filters import NonIotFilter

CPU_CORES = cpu_count()

class ModelTraining:
    mainClassifier = None
    fs = None
    printDetails = None
    labelCol = None

    removeThreshold = 10
    
    def __init__(self, fs=None, labelCol='Device', print_details=False, on_data_load=None):
        self.mainClassifier = RandomForestClassifier(n_estimators=100, n_jobs=CPU_CORES)
        self.printDetails = print_details
        self.labelCol = labelCol
        self.onDataLoad = on_data_load
        if fs:
            self.fs = fs
        else:
            raise ValueError('Please Provide a Feature Selector')

    def filterLabelData(self, data, labels):
        # Sanity Checks
        if (not isinstance(data, pd.DataFrame)):
            raise ValueError(f'Expected "data" to be type pd.DataFrame. Given {type(data)}')
        if (not isinstance(labels, list)):
            raise ValueError(f'Expected "labels" to be type list. Give {type(labels)}')
        
        data = data[data[self.labelCol].isin(labels)]
        data.reset_index(drop=True, inplace=True)
        
        data_d = set(list(data[self.labelCol].unique()))

        if (data_d == set(labels)):
            return data
        else:
            raise Exception(f"filterLabelData: Sanity Check Failed")
    def removeLessThan(self, threshold, data):
        # Explicity set to device to ensure devices less than the threshold get removed
        vc = data['Device'].value_counts()
        return data[data['Device'].isin(list(vc[vc >= threshold].index))]
    
    def loadData(self, load_train=True, load_test=True, convertDeviceType=True, prepend_dataset=False, rename_non_iot=True, rename_similar=True, normalize=True):
        if self.labelCol == 'Company-Categories':
            print('loadData: Renaming to Company')
            mappings = companyCategories
        elif self.labelCol == 'General-Categories':
            print('loadData: Renaming to General')
            mappings = generalCategories
        elif self.labelCol == "Company-Device":
            print('loadData: Renaming to Company Devices')
            mappings = companyDevices
        else:
            print(f'loadData: Using {self.labelCol}')
            mappings = None

        if load_train and self.trainDatasets:
            data_arr = []
            for dataset in self.trainDatasets:
                data = loadFeatureData(dataset, rename_non_iot=rename_non_iot, rename_similar=rename_similar, normalize=normalize)
                if convertDeviceType:
                    for device in data['Device'].unique():
                        device_type = getDeviceNameAndNumber(device)
                        data.loc[data['Device'] == device, 'Device'] = device_type
                if prepend_dataset:
                    for device in data['Device'].unique():
                        data.loc[data['Device'] == device, 'Device'] = dataset + '-' + device
                data_arr.append(data)
            train_data = pd.concat(data_arr, ignore_index=True)
            train_data = self.removeLessThan(self.removeThreshold, train_data)
            if type(mappings) != type(None):
                train_data = renameLabels(train_data, 'Device', self.labelCol, mappings)
            if self.labelCol not in list(train_data.columns):
                raise Exception(f'Label Col: {self.labelCol}, not in train_data.columns {list(train_data.columns)}')
        else:
            train_data = None
        if load_test and self.testDatasets:
            data_arr = []
            for dataset in self.testDatasets:
                data = loadFeatureData(dataset, rename_non_iot=rename_non_iot, rename_similar=rename_similar, normalize=normalize)
                if convertDeviceType:
                    for device in data['Device'].unique():
                        device_type = getDeviceNameAndNumber(device)
                        data.loc[data['Device'] == device, 'Device'] = device_type
                if prepend_dataset:
                    for device in data['Device'].unique():
                        data.loc[data['Device'] == device, 'Device'] = dataset + '-' + device
                data_arr.append(data)
            test_data = pd.concat(data_arr, ignore_index=True)
            test_data = self.removeLessThan(self.removeThreshold, test_data)
            if type(mappings) != type(None):
                test_data = renameLabels(test_data, 'Device', self.labelCol, mappings)
        else:
            test_data = None

        if (self.onDataLoad):
            if self.printDetails: print("Calling 'onDataLoad'", flush=True)
            train_data, test_data = self.onDataLoad(train_data, test_data)
        return train_data, test_data
    @staticmethod
    def getFeatureNames(clf, X, n=None, save_path=None, verbose=0):
        feature_importances = pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
        if n:
            feature_importances = feature_importances.iloc[:n,:]
        if save_path is not None:
            feature_importances.reset_index(drop=False).to_csv(save_path + '-feature_importances.csv', index=False)
        if verbose:
            print(feature_importances)
        return

    @staticmethod
    def getPerLabelMetrics(y_true, y_pred, save_path=None, verbose=0):
        per_label_metrics = pd.DataFrame()
        vc = y_true.value_counts()
        per_label_metrics.loc[:, 'Label'] = list(vc.index)
        precisions, recalls, _ ,_ = precision_recall_fscore_support(y_true, y_pred, labels=list(vc.index), average=None)
        per_label_metrics.loc[:, 'Precision'] = precisions
        per_label_metrics.loc[:, 'Recall'] = recalls
        per_label_metrics.loc[:, 'Count'] = list(vc)
        if save_path is not None:
            per_label_metrics.to_csv(save_path, index=False)
        if verbose:
            print(per_label_metrics.to_string())

    @staticmethod
    def getTopErrors(y_true, y_pred, store_file=True, print_details=False, plot_cm=False, save_path=None):
        plotCM(y_true, y_pred, store_cm=save_path, plot_cm=plot_cm)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        error_tf = (y_true != y_pred)
        true_labels = y_true[error_tf]
        pred_labels = y_pred[error_tf]
        if true_labels.size != pred_labels.size:
            raise Exception(f'Sizes should be equal {true_labels.size} and {pred_labels.size}')
        counts = pd.DataFrame({
            'True Label': true_labels,
            'Pred Label': pred_labels,
        }).groupby(['True Label', 'Pred Label']).size()
        if print_details:
            print(counts)
        if store_file:
            with open(f'{save_path}-top_errors.txt', 'a') as f:
                f.write('\n\n')
                f.write('---------------------------------' + '\n')
                f.write(datetime.now().strftime("%d %h %Y %H:%M:%S") + '\n')
                f.write(counts.to_string())
        return counts
        
    @staticmethod
    def getMetrics(y_train_true:'Series', y_train_pred:'Series', y_test_true:'Series', y_test_pred:'Series', average:'str'='macro') -> 'tst_acc, trn_acc, prs, rcl, fsc': 
        if average is None:
            average = 'macro'
        if (y_train_true is not None) and (y_train_pred is not None):
            train_accuracy = accuracy_score(y_train_true, y_train_pred)
        else:
            train_accuracy = None
        if (y_test_true is not None) and (y_test_pred is not None):
            test_accuracy = accuracy_score(y_test_true, y_test_pred)
            [precision, recall, fscore, support] = precision_recall_fscore_support(y_test_true, y_test_pred, average=average)
        else:
            test_accuracy = None
            precision = None
            recall = None
            fscore = None
        return test_accuracy, train_accuracy, precision, recall, fscore
    def printConfig(self):
        print('-------------RUN CONFIG-------------')
        print('Train Datasets    ->', self.trainDatasets)
        print('Test Datasets     ->', self.testDatasets)
        print('Cross Validation  ->', self.cv)
        print('Label             ->', self.labelCol)
        print('Run Type          ->', self.runType)
        print('-----------------------------------')
    @staticmethod
    def printMetrics(test_accuracy, train_accuracy, precision, recall, fscore):
        if train_accuracy is not None:
            print("Train Accuracy :", round(train_accuracy, 5))
        if test_accuracy is not None:
            print("Test Accuracy  :", round(test_accuracy, 5))
        if precision is not None:    
            print("Test Precision :", round(precision, 5))
        if recall is not None:
            print("Test Recall    :", round(recall, 5))
        if fscore is not None:
            print("Test Fscore    :", round(fscore, 5))
        return

    def fitMainClassifier(self, X, y, features=False, save_path=None):
        X.fillna(0, inplace=True)
        self.mainClassifier.fit(X, y)
        if features:
            self.getFeatureNames(self.mainClassifier, X, save_path=save_path)

    def predict(self, X):
        return self.mainClassifier.predict(X)
    def predict_probs(self, X):
        return self.mainClassifier.predict_proba(X)

    def main(self, train_data, test_data=None, return_metrics=False, print_metrics=True, errors=True, features=True, save_path=None, plot_cm=False, per_label_metrics=None, metric_average=None):
        if isinstance(train_data, pd.DataFrame):
            self.fs.fit(train_data, parallel=True)
            X_train = self.fs.transform(train_data)
            y_train = train_data[self.labelCol]
            self.fitMainClassifier(X_train, y_train, features=features, save_path=save_path)
            y_train_pred = self.predict(X_train)
        else:
            raise ValueError(f'train_data should be pd.DataFrame given {type(train_data)}')
        if isinstance(test_data, pd.DataFrame):
            X_test = self.fs.transform(test_data)
            y_test = test_data[self.labelCol]
            y_test_pred = self.predict(X_test)
        else:
            X_test = None
            y_test = None
            y_test_pred = None
        if per_label_metrics and y_test_pred is not None:
            self.getPerLabelMetrics(y_test, y_test_pred, save_path=save_path + '-per_label.csv', verbose=0)
        if errors:
            self.getTopErrors(y_test, y_test_pred, plot_cm=plot_cm, save_path=save_path)
        if print_metrics:
            tst_acc, trn_acc, tst_prs, tst_rcl, tst_fsc = self.getMetrics(y_train, y_train_pred, y_test, y_test_pred, average=metric_average)
            self.printMetrics(tst_acc, trn_acc, tst_prs, tst_rcl, tst_fsc)
        if return_metrics:
            return self.getMetrics(y_train, y_train_pred, y_test, y_test_pred, average=metric_average)

class MultiDatasetCombinedClassifier(ModelTraining):
    trainDatasets = None
    testDatasets = None
    cv = None
    _testSize = 0.2
    load_rename_iot = None
    load_rename_similar = None
    load_convert_device_type = None
    metrics_per_label = None
    def __init__(self, train_datasets=None, test_datasets=None, fs=None, cv=0, label_col='Device', print_details=True, test_size=0.2, on_data_load=None, load_rename_iot=True, load_rename_similar=True, load_convert_device_type=True, metrics_per_label=False):
        ModelTraining.__init__(self, fs, label_col, print_details, on_data_load=on_data_load)
        if not train_datasets:
            raise ValueError(f'Atleast 1 dataset must be provided passed = {train_datasets}')
        else:
            self.trainDatasets = train_datasets
        self.cv = cv
        self.runType = 'Single/Multi Dataset Combined Classifer'
        self.testDatasets = None
        self._testSize = test_size
        self.load_rename_iot = load_rename_iot
        self.load_rename_similar = load_rename_similar
        self.load_convert_device_type = load_convert_device_type
        self.metrics_per_label = metrics_per_label
    def run(self, resultPath=None, return_metrics=False, errors=True, runs=1, sample_size=None, features=False, labels='all', plot_cm=False):
        self.printConfig()
        data, _ = self.loadData(load_test=False, rename_non_iot=self.load_rename_iot, rename_similar=self.load_rename_similar, convertDeviceType=self.load_convert_device_type)
        
        # Filter Label Data if not going to use all labels for this run
        if labels != 'all':
            data = self.filterLabelData(data, labels)
            print('Labels: ', data[self.labelCol].unique())
        
        result = pd.DataFrame()
        
        for r in range(runs):
            data = data.sample(frac=1).reset_index(drop=True)
            if self.cv > 0:
                cv_i = 0
                kf = StratifiedKFold(n_splits=self.cv)
                metrics = {
                    'trn_acc': [],
                    'tst_acc': [],
                    'tst_prs': [],
                    'tst_rcl': [],
                    'tst_fsc': []
                }
                for train_index, test_index in kf.split(data, data[self.labelCol]):
                    cv_i += 1
                    train_data = data.iloc[train_index]
                    test_data = data.iloc[test_index]
                    if sample_size is not None:
                        train_data = train_data.groupby(self.labelCol, as_index=False).head(sample_size)
                    test_accuracy, train_accuracy, precision, recall, fscore = self.main(train_data, test_data, features=features, return_metrics=True, print_metrics=False, errors=errors, per_label_metrics=self.metrics_per_label, save_path=resultPath + f'-{r}-{cv_i}')
                    metrics['trn_acc'].append(train_accuracy)
                    metrics['tst_acc'].append(test_accuracy)
                    metrics['tst_prs'].append(precision)
                    metrics['tst_rcl'].append(recall)
                    metrics['tst_fsc'].append(fscore)
                mean_trn_acc = np.mean(metrics['trn_acc'])
                mean_tst_acc = np.mean(metrics['tst_acc'])
                mean_tst_prs = np.mean(metrics['tst_prs'])
                mean_tst_rcl = np.mean(metrics['tst_rcl'])
                mean_tst_fsc = np.mean(metrics['tst_fsc'])
                result.loc[r, 'Test Accuracy'] = mean_tst_acc
                result.loc[r, 'Test Precision'] = mean_tst_prs
                result.loc[r, 'Test Recall'] = mean_tst_rcl
                result.loc[r, 'Test Fscore'] = mean_tst_fsc
                result.loc[r, 'Train Accuracy'] = mean_trn_acc
                result.loc[r, 'Num Labels'] = len(data[self.labelCol].unique())
                result.to_csv(resultPath + '.csv', index=False)
                self.printMetrics(mean_tst_acc, mean_trn_acc, mean_tst_prs, mean_tst_rcl, mean_tst_fsc)
            else:
                train_data, test_data = train_test_split(data, test_size=self._testSize)
                
                if sample_size is not None:
                    train_data = train_data.groupby(self.labelCol, as_index=False).head(sample_size)
                    print(f'Sampled to Sample Size\n\tTotal Train Samples {train_data.shape[0]}\n\tUnique Labels {train_data[self.labelCol].nunique()}\n\tSample Size{sample_size}')
                
                tst_acc, trn_acc, tst_prs, tst_rcl, tst_fsc = self.main(train_data, test_data, return_metrics=True, print_metrics=False, errors=errors, features=features, per_label_metrics=self.metrics_per_label, plot_cm=plot_cm, save_path=resultPath + f'-{r}')
                
                result.loc[r, 'Test Accuracy'] = tst_acc
                result.loc[r, 'Test Precision'] = tst_prs
                result.loc[r, 'Test Recall'] = tst_rcl
                result.loc[r, 'Test Fscore'] = tst_fsc
                result.loc[r, 'Train Accuracy'] = trn_acc
                result.loc[r, 'Num Labels'] = int(data[self.labelCol].nunique())
                result.to_csv(resultPath + '.csv', index=False)

class MultiDatasetCommonClassifier(ModelTraining):
    trainDatasets = None
    testDatasets = None
    cv = None
    _testSize = 0.2
    dropNonIot = None
    def __init__(self, train_datasets=None, test_datasets=None, fs=None, label_col='Device', drop_non_iot=False, print_details=True):
        ModelTraining.__init__(self, fs, label_col, print_details)
        self.trainDatasets = train_datasets
        self.testDatasets = test_datasets
        self.runType = 'Single/Multi Dataset Common Classifier'
        self.dropNonIot = drop_non_iot
    def run(self, result_path=None, runs=10, return_metrics=False, errors=False, data_size=1.0, features=False):

        if data_size == 1 or data_size == 1.0:
            frac_data_size = None
            abs_data_size = None
        elif isinstance(data_size, float) and data_size < 1.0 and data_size > 0:
            frac_data_size = data_size
            abs_data_size = None
        elif isinstance(data_size, int) and data_size > 1:
            frac_data_size = None
            abs_data_size = data_size
        else:
            raise Exception(f"data_size must either be a 'float' between 0 and 1 or 'int' of greater than one, actual type: '{type(data_size)}' and value: '{data_size}'")
        
        self.printConfig()
        print('Result Path:', result_path)
        train_data, test_data = self.loadData(normalize=False)
        if self.dropNonIot:
            if self.printDetails: print('Dropping Non-IoT')
            train_data = train_data[~train_data[self.labelCol].isin('NoN-IoT', 'Non-IoT')].reset_index(drop=True)
            test_data = test_data[~test_data[self.labelCol].isin('NoN-IoT', 'Non-IoT')].reset_index(drop=True)

        if self.printDetails: print('Initial Data Train / Test', train_data.shape, test_data.shape)

        train_data, _ , test_data, _ = getCommonLabelData(train_data, test_data, label_col=self.labelCol)
        if self.printDetails: print('Common Data Train / Test', train_data.shape, test_data.shape)

        n_train = len(train_data[self.labelCol].unique())
        n_test = len(test_data[self.labelCol].unique())
        if n_train != n_test:
            raise Exception(f'Number of devices should be similar. Given {n_train} and {n_test}')
        
        results = pd.DataFrame()
        for r in range(runs):
            if frac_data_size is not None:
                if self.printDetails: print(f'frac_data_size: {frac_data_size}')
                train_data_run = train_data.sample(frac=frac_data_size).reset_index(drop=True)
            elif abs_data_size is not None:
                if self.printDetails: print(f'abs_data_size: {abs_data_size}')
                train_data_run = perLabelSample(train_data, int(abs_data_size))
            else:
                train_data_run = train_data.copy()
            if self.printDetails: print('After Sample Data Train / Test', train_data_run.shape, test_data.shape)
            tst_acc, trn_acc, tst_prs, tst_rcl, tst_fsc = self.main(train_data_run, test_data, return_metrics=True, print_metrics=False, errors=errors, features=features, save_path=f'{result_path}-{r}')
            results.loc[r, 'Accuracy'] = tst_acc
            results.loc[r, 'Precision'] = tst_prs
            results.loc[r, 'Recall'] = tst_rcl
            results.loc[r, 'F1 Score'] = tst_fsc
            results.loc[r, 'Train Accuracy'] = trn_acc
            results.loc[r, 'Devices'] = n_train
            if result_path is not None:
                results.to_csv(result_path + '.csv', index=False)
            else:
                print(results.to_string())

class KnownUnknownClassifier(ModelTraining):
    _testSize = 0.2
    def __init__(self, train_datasets=None, test_datasets=None, fs=None, cv=0, label_col='Device', print_details=True):
        print('KnownInit:', label_col)
        ModelTraining.__init__(self, fs, label_col, print_details)
        self.trainDatasets = train_datasets
        self.testDatasets = test_datasets
        self.cv = cv
        self.runType = 'Known VS Unknown Classifer'
    
    def fitSecondaryClassifier(self, data, label):
        if isinstance(label, str):
            label = data[label]
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(data, label)
        return clf
    def predictSecondaryClassifier(self, clf, data):
        return clf.predict(data)
    def predictSecondaryClassifierProbs(self, clf, data):
        return clf.predict_probs(data)
    def splitMapRandom(self, dim, devices, split, balance=False):
        known_list = []
        unknown_list = []
        devices = set(devices)
        for d in dim:
            n_elems = len(dim[d])
            if n_elems > 1 or d == 'leftover':
                n = self.ratioToN(split, len(dim[d]))
                odd = (n_elems % 2)
                if odd and balance:
                    known = set(np.random.choice(dim[d], int(n), replace=False))
                    unknown = set(dim[d]) - known
                else:
                    unknown = set(np.random.choice(dim[d], int(n), replace=False))
                    known = set(dim[d]) - unknown
                if odd:
                    balance = not balance
                known_list.extend(list(known))
                unknown_list.extend(list(unknown))
                devices = devices - set(dim[d])
        return known_list, unknown_list, devices, balance
    
    def randomSplit(self, devices, unknown_ratio=0.5, stop_level=None, non_iot=False):
        """Randomly splits the devices into two sets

        Parameters
        ----------
        devices : list
            The devices to be split
        unknown_ratio : number
            ratio or percentage of unknown devices
        stop_level : None
            Not used
        non_iot : None
            Not used

        Returns
        -------
        list
            the list of known devices
        list
            the list of unknown devices
        """
        if unknown_ratio < 0 or unknown_ratio > 1:
            raise ValueError(f'unknown_ratio must be in the range 0-1. Actual {unknown_ratio}')
        if not isinstance(devices, list):
            raise ValueError(f'devices must be a list. Actual {type(devices)}')

        if non_iot:
            non_iot_devices_set = set(devices) & set(NON_IOT)
            iot_devices = set(devices) - non_iot_devices_set
            n_unknown = self.ratioToN(unknown_ratio, len(list(non_iot_devices_set)))
            unknown_devices = list(np.random.choice(list(non_iot_devices_set), n_unknown, replace=False))
            known_devices = list(non_iot_devices_set - set(unknown_devices))
            devices = list(iot_devices)
        else:
            known_devices = []
            unknown_devices = []
            
        devices = set(devices)
        n_unknown = self.ratioToN(unknown_ratio, len(devices))
        if self.printDetails: print('Number Unknown:', n_unknown)
        unknown_devices.extend(list(np.random.choice(list(devices), n_unknown, replace=False)))
        known_devices.extend(list(devices - set(unknown_devices)))
        return known_devices, unknown_devices
    
    def ratioToN(self, ratio, total):
        if total < 1:
            raise Exception("total was less than 1", total)
        N = None
        if ratio >= 0.5:
            N = int(ratio * total)
        else:
            N = math.ceil(ratio * total)
        if N == 0:
            N = 1
        if N == total:
            N = total - 1
        return N
        

    def knownUnknownSplit(self, devices, unknown_ratio=0.5, stop_level=None, non_iot=False):
        """Strategicly splits the devices into known and unknown using additional information such as device categories

        Parameters
        ----------
        devices : list
            The devices to be split
        unknown_ratio : number
            ratio or percentage of unknown devices
        stop_level : str
            If we want to stop at some level of split e.g if we don't want to split devices further than company category
        non_iot : bool
            if we are using the non-iot filter layer

        Returns
        -------
        list
            the list of known devices
        list
            the list of unknown devices
        """
        if unknown_ratio > 1 or unknown_ratio < 0:
            raise ValueError(f'unknown_ratio must be within range 0-1. Actual {unknown_ratio}')
        
        if non_iot:
            # Put n_unknown device (according to split ratio) into unknown and rest in known
            non_iot_list = list(set(devices) & set(NON_IOT)) # List of devices in the data that are Non-IoT
            print('Non IoT Lise', non_iot_list)
            n_unknown = self.ratioToN(unknown_ratio, len(non_iot_list))
            unknown_list = list(np.random.choice(non_iot_list, n_unknown, replace=False))
            known_list = list(set(non_iot_list) - set(unknown_list))
            print(known_list, unknown_list)
        else:
            known_list = []
            unknown_list = []
        
        devices = set(devices) - set(NON_IOT)
        if stop_level == 'Device':
            n_unknown = self.ratioToN(unknown_ratio, len(devices))
            u_l = set(list(np.random.choice(devices, n_unknown, replace=False)))
            k_l = devices - u_l
            known_list = known_list + list(k_l)
            unknown_list = unknown_list + list(u_l)
            return list(known_list), list(unknown_list)
        # Company Category Split (Split YourThings-Amazon Echo Plus & UNSW-Amazon Echo Look)
        dim = getCategoryMapping(devices, companyCategories)
        k_l, u_l, devices, b = self.splitMapRandom(dim, devices, unknown_ratio)
        known_list.extend(k_l)
        unknown_list.extend(u_l)
        
        if stop_level == 'Company-Categories':
            print('Stopped at Company')
            return known_list, unknown_list
        # General Category Split (Split YourThings-Google Home & UNSW-Apple HomePod)
        dim = getCategoryMapping(devices, generalCategories)
        k_l, u_l, devices, b = self.splitMapRandom(dim, devices, unknown_ratio, b)
        known_list.extend(k_l)
        unknown_list.extend(u_l)

        if stop_level == 'General-Categories':
            print('Stopped at General')
            return known_list, unknown_list
        
        # Ones That Are Left Assign Them Randomly
        dim = {'leftover': list(devices)}
        k_l, u_l, devices, _ = self.splitMapRandom(dim, devices, unknown_ratio, b)
        known_list.extend(k_l)
        unknown_list.extend(u_l)
        
        # Sanity Check To Verify All Have Been Assigned
        mapping_list = known_list + unknown_list
        if not all([d in mapping_list for d in devices]):
            raise Exception(f'Some Device(s) Have Not Been Assigned')
        return known_list, unknown_list
    def computeCategoryMetrics(self, y_test_true_labels, y_test_pred_labels, y_test_true_binary, y_test_pred_binary, mappings):
    
        y_test_true_labels = pd.Series(y_test_true_labels)
        y_test_pred_labels = pd.Series(y_test_pred_labels)
        if y_test_true_labels.shape != y_test_pred_labels.shape:
            raise Exception(f"Shapes not equal {y_test_true_labels.shape} and {y_test_pred_labels.shape}")
        
        # Show The Accuracy of Binary Labels
        if self.printDetails:
            tst_acc, _, tst_prs, tst_rcl, tst_fsc = self.getMetrics(None, None, y_test_true_binary, y_test_pred_binary)
            print('Accuracy of Binary Labels:')
            self.printMetrics(tst_acc, None, tst_prs, tst_rcl, tst_fsc)
        
        # Show The Accuracy of Known Test Data
        if self.printDetails:
            loc_true_known = y_test_true_binary == 1
            y_test_true_labels_known = y_test_true_labels[loc_true_known]
            y_test_pred_labels_known = y_test_pred_labels[loc_true_known]
            tst_acc, _, tst_prs, tst_rcl, tst_fsc = self.getMetrics(None, None, y_test_true_labels_known, y_test_pred_labels_known)
            print('Accuracy of True Known Test Device Labels:')
            self.printMetrics(tst_acc, None, tst_prs, tst_rcl, tst_fsc)
        
        # Convert to Category Labels
        y_test_true_remapped = y_test_true_labels.apply(remapLabel, args=(mappings,))
        y_test_pred_remapped = y_test_pred_labels.apply(remapLabel, args=(mappings,))


        # Remap unknown labels to mappings provided
        loc_pred_unknown = y_test_pred_binary == 0
        y_test_true_unknown_remapped = y_test_true_remapped[loc_pred_unknown]
        y_test_pred_unknown_remapped = y_test_pred_remapped[loc_pred_unknown]

        # Some more quick sanity checks
        if y_test_true_unknown_remapped.shape != y_test_pred_unknown_remapped.shape:
            raise Exception(f"Remapped True and Pred Shape Mismatch {y_test_true_unknown_remapped.shape} and {y_test_pred_unknown_remapped.shape}")

        # Compute the metrics
        u_tst_acc, _, u_tst_prs, u_tst_rcl, u_tst_fsc = self.getMetrics(None, None, y_test_true_unknown_remapped, y_test_pred_unknown_remapped)
        if self.printDetails:
            print('Unknown Metrics:')
            self.printMetrics(u_tst_acc, None, u_tst_prs, u_tst_rcl, u_tst_fsc)

        loc_pred_known = y_test_pred_binary == 1
        if not all(loc_pred_known == (~loc_pred_unknown)):
            raise Exception(f"Known and Unknown should be extactly opposite")
        y_test_true_known = y_test_true_labels[loc_pred_known]
        y_test_pred_known = y_test_pred_labels[loc_pred_known]

        k_tst_acc, _, k_tst_prs, k_tst_rcl, k_tst_fsc = self.getMetrics(None, None, y_test_true_known, y_test_pred_known)

        if self.printDetails:
            print('Known Metrics:')
            self.printMetrics(k_tst_acc, None, k_tst_prs, k_tst_rcl, k_tst_fsc)

        return (u_tst_acc, u_tst_prs, u_tst_rcl, u_tst_fsc), (k_tst_acc, k_tst_prs, k_tst_rcl, k_tst_fsc)
    def fit_known_unknown_classifier(self, known_data, features=False, save_path=None):
        self.main(known_data, None, errors=False, features=features, save_path=save_path)
        return
        
    def predict_known_unknown_classifier(self, testval_data, known_labels, save_roc_path=None, predict_category=None, save_probs=None, run=None):
        if testval_data.shape[0] == 0:
            #Everything Filtered Out
            if self.printDetails: print("Everything filtered out. Saving 0 as all metrics")
            metric_results = pd.Series()
            metric_results['auc'] = 0
            metric_results['opt_threshold'] = 0
            metric_results['accuracy'] = 0
            metric_results['precision'] = 0
            metric_results['recall'] = 0
            metric_results['fscore'] = 0
            metric_results.to_csv(save_roc_path + '-metrics.csv')
            return None, None

        testval_binary = testval_data[self.labelCol].apply(lambda x: 1 if x in known_labels else 0)
        idx = testval_binary.groupby(by=testval_binary, group_keys=False).apply(lambda x: x.sample(frac=0.5)).index
        tfidx = np.array([x in idx for x in testval_binary.index])
        
        val_data = testval_data.loc[~tfidx]
        X_val = self.fs.transform(val_data)
        y_val_true_labels = val_data[self.labelCol]
        y_val_true_binary = testval_binary[~tfidx]
        
        test_data = testval_data.loc[tfidx]
        X_test = self.fs.transform(test_data)
        y_test_true_labels = test_data[self.labelCol]
        y_test_true_binary = testval_binary[tfidx]

        # Compute the Labels and Probabilities of Labels
        y_pred_prob_test = pd.DataFrame(self.predict_probs(X_test), columns=self.mainClassifier.classes_)
        y_pred_prob_val = pd.DataFrame(self.predict_probs(X_val), columns=self.mainClassifier.classes_)
        y_scores_test = y_pred_prob_test.apply(lambda x: np.max(x), axis=1)
        y_scores_val = y_pred_prob_val.apply(lambda x: np.max(x), axis=1)
        y_val_pred_labels = self.predict(X_val)
        y_test_pred_labels = self.predict(X_test)

        # Compute the ROC Curve and optThreshold
        fpr, tpr, threshold = roc_curve(y_val_true_binary, y_scores_val)
        auc_score = roc_auc_score(y_val_true_binary, y_scores_val)
        optFPR, optTPR, optThreshold = findOptimalThreshold(fpr, tpr, threshold)

        # Print Details
        if self.printDetails: print('Optimal Threshold:', optThreshold, 'AUC', auc_score)

        # Make Binary predictions based on optimal threshold
        y_test_pred_binary = y_scores_test.apply(lambda x: 1 if x >= optThreshold else 0)

        # Quick Sanity Checks
        if y_test_pred_binary.shape != y_test_true_binary.shape:
            raise Exception(f"True Binary Labels Shape: {y_test_true_binary.shape} must be equal to Predicted Binary Labels Shape {y_test_pred_binary.shape}")
        elif not all(list(map(lambda x: x in [1,0], y_test_pred_binary.unique()))):
            raise Exception(f"Predicted Binary Labels should be in [1,0], Found : {y_test_pred_binary.unique()}")
        elif not all(list(map(lambda x: x in [1,0], y_test_true_binary.unique()))):
            raise Exception(f"True Binary Labels should be in [1,0], Found : {y_test_true_binary.unique()}")
            # Save the selected results
        
        if save_probs is not None:
            save_df = pd.DataFrame()
            if run is None:
                raise Exception(f'Value for run was None in trainClassifer')
            y_pred_prob_test.to_csv(save_probs + f'-{run}-probs.csv', index=False)
            test_data['Pred Category']
            save_df['True Labels'] = y_test_true_labels
            save_df['True Binary'] = y_test_true_binary
            save_df['Pred Binary'] = y_test_pred_binary
            save_df['Opt Threshold'] = optThreshold
            save_df['Pred Labels'] = y_test_pred_labels
            save_df.to_csv(save_probs + f'-{run}-labels.csv', index=False)
            if predict_category is not None:
                category_df = pd.DataFrame()
                category_df['Pred Category'] = test_data['Pred Category']
                category_df['True Category'] = test_data[predict_category]
                category_df.to_csv(save_probs + f'-{run}-category.csv', index=False)
                category_df = None
            return None, None
        if save_roc_path is not None:
            roc_results = pd.DataFrame()
            roc_results['fpr'] = fpr
            roc_results['tpr'] = tpr
            roc_results['thresholds'] = threshold
            metric_results = pd.Series()
            metric_results['auc'] = auc_score
            metric_results['opt_threshold'] = optThreshold
            acc, _, prs, rcl, fsc = self.getMetrics(None, None, y_test_true_binary, y_test_pred_binary, average='binary')
            metric_results['accuracy'] = acc
            metric_results['precision'] = prs
            metric_results['recall'] = rcl
            metric_results['fscore'] = fsc
            roc_results.to_csv(save_roc_path + '-ROC.csv', index=False)
            metric_results.to_csv(save_roc_path + '-metrics.csv')
            if self.printDetails: print('ROC and Metrics Results Saved')
        if predict_category is not None:
            
            unknown_results = pd.DataFrame()
            known_results = pd.DataFrame()
            if predict_category == 'Company-Categories':
                mappings = companyCategories
            elif predict_category == 'General-Categories':
                mappings = generalCategories
            elif predict_category == True:
                print('WARNING: Not Computing the Metrics and returning')
                return None, None
            else:
                raise Exception(f"Unknown categories specified: {predict_category}")
            
            # Get The Known and Unknown Metrics
            unknown_metrics, known_metrics = self.computeCategoryMetrics(y_test_true_labels, y_test_pred_labels, y_test_true_binary, y_test_pred_binary, mappings)
            
            #Store the Unknown Metrics
            u_acc, u_prs, u_rcl, u_fsc = unknown_metrics
            unknown_results.loc[0, 'Accuracy'] = u_acc
            unknown_results.loc[0, 'Precision'] = u_prs
            unknown_results.loc[0, 'Recall'] = u_rcl
            unknown_results.loc[0, 'Fscore'] = u_fsc

            #Store the Known Metrics
            k_acc, k_prs, k_rcl, k_fsc = known_metrics
            known_results.loc[0, 'Accuracy'] = k_acc
            known_results.loc[0, 'Precision'] = k_prs
            known_results.loc[0, 'Recall'] = k_rcl
            known_results.loc[0, 'Fscore'] = k_fsc

            #Return the results
            return unknown_results, known_results

    def run(self, resultPath, predict_category=None, runs=1, two_label_split=False, split_type='strategic', ratio=0.5, non_iot_filter=False, features=False):
        if self.printDetails: print('Starting Known vs Unknown')
        if self.printDetails: print(f'Result Path: {resultPath}')
        
        # Save and Restore Old Label Col while also adding an entry for the device category
        oldLabelCol = self.labelCol
        if predict_category is not None:
            self.labelCol = predict_category
        data, _ = self.loadData(rename_non_iot=False, rename_similar=True, normalize=False, load_test=False)
        self.test_size = 0.3
        self.labelCol = oldLabelCol

       
        # Identifying the labels available in the data
        all_labels = list(data[self.labelCol].unique())
        if self.printDetails: print("Total Labels: ", len(all_labels))

        # Sanity check to make sure labels are okay
        for label in all_labels:
            if label in NON_IOT:
                continue
            foundCompany = False
            for company in companyCategories:
                if label in companyCategories[company]:
                    foundCompany = True
            if not foundCompany:
                raise Exception(f'No Company Category found for: "{label}"')
                return
            foundGeneral = False
            for general in generalCategories:
                if label in generalCategories[general]:
                    foundGeneral = True
            if not foundGeneral:
                raise Exception(f'No General Category found for: "{label}"')
                return
            

        # Add a column named 'IoTFilter' to the dataframe. This will be the labels of the non-iot vs iot filter
        if non_iot_filter:
            label_name = 'IoTFilter'
            non_iot_loc = data[self.labelCol].isin(NON_IOT)
            data.loc[non_iot_loc, label_name] = 'Non-IoT'
            data.loc[~non_iot_loc, label_name] = 'IoT'
        
        # Verify The Args
        if predict_category is None and two_label_split:
            print('WARNING: two_label_split is only used with predict_category')
        elif predict_category is not None and two_label_split:
            two_label_split = predict_category
        elif predict_category is not None and not two_label_split:
            two_label_split = None
        
        if predict_category is not None:
            all_known_results = pd.DataFrame()
            all_unknown_results = pd.DataFrame()
        # Repeat for the number of runs
        for r in range(runs):
            
            # Split The Labels into Known and Unknown set
            if self.printDetails: print('Split Type:', split_type)
            if split_type == 'strategic':
                knownLabels, unknownLabels = self.knownUnknownSplit(all_labels, stop_level=two_label_split, non_iot=non_iot_filter)
            elif split_type == 'random':
                knownLabels, unknownLabels = self.randomSplit(all_labels, unknown_ratio=ratio, non_iot=non_iot_filter)
            # all_labels should be a superset of knownUnknownLabels when two_label_split is used (Stop setting)
            knownUnknownLabels = list(set(knownLabels) | set(unknownLabels))
            
            # Sanity Check: Checks if known and unknown labels are disjoint sets and from all_labels
            if not set(knownLabels).isdisjoint(set(unknownLabels)):
                raise Exception(f'Known and unknown sets should be disjoint but found element: {set(knownLabels) & set(unknownLabels)} in both')
            if not set(knownUnknownLabels).issubset(set(all_labels)):
                raise Exception(f'Labels should be from all_labels. {set(knownUnknownLabels) - set(all_labels)} is not')
            
            # Portion of data used for this run (containing only data from labels actually used)
            run_data  = data[data[self.labelCol].isin(knownUnknownLabels)].copy()
            run_data.reset_index(drop=True, inplace=True)
            train_data, test_data = train_test_split(run_data, test_size=self.test_size)
            train_data = train_data[train_data[self.labelCol].isin(knownLabels)]
            train_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            if non_iot_filter:
                if self.printDetails: print('IoT Filter Label Counts:',train_data[label_name].value_counts(), flush=True, sep="\n")
                iot_filter_fs = FeatureSelector(n_dict=0, n_tls_tcp=500, n_udp=500, n_dns=0, n_ntp=500, n_protocol=1000, one_hot_encode=True)
                self.nonIotFilter = NonIotFilter(iot_filter_fs, filtering_method='auto')
                self.nonIotFilter.fit(train_data, train_data[label_name])
            # Get the training data of only IoT devices
            train_data = train_data[~(train_data[self.labelCol].isin(NON_IOT))]
            train_data.reset_index(drop=True, inplace=True)

            if self.printDetails:
                print('Test Data Shape:', test_data.shape)
                print('Train Data Shape:', train_data.shape)
            test_data.loc[test_data[self.labelCol].isin(NON_IOT), self.labelCol] = 'Non-IoT'
            
            # Print The Details
            if self.printDetails:
                print('\nKnown Labels:')
                print(knownLabels)
                print('\nUnknown Labels:')
                print(unknownLabels)
            

            # Train and obtain results
            self.fit_known_unknown_classifier(train_data, features=features, save_path=f'{resultPath}-{r}')
            if predict_category is not None:
                secondaryClf = self.fitSecondaryClassifier(self.fs.transform(train_data), train_data[predict_category])

            # Filter out non iot traffic in test data based on non iot filter
            if non_iot_filter:
                test_data = self.nonIotFilter.predict(test_data, test_data[label_name], save_path=resultPath)
                if self.printDetails: print('Test Data (After Filter)',test_data.shape)
            if predict_category is not None:
                test_data['Pred Category'] = self.predictSecondaryClassifier(secondaryClf, self.fs.transform(test_data))
                unknown_results_run, known_results_run = self.predict_known_unknown_classifier(test_data, known_labels=list(knownLabels), save_roc_path=None, predict_category=predict_category, save_probs=resultPath, run=r)
                if unknown_results_run is not None:
                    all_unknown_results = pd.concat([all_unknown_results, unknown_results_run], ignore_index=True)
                if known_results_run is not None:
                    all_known_results = pd.concat([all_known_results, known_results_run], ignore_index=True)
                if all_known_results.shape[0]:
                    all_known_results.to_csv(resultPath + '-Known.csv', index=False)
                if all_unknown_results.shape[0]:
                    all_unknown_results.to_csv(resultPath + '-Unknown.csv', index=False)
            else:
                self.predict_known_unknown_classifier(test_data, known_labels=list(knownLabels), save_roc_path=resultPath + f'-{ratio}-{r}', predict_category=None)

class FingerprintingDevicesExp(ModelTraining):
    devices = None
    dataset_devices = {}
    def __init__(self, train_datasets=None, test_datasets=None, fs=None, cv=0, label_col='Device', print_details=True, devices='unique'):
        ModelTraining.__init__(self, fs, label_col, print_details)
        self.trainDatasets = train_datasets
        # Doesn't Use Test Datasets
        self.testDatasets = None
        self.cv = cv
        self.devices = devices
        self.runType = f'Fingerprint Devices Experiment: Devices {devices[0].upper() + devices[1:]}'

    def selectDevices(self):
        # Inputs The Run Type and Devices Per Dataset
        # Outputs the devices to load from that dataset
        self.dataset_devices = {}
        device_to_dataset_mapping = {}
        for dataset in self.trainDatasets:
            all_dataset_devices = getDevicesDataset(dataset, rename_similar=False)
            for device in all_dataset_devices:
                name = getDeviceNameAndNumber(device)
                entry = f'{dataset}-{device}'
                if name in device_to_dataset_mapping:
                    device_to_dataset_mapping[name].append(entry)
                else:
                    device_to_dataset_mapping[name] = [entry]
        
        if self.devices == 'unique':    
            for device in device_to_dataset_mapping:
                entry_selected = np.random.choice(device_to_dataset_mapping[device])
                dataset_selected, device_selected = getDatasetAndDevice(entry_selected)
                if dataset_selected in self.dataset_devices:
                    self.dataset_devices[dataset_selected].append(device_selected)
                else:
                    self.dataset_devices[dataset_selected] = [device_selected]
        
        elif self.devices == 'multi':
            for device in device_to_dataset_mapping:
                if len(device_to_dataset_mapping[device]) >= 2:
                    print(device_to_dataset_mapping[device])
                    for entry in device_to_dataset_mapping[device]:
                        dataset_selected, device_selected = getDatasetAndDevice(entry)
                        if dataset_selected in self.dataset_devices:
                            self.dataset_devices[dataset_selected].append(device_selected)
                        else:
                            self.dataset_devices[dataset_selected] = [device_selected]
            print('\n\n')
            print(self.dataset_devices)
        elif self.devices == 'all':
            pass
        else:
            raise ValueError(f"devices has to be one of ['unique', 'multi', 'all'] given {self.devices}")
    
    def loadData(self, load_train=True, load_test=False):
        print('Using Device')
        mappings = None
        self.labelCol = 'Device'

        if load_train and self.trainDatasets:
            data_arr = []
            for dataset in self.trainDatasets:
                print('Loading', dataset)
                data = loadFeatureData(dataset, rename_similar=False, shuffle=True)
                if dataset in self.dataset_devices:
                    devices = self.dataset_devices[dataset]
                    data = data[data[self.labelCol].isin(devices)]
                else:
                    print(f'Loading All Devices From {dataset}')
                    print('Data Shape Before Removing NoN-IoT:', data.shape)
                    data = data[data[self.labelCol] != 'NoN-IoT']
                    print('Data Shape After Removing NoN-IoT:', data.shape)
                # Append Dataset Name To All Devices
                data[self.labelCol] = data[self.labelCol].apply(lambda x: f'{dataset}-{x}')
                data_arr.append(data)
            train_data = pd.concat(data_arr, ignore_index=True)
            train_data = self.removeLessThan(self.removeThreshold, train_data)
            if mappings is not None:
                train_data = renameLabels(train_data, 'Device', self.labelCol, mappings)
        else:
            train_data = None

        if load_test:
            print(f'WARNING: Test Data will not be loaded in {self.runType}')
        
        return train_data, None
    def run(self, resultPath, runs=10):
        self.selectDevices()
        data,_ = self.loadData()
        all_devices_list = data[self.labelCol].unique()
        print('Total Devices', len(all_devices_list))
        print('Devices...')
        incr = 0
        print('All Devices', all_devices_list)
        devices_list = []
        if self.devices == 'multi':
            incr = 5
            multi_devices_list = []
            for device in all_devices_list:
                _,dd = getDatasetAndDevice(device)
                dt = getDeviceNameAndNumber(dd)
                if dt not in multi_devices_list:
                    multi_devices_list.append(dt)
            devices_list = multi_devices_list
        else:
            incr = 10
            devices_list = all_devices_list
        resultsDisplay = pd.DataFrame()
        resultsStore = pd.DataFrame()
        cv = 5
        countDisplay = 0
        countStore = 0
        for i in range(0, len(devices_list), incr):
            num_devices = min(i + incr, len(devices_list))
            print('Number of Devices =', num_devices)
            cr_metrics = {
                'trn_acc': [],
                'tst_acc': [],
                'tst_prs': [],
                'tst_rcl': [],
                'tst_fsc': []
            }
            for r in range(runs):
                print('Repeating:', r)
                if self.devices == 'multi':
                    device_types = np.random.choice(devices_list, num_devices, replace=False)
                    devices = []
                    for device in all_devices_list:
                        _,dd = getDatasetAndDevice(device)
                        dt = getDeviceNameAndNumber(dd)
                        if dd in device_types:
                            devices.append(device)
                else:
                    devices = np.random.choice(devices_list, num_devices, replace=False)
                data_devices = data[data[self.labelCol].isin(devices)]
                data_devices = data_devices.sample(frac=1)
                
                count = len(data_devices[self.labelCol].unique())
                if (count != num_devices) and self.devices != 'multi':
                    raise Exception(f'num_devices ({num_devices}) should be equal to devices in data ({count})')
                kf = KFold(n_splits=cv)
                cv_metrics = {
                    'trn_acc': [],
                    'tst_acc': [],
                    'tst_prs': [],
                    'tst_rcl': [],
                    'tst_fsc': []
                }
                for train_index, test_index in kf.split(data_devices):
                    train_data = data_devices.iloc[train_index]
                    test_data = data_devices.iloc[test_index]
                    tst_acc, trn_acc, tst_prs, tst_rcl, tst_fsc = self.main(train_data, test_data, return_metrics=True, print_metrics=False, errors=False, features=False)
                    cv_metrics['trn_acc'].append(trn_acc)
                    cv_metrics['tst_acc'].append(tst_acc)
                    cv_metrics['tst_prs'].append(tst_prs)
                    cv_metrics['tst_rcl'].append(tst_rcl)
                    cv_metrics['tst_fsc'].append(tst_fsc)
                mean_trn_acc = np.mean(cv_metrics['trn_acc'])
                mean_tst_acc = np.mean(cv_metrics['tst_acc'])
                mean_tst_prs = np.mean(cv_metrics['tst_prs'])
                mean_tst_rcl = np.mean(cv_metrics['tst_rcl'])
                mean_tst_fsc = np.mean(cv_metrics['tst_fsc'])
                resultsStore.loc[countStore, 'Num Devices'] = num_devices
                resultsStore.loc[countStore, 'Train Accuracy'] = mean_trn_acc
                resultsStore.loc[countStore, 'Test Accuracy'] = mean_tst_acc
                resultsStore.loc[countStore, 'Test Precision'] = mean_tst_prs
                resultsStore.loc[countStore, 'Test Recall'] = mean_tst_rcl
                resultsStore.loc[countStore, 'Test Fscore'] = mean_tst_fsc
                if self.devices == 'multi':
                    resultsStore.loc[countStore, 'Count'] = count
                    resultsStore.loc[countStore, 'Device Types'] = str(device_types)
                    resultsStore.loc[countStore, 'Devices'] = str(devices)

                countStore += 1
                resultsStore.to_csv(resultPath + '.csv', index=False)
                cr_metrics['trn_acc'].append(mean_trn_acc)
                cr_metrics['tst_acc'].append(mean_tst_acc)
                cr_metrics['tst_prs'].append(mean_tst_prs)
                cr_metrics['tst_rcl'].append(mean_tst_rcl)
                cr_metrics['tst_fsc'].append(mean_tst_fsc)
            resultsDisplay.loc[countDisplay, 'Num Devices'] = num_devices
            resultsDisplay.loc[countDisplay, 'Train Accuracy'] = np.mean(cr_metrics['trn_acc'])
            resultsDisplay.loc[countDisplay, 'Test Accuracy'] = np.mean(cr_metrics['tst_acc'])
            resultsDisplay.loc[countDisplay, 'Test Precision'] = np.mean(cr_metrics['tst_prs'])
            resultsDisplay.loc[countDisplay, 'Test Recall'] = np.mean(cr_metrics['tst_rcl'])
            resultsDisplay.loc[countDisplay, 'Test Fscore'] = np.mean(cr_metrics['tst_fsc'])
            print(resultsDisplay.to_string())
            countDisplay += 1

class MultiDatasetCombinedClassifierIoTvsNonIoT(ModelTraining):
    trainDatasets = None
    testDatasets = None
    cv = None
    _testSize = 0.2
    def __init__(self, train_datasets=None, test_datasets=None, fs=None, cv=0, label_col='Device', print_details=True):
        ModelTraining.__init__(self, fs, label_col, print_details)
        if not isinstance(train_datasets, list):
            raise ValueError(f'Atleast train_datasets must be a list, given: {type(train_datasets)}')
        else:
            self.trainDatasets = train_datasets
        self.cv = cv
        self.runType = 'Single/Multi Dataset Combined Classifer - IoT vs Non-IoT'
        self.testDatasets = None
    def run(self, result_path=None, return_metrics=False, errors=True, runs=1, features=False):
        self.printConfig()
        data, _ = self.loadData(load_test=False)

        iot_index = ~(data[self.labelCol] == 'NoN-IoT')
        if not data[iot_index].shape[0]:
            raise Exception('No Non-IoT Labelled Traffic')
        data.loc[iot_index, self.labelCol] = 'IoT'

        result = pd.DataFrame()
        for r in range(runs):
            data = data.sample(frac=1)
            if self.cv > 0:
                kf = KFold(n_splits=self.cv)
                metrics = {
                    'trn_acc': [],
                    'tst_acc': [],
                    'tst_prs': [],
                    'tst_rcl': [],
                    'tst_fsc': []
                }
                ir = 0
                for train_index, test_index in kf.split(data):
                    train_data = data.iloc[train_index]
                    test_data = data.iloc[test_index]
                    test_accuracy, train_accuracy, precision, recall, fscore= self.main(train_data, test_data, return_metrics=True, print_metrics=False, errors=errors, features=features, save_path=f'{result_path}-{r}-{ir}')
                    metrics['trn_acc'].append(train_accuracy)
                    metrics['tst_acc'].append(test_accuracy)
                    metrics['tst_prs'].append(precision)
                    metrics['tst_rcl'].append(recall)
                    metrics['tst_fsc'].append(fscore)
                    ir += 1
                result.loc[r, 'Test Accuracy'] = np.mean(metrics['tst_acc'])
                result.loc[r, 'Test Precision'] = np.mean(metrics['tst_prs'])
                result.loc[r, 'Test Recall'] = np.mean(metrics['tst_rcl'])
                result.loc[r, 'Test Fscore'] = np.mean(metrics['tst_fsc'])
                result.loc[r, 'Train Accuracy'] = np.mean(metrics['trn_acc'])
                result.loc[r, 'Num Labels'] = len(data[self.labelCol].unique())
                result.to_csv(result_path + '.csv', index=False)
                print(result.to_string())
            else:
                if return_metrics:
                    train_data, test_data = train_test_split(data, test_size=self._testSize)
                    test_accuracy, train_accuracy, precision, recall, fscore = self.main(train_data, test_data, return_metrics=return_metrics, print_metrics=False, errors=errors, features=False)
                    return [test_accuracy, precision, recall, fscore], data[self.labelCol].nunique()
                else:
                    train_data, test_data = train_test_split(data, test_size=self._testSize)
                    self.main(train_data, test_data, errors=errors, plot_cm=False, save_path=result_path)
            
class TargetVsNonTargetClassifier(ModelTraining):
    def __init__(self, fs, train_datasets=None, label_col='Device', print_details=False, on_data_load=None):
        ModelTraining.__init__(self, fs, label_col, print_details, on_data_load)
        if not isinstance(train_datasets, list):
            raise ValueError(f"train_datasets must be a list given: {type(train_datasets)}")
        self.trainDatasets = train_datasets
        self.runType = 'Target vs Non-Target Classifier'

    def randomlySplitDevices(self, devices, allDevices, ratio):
        n = math.ceil(len(devices) * ratio)
        trainingDevices = list(np.random.choice(devices, n, replace=False))
        testDevices = list(set(devices) - set(trainingDevices))

        leftDevices = list(set(allDevices) - set(devices))
        n = math.ceil(len(leftDevices) * ratio)
        trainingDevices.extend(list(np.random.choice(leftDevices, n, replace=False)))
        testDevices.extend(list(set(leftDevices) - set(trainingDevices)))
        return trainingDevices, testDevices

    def run(self, resultPath=None, runs=10, devices=[], features=True, min_balance=True):
        fullData, _ = self.loadData(load_test=False, rename_non_iot=True, rename_similar=True)

        allDevices = fullData['Device'].unique()

        for device in devices:
            if device not in allDevices:
                raise Exception(f'Device {device} not in all devices loaded from dataset: {allDevices}')
        
        # Selected devices are known all others are unknown
        loc = fullData['Device'].isin(devices)
        fullData.loc[loc, 'Label'] = 1
        fullData.loc[~loc, 'Label'] = 0

        metrics = pd.DataFrame()
        for r in range(runs):
            copyFullData = fullData.copy()
            print('Run:', r, flush=True)
            trainingDevices, testDevices = self.randomlySplitDevices(devices, allDevices, 0.5)
            print('Training Devices', trainingDevices, flush=True)
            print('Testing Devices', testDevices, flush=True)
            trainData, testData = train_test_split(copyFullData, test_size=0.2)
            
            # Remove the test devices data from the training data
            trainData = trainData[~trainData['Device'].isin(testDevices)]

            if min_balance:

                trainMin = trainData['Label'].value_counts().min()
                testMin = testData['Label'].value_counts().min()
                print('Train Min', trainMin, 'Test Min', testMin, flush=True)
                
                trainData = trainData.groupby('Label').apply(lambda x: x.sample(trainMin)).reset_index(drop=True)
                
                testData = testData.groupby('Label').apply(lambda x: x.sample(testMin)).reset_index(drop=True)
            else:

                trainData = trainData.reset_index(drop=True)
                testData = testData.reset_index(drop=True)

            trainData['Device'] = trainData['Label']
            testData['Device'] = testData['Label']

            print('trainData Label Value Counts:', trainData['Label'].value_counts(), flush=True, sep="\n")
            print('testData Label Value Counts:', testData['Label'].value_counts(), flush=True, sep="\n")

            tst_acc, trn_acc, tst_prs, tst_rcl, tst_fsc = self.main(trainData, testData, features=features, return_metrics=True, print_metrics=False, errors=False, per_label_metrics=True, save_path=resultPath + f'-{r}', metric_average='binary')
            metrics.loc[r, 'Train Accuracy'] = trn_acc
            metrics.loc[r, 'Test Accuracy'] = tst_acc
            metrics.loc[r, 'Test Precision'] = tst_prs
            metrics.loc[r, 'Test Recall'] = tst_rcl
            metrics.loc[r, 'Test Fscore'] = tst_fsc
            metrics.to_csv(resultPath + '.csv', index=False)
        