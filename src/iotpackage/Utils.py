import json
import os
import math
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn import metrics
from sklearn.preprocessing import scale, StandardScaler, normalize
import time
import matplotlib.pyplot as plt
import progressbar
from iotpackage.__vars import features, featureGroups, dictFeatures, renameDevices, permanentRename
loadedCategories = None
storedFeatureGroups = None
devicecol = 'Device'
categorycol = 'Category'
CSVcols = ['Frame','Time','SrcIP','DstIP','Proto','tcpSrcPort','tcpDstPort','udpSrcPort','udpDstPort','Length','tcpACK','tcpSYN','tcpFIN','tcpRST','tcpPSH','tcpURG','Protocol', 'srcMAC', 'dstMAC']
NON_IOT = ['iPhone', 'Android Tablet', 'HP Printer', 'Samsung Galaxy Tab', 'Laptop', 'IPhone', 'Android Phone', 'iPad', 'Ubuntu Desktop', 'MacBook', 'MacBook/Iphone', 'Nexus Tablet', 'Android Phone', 'Desktop', 'Motog phone', 'Router', 'Pixel 2 Phone']

def addToListMapping(mapping, key, value):
    if key in mapping:
        mapping[key].append(value)
    else:
        mapping[key] = [value]
def remapLabel(device, mapping):
    for m in mapping:
        if device in mapping[m]:
            return m
    raise Exception(f"No Mapping For Device: {device}")
def getCategoryMapping(devices, mapping):
    category_mapping = {}
    devices = set(devices)
    for device in devices:
        category = findCategory(mapping, device)
        if category is None:
            raise ValueError(f'No Company Category Mapping For Device: {device}')
        else:
            addToListMapping(category_mapping, category, device)
    return category_mapping
def findCategory(category_mapping, device):
    for category in category_mapping:
        if device in category_mapping[category]:
            return category
    return None
def getCommonLabels(data1, data2, label_col='Device', print_common=True):
    if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
        data1 = data1[label_col]
        data2 = data2[label_col]
    if not isinstance(data1, pd.Series):
        data1 = pd.Series(data1)
    if not isinstance(data2, pd.Series):
        data2 = pd.Series(data2)
    uniqueDevices_data1 = set(data1.unique())
    uniqueDevices_data2 = set(data2.unique())
    uniqueDevices_data1.discard('NoN-IoT')

    common_labels = list(uniqueDevices_data1.intersection(uniqueDevices_data2))
    if print_common:
        print('Common Labels:', common_labels)
    return common_labels

def findOptimalThreshold(fpr, tpr, thresholds):
    points = {}
    for i in range(0, len(thresholds)):
        points[thresholds[i]] = [fpr[i], tpr[i]]
    min = float('inf')
    threshold = None
    for k in points:
        try:
            [[i]] = metrics.pairwise.euclidean_distances([points[k]], [[0,1]])
        except:
            continue
        if i < min:
            min = i
            threshold = k
    return points[threshold][0], points[threshold][1], threshold

# Plots the Confusion Matrix. Also used to store the values to plot later
def plotCM(y_true, y_pred, store_cm=None, plot_cm=True):
    labels = list(y_true.unique())
    labels.sort()
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if isinstance(store_cm, str):
        #Path is provided to store cmn
        pd.DataFrame(cmn).to_csv(store_cm + '-CM.csv', index=False)
        pd.Series(labels).to_csv(store_cm + '-Labels.csv', index=False)
    if plot_cm:
        fig, ax = plt.subplots()
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

# Plots all the AUC curves used in the paper
def plotAUCCurve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Unknown Devices')
    plt.legend(loc="lower right")
    plt.show()

def getCommonLabelData(X1, X2, y1=None, y2=None, label_col=None, common_both=True, return_y=False, print_common=True):
    if label_col:
        y1 = X1[label_col]
        y2 = X2[label_col]
    elif y1 is None or y2 is None:
        raise ValueError('Either y1, y2 or label_col must be defined')
    X2.reset_index(drop=True, inplace=True)
    y2.reset_index(drop=True, inplace=True)
    commonDevices = getCommonLabels(y1, y2, print_common=print_common)
    common_loc2 = y2.isin(commonDevices)
    try:
        X2 = X2[common_loc2]
    except:
        print(X2)
        print(common_loc2)
    if return_y:
        y2 = y2[common_loc2]
    else:
        y2 = None
    if common_both:
        common_loc1 = y1.isin(commonDevices)
        X1 = X1[common_loc1]
        if return_y:
            y1 = y1[common_loc1]
        else:
            y1 = None
    return X1, y1, X2, y2

def getFeatureNames():
    return list(features.keys())
def getFeatureGroups():
    return featureGroups

def normalizeFeatureData(featureData):
    listSimpleFeatures = list(set(list(features.keys())) - set(list(dictFeatures.keys())))
    featureData[listSimpleFeatures] = normalize(featureData[listSimpleFeatures], axis=0)
    return featureData

def renameLabels(featureData, labelCol, destCol, mappings, error_raise=True):
    if not isinstance(featureData, pd.DataFrame):
        raise ValueError(f'featureData must be a Pandas DataFrame given {type(featureData)}')
    if not isinstance(labelCol, str):
        raise ValueError(f'labelCol must be a str given {type(labelCol)}')
    if not isinstance(destCol, str):
        raise ValueError(f'destCol must be a str given {type(destCol)}')
    if not isinstance(mappings, dict):
        raise ValueError(f'mappings must be of type dict given {type(mappings)}')

    for label in mappings:
        loc = featureData[labelCol].isin(mappings[label])
        featureData.loc[featureData[labelCol].isin(mappings[label]), destCol] = label
    featureData.loc[featureData[labelCol].isin(NON_IOT), destCol] = 'NoN-IoT'
    if featureData[destCol].isna().sum() and error_raise:
        raise Exception(f'No Mappings For {featureData.loc[featureData[destCol].isna(), labelCol].unique()}')
    else:
        return featureData

def renameSimilarDevices(devicesSeries):
    if not isinstance(devicesSeries, pd.Series): raise ValueError(f'Expected devicesSeries to be pandas.Series given {type(devicesSeries)}')
    
    for device in renameDevices:
        devicesSeries.loc[devicesSeries.isin(renameDevices[device])] = device

    return devicesSeries

def renameNonIoTDevices(devicesSeries):
    if not isinstance(devicesSeries, pd.Series): raise ValueError(f'Expected devicesSeries to be pandas.Series given {type(devicesSeries)}')

    devicesSeries.loc[devicesSeries.isin(NON_IOT)] = 'NoN-IoT'

    return devicesSeries

# This function loads the feature data and does some processing that some experiments might require e.g renaming non_iot to 'NoN-IoT' etc
def loadFeatureData(dataset_base_path, shuffle=True, normalize=True, fillna=True, rename_similar=True, rename_non_iot=True, verbose=0):
    # Sanity Checks
    if not os.path.exists(dataset_base_path):
        dataset_base_path = os.path.join(os.getenv('IOTBASE'), dataset_base_path)
        if not os.path.exists(dataset_base_path):
            raise FileNotFoundError(f'dataset_base_path: {dataset_base_path} does not exist')
    if os.path.isdir(dataset_base_path):
        feature_data_path = os.path.join(dataset_base_path, 'featureData.pkl')
    else:
        feature_data_path = dataset_base_path
    if not os.path.exists(feature_data_path):
        raise FileNotFoundError(f'devices_file: {feature_data_path} does not exist')
    
    # Loads from the disk
    featureData = pd.read_pickle(feature_data_path)
    if verbose: print('Pickle File Loaded', flush=True)
    # Performs post-loading operations
    if rename_non_iot:
        featureData.loc[:, 'Device'] = renameNonIoTDevices(featureData['Device'].copy())
        if verbose: print('Renamed NoN-IoT', flush=True)
    if rename_similar:
        featureData.loc[:, 'Device'] = renameSimilarDevices(featureData['Device'].copy())
        if verbose: print("Renamed Similar", flush=True)
    if fillna:
        featureData.fillna(0, inplace=True)
    if normalize:
        featureData = normalizeFeatureData(featureData)
        if verbose: print("Normalized Data", flush=True)
    if shuffle:
        if verbose: print('Shuffling Data...', flush=True)
        featureData = featureData.sample(frac=1).reset_index(drop=True)
    return featureData

# A helper function to get latex style graphs from data
def DataFrame2LatexTable(df, escape=True):
    latex_string = ""
    for col in df.columns:
        latex_string += " & " + col
    latex_string = latex_string.replace(' & ', "", 1)
    latex_string += " \\\\\n"
    vals = df.values
    for i in vals:
        row_string = ""
        for j in i:
            if j is None:
                row_string += " & "
            else:
                row_string += " & " + str(j)
        row_string = row_string.replace(" & ", "", 1)
        latex_string += row_string + " \\\\\n"
    if escape:
        latex_string = latex_string.replace('_','\_')
    return latex_string

def getDatasetAndDevice(entry):
    if '-' in entry:
        dataset, device = entry.split('-',1)
        if dataset not in Datasets:
            raise Exception(f'{dataset} is not a member of Datasets')
        return dataset, device
    else:
        raise Exception(f"'-' not found in entry {entry}")

def getDeviceNameAndNumber(device):
    if '-' in device:
        split = device.split('-')
        try:
            number = int(split[-1])
            name = split[0]
        except ValueError:
            return device
        return name
    else:
        return device

def getDevicesDataset(dataset_base_path, threshold=10, prepend_dataset=False, rename_non_iot=True, rename_similar=True):
    '''Get all the devices in the dataset'''
    if not os.path.exists(dataset_base_path):
        raise FileNotFoundError(f'dataset_base_path: {dataset_base_path} does not exist')
    if os.path.isdir(dataset_base_path):
        devices_file_path = os.path.join(dataset_base_path, 'devices.pkl')
    else:
        devices_file_path = dataset_base_path
    if not os.path.exists(devices_file_path):
        raise FileNotFoundError(f'devices_file: {devices_file_path} does not exist')
    devices_counts = pd.read_pickle(devices_file_path)
    devices_counts = devices_counts.reset_index(drop=False)
    devices_counts.columns = ['Device', 'Count']
    if threshold >= 0:
        devices_counts = devices_counts[devices_counts['Count'] >= threshold]

    if rename_non_iot:
        devices_counts['Device'] = renameNonIoTDevices(devices_counts['Device'].copy())
        
    if rename_similar:
        devices_counts['Device'] = renameSimilarDevices(devices_counts['Device'].copy())
    
    devices = list(devices_counts['Device'].unique())
    if prepend_dataset:
        devices = list(map(lambda x: dataset + '-' + x, devices))
    return devices

def perLabelSample(data, sample_size, label_col='Device'):
    labels = list(data[label_col].unique())
    data_array = []
    for label in labels:
        label_loc = data[label_col] == label
        sample = min(sample_size, data[label_loc].shape[0])
        data_array.append(data.loc[label_loc,:].sample(sample))
    return pd.concat(data_array, ignore_index=True).reset_index(drop=True)

# This function gets the largest latest run of an experiment and returns the number so next run can be stored as that number + 1 or the plotting files can use the latest run.
def getLargestRunNumber(exp_id, base_dir=os.path.join('Results', 'Experiments'), name_prefix="Exp"):
    exp_dir = os.path.join(base_dir, f'{name_prefix}{exp_id}')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        return 0
    files = os.listdir(exp_dir)
    r = 0
    for file_name in files:
        file_name = file_name.replace('.csv', '')
        run_number = int(file_name.split('-')[1]) #Exp##-{run_number}**
        r = max(run_number, r)
    return r

# This returns the base path to store the results of an experiment using the function above
def getResultPath(exp_id, exp_dir=os.path.join('Results', 'Experiments')):
    r = getLargestRunNumber(exp_id=exp_id, base_dir=exp_dir)
    r += 1
    file_name_template = f'Exp{exp_id}-{r}'
    resultPath = os.path.join(exp_dir, f'Exp{exp_id}',file_name_template)
    return resultPath



