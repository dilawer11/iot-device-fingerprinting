import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

from iotpackage.FeatureSelection import FeatureSelector
from sklearn.manifold import TSNE
from iotpackage.__vars import featureGroups, dictGroups

class PlotCM:
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def findMaxCountLabels(self, labels_array):
        counts = {}
        for labels in labels_array:
            key = tuple(labels)
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
        max_count = 0
        max_key = None
        for key in counts:
            if counts[key] > max_count:
                max_count = counts[key]
                max_key = key
        return list(max_key)
    def combineExpCM(self, cm_files, label_files):
        cmn_array = []
        labels_array = []
        
        total_files = len(cm_files) if len(cm_files) == len(label_files) else None
        
        for i in range(total_files):
            cmn = pd.read_csv(cm_files[i])
            labels = pd.read_csv(label_files[i])
            labels = list(labels.values.flatten())
            cmn_array.append(cmn)
            labels_array.append(labels)
        labels = self.findMaxCountLabels(labels_array)
        for i, labels_a in enumerate(labels_array):
            if labels != labels_a:
                labels_array.pop(i)
                cmn_array.pop(i)
        cmn_df = None
        for cmn in cmn_array:
            if cmn_df is None:
                cmn_df = cmn
            else:
                cmn_df = cmn_df.add(cmn)
        cmn_df = cmn_df.divide(len(cmn_array))
        return cmn_df, labels
            
        
    def plot(self, save_path=None):
            
        exp_files = list(map(lambda x: os.path.join(self.result_dir, x), os.listdir(self.result_dir)))
        cm_files = list(filter(lambda x: 'CM.csv' in x, exp_files))
        label_files = list(filter(lambda x: 'Labels.csv' in x, exp_files))
        print(f"Found {len(cm_files)} CM files")
        cm_files.sort()
        label_files.sort()

        cmn, labels = self.combineExpCM(cm_files, label_files)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False, ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.result_dir, 'CM.pdf')
        plt.savefig(save_path, format='pdf', dpi=1200)
        print(f"Plot saved in PDF format at: {save_path}")

class TSNEScatter:
    def __init__(self, input_dir):
        if os.path.isdir(input_dir):
            self.input_dir = input_dir
            self.dataset_pkl_file = os.path.join(input_dir, 'featureData.pkl')
        else:
            raise Exception("input_dir must be a directory")

    def runTSNE(self, data):
        sample_amount = 20
        data = data.groupby('Device').apply(lambda x: x.sample(min(x.shape[0], sample_amount))).reset_index(drop=True)
        fs = FeatureSelector()
        fs.fit(data)
        X = fs.transform(data)
        y = data['Device']
        tsne = TSNE(n_components=2, verbose=2, n_jobs=-1)
        tsne_results = tsne.fit_transform(X)
        data_plot = pd.DataFrame(tsne_results)
        data_plot.columns = ['Dimension 1', 'Dimension 2']
        data_plot['Device'] = y
        return data_plot


    def plot(self, save_path):
        data = pd.read_pickle(self.dataset_pkl_file)
        data.fillna(0, inplace=True)
        data_plot = self.runTSNE(data)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x='Dimension 1',
            y='Dimension 2',
            hue='Device',
        #     style='Company',
            palette=sns.color_palette('hls', data_plot['Device'].nunique()),
            data=data_plot,
            legend='full',
            alpha=1
        )
        plt.legend()
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.input_dir, 'tsne.pdf')
        plt.savefig(save_path, format='pdf', dpi=1200)
        print(f"Plot saved in PDF format at: {save_path}")

class FeatureImportanceTable:
    def __init__(self, input_dir):
        if os.path.isdir(input_dir):
            self.input_dir = input_dir
        else:
            raise Exception('input_dir must be a directory') 

    def getGroup(self, feature):
        for featureGroup in featureGroups:
            if feature in featureGroups[featureGroup]:
                return featureGroup
        for dictGroup in dictGroups:
            for dictGroupType in dictGroups[dictGroup]:
                if dictGroupType in feature:
                    return dictGroup
        return None
   
    def table(self, save_path):
        exp_files = list(map(lambda x: os.path.join(self.input_dir, x), os.listdir(self.input_dir)))
        feature_importances_files = list(filter(lambda x: 'feature_importances.csv' in x, exp_files))
        data_arr = []
        for fn in feature_importances_files:
            data = pd.read_csv(fn)
            data_arr.append(data)
        data = pd.concat(data_arr, ignore_index=True)
        # Aggregate the feature importances and take mean values
        data = data.groupby('index').mean().reset_index()
        
        data['group'] = data['index'].apply(self.getGroup)

        data = data.groupby('group').sum().sort_values(by='importance', ascending=False)
        
        if save_path is None:
            save_path = os.path.join(self.input_dir, 'GroupFeatureImportance.csv')
        
        data.to_csv(save_path)
        print(f"Table saved in CSV format at: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_dir", help="The directory or results to be used as input")
    parser.add_argument("--cm", action="store_true", help="Plots the Confusion Matrix with the given result files")
    parser.add_argument("--tsne", action="store_true", help="Plots the TSNE scatter plot using given dataset path (either dataset base dir or featureData.pkl file path as --input-dir arg")
    parser.add_argument("--group-fi", action="store_true", help="Computes the feature group importance from individual feature importances")
    parser.add_argument("-o", dest="save_path", default=None, help="To save the resulting item in this path")
    args = parser.parse_args()

    global IOTBASE
    IOTBASE = os.getenv('IOTBASE')
    if args.cm == True:
        p = PlotCM(args.input_dir)
        p.plot(args.save_path)
    elif args.tsne == True:
        p = TSNEScatter(args.input_dir)
        p.plot(args.save_path)
    elif args.group_fi == True:
        p = FeatureImportanceTable(args.input_dir)
        p.table(args.save_path)

if __name__ == "__main__":
    main()