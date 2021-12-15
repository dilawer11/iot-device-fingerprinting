import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from iotpackage.Utils import getFeatureNames, getFeatureGroups
from iotpackage.__vars import dictGroups, featureGroups
import progressbar
from multiprocessing import Pool, Process, Queue

class FeatureSelector():
    selectedSimpleFeatureNames = None
    simplegroups = None
    dictgroups = None
    printDetails = None
    oneHotEncode = None

    dv_dict_features = {}

    n_dict_ = None
    n_tls_tcp_ = None
    n_dns_ = None
    n_udp_ = None
    n_ntp_ = None
    n_protocol_ = None

    #CallBacks
    transformCallback = None
   
    # Temp 
    __combinedSeries = None
    def __init__(self, simple_groups='all', n_dict=0, n_tls_tcp=0, n_dns=0, n_udp=0, n_ntp=0, dict_groups='all', n_protocol=0, print_details=True, one_hot_encode=False, transform_callback=None):
        self.dv_dict_features = dict()
        self.n_dict_ = n_dict
        self.n_tls_tcp_ = n_tls_tcp
        self.n_dns_ = n_dns
        self.n_udp_ = n_udp
        self.n_ntp_ = n_ntp
        self.n_protocol_ = n_protocol

        self.printDetails = print_details
        self.oneHotEncode = bool(one_hot_encode)
        self.transformCallback = transform_callback
        
        if simple_groups == 'all':
            self.simplegroups = list(featureGroups.keys())
        else:
            self.simplegroups = simple_groups
        
        if dict_groups == 'all':
            self.dictgroups = list(dictGroups.keys())
        else:
            self.dictgroups = dict_groups
    def getNumDictGroups(self):
        return len(self.dictgroups)
    def getTopN(self, series, N=None):
        if N:
            return series.sort_values(ascending=False).iloc[:N]
        else:
            return series.sort_values(ascending=False)
    def reduceDicts(self, Series, elem):
        self.__combined = {}
        def acc(d):
            for k in d:
                if k in self.__combined:
                    self.__combined[k] += d[k]
                else:
                    self.__combined[k] = d[k]
        try:
            Series.apply(acc)
        except Exception as e:
            print('Error Here', e)
            print(self.__combinedSeries)
            print(Series)
            print(elem)
        finally:
            return pd.Series(self.__combined)

    def setVectorizers_Parallel_Helper(self, series, n, elem, q):
        series = self.reduceDicts(series, elem)
        series = self.getTopN(series, n)
        dv = DictVectorizer(sparse=False)
        dv.fit([series])
        q.put((dv, elem))
    def getNForFeature(self, elem):
        if elem == 'Protocol Dict':
            return max(self.n_protocol_, self.n_dict_)
        if 'TLSTCP' in elem:
            return self.n_tls_tcp_
        elif 'DNS' in elem:
            return self.n_dns_
        elif 'UDP' in elem:
            return self.n_udp_
        elif 'NTP' in elem:
            return self.n_ntp_
        elif 'Dict' in elem:
            return self.n_dict_
        else:
            raise ValueError(f'No Value for elem: {elem}')
    def setVectorizers_Parallel(self, featureData):
        processes = []
        for group in self.dictgroups:
            for elem in dictGroups[group]:
                n = self.getNForFeature(elem)
                if n > 0:
                    q = Queue()
                    p = Process(target=self.setVectorizers_Parallel_Helper, args=(featureData[elem], n, elem, q,))
                    processes.append((p,q))
        for p,_ in processes:
            p.start()
        for p,q in progressbar.progressbar(processes):
            dv, elem = q.get()
            self.dv_dict_features[elem] = dv
            p.join()
    def setVectorizers_Serial(self, featureData):
        for group in progressbar.progressbar(self.dictgroups):
            elems = dictGroups[group]
            for elem in elems:
                elem_Series = self.reduceDicts(featureData[elem])
                elem_Series = self.getTopN(elem_Series, self.n_dict_features[group])
                self.dv_dict_features[elem] = DictVectorizer(sparse=False)
                self.dv_dict_features[elem].fit([elem_Series])

    def transformDicts(self, featureData, multiLayerMode=False):
        dicts = []
        for group in progressbar.progressbar(self.dictgroups):
            temp = []
            for elem in dictGroups[group]:
                if elem in self.dv_dict_features:
                    elem_transformed = self.dv_dict_features[elem].transform(featureData[elem])
                    if self.oneHotEncode:
                        elem_transformed = np.where(elem_transformed > 0, 1, 0)
                    cols = self.dv_dict_features[elem].get_feature_names_out()
                    cols = list(map(lambda x: str(elem) + '_' + str(x), cols))
                    elem_transformed = pd.DataFrame(elem_transformed, index=featureData.index, columns=cols)
                    temp.append(elem_transformed)
            if len(temp) == 0:
                dicts.append(pd.DataFrame([]))
            elif len(temp) == 1:
                dicts.append(temp[0])
            else:
                dicts.append(pd.concat(temp, axis=1))
        if len(dicts) == 0:
            return
        elif multiLayerMode:
            return dicts
        else:
            return pd.concat(dicts, axis=1)
    def fit(self, trainFeatureData, parallel=True):
        allDictGroups = list(dictGroups.keys())
        allFeatureGroups = list(featureGroups.keys())

        if any([x not in allFeatureGroups for x in self.simplegroups]):
            raise Exception('Error Simple Features: Please check group names one or more not in the possible group names')
        if any([x not in allDictGroups for x in self.dictgroups]):
            raise Exception('Error Group Features: Please check group names one or more not in the possible group names')
        if self.printDetails: print('Setting Vectorizers. Parallel =', parallel)
        if parallel:
            self.setVectorizers_Parallel(trainFeatureData)
        else:
            self.setVectorizers_Serial(trainFeatureData)
        self.selectedSimpleFeatureNames = []
        for group in self.simplegroups:
            try:
                groupFeatures = featureGroups[group]
            except Exception as e:
                print('''Group '{}' not found in featureGroups'''.format(group))
                continue
            for feature in groupFeatures:
                if not feature in self.selectedSimpleFeatureNames:
                    self.selectedSimpleFeatureNames.append(feature)
    def transformSimpleFeatures(self, featureData, multiLayerMode=False):
        if not isinstance(featureData, pd.DataFrame):
            raise Exception('Expected featureData to be DataFrame given {}'.format(type(featureData)))
        elif multiLayerMode:
            return featureData[self.selectedSimpleFeatureNames].reset_index(drop=True)
        else:
            return featureData[self.selectedSimpleFeatureNames]
    def transform(self, featureData):
        # Transform the dict features
        dictsFeatures = self.transformDicts(featureData)
        simpleFeatures = self.transformSimpleFeatures(featureData)
        dictsFeatures_shape = None
        simpleFeatures_shape = None
        if isinstance(dictsFeatures, pd.DataFrame) and isinstance(simpleFeatures, pd.DataFrame):
            dictsFeatures_shape = dictsFeatures.shape
            simpleFeatures_shape = simpleFeatures.shape
            if dictsFeatures_shape[0] != simpleFeatures_shape[0]:
                raise Exception("Errors Unmatching Shapes {} and {}".format(dictsFeatures.shape[0], simpleFeatures.shape[0]))
            allFeatures = pd.concat([simpleFeatures, dictsFeatures], axis=1)
        elif isinstance(dictsFeatures, pd.DataFrame):
            dictsFeatures_shape = dictsFeatures.shape
            allFeatures = dictsFeatures
        elif isinstance(simpleFeatures, pd.DataFrame):
            simpleFeatures_shape = simpleFeatures.shape
            allFeatures = simpleFeatures
        else:
            raise Exception('''No Features Selected Can't Transform Any Features''')    
        dictsFeatures = None
        simpleFeatures = None
        allFeatures.reset_index(drop=True, inplace=True)
        print('allFeaturesShape:', dictsFeatures_shape, simpleFeatures_shape, allFeatures.shape)
        if self.transformCallback:
            allFeatures = self.transformCallback(allFeatures, simpleFeatures_shape, dictsFeatures_shape)
        return allFeatures
        