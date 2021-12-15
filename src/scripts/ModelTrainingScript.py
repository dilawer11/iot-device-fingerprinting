import sys
import os

import iotpackage.ModelTraining as mt
import iotpackage.FeatureSelection as fsec
import time
import pandas as pd
import numpy as np
import iotpackage.Utils as utils
from iotpackage.__vars import dictGroups, featureGroups
import argparse
import json

VERBOSE = 1

# Loads config for the experiment runs
def loadConfigFromPath(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data
def loadConfig(config_name, config_dir=None):
    if config_dir is None:
        IOTBASE = os.getenv('IOTBASE')
        if IOTBASE is None:
            raise ValueError(f"Environment Variable 'IOTBASE' not set")
        config_dir = os.path.join(IOTBASE, 'src', 'model_configs')
    config_path = os.path.join(config_dir, config_name)
    return loadConfigFromPath(config_path)

def runFingerprintingDevicesExp(config=''):
    if config != '':
        config_data = loadConfigFromPath(config)
    else:
        config_data = loadConfig('FingerprintingDevicesExpConfig.json')
    result_path = os.path.join(IOTBASE, config_data['output_path'])
    if VERBOSE: print(f'Running FingerprintingDevicesExp')
    if VERBOSE: print(f"result_path: {result_path}", flush=True)

    fs = fsec.FeatureSelector(  n_dict=config_data['n_dict'], 
                                n_tls_tcp=config_data['n_dict'],
                                n_udp=config_data['n_udp'],
                                n_dns=config_data['n_dns'],
                                n_ntp=config_data['n_ntp'],
                                n_protocol=config_data['n_protocol'],
                                one_hot_encode=config_data['one_hot_encode'])
    model = mt.FingerprintingDevicesExp(train_datasets=config_data['train_dataset_paths'], fs=fs, devices=config_data['devices'])
    model.run(result_path, runs=config_data['runs'])

def runMultiDatasetCombinedClassifier(config=''):
    if config != '':
        config_data = loadConfigFromPath(config)
    else:
        config_data = loadConfig('MultiDatasetCombinedClassifier.json')
    result_path = os.path.join(IOTBASE, config_data['output_path'])
    if VERBOSE: print(f'Running MultiDatasetCombinedClassifier')
    if VERBOSE: print(f"result_path: {result_path}", flush=True)
    fs = fsec.FeatureSelector(  simple_groups=config_data['simple_groups'],
                                dict_groups=config_data['dict_groups'],
                                n_dict=config_data['n_dict'], 
                                n_tls_tcp=config_data['n_dict'],
                                n_udp=config_data['n_udp'],
                                n_dns=config_data['n_dns'],
                                n_ntp=config_data['n_ntp'],
                                n_protocol=config_data['n_protocol'],
                                one_hot_encode=config_data['one_hot_encode'])
    model = mt.MultiDatasetCombinedClassifier(fs=fs, train_datasets=config_data['train_dataset_paths'], cv=config_data['cv'], label_col=config_data['label_col'])
    model.run(result_path, errors=config_data['errors'], plot_cm=config_data['plot_cm'], runs=config_data['runs'])

def runKnownUnknownClassifier(config=''):
    if config != '':
        config_data = loadConfigFromPath(config)
    else:
        config_data = loadConfig('KnownUnknownClassifier.json')
    result_path = os.path.join(IOTBASE, config_data['output_path'])
    if VERBOSE: print(f'Running KnownUnknownClassifier')
    if VERBOSE: print(f"result_path: {result_path}", flush=True)
    fs = fsec.FeatureSelector(  simple_groups=config_data["simple_groups"],
                                dict_groups=config_data["dict_groups"],
                                n_dict=config_data['n_dict'], 
                                n_tls_tcp=config_data['n_dict'],
                                n_udp=config_data['n_udp'],
                                n_dns=config_data['n_dns'],
                                n_ntp=config_data['n_ntp'],
                                n_protocol=config_data['n_protocol'],
                                one_hot_encode=config_data['one_hot_encode'])
    model = mt.KnownUnknownClassifier(fs=fs, train_datasets=config_data['train_dataset_paths'], cv=config_data['cv'], label_col=config_data['label_col'])
    model.run(result_path, runs=config_data['runs'], split_type=config_data['split_type'], non_iot_filter=config_data['non_iot_filter'])
    
def runMultiDatasetCommonClassifier(config=''):
    if config != '':
        config_data = loadConfigFromPath(config)
    else:
        config_data = loadConfig('MultiDatasetCommonClassifier.json')
    result_path = os.path.join(IOTBASE, config_data['output_path'])
    if VERBOSE: print(f'Running MultiDatasetCommonClassifier')
    if VERBOSE: print(f"result_path: {result_path}", flush=True)

    fs = fsec.FeatureSelector(  simple_groups=config_data['simple_groups'],
                                dict_groups=config_data['dict_groups'],
                                n_dict=config_data['n_dict'],
                                n_tls_tcp=config_data['n_tls_tcp'],
                                n_udp=config_data['n_udp'],
                                n_dns=config_data['n_dns'],
                                n_ntp=config_data['n_ntp'],
                                n_protocol=config_data['n_protocol'],
                                one_hot_encode=config_data['one_hot_encode'])
    model = mt.MultiDatasetCommonClassifier(    train_datasets=config_data['train_dataset_paths'],
                                                test_datasets=config_data['test_dataset_paths'], 
                                                fs=fs,
                                                label_col=config_data['label_col'])
    model.run(  result_path=result_path,
                runs=config_data['runs'],
                errors=config_data['errors'],
                data_size=config_data['data_size'],
                features=config_data['features'])

def runMultiDatasetCombinedClassifierIoTvsNonIoT(config=''):
    if config != '':
        config_data = loadConfigFromPath(config)
    else:
        config_data = loadConfig('MultiDatasetCombinedClassifierIoTvsNonIoT.json')
    result_path = os.path.join(IOTBASE, config_data['output_path'])
    if VERBOSE: print(f'Running MultiDatasetCombinedClassifierIoTvsNonIoT')
    if VERBOSE: print(f"result_path: {result_path}", flush=True)
    
    fs = fsec.FeatureSelector(  simple_groups=config_data['simple_groups'],
                                dict_groups=config_data['dict_groups'],
                                n_dict=config_data['n_dict'],
                                n_tls_tcp=config_data['n_tls_tcp'],
                                n_udp=config_data['n_udp'],
                                n_dns=config_data['n_dns'],
                                n_ntp=config_data['n_ntp'],
                                n_protocol=config_data['n_protocol'],
                                one_hot_encode=config_data['one_hot_encode'])

    model = mt.MultiDatasetCombinedClassifierIoTvsNonIoT(   train_datasets=config_data['train_dataset_paths'],
                                                            test_datasets=config_data['test_dataset_paths'],
                                                            fs=fs,
                                                            label_col=config_data['label_col'],
                                                            cv=config_data['cv'],
                                                            print_details=config_data['print_details'])
    
    model.run(  result_path=result_path,
                runs=config_data['runs'],
                errors=config_data['errors'],
                features=config_data['features'])

def runTargetVsNonTargetClassifier(config=''):
    if config != '':
        config_data = loadConfigFromPath(config)
    else:
        config_data = loadConfig('TargetVsNonTargetClassifier.json')
    result_path = os.path.join(IOTBASE, config_data['output_path'])
    if VERBOSE: print(f'Running TargetVsNonTargetClassifier')
    if VERBOSE: print(f"result_path: {result_path}", flush=True)
    
    fs = fsec.FeatureSelector(  simple_groups=config_data['simple_groups'],
                                dict_groups=config_data['dict_groups'],
                                n_dict=config_data['n_dict'],
                                n_tls_tcp=config_data['n_tls_tcp'],
                                n_udp=config_data['n_udp'],
                                n_dns=config_data['n_dns'],
                                n_ntp=config_data['n_ntp'],
                                n_protocol=config_data['n_protocol'],
                                one_hot_encode=config_data['one_hot_encode'])
    
    model = mt.TargetVsNonTargetClassifier( fs=fs,
                                            train_datasets=config_data['train_dataset_paths'])
    model.run(result_path, runs=config_data['runs'], features=config_data['features'], devices=config_data['devices'], min_balance=config_data['min_balance'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--combined", action="store_true", help="Run the combined classifier, aka multi dataset combined classifier")
    group.add_argument("--common", action="store_true", help="Run the common classifier, aka multi dataset common classifier")
    group.add_argument("--known-unknown", action="store_true", help="Run the known unknown classifier")
    group.add_argument("--combined-iot-vs-noniot", action="store_true", help="Run the multi dataset combined iot vs non-iot classifier")
    group.add_argument("--target", action="store_true", help="Run the target vs non target classifier")
    group.add_argument("--fingerprint", action="store_true", help="Run the fingerprinting devices experiment")
    parser.add_argument("--config", default="", type=str, help="Path to config file to use for this experiment")
    args = parser.parse_args()

    global IOTBASE
    IOTBASE = os.getenv('IOTBASE')
    if args.combined:
        runMultiDatasetCombinedClassifier(args.config)
    elif args.common:
        runMultiDatasetCommonClassifier(args.config)
    elif args.known_unknown:
        runKnownUnknownClassifier(args.config)
    elif args.combined_iot_vs_noniot:
        runMultiDatasetCombinedClassifierIoTvsNonIoT(args.config)
    elif args.target:
        runTargetVsNonTargetClassifier(args.config)
    elif args.fingerprint:
        runFingerprintingDevicesExp(args.config)
    else:
        raise ValueError("unknown run type")
    
