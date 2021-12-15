import sys
import os
import argparse

from iotpackage.Utils import CSVcols
from iotpackage.FeatureExtraction import FeatureExtracter
from iotpackage.PreProcessing import PreProcessor
from iotpackage.__vars import permanentRename
from psutil import cpu_count
from multiprocessing import Pool
import pandas as pd

VERBOSE = 1
DEVICE_MAPPINGS = None
OUTPUT_DIR = None

def loadFeatureData(feature_files):
    feature_data_arr = []
    for feature_file in feature_files:
        feature_data = pd.read_pickle(feature_file)
        feature_data_arr.append(feature_data)
    feature_data = pd.concat(feature_data_arr, ignore_index=True)
    feature_data.reset_index(drop=True, inplace=True)
    return feature_data

def permanentRenameDevices(feature_data):
    for device in permanentRename:
        loc = feature_data['Device'].isin(permanentRename[device])
        feature_data.loc[loc, 'Device'] = device
    return feature_data

def getDevicesFromData(feature_data):
    return feature_data['Device'].value_counts()

def saveData(feature_data, feature_data_path):
    if os.path.exists(feature_data_path):
        print(f"WARNING: feature_data_path: {feature_data_path} already exists and is being overwritten")
    if VERBOSE: print('Saving Data...')
    feature_data.to_pickle(feature_data_path)
    if VERBOSE: print('Data Saved')
    return

def saveDevices(devices, devices_path):
    if os.path.exists(devices_path):
        print(f"WARNING: devices_path: {devices_path} already exists and is being overwritten")
    if VERBOSE: print('Saving Devices...')
    devices.to_pickle(devices_path)
    if VERBOSE: print('Devices Saved')
    return

def concatFeatureDataFiles(feature_files, feature_data_path, devices_path):
    feature_data = loadFeatureData(feature_files)
    feature_data = permanentRenameDevices(feature_data)
    saveData(feature_data, feature_data_path)
    devices = getDevicesFromData(feature_data)
    saveDevices(devices, devices_path)
    return

def defaultCSVFileListLoader(base_path, max_files_per_job=10, file_exts=['.csv']):
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        raise ValueError(f'base_path: {base_path} is not a directory')
    
    # Walks in the directory and gets all the files with file_exts
    file_list_list = []
    for root, dirs, files in os.walk(base_path):
        cur_list = []
        for name in files:
            if os.path.splitext(name)[1] in file_exts:
                cur_list.append(os.path.join(root, name))
            if max_files_per_job > 0 and len(cur_list) >= max_files_per_job:
                file_list_list.append(cur_list)
                cur_list = []
        if len(cur_list): file_list_list.append(cur_list)

    return file_list_list

def extractFeaturesFromCSV(idx, CSVFiles):
    # Sanity Checks
    if VERBOSE: print('Starting The Process..')
    if not isinstance(CSVFiles, list):
        raise Exception('CSVFiles should be in an iterable list form given {}'.format(type(CSVFiles)))

    if not len(CSVFiles): return
    # Load The packets from all CSV Files into one pandas dataframe
    packets_arr = []
    for CSV in CSVFiles:
        packets_arr.append(pd.read_csv(CSV, names=CSVcols))
    packets = pd.concat(packets_arr, ignore_index=True)
    if VERBOSE: print('Main: CSV Files Loaded', flush=True)
    del packets_arr

    # Load the device mappings
    if not os.path.exists(DEVICE_MAPPINGS):
        raise FileNotFoundError(f'device_mappings: File not found: {DEVICE_MAPPINGS}')
    device_mappings = pd.read_csv(DEVICE_MAPPINGS)
    
    pp = PreProcessor(src_name='SrcIP', dst_name='DstIP', col_name_mappings='IP', device_mappings=device_mappings)
    cleanedPackets = pp.run(packets)

    del packets
    
    if VERBOSE: print('Main: Dataset Preprocessed and cleaned', flush=True)
    
    # Extract The Features
    fe = FeatureExtracter()
    featureData = fe.run(cleanedPackets, reset_index=True)
    if VERBOSE: print('Main: Features Extracted', flush=True)
    
    # Save the Feature Data
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_file_path = os.path.join(OUTPUT_DIR, str(idx) + '.pkl')
    featureData.to_pickle(output_file_path)
    
    return

def CSV2FeatureData(csv_dir, device_mapping, feature_data_dir, feature_data_path, devices_path, max_jobs, max_files_per_job, rm_feature_data_files=True):
    if VERBOSE: print('Starting Feature Extraction...', flush=True)
    file_list_list = defaultCSVFileListLoader(csv_dir, max_files_per_job=max_files_per_job)
    if VERBOSE: print('Loaded CSV File List', flush=True)
    count_file_list_list = [(i, CSVList) for i, CSVList in enumerate(file_list_list)]

    if device_mapping is None or not os.path.exists(device_mapping):
        raise FileNotFoundError(f"device_mappings: {device_mapping} file not found")
    if not os.path.isdir(feature_data_dir):
        os.makedirs(feature_data_dir)

    global OUTPUT_DIR, DEVICE_MAPPINGS
    OUTPUT_DIR = feature_data_dir
    DEVICE_MAPPINGS = device_mapping

    pool = Pool(max_jobs)
    if VERBOSE: print('Pool Initialized', flush=True)
    pool.starmap(extractFeaturesFromCSV, count_file_list_list)
    pool.close()
    pool.join()
    if VERBOSE: print('Feature Extraction Done...', flush=True)

    feature_files = list(map(lambda x: os.path.join(feature_data_dir, x), os.listdir(feature_data_dir)))
    concatFeatureDataFiles(feature_files, feature_data_path, devices_path)

    if rm_feature_data_files:
        if VERBOSE: print('Removing intermediate feature files')
        for feature_file in feature_files:
            os.system(f'rm {feature_file}')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--base-dir', required=True, type=str, help="The base directory of the dataset. By default the CSV subdirectory should contain CSV files and featureData.pkl and devices.pkl would be created in this directory for as the output files")
    parser.add_argument('--csv-dir-path', default=None, help="The path of the CSV source directory if not the 'CSV' folder in base-dir")
    parser.add_argument('--device-mapping', default=None, help="The path of the CSV device mapping file if not the 'device_mappings.csv' in the base-dir")
    parser.add_argument('--intermediate-dir', default='.tmp', help="The directory for storing individual pkl files before combining them (.tmp)")
    parser.add_argument('-o', '--output-dir', default=None, help="This overrides the base-dir as the output directory. The files featureData.pkl and devices.pkl would be created in this.")
    parser.add_argument('--max-jobs', default=cpu_count(), type=int, help="The max number of processes to create in the pool")
    parser.add_argument('--max-files-per-job', default=10, type=int, help="The maximum number of CSV files processed in one job (Default: 10)")
    args = parser.parse_args()

    if args.csv_dir_path is None:
        csv_dir_path = os.path.join(args.base_dir, 'CSV')
    else:
        csv_dir_path = args.csv_dir_path

    if args.device_mapping is None:
        device_mapping = os.path.join(args.base_dir, 'device_mappings.csv')
    else:
        device_mapping = args.device_mapping
    
    if args.output_dir is None:
        output_dir = args.base_dir
    else:
        output_dir = args.output_dir
    
    feature_data_path = os.path.join(args.base_dir, 'featureData.pkl')
    devices_path = os.path.join(args.base_dir, 'devices.pkl')

    created_intermediate = False
    # Create a temporary directory to store feature files
    tmp_feature_data_dir = os.path.join(args.base_dir, args.intermediate_dir)
    if not os.path.exists(tmp_feature_data_dir):
        created_intermediate = True
        os.makedirs(tmp_feature_data_dir)

    if not os.path.isdir(tmp_feature_data_dir):
        raise ValueError(f'{tmp_feature_data_dir} already exists but not as a directory this path is needed to store files or you can rename this to another name')

    # Extract features and combine feature files
    CSV2FeatureData(csv_dir_path, device_mapping, tmp_feature_data_dir, feature_data_path, devices_path, args.max_jobs, args.max_files_per_job)

    # Clear the intermediate directory if we created
    if created_intermediate: os.system(f"rm -r {tmp_feature_data_dir}")