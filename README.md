# IoT Device Fingerprinting

## Introduction
This repository contains the end-to-end code to process PCAP files and train the models used in the PETS'22 paper: 
> Analyzing the Feasibility and Generalizability of
Fingerprinting Internet of Things Devices



## Setup

There are many different setup options. The 3 we tested are briefly described below.

### Option 1. Using pre-built docker image

You can use the provided Docker image to start a container. This docker image also contains the *sample* dataset (more details on this below) and can be used to test the configurations. The Docker image can be downloaded from [here](https://drive.google.com/file/d/1mxcOgxhht5LIeBYMS2i6nrrd3PsE02DP/view?usp=sharing). 

To load the Docker image use the following command

`sudo docker load < /path/to/dockerimage.tar.gz`

It can then be run using:

`sudo docker run -it iot-device-fingerprinting`

If you want to load additional datasets you can mount the directory containing the datasets at the correct mount point i.e `/` in the docker image

### Option 2. Building docker image from Dockerfile

You can also build and run your own Docker container from the Dockerfile. There are multiple ways to achieve this based upon your particular use. Datasets would have to be downloaded and placed in the correct directory structure (more details on this in the *Datasets* section of this README).

If you want everything strictly within the Docker container and not mount any volumes (maybe to just test code on sample). This requires that datasets are placed in the datasets subdirectory on the project base directory. If this is not the case you can alter the Dockerfile accordingly. After datasets have been correctly setup you can run the following command in the project base directory

`sudo docker build -t iot-device-fingerprinting`

`sudo docker run -it iot-device-fingerprinting`

If you want it store the results for later on the local file system you can mount the results directory. Also the datasets directory as it saves time and space while building. You would have to delete/comment out the following 2 lines from Dockerfile 

```
COPY datasets datasets
COPY results results
```

After this you can run the following command to build

`sudo docker build -t iot-device-fingerprinting .`

Docker can then be started by using the following command (with correct paths):

`sudo docker run -it -v /path/to/datasets:/datasets -v /path/to/results:/results iot-device-fingerprinting`

The command above mounts the datasets and results to the docker to save the results across docker runs.

### Option 3. Manually installing dependencies on Ubuntu
You can also run on ubuntu directly. First you must install all the dependencies and then set the enviroment variables. You might have to configure tshark as well depending on the machine. The default configuration shouldn't cause any problems while running this

`sudo apt install python3 python3-pip tshark`

After this you can install the required python packages by using the following command:

`pip3 install -r requirements.txt`

Finally, the two environment variables have to be set. You can use the following commands:

`export IOTBASE=/path/to/projectdir`
`export PYTHONPATH=$IOTBASE/src:$PYTHONPATH`

The setup should be complete at this point and you can continue running the commands below. Datasets would have to be set the instructions can be found at the Datasets section of this README file.

# Datasets

We used many different datasets for our experiments and evaluations. Many of them are openly available others might be available upon request. Below is a list of all datasets and the links where they can be fetched from. We used RAW PCAP files and then processed them ourselves and all the scripts we used are available in the code. We created multiple different scripts to convert from one form to another. We convert PCAP to CSV, which cleans the data portion of the packet and only extracts the fields we need. It takes up less diskspace and is easier and faster to work with using python libraries such as pandas compared to pcaps. We then use this CSV to extract features using multiprocessing which is stored as seperate '.pkl' files and then combines the result and creates other required files.

## Adding datasets

You would need two things to add a dataset to this project

1. A directory containing all the raw PCAP files
2. A CSV file **device_mappings.csv** with two columns *Device* and *IP* containing the mapping between IP and Device name (see *Our* dataset's device_mappings.csv file for details). 

To test you can download sample dataset from the link provided below and download the zip file containing pcaps and device_mappings.csv. The datasets are to be placed under the datasets directory in project base directory. You would need to copy zip files and device_mappings.csv to the *datasets/{dataset_name}* path. For example, Our dataset would be under *datasets/Our*. You can then extract the zip to *datasets/{dataset_name}/RAW* subdirectory. Sample dataset is already setup in this way. For others something along the lines of this has to be done

`unzip {dataset_pcaps}.zip -d /path/to/projectbase/datasets/{dataset_name}/RAW`

`cp {dataset_device_mappings}.csv /path/to/projectbase/datasets/{dataset_name}/device_mappings.csv`

Before proceeding further, make sure that you create a 'RAW' subdirectory and 'device_mappings.csv' file under the dataset you want to use.

- **Sample Dataset (Recommended for testing code)**: This dataset is a limited subset of *Our2* dataset. It is only recommended for evaluating the code as it runs much faster than other datasets and requires less memory. This would be faster and takes less memory. It can be downloaded from [here](https://drive.google.com/file/d/1sUq2e2Y104trFMhj700Lb9DluWld8RW4/view?usp=sharing). To setup this dataset download the file and copy/place it in the `/path/to/projectdir/datasets` then you can unzip the contents using: `unzip /path/to/projectdir/datasets -d /path/to/projectdir/datasets`. This should setup the files correctly

- **NCSU IoT Lab Datasets (Our and Our2)**: The datasets can be downloaded from [here](https://drive.google.com/drive/folders/1WUVK9BQFCZCq-9xTjmT22SOC9MIyW8hG?usp=sharing).

- **YourThings Scorecard (YT)**: The dataset can be downloaded from [here](https://yourthings.info/data/). 

- **UNSW IoT Traces (UNSW)**: The dataset can be downloaded from [here](https://iotanalytics.unsw.edu.au/iottraces).

- **Mon(IoT)r Lab Dataset (NE_US and NE_UK)**: The dataset can be accessed from [here](https://moniotrlab.ccis.neu.edu/imc19dataset/).

- **PingPong Dataset (PP)**: The dataset can be accessed from [here](https://athinagroup.eng.uci.edu/projects/pingpong/data/).

- **HomeSnitch Datasets (HS and HS2)**: The datasets were made available to us upon request. Most recent version can be found from this [link](https://research.fit.edu/iot/)

- **Adding additional datasets**

    Additional datasets can be added. Follow the current directory structure and add the PCAP files in the Datasets/{dataset_name}/RAW folder and then pass this name to the pipefile for processing of data

# Pipeline

The Pipeline for this project is simply 4 different steps:

1) Raw PCAPs to CSV (PCAP2CSV)
2) CSV to Feature Data (CSV2FeatureData)
3) Feature Data to Raw Classification data (ModelTraining)
4) Raw Classification data to plots, figures and tables (Plotting)

We explain each step and the required files for each step below.

## PCAP2CSV

In this step, the PCAPs data from the datasets are converted to a CSV file to have faster access in python and low disk space usage. We created a script which can be found at Scripts/PCAP2CSV.py. The script takes an input directory (-i /path/to/rawpcaps) and an output directory (-o /path/to/csv). By default it walks through the input directory and searches for all files with '.pcap' extension. It processes each file using tshark and outputs the results in CSV file format in the output directory using the same directory structure.

Example Usage:
`python3 src/scripts/PCAP2CSV.py -i datasets/sample/RAW -o datasets/sample/CSV`

## CSV2FeatureData

Feature extraction is to extract the features from data which will be helpful later on to train te model on. This takes some time since reverse dns lookups need to be performed as well and it is otherwise a CPU intensive operation as well. We have parrallelized this step to execute faster. This step may also take up a lot of memory.

Example Usage:
`python3 src/scripts/CSV2FeatureData.py -i datasets/sample`

## ModelTraining

To train the models the a script "ModelTrainingScript" is provided. This is simplified for future use where simple models can be trained using configurations provided. More complex setups can also be done by tweaking the ModelTrainingScript. We used different configurations and scripts to train the models but these are a good place to start. Default configurations have been provided in the src/model_configs directory. By default these configs are used by the model. There are 5 different types of models that we used in our work.

- Multi Dataset Combined Classifier
    This aggregates the data from multiple datasets into one combined dataset and then trains the model based on that. Important to note that testing dataset also comes from this dataset based on the 'cv' parameter. If 'cv' is 0 then 80:20 train-test split is used otherwise 'cv' is parsed as cross-validation parameter

    Example Usage: `python3 src/scripts/ModelTrainingScript.py --combined`

- Multi Dataset Common Classifier
    This aggregates all the data from training datasets into one combined train dataset and all data from testing datasets as combined test dataset. Then it finds the common devices and trains on the common devices from the training dataset and tests on the common devices from the test dataset

    Example Usage: `python3 src/scripts/ModelTrainingScript.py --common`

- Known vs Unknown Classifier
    This model is similar to known unknown model mentioned in the paper. It sets some devices as known and trains from a portion of known and tries to get additional information about unknown devices. This is setting for the open-world experiment.

    Example Usage: `python3 src/scripts/ModelTrainingScript.py --known-unknown`

- Target vs NonTarget Classifier
    This model takes training datasets and target devices as input and splits the devices equally among known and unknown target devices. It trains using the known target and portion on non-target device

    Example Usage: `python3 src/scripts/ModelTrainingScript.py --target`

- Fingerprinting Devices
    This model fingerprints the individual devices 

    Example Usage: `python3 src/scripts/ModelTrainingScript.py --fingerprint`

Output of these should be by default under results folder. The files are the results based on the configuration used. They can be confusion matrices (plot_cm) and the misclassifications (errors) and top_features (features). Depending on a specific model it can have additional parameters that can be tuned. Some are highly specialized for one corner case experiment in our work. Most however are general