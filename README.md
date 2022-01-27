# IoT Device Fingerprinting

## Introduction
This repository contains the end-to-end code to process PCAP files and train the models used in the PETS'22 paper: 
> Analyzing the Feasibility and Generalizability of Fingerprinting Internet of Things Devices. Dilawer Ahmed, Anupam Das and Fareed Zaffar. Proceedings on Privacy Enhancing Technologies (PoPETs), 2022

The paper can be found at this [link](https://anupamdas.org/paper/PETS2022.pdf)

## Setup

There are many different setup options. The 3 we tested are briefly described below.

### Option 1. Using pre-built docker image

You can use the provided Docker image to start a container. This docker image also contains the *sample* dataset (more details on this below) and can be used to test the configurations. The Docker image can be downloaded from [here](https://drive.google.com/file/d/1n4kXuYKDDsx1EWZrtAqe_G7l5o1CTkCy/view?usp=sharing). The SHA256 Id of the docker image is:
`sha256:0a8cd174f997350bd67e676e896d34c0e5eb5bbb1738cd860cb620ec71e16497`

To load the Docker image use the following command:

`sudo docker load < /path/to/dockerimage.tar.gz`

The SHA256 hash can be verified by running the following command:

`sudo docker inspect --format="{{.Id}}" iot-device-fingerprinting`

To run the docker container use:

`sudo docker run -it iot-device-fingerprinting`

Note you would need to setup all the datasets (except *sample*) datasets as described in the datasets section below.

### Option 2. Building docker image from Dockerfile

You can also build and run your own Docker container from the Dockerfile. There are multiple ways to achieve this based upon your particular use. Datasets would have to be downloaded and placed in the correct directory structure (more details on this in the *Datasets* section of this README).

After the datasets have been correctly setup, you have two options on how to build and run the Docker container, depending on whether you want to retain the results on the host(local file system) *after* the Docker container has stopped, or not.

1. If you want the results to only be within the Docker container while the container runs, and *not* on the host once the container stops (maybe to just test the code on the sample dataset), then 
If you want everythiyou can build the Dockerfile without any changes, as follows: (Note that this default build of the Dockerfile may take some more time than the option below, as the entire "datasets/" folder will be copied to the Docker image)

Build the Dockerfile and tag it "iot-device-fingerprinting":

`sudo docker build -t iot-device-fingerprinting .`

Run a Docker container with the built image:

`sudo docker run -it iot-device-fingerprinting`

2. If you want to retain the results for later on the host (local file system), you will need to change the Dockerfile before building it. You would have to delete/comment out the following two files from the Dockefile. Since we will not be copying the dataset files to the Docker iamge, we will save some time and space while building the image. Also the datasets directory as it saves time and space while building. You would have to delete/comment out the following 2 lines from Dockerfile 

```
COPY datasets datasets
COPY results results
```

You can then build and tag the Docker image as above:

`sudo docker build -t iot-device-fingerprinting .`

To run the Docker container, you will need to mount the datasets and results directories, as in the following command run from the project base directory:

`sudo docker run -it -v "$(pwd)"/datasets:/datasets -v "$(pwd)"/results:/results iot-device-fingerprinting`

The command above mounts the datasets and results to the docker to save the results across docker runs.

### Option 3. Manually installing dependencies on Ubuntu
You can also run on ubuntu directly. First you must install all the dependencies and then set the enviroment variables. You might have to configure tshark as well depending on the machine. The default configuration shouldn't cause any problems while running this

To innstall apt packages:

`sudo apt install python3 python3-pip tshark`

After this you can install the required python packages by using the following command:

`pip3 install -r requirements.txt`

Finally, the two environment variables have to be set. You can use the following commands:

`export IOTBASE=/path/to/projectdir`
`export PYTHONPATH=$IOTBASE/src:$PYTHONPATH`

The setup should be complete at this point and you can continue running the commands below. Datasets would have to be set the instructions can be found at the Datasets section of this README file.

# Datasets

We used many different datasets for our experiments and evaluations. Many of them are openly available others might be available upon request. Below is a list of all datasets and the links where they can be fetched from. We suggest downloading the sample dataset for testing.

We used RAW PCAP files and then processed them ourselves and all the scripts we used are available in the code. We created multiple different scripts to convert from one form to another. We convert PCAP to CSV, which cleans the data portion of the packet and only extracts the fields we need. It takes up less diskspace and is easier and faster to work with using python libraries such as pandas compared to pcaps. We then use this CSV to extract features using multiprocessing which is stored as seperate '.pkl' files and then combines the result and creates other required files.

## Adding datasets

You would need two things to add a dataset to this project

1. A directory containing all the raw PCAP files (datasets/{dataset_name}/RAW)
2. A CSV file **device_mappings.csv** with two columns *Device* and *IP* containing the mapping between IP and Device name (see *Our* dataset's device_mappings.csv file for details) (datasets/{dataset_name}/device_mappings.csv). 

The sample dataset already has the correct directory structure and content in the zip file so you will only need to unzip the zip file as below and it will be setup correctly:

To setup the *sample* dataset only:

`unzip /path/to/sample.zip -d /path/to/projectbase/datasets`

For all other datasets, one way to correctly setup would be to create the directory structure first and then copy the files into that structure. To setup the parent directory and the 'RAW' directory (which has to contain the pcap files). You can use the following command (including the -p flag)

To create the directories:

`mkdir -p /path/to/projectbase/datasets{dataset_name}/RAW`

After this you would need to copy zip files and device_mappings.csv to the *datasets/{dataset_name}* path. For example, the sample dataset would be under *datasets/sample*. You can then extract the zip to *datasets/{dataset_name}/RAW* subdirectory:

To copy pcap files (assuming source file is zip file):

`unzip {dataset_pcaps}.zip -d /path/to/projectbase/datasets/{dataset_name}/RAW`

To copy the device_mappings.csv file to correct place:

`cp {dataset_device_mappings}.csv /path/to/projectbase/datasets/{dataset_name}/device_mappings.csv`

The above commands assumes zip source file for others you can *extract* them seperately and then move/copy accordingly

- **Sample Dataset (Recommended for testing code)**: This dataset is a limited subset of *Our2* dataset. It is only recommended for evaluating the code as it runs much faster than other datasets and requires less memory. It can be downloaded from [here](https://drive.google.com/file/d/1sUq2e2Y104trFMhj700Lb9DluWld8RW4/view?usp=sharing). To setup this dataset follow instructions above

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
4) Creating plots, tables and figures (Post-processing)

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

To train the models the a script "ModelTrainingScript" is provided. This is simplified for future use where simple models can be trained using configurations provided. More complex setups can also be done by tweaking the ModelTrainingScript. 

ModelTrainingScript requires a corresponding configuration file which contains information and values to tweak the model. Configuration files control the input datasets and weather they are used as training/testing datasets. They also control where results are stored and what commonly tweaked values the model uses. Default configurations for models are provided in the 'src/model_configs' directory. We used many different configurations and scripts to train the models but the these are a good place to start. Output of these should be by default under *results* folder. The files are the results based on the configuration used. They can be confusion matrices (plot_cm) and the misclassifications (errors) and top_features (features). Depending on a specific model it can have additional parameters that can be tuned. Some are highly specialized for one corner case experiment in our work. Most, however, are general

There are 5 different types of models that we used in our work.

- Multi Dataset Combined Classifier
    This aggregates the data from multiple datasets into one combined dataset and then trains the model based on that. Important to note that testing dataset also comes from this dataset based on the 'cv' parameter. If 'cv' is 0 then 80:20 train-test split is used otherwise 'cv' is parsed as cross-validation parameter. This model by default uses configuration file located at "src/model_configs/MultiDatasetCombinedClassifier.json"

    Example Usage:
    
    `python3 src/scripts/ModelTrainingScript.py --combined`

- Multi Dataset Common Classifier
    This aggregates all the data from training datasets into one combined train dataset and all data from testing datasets as combined test dataset. Then it finds the common devices and trains on the common devices from the training dataset and tests on the common devices from the test dataset. This model by default uses configuration file located at "src/model_configs/MultiDatasetCommonClassifier.json"

    Example Usage:
    
    `python3 src/scripts/ModelTrainingScript.py --common`

- Known vs Unknown Classifier
    This model is similar to known unknown model mentioned in the paper. It sets some devices as known and trains from a portion of known and tries to get additional information about unknown devices. This is setting for the open-world experiment. This model by default uses configuration file located at "src/model_configs/KnownUnknownClassifier.json"

    Example Usage:
    
    `python3 src/scripts/ModelTrainingScript.py --known-unknown`

- Target vs NonTarget Classifier
    This model takes training datasets and target devices as input and splits the devices equally among known and unknown target devices. It trains using the known target and portion on non-target device. This model by default uses configuration file located at "src/model_configs/TargetVsNonTargetClassifier.json"

    Example Usage:
    `python3 src/scripts/ModelTrainingScript.py --fingerprint`

- Fingerprinting Devices
    This model fingerprints the individual devices. This model by default uses configuration file located at "src/model_configs/FingerprintingDevicesExpConfig.json"

    Example Usage:
    
    `python3 src/scripts/ModelTrainingScript.py --fingerprint`

Output of these should be by default under results folder. The files are the results based on the configuration used. They can be confusion matrices (plot_cm) and the misclassifications (errors) and top_features (features). Depending on a specific model it can have additional parameters that can be tuned. Some are highly specialized for one corner case experiment in our work. Most, however, are general

## Post-processing (Tables and Plots)

There are multitudes of possible ways to interpret any result and create different plots with it. A few basic examples are given below to help you get started. To create a device. We have already included the plotting or table libraries we used to create all the plots. Our examples cover different types of figures and tables to serve as an example of different types.

### Confusion Matrix Plot
Confusion Matrices relies on the correct and errors during prediction from the classifier. After you have trained the model e.g the MultiDatasetCombinedClassifier as  (using the default configuration paths) you can then use the results to plot a Confusion Matix as follows:

To train the classifier as above (you can skip this step if you have already trained the model and the results are ready to be processed):

`python3 src/scripts/ModelTrainingScript.py --combined`

To generate CM using the PostProcessing.py script:

`python3 src/scripts/PostProcessing.py -i /results/multidatasetcombinedclassifier --cm`

It will generate the results using CM files from the provided directory (The input directory can be changed if the results are in a different location). The result by default will be saved in the same directory in form of a PDF file named "CM.pdf". You can change this by providing a seperate save path

### Group Feature Importance Table
Group feature importance is to see the feature importance of a group as a whole based on the individual features used by the training model. When the classifier is trained it will dump the feature importances for each run (as long a 'features: true' is set in the config file). We can use these feature importances to get the group feature importances for all model training runs.

To train the classifier as above (you can skip this step if you have already trained the model and the results are ready to be processed):

`python3 src/scripts/ModelTrainingScript.py --combined`

To generate the Group Feature importance table using the PostProcessing.py script:

`python3 src/scripts/PostProcessing.py -i /results/multidatasetcombinedclassifier --group-fi`

It will generate the results using feature importances files from the provided directory (The input directory can be changed if the results are in a different location). The result by default will be saved in the same directory in form of a CSV file. You can change this by providing a seperate save path

### t-SNE Scatter Plot for datasets
t-SNE scatter plots can help visualize the device samples from the given dataset. These require the feature data (with the run-time feature serializer to select and serialize the multi-valued features). We don't need the model to be trained for this but require the feature data from the datasets.

To generate the t-SNE scatter plot for the *sample* dataset:

`python3 src/scripts/PostProcessing.py -i /datasets/sample --tsne`

It will generate the results using dataset path provided (dataset path can be replaced to generate the plot for other datasets). The result by default will be saved in the same input directory in form of a PDF file. You can change this by providing a seperate save path
