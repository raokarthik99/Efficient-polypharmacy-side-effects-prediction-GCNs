# Efficient-polypharmacy-side-effects-predictions-GCNs
 This repository contains the source code and jupyter notebooks associated with our work on "Efficiently Predicting Pharmacological Side-effects Resulting from Pair-wise  Consumption of Drugs, using Graph Convolutional Networks". The source code is written and tested in Python3. It leverages PyTorch and several of its associated packages. End-to-end development of this project was done using [Google Colaboratory (GPU-backend)]() along with the help of Google Cloud Platform for storing, analysing and visualizing the results. It may be advisable to go through our [paper]() for a more refined understanding of this implementation.

# About 
 Polypharmacy has grown rampantly in the past few decades as an attempt to treat patients with complex diseases or co-existing conditions. Nonetheless, a dangerous consequence of polypharmacy is unanticipated adverse side-effects. These side-effects manifest due to drug-drug interactions, where one drug influences the behaviour of the other. This problem can be modelled as a multi-relational link prediction task that leverages graph convolutional networks. A recent research proposed a model named decagon that uses a multimodal graph of protein-protein interactions, drug-protein interactions and drug-drug interactions labelled with polypharmacy side-effects. However, this method leads to high computational costs and memory demands. A follow-up study suggested a ‘tri-information propagation (TIP)’ approach that simplified the processing by breaking down the task into three modules. Although this attempt significantly reduced resource usage, there was no evidence that the results put forward captured all possible side-effects for all possible pair-wise combinations of drugs. Such evidence is crucial to facilitate the translational science and the discovery of novel drug combinations which was the basic premise for the decagon model. Our work aims to efficiently meet the objectives of the decagon model by building on the TIP-based architecture and leveraging big data solutions on the cloud. 
 
# Sample results
## Querying on a particular drug pair
![Sample query]()  
![Sample results]()  
## Querying on a particular side-effect 
![Sample query]()  
![Sample results]()  
## Querying on a particular side-effect 
![Sample query]()  
![Sample results]()  

# Data sets
The data sets used for this implementation were taken directly from http://snap.stanford.edu/decagon. It is not included in this repo due to size limitations. If one wishes to use the same data sets without modifications, it is not neccessary to acquire them and integrate it into this implementation. The data folder contains all the preprocessed files that would be sufficient for training and evaluating the model.    
Number of drugs = 645.  
Number of side-effects = 1097.  

# Architecture 
## Encoder 
![Encoder]()
## Decoder
![Decoder]()

# 6 model variants 
1. DF-DistMult
2. DDM-DF-RGCN
3. DDM-NN
4. PPM-GGM-NN
5. DDM-PPM-GGM-DF-Cat (TIP(tri information propagation)-Cat)
6. DDM-PPM-GGM-DF-Sum (TIP(tri information propagation)-Sum)

# Folder descriptions
* analysis - Contains jupyter notebooks for both data exploration (done first!) and results compilation (done last!). It also has helper scripts for loading the data and finding top 10 most accurately predicted side-effects
* data - Contains the preprocessed data to be used downstream for training and evaluation 
* evaluation - Contains the scripts that evaluate a trained model to produce probabilistic predictions for all possible side-effects for all possible pairwise drug combinations 
* model template - Contains the template for the training script that provides a common basis for the model variants
* model - Contains the training scripts for the model variants 
* output - Contains the trained models and accuracy reports obtained training/validation of the model variants 
* requirements - Contains the requirements for cpu and gpu backends (need not bother with this if running on Google Colab with instructions provided here)
* src - Contains additional helper scripts that implement the encoder and decoder for the model variants

# Getting started 
The instructions provided below are best applicable when using Google Colab and Google Cloud Platform. Nonetheless, it can be easily generalized to suit different contexts.
(Optional: Download the raw data sets from [BioSNAP](), unzip them accordingly, create a new folder called raw_data and upload all the CSV files. Run the data_exploration.ipynb notebook. This step is crucial if one wishes to modify the data sets or use data sets other than the ones specified.)
1. Clone this repo and upload all the items to a Google Drive account. Make note of the location where the items are uploaded on the drive.  
2. Navigate to the Training_and_Evaluation.ipynb (on the drive), and open it with Google Colaboratory.  
3. The first cell executes a commnad that mounts the Drive on Colab. Provide authorization as prompted.  
4. Next is a change directory command where one should put the relative path to the noted location of Step 1.  
5. The remainder of the jupyter notebook can be executed as is. If any problems pertaining to memory overflow is encountered, it may be resolved by restarting the   kernel and continuing execution from the cell at which the error was encountered. In case, this fix fails, the error is mainly due to the massive number of combinations that need to be evaluated. Tweaking the batch size during evaluation by altering the number of side-effects processed at each iteration(need to alter the evaluation scripts that are present in the evaluation folder) may help in fine-tuning the process to the resources at hand. Although this appears to be an extra hassle, it ends up making the implementation highly flexible and portable.
6. Run the results_compilation.ipynb notebook in a similar fashion. This notebook also has iteration variables that could be manipulated as needed to meet the resource constraints.  For example, in the current provided setting, we generate two CSV files that span the result set and tie them up together at the end. Nonetheless, itt is possible to create several such smaller CSV chunk files that span the result set and append them all into a single CSV file at the end.
*Excercise caution as all of the storage happens on the linked Google Drive account (It is possible that the default storage capacity provided by Google Drive is insufficient to deal with the order of this big data).*
7. Upload the 1-in-all combined CSV file to Google Cloud Storage (under a bucket). 
8. Create a BigQuery table from this CSV file and query the result set as desired. 
*Excercise caution while providing the drug identifiers in the queries. The Chemical IDentifier (CID) used is the same as that used by [PubChem](). One could use PubChem to obtain details of any of the 645 drugs that we can use (Remember, we are limited to the BioSNAP data set used which has protein-protein, protein-drug and drug-drug interactions for the 645 most common drugs that occur in polypharamacy contexts. Nonetheless, the model itself is flexible to work for any new protein, drug and/or side-effect without having to be re-trained. It is just necessary to model these new entities and represent them in the input subgraphs before passing it through the trained encoder/decoder framework.*




 
