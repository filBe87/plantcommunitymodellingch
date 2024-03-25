# PlantCommunityModellingCH

The files contained in this folder represent a minimal working example of the core DNN fitting and prediction as conducted in the paper "Deep learning from citizen-science data to model plant communities" by Brun et al. They can be used to fit the DNN architecutre with different cost functions, predictor sets, and observation weights, as described in the manuscript and to make annualy averaged spatial predictions of the model ensemble.

## Files supplied

**Training.ipynb**:
Jupyter notebook used to fit the low-resolution DNNs, both with the cross-entropy loss (cel) and the Normalized Discounted Cumulative Gain (ndcg) cost functions. In the section "Global definitions" the list 'mtype' can be modified to specify which predictor set, which observation weights, and which cost function should be used. See comments for details. Also, in this section, make sure that the working directory is set correctly.

**Spatial_projection.ipynb**:
Jupyter notebook used to predict the DNN ensemble (geometric mean of the cel and the ndcg fit) to all wooded low-resolution (100x100m) pixels of the canton of Ticino. It uses the low-resolution models from the manuscript (see below) and illustrates how the averaging over yeardays is done.

**models.py**:
Contains the class definitons for the DNN architecture.

**Sample_dat2.csv**:
CSV-table containing ca. 2900 observations of Ranunculus species in Switzerland from, extracted from GBIF (referenced below), which have been quality filtered as described in the manuscript and matched with the low-resolution predictor set. As test set, we defined 10% of the points at random. This data set is used by for training.

GBIF.org (23 February 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.fmq5x3

**Env_data_Tici.csv**:
Low-resolution predictor set for >80'000 wooded pixel in the northern part of the canton Ticino. The data has been used to create Supplementary Figure 4 in the manuscript. This data file is used for spatial projections.

**Mod_LR_cent.pth**:
Trained low-resolution DNN trained with the cross-entropy loss, as presented in the manuscript. It is used for spatial projections.

**Mod_LR_cent.pth**:
Trained low-resolution DNN trained with the normalized discounted cumulative gain, as presented in the manuscript. It is used for spatial projections.

## Dependencies

* jupyter notebook
* python version 3.8.5
* pytorch version 1.7.1
* pandas version 1.1.3
* numpy version 1.19.2
* pytorchltr version 0.2.1
	
## Hardware

Currelntly the script is expecting a cuda-ready GPU to work. If this is not available, set params['device'] = 'cpu' 
in the Global Definitions section. The script should run fine like this as well, but considerably slower. 

## Installation

The code itself does not need compilation/installation. The code depends on commonly-used open-source software only, 
and ample help for installation can be found on the internet. Installation of the dependencies should not take more 
than 45 min.

## Expected output

**Training.ipynb** returns accuracy and loss scores of training and test sets over the training epochs and fitted model objects that can be used for predictions.

**Spatial_projection.ipynb** returns an array with predicted observation probabilities (averaged over the season) for each inputted wooded pixel (row in the input array) for a subset of ca. 40 canopy-forming tree species. 

## Expected run time

The models in **Training.ipynb** take less than one minute to train on a laptop with a cuda-ready GPU. On a desktop computer without a 
cuda-ready GPU 15-30 minutes may be expected.

The predictions in **Spatial_projection.ipynb** take about one minute to run on a laptop with a cuda-ready GPU.
