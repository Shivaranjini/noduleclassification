# noduleclassification
Program to classify lungs nodules

Follow the below steps to execute the program:
1. python build_dataset.py --> to extract feature descriptors from the input images and store them in HDF5 on disk. 
2. python train_model.py --> train the model with required ML classifier, using features extracted in the first step. This also saves model on disk.

config/nodule_config.py -> stores all the configuration parameters where in user can specify:
- Selection of multiple Image Descriptors (stacked in order to extract the features from input image)
- Nodule size
- Selection of ML Classifier to classify nodules


