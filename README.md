# noduleclassification
Program to classify the nodules of lungs

Follow the below steps to execute the program:
1. python build_dataset.py --> to extract feature descriptors from the input image and store them in HDF5 on disk. 
2. python train_model.py --> train the model with required ML classifier, using features extracted in the first step. This also saves model on disk.

