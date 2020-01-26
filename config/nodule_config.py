#Author: Shivaranjini
# import the necessary packages
from os import path

# define the base path to the emotion dataset
BASE_PATH = "data"

# use the base path to define the path to the input emotions file
RAW_IMAGE_PATH = path.sep.join([BASE_PATH, "annotations.csv"])
PIXEL_PER_MM = 0.7

#Assuming the radius of the nodule as 6
NODULE_RADIUS_IN_PIXEL = 6	

#provide list of image descriptors to be applied (in-order) 
#["mean_std", "grayhist", "haralik", "lbp"]
IMAGE_DESC_LIST = ["mean_std","grayhist","lbp"]

#specify any classifier: LogisticRegression/RandomForests/DecisionTree/SVM
CLASSFIER = "SVM"

# define the path to where output logs will be stored
HDF5_OUTPUT_PATH = path.sep.join([BASE_PATH, "output/features.hdf5"])

# define the path to where output trained model will be stored
MODEL_OUTPUT_PATH = path.sep.join([BASE_PATH, "output/nodules.cpickle"])


