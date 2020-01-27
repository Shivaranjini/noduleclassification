#Author: Shivaranjini
# USAGE
# python build_dataset.py 

# import the necessary packages
from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

import pandas as pd
from imutils import paths
import numpy as np
import cv2
from os import path

#internal modules
from modules.descriptors import LocalBinaryPatterns
from modules.descriptors import GrayHistogram
from modules.descriptors import Haralik
from modules.descriptors import Mean_Std
from modules.io import HDF5DatasetWriter

from config import nodule_config as config


#Read CSV annotations file 
dataFrame = pd.read_csv(config.RAW_IMAGE_PATH, sep=r'\s*,\s*',delimiter=';',
                           header=0, encoding='ascii', engine='python')

#shuffle data frames
dataFrame = shuffle(dataFrame)

#Gets the list of image descriptor specified in the config file in the same order
def Get_Image_Desc_Objs_List():
	featureDesc = []
	desc = None
	for p in config.IMAGE_DESC_LIST:
		if p == "mean_std":
			desc = Mean_Std()

		elif p == "grayhist":
			desc = GrayHistogram([4])

		elif p == "lbp":
			desc = LocalBinaryPatterns(24, 3)	

		elif p == "haralik":
			desc = Haralik()

		if(desc != None):
			featureDesc.append(desc)
			desc = None
	
	return featureDesc
	


# initialize the image descriptors
featureDesc = Get_Image_Desc_Objs_List()

#dynamically compute  feature vector size
feature_vector_size = 0
for desc in featureDesc:
	feature_vector_size += desc.getFeatureVectorSize()

#read annotaion information from data frame
imagePaths = dataFrame["Name"]
x_coordinates = dataFrame['x']
y_coordinates = dataFrame['y']
z_coordinates = dataFrame['z']
y = dataFrame['class']

#encode labels
le = LabelEncoder()
labels = le.fit_transform(y)

#Initialize HDF5 writer
dataset = HDF5DatasetWriter((len(imagePaths), feature_vector_size), config.HDF5_OUTPUT_PATH, dataKey="features")
dataset.storeClassLabels(le.classes_)


features = np.arange(0)
ls=[]

for (imagePath, x, y, z, label) in zip(imagePaths, x_coordinates, y_coordinates, z_coordinates, labels):
	x = int(x)
	y = int(y)
	
	image = cv2.imread(path.sep.join([config.BASE_PATH, imagePath]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = image.shape
	mask = np.zeros((height,width), np.uint8)
	circle_img = cv2.circle(mask, (x, y), config.NODULE_RADIUS_IN_PIXEL, (255,255,255), thickness=-1)
	masked_data = cv2.bitwise_and(image, image, mask = circle_img)
	crop = masked_data[y - config.NODULE_RADIUS_IN_PIXEL : y + config.NODULE_RADIUS_IN_PIXEL,
	x - config.NODULE_RADIUS_IN_PIXEL : x + config.NODULE_RADIUS_IN_PIXEL]
	
	#cv2.imshow(imagePath,crop)
	#cv2.waitKey(0)

	# describe the image using multiple image descriptors	
	for desc in featureDesc:
		features= np.append(features, desc.describe(crop))
	
	ls.append(label)
	

features = np.reshape(features, (imagePaths.shape[0], -1))
print(features)
dataset.add(features, ls)

dataset.close()


