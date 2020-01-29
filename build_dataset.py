#Author: Shivaranjini
# USAGE
# python build_dataset.py 

# import the necessary packages
from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from skimage import measure
from imutils import contours
import pandas as pd
from imutils import paths
import imutils 
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

def process_in_parts(image, radius, x, y):
	background = 0
	width = 2 * radius + 1
	area = width * width 	
	
	if(image[x,y] == 0):
		background = 255 

	crop = image[y - radius : y + radius, x - radius : x + radius]
	labels = measure.label(crop, neighbors=4, background=background)
	mask = np.zeros(crop.shape, dtype="uint8")	

	for label in np.unique(labels):
	
		labelMask = np.zeros(crop.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if(area *0.45 < numPixels):
			mask = cv2.add(mask, labelMask) 

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if len(cnts) > 0 :
		cnts = contours.sort_contours(cnts)[0]

		new_radius = 0
		max_extent = 0
		# loop over the contours
		for (i, c) in enumerate(cnts):
			# draw the bright spot on the image
			(x1, y1, w, h) = cv2.boundingRect(c)

			extent = w * h/ area			
			
			if(extent > max_extent):				
				max_extent = extent
				((cX, cY), new_radius) = cv2.minEnclosingCircle(c)
				
			
		if(max_extent > 0):
			return new_radius	 
		else: 
			return 0
	return 0

def image_processing(imagePath, x, y):
	image = cv2.imread(path.sep.join([config.BASE_PATH, imagePath]))	
	height, width, channel = image.shape	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	mid = cv2.Canny(blurred, 30, 150)
	eroded = cv2.erode(mid.copy(), None, iterations=2)
	dilated = cv2.dilate(mid.copy(), None, iterations=2)
	
	k = 3
	radius = 0
	while (k < 15):
		rad = process_in_parts(dilated, k, x, y)	
		if(rad > radius):
			radius = rad
		else :
			break
		k = k + 1	
	
	if(radius > 0):
		radius = int(radius)
		mask = np.zeros((height,width), np.uint8)
		circle_img = cv2.circle(mask, (x, y),radius, (255,255,255), thickness=-1)
		masked_data = cv2.bitwise_and(image, image, mask = circle_img)
		crop = masked_data[y - radius: y + radius,
		x - radius : x + radius]		
		return crop
	return None	


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
			desc = LocalBinaryPatterns(8, 3)	

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
	crop = image_processing(imagePath, x, y)	
	crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("min circle",crop)	
	#cv2.waitKey(0)	
	# describe the image using multiple image descriptors	
	
	for desc in featureDesc:
		features= np.append(features, desc.describe(crop))
	
	ls.append(label)
	

features = np.reshape(features, (imagePaths.shape[0], -1))
dataset.add(features, ls)

dataset.close()

	
	


