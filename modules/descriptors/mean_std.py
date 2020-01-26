#Author: Shivaranjini
# import the necessary packages
import cv2
import numpy as np

class Mean_Std:


	def describe(self, image):
		(means, stds) = cv2.meanStdDev(image)
		colorStats = np.concatenate([means, stds]).flatten()		

				
		# return out 3D histogram as a flattened array
		return colorStats

	def getFeatureVectorSize(self):
		return 2
