# import the necessary packages
from skimage import feature
import mahotas
import numpy as np

class Haralik:

			

	def describe(self, image):		
		haralik  = mahotas.features.haralick(image).mean(axis=0)		
		return haralik

	def getFeatureVectorSize(self):
		return 0 
