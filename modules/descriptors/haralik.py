# import the necessary packages
from skimage import feature
import mahotas
import numpy as np

class Haralik:

	def __init__(self, ignore_zeros = True):
		self.ignore_zeros = ignore_zeros	

	def describe(self, image):		
		haralik  = mahotas.features.haralick(image, ignore_zeros = self.ignore_zeros).mean(axis=0)		
		return haralik

	def getFeatureVectorSize(self):
		return 13 
