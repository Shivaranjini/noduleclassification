#Author: Shivaranjini
# import the necessary packages
import cv2

class GrayHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		self.bins = bins

	def describe(self, image):
		# compute a histogram in the then normalize the histogram so
		# that images with the same content, but either scaled larger or smaller will
		# have (roughly) the same histogram
		hist = cv2.calcHist([image], [0],
			None, self.bins, [0, 256])
		cv2.normalize(hist, hist)

		# return out 3D histogram as a flattened array
		return hist.flatten()

	def getFeatureVectorSize(self):
		return self.bins[0]
