#Author: Shivaranjini
# USAGE
# python train_model.py 


# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle
import h5py
from config import nodule_config as config


# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(config.HDF5_OUTPUT_PATH, "r")
i = int(db["labels"].shape[0] *0.75)

def Define_Model(classifier):
	
	if classifier == "LogisticRegression":
		# define the set of parameters that we want to tune then start a
		# grid search where we evaluate our model for each value of C
		params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
		model = GridSearchCV(LogisticRegression(), params, cv=3,
			n_jobs=-1)

	elif classifier == "SVM":
		model = SVC(kernel="poly", degree=2, coef0=1)

	elif classifier == "DecisionTree":
		model = DecisionTreeClassifier(random_state=84)

	else:
		model = RandomForestClassifier(n_estimators=20, random_state=42)

	return model


def SaveModel(model, classifier):
	
	if classifier == "LogisticRegression":
		f = open(config.MODEL_OUTPUT_PATH, "wb")
		f.write(pickle.dumps(model.best_estimator_))
		f.close()
	else:
		f = open(config.MODEL_OUTPUT_PATH, "wb")
		f.write(pickle.dumps(model))
		f.close()
	
#Create model based on configuration parameters
model = Define_Model(config.CLASSFIER)

#Fit model with train data
model.fit(db["features"][:i], db["labels"][:i])

# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
	target_names=db["label_names"]))

# serialize the model to disk
print("[INFO] saving model...")
SaveModel(model, config.CLASSFIER)


# close the database
db.close()
