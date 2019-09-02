#!/usr/bin/python3

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def valuePrediction(fileName, columnNum):
	wholeDataSet = readFile(fileName, columnNum)
	x = wholeDataSet[:,[2,5]] # Let number of doors and safety be the attributes for prediction
	y = wholeDataSet[:,[0]] # Set buying value as the target for this prediction
	xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.3,random_state=42)
	print(xTrain.shape)
	print(xTest.shape)
	print(x)
	print(y)
	kRange = range(1,51)
	score = []

	for k in range(1,101):
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(xTrain, yTrain)
		yPred = knn.predict(xTest)
		score.append(metrics.accuracy_score(yTest, yPred))
	print(max(score))
	
	return None







# Read file and return a numpy array object
def readFile(fileName, columnNum):
	result = np.zeros((1,columnNum))
	f = open(fileName, "r")
	for x in f:
		info = np.array(x.split(","))
		for i in range(0,columnNum):
			if info[i] == 'vhigh':
				info[i] = 4
			elif info[i] == 'high':
				info[i] = 3
			elif info[i] == 'med':
				info[i] = 2
			elif info[i] == 'low':
				info[i] = 1
			elif info[i] == '5more':
				info[i] = 5
		result = np.vstack((result, info))
	result = np.delete(result, (0), axis=0)
	return result


valuePrediction("car.txt", 7)