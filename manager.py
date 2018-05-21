

## Manage the dataset

import random

import matplotlib
matplotlib.use('TkAgg')

def split_dataset(data_filename, train_proportion):
	"""
	-> Split data_filename to train and test data file, 
	according to train_proportion (belong to 0 - 1)

	-> Assume data_file is a csv file with no header
	"""

	train_data_filename = data_filename.split(".")
	train_data_filename = train_data_filename[0]+"_train.csv"
	train_data = open(train_data_filename, "w")
	train_data.close()

	test_data_filename = data_filename.split(".")
	test_data_filename = test_data_filename[0]+"_test.csv"
	test_data = open(test_data_filename, "w")
	test_data.close()

	## Get the number of entries, assume the file has no header
	data_file = open(data_filename, "r")
	number_of_lines = 0
	for line in data_file:
		number_of_lines += 1
	data_file.close()

	## compute the number of line to keep in train data
	number_of_lines_to_keep = train_proportion * number_of_lines

	## split the data
	number_of_lines_in_train_dataset = 0
	selected_lines = []
	while(number_of_lines_in_train_dataset < number_of_lines_to_keep):
		line_selected = random.randint(0,number_of_lines)
		if(line_selected not in selected_lines):

			## Find the corresponding line
			data_file = open(data_filename, "r")
			cmpt = 0
			for line in data_file:
				
				line = line.split("\n")
				line = line[0]

				if(cmpt == line_selected):
				
					if(number_of_lines_in_train_dataset == number_of_lines_to_keep - 1):
						## write line in train data file
						train_data = open(train_data_filename, "a")
						train_data.write(line)
						train_data.close()
					else:
						## write line in train data file
						train_data = open(train_data_filename, "a")
						train_data.write(line+"\n")
						train_data.close()

					selected_lines.append(line_selected)
					number_of_lines_in_train_dataset += 1
				cmpt += 1

			data_file.close()

	## store the rest of lines in test data file
	data_file = open(data_filename, "r")
	number_of_lines_in_test_dataset = 0
	cmpt = 0
	for line in data_file:
		line = line.split("\n")
		line = line[0]
		if(cmpt not in selected_lines):
			if(number_of_lines_in_test_dataset == number_of_lines - number_of_lines_in_train_dataset - 1):

				test_data = open(test_data_filename, "a")
				test_data.write(line)
				test_data.close()
			else:
				test_data = open(test_data_filename, "a")
				test_data.write(line+"\n")
				test_data.close()
		cmpt += 1
	data_file.close()


split_dataset("data.csv", 0.7)



## Model part

from numpy import loadtxt
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
train_dataset = loadtxt('data_train.csv', delimiter=",")
test_dataset = loadtxt('data_test.csv', delimiter=",")

# split data into X and y
train_X = train_dataset[:,1:-1]
train_Y = train_dataset[:,0]

test_X = test_dataset[:,1:-1]
test_Y = test_dataset[:,0]



model = XGBClassifier()
model.fit(train_X, train_Y)

xgboost.plot_importance(model)


# make predictions for test data
y_pred = model.predict(test_X)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(test_Y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
