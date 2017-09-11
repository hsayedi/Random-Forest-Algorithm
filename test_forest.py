# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:37:34 2017

@author: Husna
"""

# This is my re-coding of Browniee's algorithm for Random Forest


# Import the necessary packages 
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# First we will import the CSV file of the Sonar Dataset
# data is the name of the Sonar Dataset
def import_csv(filename):
	data = list()
	with open(filename, 'r') as sonar: 
		csv_reader = reader(sonar)
		for row in csv_reader:
			if not row:
				continue
			data.append(row)
	return data
 
 # Convert string column to float
def convert(data, column):
	for row in data:
		row[column] = float(row[column].strip())
  
# Convert string column to integer
def convert1(data, column):
	values = [row[column] for row in data]
	unique = set(values)
	find = dict()
	for i, value in enumerate(unique):
		find[value] = i
	for row in data:
		row[column] = find[row[column]]
	return find
 
 # Split a data into k models
def kmodels(data, n_folds):
	data_split = list()
	data_copy = list(data)
	fold_size = int(len(data) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split
 
 # Calculate accuracy percentage
def accuracy(actual, predicted):
	accurate = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			accurate += 1
	return accurate / float(len(actual))*100.0
 
 # Evaluate an algorithm using a cross validation split
def EvalAlgorithm(data, algorithm, n_folds, *args):
	folds = kmodels(data, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy1 = accuracy(actual, predicted)
		scores.append(accuracy1)
	return scores
 
 # Split a dataset based on an attribute and an attribute value
def HelpSplit(index, value, data):
	left, right = list(), list()
	for row in data:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def ginii(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
 # Select the best split point for a dataset
def FindSplit(data, n_feats):
	class_values = list(set(row[-1] for row in data))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	feats = list()
	while len(feats) < n_feats:
		index = randrange(len(data[0])-1)
		if index not in feats:
			feats.append(index)
	for index in feats:
		for row in data:
			groups = HelpSplit(index, row[index], data)
			gini = ginii(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

 
# Create a terminal node value
def terminal(group):
	results = [row[-1] for row in group]
	return max(set(results), key=results.count)
 
 # Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_feats, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = terminal(left), terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = terminal(left)
	else:
		node['left'] = FindSplit(left, n_feats)
		split(node['left'], max_depth, min_size, n_feats, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = terminal(right)
	else:
		node['right'] = FindSplit(right, n_feats)
		split(node['right'], max_depth, min_size, n_feats, depth+1)
 
# We now build a decision tree based on depth, size, feature, train
def TreeBuild(train, max_depth, min_size, n_feats):
	root = FindSplit(train, n_feats)
	split(root, max_depth, min_size, n_feats, 1)
	return root
 
# Uce decision tree to make prediction with nodes and row
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Generate a random subsample from the data with replacement
def SubSample(data, ratio):
	sample = list()
	n_sample = round(len(data) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(data))
		sample.append(data[index])
	return sample
 
# Make a prediction with a list of bagged trees
def BaggingPredict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)
 
# Random Forest Algorithm
def RandomForest(train, test, max_depth, min_size, sample_size, n_trees, n_feats):
	trees = list()
	for i in range(n_trees):
		sample = SubSample(train, sample_size)
		tree = TreeBuild(sample, max_depth, min_size, n_feats)
		trees.append(tree)
	predictions = [BaggingPredict(trees, row) for row in test]
	return(predictions)
 
# We now est the random forest algorithm
seed(2)

# Now load and prepare data
filename = 'sonar.all-data.csv'
data = import_csv(filename)

# Convert strings to integers
for i in range(0, len(data[0])-1):
	convert(data, i)

# Convert class column to integers
convert1(data, len(data[0])-1)

# Input values, evaluate algorithm for 1, 5, 10 trees
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_feats = int(sqrt(len(data[0])-1))
for n_trees in [1, 5, 10]:
	scores = EvalAlgorithm(data, RandomForest, n_folds, max_depth, min_size, sample_size, n_trees, n_feats)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))




 
 
 
 
 
 
 
 
 
 
 