# This is a test script in Python that creates a decision tree regressor from the 15k_ga_adaptive.csv dataset
# We are using sklearn to see what result we should expect for our own decision tree

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import tree
import csv
from matplotlib import pyplot as plt

csv_file = '../datasets/15k_ga_adaptive.csv'

# Creating an array for each row in the file

p1=[]
p2=[]
p3=[]
p4=[]
p5=[]
p6=[]
p7=[]
p8=[]
matrix_size_x=[]
matrix_size_y=[]
performance=[]

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row if present
    for row in reader:
        p1.append(float(row[5]))  # Careful, the p1 values are in the 6th row, not the first!
        p2.append(float(row[4]))
        p3.append(float(row[8]))        
        p4.append(float(row[9]))
        p5.append(float(row[0]))
        p6.append(float(row[1]))
        p7.append(float(row[7]))
        p8.append(float(row[6]))
        matrix_size_x.append(float(row[2]))
        matrix_size_y.append(float(row[3]))
        performance.append(float(row[10]))

# Printing the first 5 values of each array to check the rows match
print(p1[:5])
print(p2[:5])       
print(p3[:5])
print(p4[:5])
print(p5[:5])
print(p6[:5])
print(p7[:5])
print(p8[:5])
print(matrix_size_x[:5])
print(matrix_size_y[:5])
print(performance[:5])

# Creating the labels array
labels = []
# Each label is the performance value within a certain range
for i in range(len(performance)):
    if performance[i] < 0.01:
        labels.append(0)
    elif performance[i] < 0.02:
        labels.append(1)
    elif performance[i] < 0.03:
        labels.append(2)
    elif performance[i] < 0.04:
        labels.append(3)
    elif performance[i] < 0.05:
        labels.append(4)
    elif performance[i] < 0.06:
        labels.append(5)
    elif performance[i] < 0.07:
        labels.append(6)            
    elif performance[i] < 0.08:
        labels.append(7)
    elif performance[i] < 0.09:
        labels.append(8)
    else:
        labels.append(9)    

# Creating the decision tree regressor
dataset = list(zip(p1, p2, p3, p4, p5, p6, p7, p8, matrix_size_x, matrix_size_y))
Y = labels
clf = tree.DecisionTreeRegressor()
clf = clf.fit(dataset, Y)
predictions = clf.predict(dataset)
print(predictions[:100])  # Print first 20 predictions to check

# Showing the tree in a text form to make it clearer
text_representation = tree.export_text(clf)
print(text_representation)

