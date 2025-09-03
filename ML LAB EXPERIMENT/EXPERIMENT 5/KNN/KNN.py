import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names=["sepallength","sepalwidth","petallength","petalwidth","class"]
dataset=pd.read_csv(url,names=col_names)
dataset
dataset_knn=dataset.values[:,0:4]
labels=dataset.values[:,-1]

dataset_knn
import math
def distance(x1,x2):
  distances=0.0
  for i in range(len(x1)):
    distances+=(x1[i]-x2[i])**2
  return (math.sqrt(distances))
testing_data=dataset.values[54,0:4]


distances_with_labels = []
for i in range(len(dataset_knn)):
  dist = distance(testing_data, dataset_knn[i])
  distances_with_labels.append((dist, labels[i]))

distances_with_labels.sort()
distances_cumm.sort()
distances_with_labels
k = 10
top_k_neighbors = distances_with_labels[:k]

class_counts = {}
for dist, label in top_k_neighbors:
  class_counts[label] = class_counts.get(label, 0) + 1

predicted_class = max(class_counts, key=class_counts.get)

print(f"The predicted class for the testing data is: {predicted_class}")
max_key = max(class_counts, key=class_counts.get)
print("predicated class :", max_key)





