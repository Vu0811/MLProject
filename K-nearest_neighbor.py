import numpy as np
import pandas as pd
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

#KNN Algorithm
#Find most similar neighbors
def Find_similar_Neighbors(trained, test, number_of_neighbors):
    distances = list()
    for train_row in trained:
        dist = np.linalg.norm(np.array(train_row[:-1])-np.array(test[:-1]))
        distances.append((train_row,dist))
    distances.sort(key = lambda row : row[1])
    neighbors = list()
    for i in range(number_of_neighbors - 1):
        neighbors.append(distances[i])
    return neighbors

#get prediction
def get_prediction(trained, test, number_of_neighbors):
    neighbors = Find_similar_Neighbors(trained, test, number_of_neighbors)
    output_cluster = [rows[0][-1] for rows in neighbors]
    prediction = max(set(output_cluster),key= output_cluster.count)
    return prediction
#data
data = pd.read_csv (r'E:/Works/Visual Studio/Artificial Intelligence/Data/K-Neareat-Neighbor/IRIS.csv')
iris = np.array(data)
dataset = iris.tolist();
i = 134
prediction = get_prediction(dataset[1:150], dataset[i], 5)
print('Expected %s, Got %s.' % (dataset[i][-1], prediction))
accurate = list()
for k in range(3,10):
    actural = [acturals[-1] for acturals in dataset]
    predicted = list()
    for a in range(len(dataset)) : 
        predicted.append(get_prediction(dataset, dataset[a], k))
    print('Correct rate : %d %% with k = %d' % (accuracy_metric(actural,predicted),k))

