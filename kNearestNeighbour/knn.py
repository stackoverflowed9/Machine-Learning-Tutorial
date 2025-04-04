import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = np.array(pd.read_csv("knn_dataset.csv"))
test_set = np.array([[4, 4], [5,4], [1,2], [7,3]])  # We will predict to which class these points belong

color_array = np.where(df[:, 2] == 0, "r", "b")
plt.scatter(df[:, 0], df[:, 1], c=color_array)      # Class 0 = Red, Class 1 = Blue points
plt.scatter(test_set[:, 0], test_set[:, 1], c="grey")   # Test points = Grey points
plt.show()

# Euclidean distance = ((x1-x2)^2 + (y1-y2)^2)^(1/2)

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))  # axis=1 meaning sum along a row
                                                    # axis=0 would do the sum along a column(x1+x2+ ...)


def k_nearest_neighbors(train_set, test_set, k=3):
    predictions = []
    
    for point in test_set:
        
        point = point.reshape(1, -1)    # Convert the point array into a row vector, since train set is a collection of row vectors

        distances = euclidean_distance(train_set[:, :2], point)

        k_indices = np.argsort(distances)[:k]

        k_labels = train_set[k_indices, 2]

        prediction = np.bincount(k_labels.astype(int)).argmax()
        predictions.append(prediction)
    
    return predictions

def main():
    predictions = k_nearest_neighbors(df, test_set, k=3)

    for i in range(test_set.shape[0]):
        print(f"({test_set[i][0]}, {test_set[i][1]}) --> Class: {int(predictions[i])}")

main()