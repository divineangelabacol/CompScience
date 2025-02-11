# Step 1: Define the dataset
# Format: [feature1, feature2, ..., featureN, class_label]
dataset = [
    [1, 1, 1],  # Class 1
    [1, 1, 0],  # Class 0
    [1, 0, 1],  # Class 1
    [0, 1, 1],  # Class 0
    [0, 0, 0],  # Class 1
    [1, 0, 0],  # Class 1
    [0, 1, 0],  # Class 0
]

# Step 2: Compute Euclidean Distance (without math package)
def euclidean_distance(instance1, instance2):
    sum_squared = 0
    for i in range(len(instance1) - 1):  # Exclude class label
        sum_squared += (instance1[i] - instance2[i]) ** 2
    return sum_squared ** 0.5  # Square root manually

# Step 3: Find K Nearest Neighbors
def get_neighbors(dataset, new_instance, k):
    distances = []
    
    for row in dataset:
        dist = euclidean_distance(row, new_instance)
        distances.append((row, dist))  # Store (data point, distance)
    
    # Sort by distance (smallest first)
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i][1] > distances[j][1]:  # Simple sorting
                distances[i], distances[j] = distances[j], distances[i]
    
    # Return the first K neighbors
    return [distances[i][0] for i in range(k)]

# Step 4: Predict the class
def predict(dataset, new_instance, k):
    neighbors = get_neighbors(dataset, new_instance, k)
    class_counts = {}

    # Count occurrences of each class in the neighbors
    for neighbor in neighbors:
        label = neighbor[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Determine the most frequent class
    best_label = None
    best_count = -1
    for label in class_counts:
        if class_counts[label] > best_count:
            best_count = class_counts[label]
            best_label = label

    return best_label

# Step 5: Run the KNN Classifier
def knn_classifier(dataset, new_instance, k):
    return predict(dataset, new_instance, k)

# Test the classifier
new_instance = [1, 0, 0]  # Example instance
k = 3  # Number of neighbors to consider
result = knn_classifier(dataset, new_instance, k)
print("Predicted class for", new_instance, "with k =", k, "is:", result)
