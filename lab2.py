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

# Step 2: Calculate prior probabilities (P(Class))
def calculate_prior(dataset):
    class_counts = {}
    total_samples = len(dataset)
    
    for row in dataset:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    prior = {}
    for label in class_counts:
        prior[label] = class_counts[label] / total_samples
    return prior

# Step 3: Calculate likelihood (P(Feature | Class))
def calculate_likelihood(dataset):
    class_counts = {}
    feature_counts = {}

    for row in dataset:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
            feature_counts[label] = {i: [0, 0] for i in range(len(row) - 1)}
        
        class_counts[label] += 1
        for i in range(len(row) - 1):
            feature_counts[label][i][row[i]] += 1  # Count occurrences of 0s and 1s

    likelihood = {}
    for label in class_counts:
        likelihood[label] = {}
        for feature in feature_counts[label]:
            total = class_counts[label]
            likelihood[label][feature] = [
                feature_counts[label][feature][0] / total,  # P(Feature=0 | Class)
                feature_counts[label][feature][1] / total   # P(Feature=1 | Class)
            ]
    
    return likelihood

# Step 4: Apply Bayes' Theorem for prediction
def predict(instance, prior, likelihood):
    posteriors = {}

    for label in prior:
        posterior = prior[label]  # Start with prior probability
        
        for i in range(len(instance)):
            posterior *= likelihood[label][i][instance[i]]  # Multiply by likelihood
        
        posteriors[label] = posterior

    # Choose the class with the highest probability
    best_label = None
    best_prob = -1
    for label in posteriors:
        if posteriors[label] > best_prob:
            best_prob = posteriors[label]
            best_label = label
    
    return best_label

# Step 5: Run the classifier
def naive_bayes(dataset, new_instance):
    prior = calculate_prior(dataset)
    likelihood = calculate_likelihood(dataset)
    return predict(new_instance, prior, likelihood)

# Test the classifier
new_instance = [1, 0, 0]  # Example instance
result = naive_bayes(dataset, new_instance)
print("Predicted class for", new_instance, "is:", result)
