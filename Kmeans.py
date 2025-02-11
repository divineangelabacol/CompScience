# Step 1: Define the dataset
dataset = [
    [2, 3], [3, 4], [4, 5], [10, 12], [11, 13], [12, 14],
    [50, 52], [51, 53], [52, 54]
]

# Step 2: Initialize Centroids (First K points in dataset)
def initialize_centroids(dataset, k):
    centroids = []
    for i in range(k):
        centroids.append(dataset[i])  # Pick first K points as initial centroids
    return centroids

# Step 3: Compute Euclidean Distance (Without Using Libraries)
def euclidean_distance(point1, point2):
    sum_squared = 0
    for i in range(len(point1)):  # Assume all points have the same dimensions
        sum_squared += (point1[i] - point2[i]) ** 2
    return sum_squared ** 0.5  # Square root manually

# Step 4: Assign Data Points to the Nearest Cluster
def assign_clusters(dataset, centroids):
    clusters = {}
    for i in range(len(centroids)):
        clusters[i] = []  # Create empty lists for each cluster
    
    for point in dataset:
        min_dist = float("inf")
        best_cluster = None
        for i in range(len(centroids)):
            dist = euclidean_distance(point, centroids[i])
            if dist < min_dist:
                min_dist = dist
                best_cluster = i
        clusters[best_cluster].append(point)  # Assign point to nearest cluster
    
    return clusters

# Step 5: Recalculate New Centroids
def compute_new_centroids(clusters):
    new_centroids = []
    
    for cluster in clusters:
        cluster_points = clusters[cluster]
        if len(cluster_points) == 0:
            continue  # Avoid empty clusters

        new_centroid = []
        for i in range(len(cluster_points[0])):  # Iterate over dimensions
            sum_values = 0
            for point in cluster_points:
                sum_values += point[i]
            new_centroid.append(sum_values / len(cluster_points))  # Compute mean
        
        new_centroids.append(new_centroid)
    
    return new_centroids

# Step 6: K-Means Algorithm
def kmeans(dataset, k, max_iterations=100):
    centroids = initialize_centroids(dataset, k)
    
    for _ in range(max_iterations):
        clusters = assign_clusters(dataset, centroids)
        new_centroids = compute_new_centroids(clusters)
        
        if new_centroids == centroids:  # Stop if centroids don't change
            break
        
        centroids = new_centroids  # Update centroids
    
    return centroids, clusters

# Run the K-Means Algorithm
k = 3  # Number of clusters
final_centroids, final_clusters = kmeans(dataset, k)

# Print the results
print("Final Centroids:", final_centroids)
print("Clustered Points:")
for cluster_id, points in final_clusters.items():
    print(f"Cluster {cluster_id}: {points}")
