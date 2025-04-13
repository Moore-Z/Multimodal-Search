import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

# Assuming document_embeddings is a list of 200 numpy arrays, each with shape (384, 150)
# We need to reshape this to get a single vector per document

document_embeddings = np.array()

# Method 1: Mean pooling across words (axis=1)
document_vectors = []
for doc_embedding in document_embeddings:
    # Average across all words (axis=1)
    # This gives us a vector of shape (384,)
    doc_vector = np.mean(doc_embedding, axis=1)
    document_vectors.append(doc_vector)

# Convert to numpy array
document_vectors = np.array(document_vectors)  # Shape: (200, 384)

# Now we can proceed with Affinity Propagation using cosine similarity
cosine_similarities = 1 - pairwise_distances(document_vectors, metric='cosine')

# Apply Affinity Propagation
af = AffinityPropagation(affinity='precomputed', random_state=42)
cluster_labels = af.fit_predict(cosine_similarities)

# Get cluster centers and organize results
cluster_centers_indices = af.cluster_centers_indices_
n_clusters = len(cluster_centers_indices)

# Group documents by cluster
clusters = {}
for i, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(i)

# Print results
print(f"Number of clusters: {n_clusters}")
for cluster_id, document_indices in clusters.items():
    print(f"Cluster {cluster_id} contains {len(document_indices)} documents")
    print(f"Representative document index: {cluster_centers_indices[cluster_id]}")
    print(f"Document indices: {document_indices}")
    print("---")