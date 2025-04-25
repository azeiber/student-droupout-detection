import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Read the data
print("Loading data...")
df = pd.read_csv('data_backup.csv', sep=';')

# Select more features for analysis
features = [
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)'
]

# Prepare the data
print("Preparing data...")
X = df[features].copy()

# Handle missing values
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using silhouette analysis
print("Determining optimal number of clusters...")
silhouette_scores = []
K = range(2, 7)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Silhouette score for k={k}: {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# Use optimal number of clusters (choose k with highest silhouette score)
optimal_k = K[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

# Perform K-means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Create scatter plot using actual features
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Curricular units 1st sem (grade)'], 
                     df['Curricular units 2nd sem (grade)'],
                     c=df['Cluster'],
                     cmap='viridis',
                     alpha=0.6)

# Add cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3, label='Cluster Centers')

plt.xlabel('First Semester Grade')
plt.ylabel('Second Semester Grade')
plt.title('Student Performance Clusters')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

# Analyze cluster characteristics
print("\nCluster Characteristics:")
cluster_stats = df.groupby('Cluster').agg({
    'Curricular units 1st sem (grade)': ['mean', 'count'],
    'Curricular units 2nd sem (grade)': 'mean',
    'Target': ['mean', 'count']
}).round(2)

print("\nCluster Statistics:")
print(cluster_stats)

# Create feature importance visualization
plt.figure(figsize=(12, 6))
feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
feature_importance = pd.Series(feature_importance, index=features)
feature_importance.sort_values(ascending=True).plot(kind='barh')
plt.title('Feature Importance in Clustering')
plt.xlabel('Absolute Mean Cluster Center Value')
plt.tight_layout()
plt.show()

# Create heatmap of cluster centers
plt.figure(figsize=(15, 8))
cluster_centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
sns.heatmap(cluster_centers_df, annot=True, cmap='coolwarm', center=0)
plt.title('Cluster Centers Heatmap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Create dropout rate visualization
plt.figure(figsize=(10, 6))
dropout_rates = df.groupby('Cluster')['Target'].agg(['mean', 'count'])
dropout_rates['mean'].plot(kind='bar')
plt.title('Dropout Rate by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Dropout Rate')
for i, v in enumerate(dropout_rates['mean']):
    plt.text(i, v, f'n={dropout_rates["count"][i]}', ha='center', va='bottom')
plt.show()

# Save the clustered data
df.to_csv('clustered_grades.csv', index=False)
print("\nResults have been saved to 'clustered_grades.csv'") 