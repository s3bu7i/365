# Mall Customer Segmentation using K-Means and Hierarchical Clustering
# Complete solution for unsupervised learning assignment

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
# Make sure to place the Mall_Customers.csv file in the same directory
df = pd.read_csv('C:/Users/Dino/Desktop/ai/task 6/Mall Customers.csv')

print("=== MALL CUSTOMER SEGMENTATION ANALYSIS ===\n")
print("1. DATASET OVERVIEW")
print("=" * 50)
print(f"Dataset shape: {df.shape}")
print(f"\nDataset info:")
print(df.info())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nBasic statistics:")
print(df.describe())

# Data preprocessing
print("\n2. DATA PREPROCESSING")
print("=" * 50)

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Prepare features for clustering (using Age, Annual Income, and Spending Score)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

print(f"\nFeatures selected for clustering: {features}")
print(f"Feature matrix shape: {X.shape}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print("\nFeatures have been standardized for clustering")

# ============================================================================
# PART A & B: K-MEANS CLUSTERING WITH OPTIMAL CLUSTER DETERMINATION
# ============================================================================

print("\n3. K-MEANS CLUSTERING ANALYSIS")
print("=" * 50)

# Elbow Method for K-Means


def elbow_method(X, max_clusters=10):
    """Calculate within-cluster sum of squares (WCSS) for different cluster numbers"""
    wcss = []
    K_range = range(1, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    return K_range, wcss

# Silhouette Score Analysis for K-Means


def silhouette_analysis(X, max_clusters=10):
    """Calculate silhouette scores for different cluster numbers"""
    silhouette_scores = []
    # Silhouette score requires at least 2 clusters
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    return K_range, silhouette_scores


# Apply Elbow Method and Silhouette Analysis
k_range, wcss = elbow_method(X_scaled, max_clusters=10)
sil_k_range, silhouette_scores = silhouette_analysis(X_scaled, max_clusters=10)

# Find optimal number of clusters
optimal_k_silhouette = sil_k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters (Silhouette Score): {optimal_k_silhouette}")
print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")

# Visualize Elbow Method and Silhouette Scores
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Elbow Method Plot
axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

# Silhouette Score Plot
axes[1].plot(sil_k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Different k')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=optimal_k_silhouette, color='red', linestyle='--',
                alpha=0.7, label=f'Optimal k = {optimal_k_silhouette}')
axes[1].legend()

plt.tight_layout()
plt.show()

# Apply K-Means with optimal number of clusters
# You can also manually set this to 5 if elbow suggests 5
optimal_k = optimal_k_silhouette
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print(f"\nK-Means clustering completed with {optimal_k} clusters")
print(f"Cluster distribution: {np.bincount(kmeans_labels)}")

# ============================================================================
# HIERARCHICAL CLUSTERING
# ============================================================================

print("\n4. HIERARCHICAL CLUSTERING ANALYSIS")
print("=" * 50)

# Create dendrogram to visualize hierarchical clustering


def plot_dendrogram(X, method='ward'):
    """Create and plot dendrogram"""
    plt.figure(figsize=(12, 6))

    if method == 'ward':
        linkage_matrix = linkage(X, method='ward')
    else:
        distances = pdist(X)
        linkage_matrix = linkage(distances, method=method)

    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title(
        f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.show()

    return linkage_matrix


# Plot dendrogram
linkage_matrix = plot_dendrogram(X_scaled, method='ward')

# Apply Hierarchical Clustering with same number of clusters as K-Means
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

print(f"Hierarchical clustering completed with {optimal_k} clusters")
print(f"Cluster distribution: {np.bincount(hierarchical_labels)}")

# ============================================================================
# PART C: VISUALIZATION AND COMPARISON
# ============================================================================

print("\n5. CLUSTER VISUALIZATION AND COMPARISON")
print("=" * 50)

# Add cluster labels to original dataset
df_results = df.copy()
df_results['KMeans_Cluster'] = kmeans_labels
df_results['Hierarchical_Cluster'] = hierarchical_labels

# 3D Scatter plots for both methods
fig = plt.figure(figsize=(18, 7))

# K-Means 3D plot
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
                       c=kmeans_labels, cmap='viridis', s=60, alpha=0.7)
ax1.set_xlabel('Age')
ax1.set_ylabel('Annual Income (k$)')
ax1.set_zlabel('Spending Score (1-100)')
ax1.set_title('K-Means Clustering Results')

# Hierarchical 3D plot
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
                       c=hierarchical_labels, cmap='viridis', s=60, alpha=0.7)
ax2.set_xlabel('Age')
ax2.set_ylabel('Annual Income (k$)')
ax2.set_zlabel('Spending Score (1-100)')
ax2.set_title('Hierarchical Clustering Results')

plt.tight_layout()
plt.show()

# 2D visualizations for each pair of features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

feature_pairs = [('Age', 'Annual Income (k$)'),
                 ('Age', 'Spending Score (1-100)'),
                 ('Annual Income (k$)', 'Spending Score (1-100)')]

for i, (feature1, feature2) in enumerate(feature_pairs):
    # K-Means plots
    axes[0, i].scatter(df[feature1], df[feature2],
                       c=kmeans_labels, cmap='viridis', alpha=0.7)
    axes[0, i].set_xlabel(feature1)
    axes[0, i].set_ylabel(feature2)
    axes[0, i].set_title(f'K-Means: {feature1} vs {feature2}')

    # Hierarchical plots
    axes[1, i].scatter(df[feature1], df[feature2],
                       c=hierarchical_labels, cmap='viridis', alpha=0.7)
    axes[1, i].set_xlabel(feature1)
    axes[1, i].set_ylabel(feature2)
    axes[1, i].set_title(f'Hierarchical: {feature1} vs {feature2}')

plt.tight_layout()
plt.show()

# Compare clustering results

ari_score = adjusted_rand_score(kmeans_labels, hierarchical_labels)
nmi_score = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)

print(f"Clustering Comparison Metrics:")
print(f"Adjusted Rand Index: {ari_score:.3f}")
print(f"Normalized Mutual Information: {nmi_score:.3f}")
print("(Higher values indicate more similar clustering results)")

# ============================================================================
# PART D: CUSTOMER SEGMENT INTERPRETATION AND MARKETING STRATEGY
# ============================================================================

print("\n6. CUSTOMER SEGMENT ANALYSIS & MARKETING INSIGHTS")
print("=" * 50)

# Analyze K-Means clusters (using K-Means as primary method)
cluster_analysis = df_results.groupby('KMeans_Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'CustomerID': 'count'
}).round(2)

cluster_analysis.columns = ['Age_Mean', 'Age_Std', 'Income_Mean', 'Income_Std',
                            'Spending_Mean', 'Spending_Std', 'Count']

print("Detailed Cluster Analysis (K-Means):")
print(cluster_analysis)

# Create customer segment profiles


def interpret_clusters(df_results, method='KMeans_Cluster'):
    """Interpret and name customer segments"""

    segments = {}
    cluster_col = method

    for cluster in sorted(df_results[cluster_col].unique()):
        cluster_data = df_results[df_results[cluster_col] == cluster]

        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        size = len(cluster_data)

        # Determine segment characteristics
        income_level = 'High' if avg_income > 70 else 'Medium' if avg_income > 40 else 'Low'
        spending_level = 'High' if avg_spending > 70 else 'Medium' if avg_spending > 40 else 'Low'
        age_group = 'Young' if avg_age < 35 else 'Middle-aged' if avg_age < 55 else 'Senior'

        segments[cluster] = {
            'name': f"Cluster {cluster}",
            'size': size,
            'avg_age': round(avg_age, 1),
            'avg_income': round(avg_income, 1),
            'avg_spending': round(avg_spending, 1),
            'income_level': income_level,
            'spending_level': spending_level,
            'age_group': age_group,
            'percentage': round(size/len(df_results)*100, 1)
        }

    return segments


# Get segment interpretations
segments = interpret_clusters(df_results)

print("\nCUSTOMER SEGMENT PROFILES:")
print("=" * 60)

segment_names = {}
marketing_strategies = {}

for cluster_id, info in segments.items():
    # Create meaningful segment names
    if info['income_level'] == 'High' and info['spending_level'] == 'High':
        segment_name = "Premium Customers"
        strategy = "VIP programs, luxury products, personalized service, exclusive events"
    elif info['income_level'] == 'High' and info['spending_level'] == 'Low':
        segment_name = "Cautious Affluent"
        strategy = "Value propositions, quality assurance, investment-focused products"
    elif info['income_level'] == 'Low' and info['spending_level'] == 'High':
        segment_name = "Impulsive Spenders"
        strategy = "Budget-friendly options, payment plans, impulse-buy promotions"
    elif info['income_level'] == 'Medium' and info['spending_level'] == 'High':
        segment_name = "Enthusiastic Shoppers"
        strategy = "Seasonal campaigns, loyalty programs, trendy products"
    elif info['income_level'] == 'Medium' and info['spending_level'] == 'Medium':
        segment_name = "Balanced Consumers"
        strategy = "Balanced offerings, family packages, practical products"
    else:
        segment_name = "Budget Conscious"
        strategy = "Discounts, bulk offers, essential products, clearance sales"

    segment_names[cluster_id] = segment_name
    marketing_strategies[cluster_id] = strategy

    print(f"\n {segment_name} (Cluster {cluster_id})")
    print(
        f"   Size: {info['size']} customers ({info['percentage']}% of total)")
    print(
        f"   Profile: {info['age_group']}, {info['income_level']} Income, {info['spending_level']} Spending")
    print(f"   Avg Age: {info['avg_age']} years")
    print(f"   Avg Income: ${info['avg_income']}k")
    print(f"   Avg Spending Score: {info['avg_spending']}/100")
    print(f"   Marketing Strategy: {strategy}")

# Create a comprehensive visualization of segments
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Segment size distribution
segment_sizes = [info['size'] for info in segments.values()]
segment_labels = [
    f"{segment_names[i]}\n({info['percentage']}%)" for i, info in segments.items()]

axes[0, 0].pie(segment_sizes, labels=segment_labels,
               autopct='%1.0f%%', startangle=90)
axes[0, 0].set_title('Customer Segment Distribution')

# Average metrics by segment
cluster_ids = list(segments.keys())
ages = [segments[i]['avg_age'] for i in cluster_ids]
incomes = [segments[i]['avg_income'] for i in cluster_ids]
spendings = [segments[i]['avg_spending'] for i in cluster_ids]

x = np.arange(len(cluster_ids))
width = 0.25

axes[0, 1].bar(x - width, ages, width, label='Age', alpha=0.8)
axes[0, 1].bar(x, incomes, width, label='Income (k$)', alpha=0.8)
axes[0, 1].bar(x + width, spendings, width, label='Spending Score', alpha=0.8)
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Average Values')
axes[0, 1].set_title('Average Metrics by Segment')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([f'Cluster {i}' for i in cluster_ids])
axes[0, 1].legend()

# Income vs Spending scatter with segment names
for cluster_id in cluster_ids:
    cluster_data = df_results[df_results['KMeans_Cluster'] == cluster_id]
    axes[1, 0].scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                       label=segment_names[cluster_id], alpha=0.7, s=60)

axes[1, 0].set_xlabel('Annual Income (k$)')
axes[1, 0].set_ylabel('Spending Score (1-100)')
axes[1, 0].set_title('Customer Segments: Income vs Spending')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Age distribution by segment
for cluster_id in cluster_ids:
    cluster_data = df_results[df_results['KMeans_Cluster'] == cluster_id]
    axes[1, 1].hist(cluster_data['Age'], alpha=0.7,
                    label=segment_names[cluster_id], bins=15)

axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Age Distribution by Segment')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("COMPREHENSIVE MARKETING STRATEGY RECOMMENDATIONS")
print("="*80)

print("""
OVERALL MARKETING INSIGHTS:

1. SEGMENT-SPECIFIC STRATEGIES:
   • Focus premium marketing on high-income, high-spending segments
   • Develop budget-friendly options for price-sensitive segments  
   • Create targeted age-appropriate campaigns
   • Implement loyalty programs for medium-spending customers

2. PRODUCT POSITIONING:
   • Premium products for affluent customers
   • Value products for budget-conscious segments
   • Trendy items for younger demographics
   • Practical products for middle-aged customers

3. CHANNEL STRATEGY:
   • Digital marketing for younger segments
   • Traditional marketing for older segments
   • Personalized recommendations for high-spenders
   • Mass marketing for large segments

4. PRICING STRATEGY:
   • Premium pricing for high-income segments
   • Competitive pricing for price-sensitive groups
   • Bundle offers for medium-spending customers
   • Seasonal promotions for impulse buyers

5. CUSTOMER RETENTION:
   • VIP programs for top customers
   • Loyalty rewards for regular shoppers
   • Win-back campaigns for low-activity segments
   • Referral programs across all segments
""")
