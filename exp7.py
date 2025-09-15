import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np

# ------------------------------
# Student Info
student_name = "Kaustubh Bane"
roll_no = 4
# ------------------------------

# Step 1: Define Football Teams
teams = ["Barcelona", "Real Madrid", "Liverpool", "Manchester City", "Chelsea",
         "Bayern Munich", "Juventus", "PSG", "Inter Milan", "Arsenal"]
num_teams = len(teams)

# Step 2: Create Graph where nodes = teams, edges = matches
G = nx.Graph()

# Add nodes
for team in teams:
    G.add_node(team)

# Add random edges to represent matches played
np.random.seed(42)
for i in range(num_teams):
    for j in range(i + 1, num_teams):
        if np.random.rand() < 0.4:  # 40% chance match happened
            G.add_edge(teams[i], teams[j], weight=np.random.randint(1, 5))  # 1-4 matches

# Step 3: Convert to adjacency matrix
adj_matrix = nx.to_numpy_array(G)

# Apply Spectral Clustering
n_clusters = 3
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                        assign_labels='kmeans', random_state=42)
labels = sc.fit_predict(adj_matrix)

# Step 4: Visualize clusters
colors = ['red', 'green', 'blue', 'orange', 'purple']
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 8))

# Add student info at the top of window
plt.suptitle(f"{student_name} - Roll No {roll_no}", fontsize=16, fontweight='bold', color='darkblue')

# Draw graph with partitions
nx.draw(G, pos, with_labels=True,
        node_color=[colors[label] for label in labels],
        node_size=1200, edge_color='gray', font_size=10, font_weight='bold')

plt.title("Football Teams Clustering (Matches Based)", fontsize=14)
plt.show()

# Print cluster labels
for team, label in zip(teams, labels):
    print(f"Team: {team} -> Cluster {label}")
