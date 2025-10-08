import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from networkx.algorithms import community
import numpy as np

# ============ Configuration ============
INPUT_FILE = 'network_collab_hms.graphml'  # Change to your file path
OUTPUT_HTML = 'network_visualization.html'
ALGORITHM = 'louvain'  # Options: 'louvain', 'label_propagation', 'greedy_modularity'

# ============ 1. Load Network ============
print("Loading network file...")
G = nx.read_graphml(INPUT_FILE)
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Convert to undirected graph if directed
if G.is_directed():
    G = G.to_undirected()

# ============ 2. Community Detection ============
print(f"\nPerforming community detection using {ALGORITHM} algorithm...")

if ALGORITHM == 'louvain':
    communities = community.louvain_communities(G, seed=42)
elif ALGORITHM == 'label_propagation':
    communities = community.label_propagation_communities(G)
elif ALGORITHM == 'greedy_modularity':
    communities = community.greedy_modularity_communities(G)
else:
    raise ValueError(f"Unknown algorithm: {ALGORITHM}")

# Create node to community ID mapping
node_to_community = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_to_community[node] = i

print(f"Detected {len(communities)} communities")
for i, comm in enumerate(communities):
    print(f"Community {i}: {len(comm)} nodes")

# Calculate modularity
modularity = community.modularity(G, communities)
print(f"Modularity: {modularity:.4f}")

# ============ 3. Calculate Layout ============
print("\nCalculating network layout...")
# Use layout algorithm that doesn't depend on scipy
try:
    # Prefer spring_layout (requires scipy)
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
except ImportError:
    print("scipy not installed, using kamada_kawai_layout as fallback...")
    # kamada_kawai doesn't require scipy (but slower)
    pos = nx.kamada_kawai_layout(G)

# ============ 4. Create Interactive Visualization ============
print("Generating visualization...")

# Prepare edge data
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Prepare node data
node_x = []
node_y = []
node_color = []
node_text = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(node_to_community[node])
    
    # Node degree
    degree = G.degree(node)
    community_id = node_to_community[node]
    node_text.append(f'Node: {node}<br>Community: {community_id}<br>Degree: {degree}')

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        color=node_color,
        size=8,
        colorbar=dict(
            thickness=15,
            title=dict(text='Community ID', side='right'),
            xanchor='left'
        ),
        line=dict(width=0.5, color='white')
    )
)

# Create figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f'<br>Community Detection Results - {ALGORITHM.upper()}<br>Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Communities: {len(communities)} | Modularity: {modularity:.3f}',
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=80),
                    annotations=[dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                ))

# Save as HTML
fig.write_html(OUTPUT_HTML)
print(f"\nVisualization saved to: {OUTPUT_HTML}")

# ============ 5. Generate Statistics Charts ============
print("\nGenerating statistics charts...")

# Community size distribution
community_sizes = [len(comm) for comm in communities]

fig_stats = plt.figure(figsize=(15, 5))

# Subplot 1: Community size
plt.subplot(1, 3, 1)
plt.bar(range(len(community_sizes)), sorted(community_sizes, reverse=True))
plt.xlabel('Community Rank')
plt.ylabel('Community Size (Number of Nodes)')
plt.title('Community Size Distribution')
plt.grid(axis='y', alpha=0.3)

# Subplot 2: Degree distribution
plt.subplot(1, 3, 2)
degrees = [G.degree(n) for n in G.nodes()]
plt.hist(degrees, bins=50, edgecolor='black')
plt.xlabel('Node Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
plt.grid(axis='y', alpha=0.3)

# Subplot 3: Average degree per community
plt.subplot(1, 3, 3)
avg_degrees = []
for comm in communities:
    avg_deg = np.mean([G.degree(n) for n in comm])
    avg_degrees.append(avg_deg)
plt.bar(range(len(avg_degrees)), avg_degrees)
plt.xlabel('Community ID')
plt.ylabel('Average Degree')
plt.title('Average Degree by Community')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('community_statistics.png', dpi=150, bbox_inches='tight')
print("Statistics chart saved to: community_statistics.png")

# ============ 6. Output Community Details ============
print("\n============ Community Details ============")
for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
    print(f"\nCommunity {i} ({len(comm)} nodes):")
    # Calculate internal density
    subgraph = G.subgraph(comm)
    internal_edges = subgraph.number_of_edges()
    possible_edges = len(comm) * (len(comm) - 1) / 2
    density = internal_edges / possible_edges if possible_edges > 0 else 0
    print(f"  Internal density: {density:.4f}")
    
    # Show first 10 nodes (if node IDs are not too long)
    sample_nodes = list(comm)[:10]
    print(f"  Sample nodes: {sample_nodes}{'...' if len(comm) > 10 else ''}")

print("\nComplete!")
print(f"Please open {OUTPUT_HTML} to view the interactive visualization")