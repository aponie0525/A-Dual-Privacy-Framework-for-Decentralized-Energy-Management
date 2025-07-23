"""
This script creates a directed graph and saves it to a JSON file.
生成电网联络线信息，并将其保存为JSON文件。
"""
import networkx as nx
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


# Create directory if it doesn't exist
path = 'data/6-bus'
os.makedirs(path, exist_ok=True)

config_file = 'data/6-bus/network_config.csv'
config_df = pd.read_csv(config_file)

# Add edges，这里的edge是指不同节点之间的连线
edges = [(int(config_df['u'][i]), int(config_df['v'][i]), {'R': float(config_df['R'][i]), 'X': float(config_df['X'][i])})
         for i in range(len(config_df))]

# Create a directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

bfs = list(nx.bfs_tree(G, 0))
i_ = 0
for i in range(len(bfs)):
    v = bfs.pop()
    G.add_node(v, height=0)
    G.nodes[v]['height'] = 0 if G.out_degree(v) == 0 else max([G.nodes[u]['height'] for u in G.successors(v)]) + 1
    G.add_node(v, length=0)
    G.nodes[v]['length'] = i_
    i_ += 1

for u in G.nodes:
    print(f"节点{u}, 高度{G.nodes[u]['height']}")
    print(f"节点{u}, 长度{G.nodes[u]['length']}")

# Prepare data for JSON
data = nx.node_link_data(G)


# Save the graph to a JSON file
with open('data/6-bus/network_config.json', 'w') as f:
    json.dump(data, f)

# Load the graph from a JSON file
with open('data/6-bus/network_config.json', 'r') as f:
    data = json.load(f)

H = nx.node_link_graph(data)
# 显示graph
pos = nx.shell_layout(H, scale=100)
nx.draw(H, pos, with_labels=True)
plt.axis('on')
plt.xticks([])
plt.yticks([])
plt.show()
