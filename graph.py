import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
class Graph:
    def __init__(self, adjacency_matrix) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.n = len(adjacency_matrix)
    def display(self, edge_labels=None, node_labels=None, node_colors=None):
        G = nx.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        if edge_labels is not None:
            edge_labels = {(i, j): int(100*edge_labels[i][j])/100 for i in range(self.n) for j in range(self.n) if self.adjacency_matrix[i][j] != 0}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        if node_labels is not None:
            node_labels = {i: node_labels[i] for i in range(self.n)}
            nx.draw_networkx_labels(G, pos, labels=node_labels)
        if node_colors is not None:
            node_colors = [node_colors[i] for i in range(self.n)]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        plt.show()

    