"""
Module storing the classes having UML abstraction relationship type as
'realization' between `common.gnn.GraphCNN`, which is an abstract class.
"""

import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv
from models.common import gnn


class GCNconv(gnn.GraphCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def point_clouds_pooling(self, input_dim):
        # point clouds pooling for nodes
        self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2)
        # point clouds pooling for graphs
        self.score_graph_layer = GCNConv(input_dim, 2)


class GINconv(gnn.GraphCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def point_clouds_pooling(self, input_dim):
        # point clouds pooling for nodes
        node_nn = nn.Sequential(
            nn.Linear(input_dim, self.num_neighbors * 2),
            nn.ReLU(),
            nn.Linear(self.num_neighbors * 2, self.num_neighbors * 2)
        )
        self.score_node_layer = GINConv(node_nn, train_eps=True)
        
        # point clouds pooling for graphs
        graph_nn = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )
        self.score_graph_layer = GINConv(graph_nn, train_eps=True)
