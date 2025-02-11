import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.autograd import Variable
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_adj
import pdb

from models.common.log_setup import logger
from models.common.mlp import MLP
from TDA.tda import *


class GraphCNN(nn.Module, ABC):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 final_dropout,
                 learn_eps,
                 graph_pooling_type,
                 neighbor_pooling_type,
                 device,
                 num_neighbors,
                 num_landmarks,
                 first_pool_ratio,
                 PI_resolution_sq: int,
                 PI_hidden_dim: int):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.num_neighbors = int(num_neighbors)
        self.num_landmarks = int(num_landmarks)
        self.first_pool_ratio: float | str = first_pool_ratio

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # for attentional second-order pooling
        self.PI_dim = PI_resolution_sq ** 2
        self.PI_hidden_dim = PI_hidden_dim
        self.total_latent_dim = input_dim + hidden_dim * (num_layers - 1) + self.PI_hidden_dim
        self.dense_dim = self.total_latent_dim
        self.attend = nn.Linear(self.total_latent_dim - self.PI_hidden_dim, 1)
        self.linear1 = nn.Linear(self.dense_dim, output_dim)
        self.mlp_PI_witnesses = nn.Linear(self.PI_dim, self.PI_hidden_dim)

        # point clouds pooling for nodes and graphs
        self.point_clouds_pooling(input_dim)

    @abstractmethod
    def point_clouds_pooling(self) -> None:
        pass
    
    # Keeping all preprocessing methods unchanged as they don't depend on the convolution type
    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max(graph.max_neighbor for graph in batch_graph)

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool_witnesses(self, batch_graph):
        """
        Process graph batch with witness complex computation and pooling operations.
        Creates block diagonal sparse matrix and handles witness complex computations.
        """
        edge_attr = None  # Initialize edge attributes as None
        edge_mat_list = []
        start_idx = [0]
        pooled_x = []
        pooled_graph_sizes = []
        PI_witnesses_dgms = []
        
        for i, graph in enumerate(batch_graph):
            try:
                # Initial feature processing
                x = graph.node_features.to(self.device)
                edge_index = graph.edge_mat.to(self.device)
                num_nodes = x.size(0)

                # Compute node embeddings and point clouds using GINConv
                witnesses = self.score_graph_layer(x, edge_index).to(self.device)
                node_embeddings = self.score_node_layer(x, edge_index).to(self.device)
                node_point_clouds = node_embeddings.view(-1, self.num_neighbors, 2).to(self.device)
                
                # Calculate score lifespan for initial pooling
                score_lifespan = torch.FloatTensor([
                    sum_diag_from_point_cloud(node_point_clouds[j,...]) 
                    for j in range(node_point_clouds.size(0))
                ]).to(self.device)

                # First pooling operation
                batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
                # perm = topk(score_lifespan, self.first_pool_ratio, batch)
                perm = topk(score_lifespan, num_nodes, batch)
                x = x[perm]
                edge_index, _ = filter_adj(edge_index, edge_attr, perm, num_nodes=num_nodes)

                # Witness complex computation
                network_statistics = torch.sum(to_dense_adj(graph.edge_mat)[0,:,:], dim=1).to(self.device)
                batch_for_witnesses = torch.zeros(network_statistics.size(0), dtype=torch.long, device=self.device)
                    # ↳ batch tensor that matches network_statistics size 
                
                # Select landmarks with size validation
                num_landmarks = min(self.num_landmarks, network_statistics.size(0))
                    # ↳ Make sure not to request more landmarks than node quantity
                if num_landmarks < 2:
                    print(f"Warning: Graph {i} has insufficient nodes for witness complex ({num_landmarks})")
                    
                witnesses_perm = topk(network_statistics, num_landmarks, batch_for_witnesses)
                landmarks = witnesses[witnesses_perm]

                # Create witness complex and compute persistence
                try:
                    witness_complex = gd.EuclideanStrongWitnessComplex(
                        witnesses=witnesses.cpu().detach().numpy(),
                        landmarks=landmarks.cpu().detach().numpy()
                    )
                    simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=1, limit_dimension=1)
                    simplex_tree.compute_persistence(min_persistence=-1.)
                    witnesses_dgm = simplex_tree.persistence_intervals_in_dimension(0)[:-1,:]
                    
                    # Check if we got valid intervals
                    # **Filter out intervals with infinite death times**
                    if len(witnesses_dgm) > 0 and np.isfinite(witnesses_dgm).all():
                        # Compute persistence image
                        PI_witnesses_dgm = torch.FloatTensor(
                            persistence_images(
                                dgm=witnesses_dgm,
                                resolution=[None,  # make sure to be non-negative
                                            None],
                                # resolution=[int(np.sqrt(self.PI_resolution_sq)),  # make sure to be non-negative
                                #             int(np.sqrt(self.PI_resolution_sq))],
                                normalization=False
                            ).reshape(1,-1)
                        ).to(self.device)
                    else:
                        logger.debug(f"Warning: Invalid persistence diagram for graph {i}")
                        PI_witnesses_dgm = torch.zeros((1, self.PI_dim), device=self.device)
                    
                except Exception as e:
                    logger.debug(f"Warning: Error in witness complex computation for graph {i}: {e}")
                    PI_witnesses_dgm = torch.zeros((1, self.PI_dim), device=self.device)
                
                PI_witnesses_dgms.append(PI_witnesses_dgm)

                # Update graph information
                start_idx.append(start_idx[i] + x.size(0))
                edge_mat_list.append(edge_index + start_idx[i])
                pooled_x.append(x)
                pooled_graph_sizes.append(x.size(0))

            except Exception as e:
                print(f"Error processing graph {i}: {e}")
                # Add minimal valid data
                PI_witnesses_dgms.append(torch.zeros((1, self.PI_dim), device=self.device))
                pooled_x.append(x[:1])
                pooled_graph_sizes.append(1)
                continue

        # Create adjacency block matrix
        try:
            pooled_X_concat = torch.cat(pooled_x, 0).to(self.device)
            Adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device)
            Adj_block_elem = torch.ones(Adj_block_idx.shape[1]).to(self.device)

            if not self.learn_eps:
                num_node = start_idx[-1]
                self_loop_edge = torch.LongTensor([range(num_node), range(num_node)]).to(self.device)
                elem = torch.ones(num_node).to(self.device)
                Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1).to(self.device)
                Adj_block_elem = torch.cat([Adj_block_elem, elem], 0).to(self.device)

            Adj_block = torch.sparse.FloatTensor(
                Adj_block_idx, 
                Adj_block_elem, 
                torch.Size([start_idx[-1], start_idx[-1]])
            )

        except Exception as e:
            print(f"Error creating adjacency block matrix: {e}")
            raise

        return Adj_block.to(self.device), pooled_X_concat, PI_witnesses_dgms, pooled_graph_sizes

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        #compute the padded neighbor list
        start_idx.extend(
            start_idx[i] + len(graph.g) for i, graph in enumerate(batch_graph)
        )
        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))

            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1])])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        return self._output_of_layer(layer, pooled)


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        return self._output_of_layer(layer, pooled)
    
    
    def _output_of_layer(self, layer, pooled):
        #    ↱ representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)  # non-linearity
        return h


    def forward(self, batch_graph):
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block, pooled_X_concat, PI_witnesses_dgms, pooled_graph_sizes = self.__preprocess_neighbors_sumavepool_witnesses(batch_graph)

        # List of hidden representation at each layer (including input)
        hidden_rep = [pooled_X_concat]
        h = pooled_X_concat

        # Process through layers with gradient clipping
        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif self.neighbor_pooling_type != "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max":
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            else:
                # operation
                h = self.next_layer(h, layer, Adj_block = Adj_block)
            hidden_rep.append(h)

        hidden_rep = torch.cat(hidden_rep, 1)

        # Process graphs
        graph_sizes = pooled_graph_sizes
        batch_size = len(graph_sizes)
        
        # Initialize tensors with zeros
        batch_graphs = torch.zeros(batch_size, self.dense_dim - self.PI_hidden_dim, device=self.device)
        batch_graphs_out = torch.zeros(batch_size, self.dense_dim, device=self.device)

        # Process each graph
        node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)

        for g_i in range(len(graph_sizes)):
            cur_node_embeddings = node_embeddings[g_i]
            
            # Attention
            attn_coef = self.attend(cur_node_embeddings)
            attn_weights = torch.transpose(attn_coef, 0, 1)
            cur_graph_embeddings = torch.matmul(attn_weights, cur_node_embeddings)
            batch_graphs[g_i] = cur_graph_embeddings.view(self.dense_dim - self.PI_hidden_dim)
            witnesses_PI_out = (self.mlp_PI_witnesses(PI_witnesses_dgms[g_i])).view(-1)
            batch_graphs_out[g_i] = torch.cat([batch_graphs[g_i], witnesses_PI_out], dim=0)

        score = F.dropout(self.linear1(batch_graphs_out), self.final_dropout, training=self.training)
        score = torch.clamp(score, min=-88.0, max=88.0)  # Add clamping
        return score
