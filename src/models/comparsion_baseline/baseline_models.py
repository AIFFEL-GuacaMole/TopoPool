class BaselineGraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout,
                 device):
        super(BaselineGraphCNN, self).__init__()
        
        self.device = device
        self.num_layers = num_layers
        self.dropout = dropout

        # Simple layer structure without witness complex
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Simple global pooling
        self.pool = global_add_pool  # or global_mean_pool

    def forward(self, batch_graph):
        x, edge_index = batch_graph.x, batch_graph.edge_index
        batch = batch_graph.batch
        
        # Initial features
        h = x
        
        # Process through layers
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Final layer
        h = self.convs[-1](h, edge_index)
        
        # Global pooling
        out = self.pool(h, batch)
        return out


class TopKBaseline(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout, device, pool_ratio=0.5):
        super(TopKBaseline, self).__init__()
        
        self.device = device
        self.pool_ratio = pool_ratio
        
        # GNN layers with TopK pooling
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=pool_ratio)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=pool_ratio)
        
        # Final prediction layers
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, batch_graph):
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        
        # First block
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = global_mean_pool(x, batch)
        
        # Second block
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = global_mean_pool(x, batch)
        
        # Combine features from different levels
        x = torch.cat([x1, x2], dim=1)
        
        # Final prediction
        x = F.relu(self.lin1(x))
        x = self.batch_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x


class TraditionalGraphBaseline:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        
    def extract_graph_features(self, graph):
        # Calculate traditional graph metrics
        G = to_networkx(graph)  # Convert to NetworkX graph
        features = {
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'clustering_coef': nx.average_clustering(G),
            'density': nx.density(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else -1,
            'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else -1,
            'num_triangles': sum(nx.triangles(G).values()) / 3,
            'spectral_radius': max(abs(np.linalg.eigvals(nx.adjacency_matrix(G).todense())))
        }
        return np.array(list(features.values()))

    def fit(self, graphs, labels):
        # Extract features from all graphs
        features = np.array([self.extract_graph_features(g) for g in graphs])
        self.classifier.fit(features, labels)
        
    def predict(self, graphs):
        features = np.array([self.extract_graph_features(g) for g in graphs])
        return self.classifier.predict(features)