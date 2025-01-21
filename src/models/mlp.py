from models.log_setup import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # Initialize weights with smaller values
            for linear in self.linears:
                nn.init.xavier_uniform_(linear.weight, gain=0.1)  # Use smaller gain
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)  # Initialize bias to zero
            
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                # Check weights before linear operation
                logger.debug("Linear layer %d weights:", layer)
                logger.debug("range=[%f, %f]",
                             self.linears[layer].weight.min(),
                             self.linears[layer].weight.max())
                logger.debug("has_nan=%s",
                             torch.isnan(self.linears[layer].weight).any())
                if self.linears[layer].bias is not None:
                    logger.debug("Linear layer %d bias:", layer)
                    logger.debug("range=[%f, %f]",
                                 self.linears[layer].bias.min(),
                                 self.linears[layer].bias.max())
                    logger.debug("has_nan=%s",
                                 torch.isnan(self.linears[layer].bias).any())
                
                # Break down the operations
                h_linear = self.linears[layer](h)
                h_bn = self.batch_norms[layer](h_linear)
                h = F.relu(h_bn)
                
            return self.linears[self.num_layers - 1](h)