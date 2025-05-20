import torch.nn as nn
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        input_dim=3,
        weight_norm=True,
        norm_layers=None,
        leaky_relu=None,
        dropout=None,
        dropout_prob=0.1,
        latent_in=None, # List of layers that will have the latent operation conducted
        use_tanh=True,
        operations=None # List of operations to be applied
    ):
        super(Decoder, self).__init__()
        self.ops = operations

        # Set default values for layer parameters
        self.dropout_prob = dropout_prob
        # Input dimensions
        self.input_dim = input_dim

        self.layers = nn.ModuleList()
        # Which and if an opperation is conducted before a layer
        self.doops = []

        opsused = 0
        in_dim = dims.pop(0)
        for i in range(len(dims)):
            out_dim = dims.pop(0)
            # Check if the current layer is an extra operation layer
            if i in latent_in:
                in_dim += self.input_dim
                self.doops.append(opsused)
                opsused += 1

            # Standard FC layer
            layer = nn.Linear(in_dim, out_dim)
            if i in norm_layers:
                # vAply normalization if needed
                layer = nn.utils.weight_norm(layer)
            self.layers.append(layer)
            self.doops.append(-1)
            
            if i in leaky_relu:
                # Add relu layer if needed
                self.layers.append(nn.LeakyReLU())
                self.doops.append(-1)

            if i in dropout:
                # Add dropout layer if needed
                self.layers.append(nn.Dropout(p=self.dropout_prob))    
                self.doops.append(-1)
            
            in_dim = out_dim
        
        if use_tanh:
            # Add final tanh layer if needed
            self.layers.append(nn.Tanh()) 
    
    # input: N x 3
    def forward(self, input):
        x = input
        # Pass the input through each layers
        for i, layer in enumerate(self.layers):
            # Apply the extra operation if it exists for this layer
            if self.doops[i] > -1:
                x = self.ops[self.doops[i]](x, input)
            x = layer(x)
        return x