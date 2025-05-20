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
        leaky_relu_slope=0.2,
        dropout=None,
        dropout_prob=0.1,
        latent_in=None,
        use_tanh=True,
        operations=None
    ):
        super(Decoder, self).__init__()

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Leaky ReLU layers, Dropout layers and a tanh layer.\
        self.ops = operations

        self.dropout_prob = dropout_prob
        self.leaky_relu_slope = leaky_relu_slope
        self.input_dim = input_dim

        self.layers = nn.ModuleList()
        self.doops = []
        opsused = 0
        in_dim = dims.pop(0)
        for i in range(len(dims)):
            out_dim = dims.pop(0)
            # If this is the layer after which we concatenate, increase in_dim
            if i in latent_in:
                in_dim += self.input_dim
                self.doops.append(opsused)
                opsused += 1

            layer = nn.Linear(in_dim, out_dim)
            if i in norm_layers:
                layer = nn.utils.weight_norm(layer)
            self.layers.append(layer)
            self.doops.append(-1)
            
            if i in leaky_relu:
                self.layers.append(nn.LeakyReLU(negative_slope=self.leaky_relu_slope))
                self.doops.append(-1)

            if i in dropout:
                self.layers.append(nn.Dropout(p=self.dropout_prob))    
                self.doops.append(-1)
            
            in_dim = out_dim
        
        if use_tanh:
            self.layers.append(nn.Tanh()) 

        # self.fc.append(nn.Sequential(*self.layers))
        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        
        x = input
        for i, layer in enumerate(self.layers):
            if self.doops[i] > -1:
                x = self.ops[self.doops[i]](x, input)
            x = layer(x)
        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################
        return x