import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weight_variable_glorot(inputDim, outputDim):
    """Create a weight variable with Glorot & Bengio initialization"""

    init_range = np.sqrt(6.0 / (inputDim + outputDim))
    initial    = torch.FloatTensor(inputDim, outputDim).uniform_(-init_range,init_range)
    return initial

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape    = [num_nonzero_elems]
    random_tensor  = keep_prob
    random_tensor += torch.rand(noise_shape)
    mask           = torch.floor(random_tensor).type(torch.floor)
    rc             = x._indices()[:,mask]
    val            = x._values()[mask]*(1.0/keep_prob)
    return torch.sparse.FloatTensor(rc,val)


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, inputDim, outputDim, adjacenyMatrix, dropout=0, **kwargs):
        super(GraphConvolution, self).__init__()
        self.weights        = weight_variable_glorot(inputDim, outputDim)
        self.dropout        = dropout
        self.adjacenyMatrix = adjacenyMatrix

    def forward(self, inputs):
        x = inputs
        x = F.dropout(x,p=1-self.dropout, training=True)
        x = torch.matmul(x, self.weights)
        x = torch.sparse.mm(self.adjacenyMatrix, x)
        outputs = F.relu(x)
        return outputs


class GraphConvolutionSparse(nn.Module):
    """ Graph convolution layer for sparse inputs """
    def __init__ (self, inputDim, outputDim, adjacenyMatrix, featureMatrix, dropout=0, **kwargs):
        super(GraphConvolutionSparse, self).__init__()
        self.weights        = weight_variable_glorot(inputDim, outputDim)
        self.dropout        = dropout
        self.adjacenyMatrix = adjacenyMatrix
        self.isSparse       = True
        self.featureMatrix  = featureMatrix

    def forward(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.featureMatrix)
        x = torch.sparse.mm(x, self.weights)
        x = torch.sparse.mm(self.adjacenyMatrix, x)
        outputs = F.relu(x)
        return outputs


class InnerProductDecoder(nn.Module):
    """ Decoder model layer for link prediction. """
    def __init__(self, inputDim, dropout=0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, inputs):
        inputs = F.dropout(inputs, 1-self.dropout)
        x      = F.transpose(inputs, 0, 1) # Let's hope this works
        inputs = torch.matmul(inputs, x)
        inputs = torch.reshape(inputs, (-1,))
        outputs= F.sigmoid(inputs)
        return outputs
        