from ekan import KAN as eKAN
from fastkan import FastKAN
import torch

import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

def make_mlp(num_features, hidden_dim, out_dim, hidden_layers, batch_norm=True):
    if hidden_layers>=2:
        if batch_norm:
            list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim))]
        else:
            list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            if batch_norm:
                list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))
            else:
                list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, out_dim, nn.ReLU())))
    else:
        list_hidden = [nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())]
    MLP = nn.Sequential(*list_hidden)
    return(MLP)

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-2) + [out_dim]
    return(eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order))

def make_fastkan(num_features, hidden_dim, out_dim, hidden_layers, grid_size):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-2) + [out_dim]
    return(FastKAN(layers_hidden=sizes, num_grids=grid_size))

class GCKANLayer(torch.nn.Module):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3):
        super(GCKANLayer, self).__init__()
        self.kan = eKAN([in_feat, out_feat], grid_size=grid_size, spline_order=spline_order)

    def forward(self, X, A_hat_normalized):
        return self.kan(A_hat_normalized @ X)

class GIKANLayer(GINConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3,
                 hidden_dim:int=16,
                 nb_layers:int=2):
        kan = make_kan(in_feat, hidden_dim, out_feat, nb_layers, grid_size, spline_order)
        GINConv.__init__(self, kan)

class GCFASTKANLayer(torch.nn.Module):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4):
        super(GCFASTKANLayer, self).__init__()
        self.kan = FastKAN([in_feat, out_feat], num_grids=grid_size)

    def forward(self, X, A_hat_normalized):
        return self.kan(A_hat_normalized @ X)

class GIFASTKANLayer(GINConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 hidden_dim:int=16,
                 nb_layers:int=2):
        kan = make_fastkan(in_feat, hidden_dim, out_feat, nb_layers, grid_size)
        GINConv.__init__(self, kan)

class GNN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 hidden_layers:int=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(mp_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                else:
                    self.convs.append(GINConv(make_mlp(num_features, hidden_channels, hidden_channels, hidden_layers, False)))
            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels, hidden_channels, hidden_layers, False)))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers-1)*hidden_channels if skip else hidden_channels
        if conv_type == "gcn":
            self.conv_out = GCNConv(dim_out_message_passing, num_classes)
        else:
            self.conv_out = GINConv(make_mlp(dim_out_message_passing, hidden_channels, num_classes, hidden_channels, False))

    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.relu(x)
        return x

class GKAN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4,
                 spline_order:int = 3,
                 hidden_layers:int=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(mp_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCKANLayer(num_features, hidden_channels, grid_size, spline_order))
                else:
                    self.convs.append(GIKANLayer(make_kan(num_features, hidden_channels, hidden_channels, hidden_layers, grid_size, spline_order)))
            else:
                if conv_type == "gcn":
                    self.convs.append(GCKANLayer(hidden_channels, hidden_channels, grid_size, spline_order))
                else:
                    self.convs.append(GIKANLayer(make_kan(hidden_channels, hidden_channels, hidden_channels, hidden_layers, grid_size, spline_order)))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers-1)*hidden_channels if skip else hidden_channels
        if conv_type == "gcn":
            self.conv_out = GCKANLayer(dim_out_message_passing, num_classes, grid_size, spline_order)
        else:
            self.conv_out = GINConv(make_kan(dim_out_message_passing, hidden_channels, num_classes, hidden_channels, grid_size, spline_order))

    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.relu(x)
        return x

class GFASTKAN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4,
                 spline_order:int = 3,
                 hidden_layers:int=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(mp_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCFASTKANLayer(num_features, hidden_channels, grid_size+1))
                else:
                    self.convs.append(GIFASTKANLayer(make_fastkan(num_features, hidden_channels, hidden_channels, hidden_layers, grid_size+1, spline_order)))
            else:
                if conv_type == "gcn":
                    self.convs.append(GCFASTKANLayer(hidden_channels, hidden_channels, grid_size+1))
                else:
                    self.convs.append(GIFASTKANLayer(make_fastkan(hidden_channels, hidden_channels, hidden_channels, hidden_layers, grid_size+1)))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers-1)*hidden_channels if skip else hidden_channels
        if conv_type == "gcn":
            self.conv_out = GCFASTKANLayer(dim_out_message_passing, num_classes, grid_size+1)
        else:
            self.conv_out = GINConv(make_fastkan(dim_out_message_passing, hidden_channels, num_classes, hidden_channels, grid_size+1))

    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.relu(x)
        return x