import torch
import torch.nn as nn
from torch_geometric.nn import  GCNConv, GATConv
from ekan import KANLinear


def make_mlp(num_features, hidden_dim, out_dim, hidden_layers):
    if hidden_layers>=2:
        list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, out_dim, nn.ReLU())))
    else:
        list_hidden = [nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())]
    mlp = nn.Sequential(*list_hidden)
    return(mlp)

class KANLayer(KANLinear):
    def __init__(self, input_dim, output_dim):
        super(KANLayer, self).__init__(in_features=input_dim, out_features=output_dim)

class KAGCNConv(GCNConv):
    def __init__(self, in_feat:int,
                 out_feat:int):
        super(KAGCNConv, self).__init__(in_feat, out_feat)
        self.lin = KANLayer(in_feat, out_feat)

class KAGATConv(GATConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 heads:int):
        super(KAGATConv, self).__init__(in_feat, out_feat, heads)
        self.lin = KANLayer(in_feat, out_feat*heads)




class GNN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        torch.nn.BatchNorm1d(hidden_channels)
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(num_features, hidden_channels, heads))
                elif conv_type == "gin":
                    print("gin is here")
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads))
                elif conv_type == "gin":
                    print("gin is here")
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers)*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+(mp_layers)*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = torch.nn.Linear(dim_out_message_passing, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        if self.skip:
            l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            if self.skip:
                l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        return x

class GKAN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.softmax = nn.Softmax(dim=1) ##2025.2.14 val loss appear Nan, append softmax layer
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(num_features, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(KAGATConv(num_features, hidden_channels, heads))
                elif conv_type == "gin":
                    print("gin is here")
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(hidden_channels, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(KAGATConv(hidden_channels*heads, hidden_channels, heads))
                else:
                    print("gin is here")
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+mp_layers*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+mp_layers*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = KANLinear(dim_out_message_passing, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        x = self.softmax(x) #2025.2.14 val loss appear Nan, append softmax layer
        
        return x
