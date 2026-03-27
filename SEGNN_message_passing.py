import torch
import torch.nn as nn
from torch.nn.functional import silu
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from e3nn.o3 import (FullyConnectedTensorProduct,
                     FullTensorProduct,
                     Irreps,
                     spherical_harmonics,
                     rand_matrix)
from e3nn.nn import Gate


class TensorProduct(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, l_f, irreps_out=None,
                 additional_scalars=True):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2

        self.additional_scalars = additional_scalars

        if irreps_out is not None:
            irreps_out = irreps_out
        else:
            irreps_out = self._calculate_irreps_out(self.irreps_in1,
                                                    self.irreps_in2,
                                                    l_f)
        if self.additional_scalars:
            add_scalar = (irreps_out.num_irreps
                          - irreps_out.filter(keep=["0e"]).num_irreps)
        else:
            add_scalar = 0
        irreps_out = irreps_out + Irreps(f"{add_scalar}x0e")
        self.irreps_out = irreps_out.regroup()

        self.tp = FullyConnectedTensorProduct(self.irreps_in1,
                                              self.irreps_in2,
                                              self.irreps_out,
                                              shared_weights=True)
        self.ps_scalars = self.irreps_out.filter(keep=["0o"]).num_irreps
        self.scalars = self.irreps_out.filter(keep=["0e"]).num_irreps
        self.bias = nn.Parameter(torch.empty(self.scalars))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, input_1, input_2):
        out = self.tp(input_1, input_2)
        bias = torch.cat([
            torch.zeros(self.ps_scalars),
            self.bias,
            torch.zeros(out.size(-1) - self.ps_scalars - self.scalars)
        ], dim=-1)
        out += bias
        return out

    def _calculate_irreps_out(self, irreps_in1, irreps_in2, l_f):
        tp = FullTensorProduct(irreps_in1, irreps_in2)
        if l_f is None:
            return tp.irreps_out
        else:
            return tp.irreps_out.filter(lmax=l_f)


class TensorProductSwishGate(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, l_f, irreps_out=None):
        super().__init__()
        self.tp = TensorProduct(irreps_in1, irreps_in2, l_f,
                                irreps_out=irreps_out)
        self.ps_scalars = self.tp.ps_scalars
        self.gates = self.tp.irreps_out.num_irreps - self.tp.scalars
        self.scalars = self.tp.scalars - self.gates
        self.irreps_gated = self.tp.irreps_out.filter(drop=["0e"])
        self.gate = Gate(
            f"{self.scalars}x0e", [silu],
            f"{self.gates}x0e", [silu],
            self.irreps_gated
        )
        self.irreps_out = self.gate.irreps_out
    
    def forward(self, input_1, input_2):
        out = self.tp(input_1, input_2)
        out = self.gate(out)
        return out


class NodeAttributes(MessagePassing):
    def __init__(self, l_a):
        super().__init__(aggr='mean')
        self.l_a = l_a
    
    def forward(self, pos, gp_features, edge_index):
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]
        rel_pos_embed = spherical_harmonics(self.l_a,
                                            rel_pos,
                                            True)
        gp_features_embed = torch.sum(spherical_harmonics(self.l_a,
                                                          gp_features,
                                                          True), dim=0)
        out = self.propagate(edge_index, rel_pos_embed=rel_pos_embed)
        out = out + gp_features_embed
        return out, rel_pos_embed, rel_pos

    def message(self, rel_pos_embed):
        return rel_pos_embed


class SELayer(MessagePassing):
    def __init__(self, feature_irreps, l_f, l_a, aggr='sum'):
        super().__init__(aggr=aggr)
        self.l_f = l_f
        self.l_a = l_a
        self.node_attr = NodeAttributes(l_a)

        irreps_rel_pos = Irreps(f"{self.l_a}y")
        irreps_h = feature_irreps + feature_irreps + Irreps("1x0e")

        self.message_tp1 = TensorProductSwishGate(irreps_h,
                                                  irreps_rel_pos,
                                                  self.l_f)
        self.message_tp2 = TensorProductSwishGate(
            self.message_tp1.irreps_out,
            irreps_rel_pos,
            self.l_f
        )
        self.update_tp1 = TensorProductSwishGate(
            feature_irreps + self.message_tp2.irreps_out,
            irreps_rel_pos,
            self.l_f
        )
        self.update_tp2 = TensorProduct(
            self.update_tp1.irreps_out,
            irreps_rel_pos,
            None,
            irreps_out=feature_irreps,
            additional_scalars=False
        )

    def forward(self, s_features, pos, gp_features, edge_index):
        a, rel_pos_embed, rel_pos = self.node_attr(pos,
                                                   gp_features,
                                                   edge_index)
        h = torch.cat([s_features[edge_index[1]],
                       s_features[edge_index[0]],
                       torch.sum(rel_pos ** 2, dim=-1)[..., None]],
                       dim=-1)
        out = self.propagate(edge_index, h=h, rel_pos_embed=rel_pos_embed,
                             s_features=s_features, a=a)
        return out

    def message(self, h, rel_pos_embed):
        m = self.message_tp1(h, rel_pos_embed)
        m = self.message_tp2(m, rel_pos_embed)
        return m

    def update(self, inputs, s_features, a):
        out = self.update_tp1(torch.cat([s_features, inputs], dim=-1), a)
        out = s_features + self.update_tp2(out, a)
        return out

def main():  # test for equivariance
    with torch.no_grad():
        feature_irreps = Irreps("1x0o + 1x0e + 1x1o")
        s_features = feature_irreps.randn(4, -1)
        pos = torch.randn(4, 3)
        gp_features = torch.randn(2, 4, 3)
        edge_index = torch.tensor([[0, 0, 0, 1, 2, 3],
                                   [1, 2, 3, 0, 0, 0]])
        R = rand_matrix(1).squeeze()
        layer = SELayer(feature_irreps, l_f=1, l_a=1)
        D_R = feature_irreps.D_from_matrix(R)
        out_1 = torch.matmul(layer(s_features, pos, gp_features,
                                   edge_index),
                             D_R.t())
        out_2 = layer(torch.matmul(s_features, D_R.t()),
                      torch.matmul(pos, R.t()),
                      torch.matmul(gp_features, R.t()), edge_index)

    assert torch.allclose(out_1, out_2)  # assert equivariance

if __name__ == '__main__':
    main()
