import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected, softmax as segment_softmax
from torchvision.models import resnet50
from torch.nn import Parameter
import math
import numpy as np
import pickle
from .ehr_transformer import EHRTransformer

def gen_A(num_classes, t, adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int64)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class CGAFusion(nn.Module):
    def __init__(self, hidden_size, tau=2.0, use_proj=True):
        super().__init__()
        self.tau = tau
        self.use_proj = use_proj
        if use_proj:
            self.proj_k = nn.Linear(hidden_size, hidden_size)
            self.proj_q = nn.Linear(hidden_size, hidden_size)
            self.proj_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor, label_proto: torch.Tensor, token_mask=None):
        if self.use_proj:
            K_tok = F.normalize(self.proj_k(tokens), dim=-1)
            Q_lbl = F.normalize(self.proj_q(label_proto), dim=-1)
            V_tok = self.proj_v(tokens)
        else:
            K_tok = F.normalize(tokens, dim=-1)
            Q_lbl = F.normalize(label_proto, dim=-1)
            V_tok = tokens
        scores = torch.einsum('bmh,kh->bmk', K_tok, Q_lbl)
        if token_mask is not None:
            scores = scores.masked_fill(~token_mask.unsqueeze(-1), float('-inf'))
        attn = torch.softmax(scores / self.tau, dim=1)
        z = torch.einsum('bmk,bmh->bkh', attn, V_tok)
        return z

class TimeWeightedCxrToEhrConv(nn.Module):
    def __init__(self, hidden_size: int, tau: float = 0.5, mode: str = 'softmax', invert_time: bool = False):
        super().__init__()
        assert mode in {'softmax', 'exp_decay'}
        self.post_lin = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = tau
        self.mode = mode
        self.invert_time = invert_time

    @torch.no_grad()
    def _normalize_per_dst(self, w: torch.Tensor, dst: torch.Tensor, num_dst: int) -> torch.Tensor:
        denom = torch.zeros(num_dst, device=w.device, dtype=w.dtype)
        denom.index_add_(0, dst, w)
        denom = denom.clamp_min_(1e-12)
        return w / denom[dst]

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_index: torch.Tensor, edge_attr_time: torch.Tensor):
        if edge_index.numel() == 0:
            return torch.zeros_like(x_dst)
        src, dst = edge_index[0], edge_index[1]
        t = edge_attr_time.view(-1).to(x_src.dtype)
        if self.mode == 'softmax':
            scores = -t if self.invert_time else t
            alpha = segment_softmax(scores / (self.tau + 1e-8), dst, num_nodes=x_dst.size(0))
        else:
            w = torch.exp(t / (self.tau + 1e-8))
            alpha = self._normalize_per_dst(w, dst, num_dst=x_dst.size(0))
        msg = self.post_lin(x_src[src]) * alpha.unsqueeze(1)
        out = torch.zeros_like(x_dst)
        out.index_add_(0, dst, msg)
        return out

@torch.no_grad()
def build_hetero_graph_ehr_cxr(
    ehr_feats: torch.Tensor,
    cxr_feats: torch.Tensor,
    cxr_to_ehr_dst: torch.Tensor,
    cxr_times: torch.Tensor,
    sim_method: str = 'cosine',
    threshold: float = 0.6
):
    graph = HeteroData()
    device = ehr_feats.device
    graph['ehr'].x = ehr_feats
    graph['cxr'].x = cxr_feats
    if cxr_feats.numel() > 0:
        src = torch.arange(cxr_feats.size(0), device=device, dtype=torch.long)
        dst = cxr_to_ehr_dst.to(device).long()
        ei = torch.stack([src, dst], dim=0)
        graph['cxr', 'intra', 'ehr'].edge_index = ei
        graph['cxr', 'intra', 'ehr'].edge_attr = cxr_times.to(device).view(-1, 1)
    else:
        graph['cxr', 'intra', 'ehr'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        graph['cxr', 'intra', 'ehr'].edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
    if sim_method == 'cosine':
        norm_ehr = F.normalize(ehr_feats, p=2, dim=1)
        sim_matrix = torch.matmul(norm_ehr, norm_ehr.T)
    elif sim_method == 'euclidean':
        dist = torch.cdist(ehr_feats, ehr_feats, p=2)
        sim_matrix = 1.0 / (dist + 1e-5)
    else:
        raise ValueError(f"Unknown sim_method: {sim_method}")
    ee = (sim_matrix > threshold).nonzero(as_tuple=False).T
    ee = ee[:, ee[0] != ee[1]]
    graph['ehr', 'inter', 'ehr'].edge_index = to_undirected(ee)
    return graph

class MyGNN(nn.Module):
    def __init__(self, hidden_size, num_heads,
        time_weight_mode: str = 'softmax', time_tau: float = 0.5, invert_time: bool = False,
        cma_tau: float = 1.0, cma_use_proj: bool = True):
        super().__init__()
        self.gat_ehr_ehr = GATConv(
            hidden_size, hidden_size // num_heads, heads=num_heads, concat=True, add_self_loops=False
        )
        self.time_cxr_ehr = TimeWeightedCxrToEhrConv(hidden_size, tau=time_tau, mode=time_weight_mode, invert_time=invert_time)
        self.cma = CGAFusion(hidden_size, tau=cma_tau, use_proj=cma_use_proj)
        self._label_proto = None
        self.pre_tok_ln = nn.LayerNorm(hidden_size)
        self.post_z_ln = nn.LayerNorm(hidden_size)

    @torch.no_grad()
    def set_label_prototypes(self, label_proto: torch.Tensor):
        self._label_proto = label_proto

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        device = x_dict['ehr'].device
        ei_ehr = edge_index_dict.get(('ehr', 'inter', 'ehr'), torch.empty((2, 0), dtype=torch.long, device=device))
        msg_ehr_ehr = self.gat_ehr_ehr(x_dict['ehr'], ei_ehr)
        ei_cxr = edge_index_dict.get(('cxr', 'intra', 'ehr'), torch.empty((2, 0), dtype=torch.long, device=device))
        B = x_dict['ehr'].size(0); device = x_dict['ehr'].device
        has_cxr = torch.zeros(B, dtype=torch.bool, device=device)
        if ei_cxr is not None and ei_cxr.numel() > 0:
            has_cxr.scatter_(0, ei_cxr[1], True)
        token_mask = torch.stack([
            torch.ones(B, dtype=torch.bool, device=device),
            torch.ones(B, dtype=torch.bool, device=device),
            has_cxr
        ], dim=1)
        if edge_attr_dict is None:
            cxr_ehr_t = None
        else:
            cxr_ehr_t = edge_attr_dict.get(('cxr', 'intra', 'ehr'), None)
        if cxr_ehr_t is None:
            msg_cxr_ehr = torch.zeros_like(x_dict['ehr'])
        else:
            msg_cxr_ehr = self.time_cxr_ehr(x_dict['cxr'], x_dict['ehr'], ei_cxr, cxr_ehr_t)
        tokens = torch.stack([x_dict['ehr'], msg_ehr_ehr, msg_cxr_ehr], dim=1)
        tokens = self.pre_tok_ln(tokens)
        assert self._label_proto is not None, "label prototypes not set; call set_label_prototypes(x_[K,H]) before forward"
        z = self.cma(tokens, self._label_proto, token_mask=token_mask)
        z = self.post_z_ln(z)
        return {'z': z, 'msg_ehr_ehr': msg_ehr_ehr, 'msg_cxr_ehr': msg_cxr_ehr}

class HeteroGraphMultiModalModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head, t, adj_file):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.ehr_encoder = EHRTransformer(
            input_size=76,
            num_classes=num_classes,
            d_model=hidden_size,
            n_head=ehr_n_head,
            n_layers_feat=1,
            n_layers_shared=ehr_n_layers,
            n_layers_distinct=ehr_n_layers,
            dropout=ehr_dropout
        )
        base_resnet = resnet50(pretrained=True)
        self.cxr_encoder = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
            base_resnet.avgpool,
            nn.Flatten(),
            nn.Linear(2048, hidden_size)
        )
        self.time_weight_mode = 'softmax'
        self.time_tau = 1
        self.invert_time = False
        self.num_heads = 4
        self.gnn = MyGNN(hidden_size, self.num_heads,
                          time_weight_mode=self.time_weight_mode,
                          time_tau=self.time_tau,
                          invert_time=self.invert_time)
        self.gc1 = GraphConvolution(25, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, hidden_size)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.relu = nn.ReLU()
        self.update_linear = nn.Linear(hidden_size, hidden_size)
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            for _ in range(self.num_classes)
        ])
        nn.init.eye_(self.gnn.time_cxr_ehr.post_lin.weight)

    def set_time_weight(self, mode: str = 'softmax', tau: float = 0.5, invert_time: bool = False):
        assert mode in {'softmax', 'exp_decay'}
        self.time_weight_mode = mode
        self.time_tau = tau
        self.invert_time = invert_time
        self.gnn.time_cxr_ehr.mode = mode
        self.gnn.time_cxr_ehr.tau = tau
        self.gnn.time_cxr_ehr.invert_time = invert_time

    def forward(self, ehr_inputs, cxr_imgs_list, cxr_time_encodings_list, seq_lengths, pairs, inp):
        ehr_feats, _, _ = self.ehr_encoder(ehr_inputs, seq_lengths)
        device = ehr_feats.device
        B = ehr_feats.size(0)
        all_cxr_imgs = []
        cxr_to_ehr_dst = []
        cxr_times = []
        for i, (cxr_imgs, time_enc) in enumerate(zip(cxr_imgs_list, cxr_time_encodings_list)):
            if len(cxr_imgs) == 0:
                continue
            t_col = time_enc[:, 0] if time_enc.ndim > 1 else time_enc
            for j, img in enumerate(cxr_imgs):
                all_cxr_imgs.append(img)
                cxr_to_ehr_dst.append(i)
                cxr_times.append(t_col[j])
        if len(all_cxr_imgs) > 0:
            all_cxr_batch = torch.stack(all_cxr_imgs).to(device)
            all_cxr_feats = self.cxr_encoder(all_cxr_batch)
            cxr_to_ehr_dst = torch.tensor(cxr_to_ehr_dst, device=device, dtype=torch.long)
            cxr_times = torch.tensor(cxr_times, device=device, dtype=torch.float32)
        else:
            all_cxr_feats = torch.empty((0, self.hidden_size), device=device)
            cxr_to_ehr_dst = torch.empty((0,), dtype=torch.long, device=device)
            cxr_times = torch.empty((0,), dtype=torch.float32, device=device)
        graph = build_hetero_graph_ehr_cxr(
            ehr_feats, all_cxr_feats, cxr_to_ehr_dst, cxr_times,
            sim_method='cosine', threshold=0.6
        )
        adj = gen_adj(self.A).detach()
        inp = inp.to(dtype=torch.float32)
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        self.gnn.set_label_prototypes(x)
        edge_attr_dict = {
            ('cxr','intra','ehr'): graph[('cxr','intra','ehr')].edge_attr.view(-1)
        }
        out_dict = self.gnn(graph.x_dict, graph.edge_index_dict, edge_attr_dict)
        z = out_dict['z']
        logits = torch.cat([self.pred_heads[k](z[:, k, :]) for k in range(self.num_classes)], dim=1)
        pred = logits.sigmoid()
        return pred, out_dict
