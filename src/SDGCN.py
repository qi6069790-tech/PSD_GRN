import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.sparse.mm(mul_L_real, X_real)
    real = torch.matmul(data, weight)

    data = -1.0 * torch.sparse.mm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight)

    data = torch.sparse.mm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)

    data = torch.sparse.mm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)

    return real, imag


class SDConv(nn.Module):
    def __init__(self, in_c, out_c, K, L_norm_real, L_norm_imag, bias=True):
        super().__init__()
        self.mul_L_real = L_norm_real
        self.mul_L_imag = L_norm_imag

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))
        stdv = 1.0 / math.sqrt(out_c)
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_c))
        else:
            self.register_parameter("bias", None)

    def forward(self, X_real, X_imag):
        real_list = []
        imag_list = []

        for i in range(len(self.mul_L_real)):
            real_i, imag_i = process(
                self.mul_L_real[i],
                self.mul_L_imag[i],
                self.weight[i],
                X_real,
                X_imag
            )
            real_list.append(real_i)
            imag_list.append(imag_i)

        real = torch.stack(real_list, dim=0).sum(dim=0)
        imag = torch.stack(imag_list, dim=0).sum(dim=0)

        if self.bias is not None:
            real = real + self.bias
            imag = imag + self.bias

        return real, imag


class complex_relu_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, imag):
        mask = (real >= 0).float()
        return mask * real, mask * imag


class SDGCN_link_prediction(nn.Module):
    def __init__(
        self,
        num_features,
        hidden,
        K,
        L_norm_real,
        L_norm_imag,
        label_dim=5,
        layer=2,
        dropout=0.5
    ):
        super().__init__()

        chebs = []
        chebs.append(SDConv(num_features, hidden, K, L_norm_real, L_norm_imag))
        chebs.append(complex_relu_layer())

        for _ in range(1, layer):
            chebs.append(SDConv(hidden, hidden, K, L_norm_real, L_norm_imag))
            chebs.append(complex_relu_layer())

        self.chebs = nn.ModuleList(chebs)
        self.dropout = dropout

        # 两层 MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden*4 + 60, label_dim)
        )

    def encode(self, X_real, X_imag):
        real, imag = X_real, X_imag
        for layer in self.chebs:
            real, imag = layer(real, imag)
        return real, imag

    def forward(self, X_real, X_imag, query_edges, one_graph_features):
    # 统一为 [E, 2]
        if query_edges.dim() != 2:
            raise ValueError(f"query_edges 维度错误: {query_edges.shape}")

        if query_edges.size(0) == 2 and query_edges.size(1) != 2:
            query_edges = query_edges.t().contiguous()
        elif query_edges.size(1) != 2:
            raise ValueError(f"query_edges 形状应为 [E,2] 或 [2,E]，实际为 {query_edges.shape}")

        real, imag = self.encode(X_real, X_imag)

    # 统一设备
        device = real.device
        query_edges = query_edges.to(device)
        one_graph_features = one_graph_features.to(device).float()

        src = query_edges[:, 0].long()
        dst = query_edges[:, 1].long()

        x = torch.cat(
            [
             real[src],
                real[dst],
                imag[src],
                imag[dst],
                one_graph_features[src],
                one_graph_features[dst]
            ],
            dim=-1
        )

        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x