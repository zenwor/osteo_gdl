import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_covariance(x, eps=1e-1, max_rank=16):
    B, N, D = x.shape
    N_use = min(N, max_rank)
    x = x[:, :N_use, :]
    x = x - x.mean(dim=1, keepdim=True)
    cov = torch.matmul(x.transpose(1, 2), x) / (N_use - 1)
    cov = cov + eps * torch.eye(D, device=x.device).unsqueeze(0)
    return cov


def safe_eigh(x, eps=1e-3, max_eig=1e3):
    x_fp64 = x.double()
    eigvals, eigvecs = torch.linalg.eigh(x_fp64)
    eigvals = torch.clamp(eigvals, min=eps, max=max_eig)
    return (eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)).float()


class SPDRectifiedLinear(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return safe_eigh(x, eps=self.eps)


class SPDTangentProjection(nn.Module):
    def forward(self, x):
        x = safe_eigh(x, eps=1e-3)
        eigvals, eigvecs = torch.linalg.eigh(x.double())
        eigvals = torch.clamp(eigvals, min=1e-3, max=1e3)
        log_eig = torch.log(eigvals)
        x_log = (
            eigvecs @ torch.diag_embed(log_eig) @ eigvecs.transpose(-1, -2)
        ).float()
        return x_log


class SPDFlatten(nn.Module):
    def forward(self, x):
        B, D, _ = x.shape
        idx = torch.triu_indices(D, D)
        return x[:, idx[0], idx[1]]


class SPDLayer(nn.Module):
    def __init__(self, dim_in, dim_out, eps=1e-3, fast_exp=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.relu = SPDRectifiedLinear(eps=eps)
        self.dim_out = dim_out
        self.eps = eps
        self.fast_exp = fast_exp

    def forward(self, X):
        X = X + self.eps * torch.eye(X.shape[1], device=X.device).unsqueeze(0)

        X_tan = SPDTangentProjection()(X)
        Y_tan = self.linear(X_tan)

        # Optional reduce dimension
        if self.dim_out < X.shape[1]:
            Y_tan = Y_tan[:, : self.dim_out, : self.dim_out]

        # Exponential map
        if self.fast_exp:
            # 1st order approx
            Y_hyp = torch.eye(Y_tan.shape[1], device=Y_tan.device).unsqueeze(0) + Y_tan
        else:
            Y_hyp = torch.matrix_exp(Y_tan)

        # ReEig to ensure SPD
        Y_hyp = self.relu(Y_hyp)
        return Y_hyp
