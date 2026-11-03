import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 condition_dim, 
                 method="bilinear"):
        """
        Fusion layer supporting FiLM, Bilinear Fusion, and Gated Fusion.

        Args:
            input_dim (int): Dimension of the main input vector.
            condition_dim (int): Dimension of the conditioning vector.
            method (str): One of ["film", "bilinear", "gated"].
        """
        super().__init__()
        self.method = method.lower()
        self.input_dim = input_dim

        if self.method == "film":
            # Generate gamma and beta from condition
            self.mlp = nn.Sequential(
                nn.Linear(condition_dim, 2 * input_dim),
                nn.ReLU(),
                nn.Linear(2 * input_dim, 2 * input_dim)
            )

        elif self.method == "bilinear":
            # Low-rank bilinear pooling: x^T U V^T y
            self.U = nn.Linear(input_dim, input_dim, bias=False)
            self.V = nn.Linear(condition_dim, input_dim, bias=False)

        elif self.method == "gated":
            # Project condition into input space
            self.cond_proj = nn.Linear(condition_dim, input_dim)
            # Gate from concatenated [x, cond_proj]
            self.gate = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, x, cond):
        """
        Args:
            x (Tensor): Input vector of shape (batch, input_dim).
            cond (Tensor): Conditioning vector of shape (batch, condition_dim).
        Returns:
            fused (Tensor): Fused representation of shape (batch, input_dim) or (batch, rank).
        """
        if self.method == "film":
            cond_expanded = cond
            # FiLM: gamma * x + beta
            params = self.mlp(cond_expanded)  # (batch, 2*input_dim)
            gamma, beta = params.chunk(2, dim=-1)
            fused = gamma * x + beta

        elif self.method == "bilinear":
            cond_expanded = cond
            # Low-rank bilinear pooling
            u = self.U(x)            # (batch, rank)
            v = self.V(cond_expanded)         # (batch, rank)
            fused = u * v            # elementwise interaction

        elif self.method == "gated":
            cond_expanded = cond
            # Gated fusion
            cond_proj = self.cond_proj(cond_expanded)  # (batch, input_dim)
            g = self.gate(torch.cat([x, cond_proj], dim=-1))  # (batch, input_dim)
            fused = g * x + (1 - g) * cond_proj

        return fused