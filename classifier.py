import torch.nn as nn
import torch.nn.functional as F 
import torch
from fusion import FusionLayer

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, norm_type='layer', act=nn.LeakyReLU()):
        """
        Multi-Layer Perceptron (MLP)

        Args:
            input_dim (int): Size of input features
            hidden_dims (list of int): Sizes of hidden layers.
            output_dim (int): Size of output layer.
            act (nn.Module or None): Activation module.
        """
        super().__init__()

        self.act = act

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            
            if norm_type.lower() == "layer":
                layers.append(nn.LayerNorm(h_dim))
            elif norm_type.lower() == "batch":
                layers.append(nn.BatchNorm1d(h_dim))
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
            
            layers.append(self.act)
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
def get_normalization(norm_type, dim):
    """
    Helper function to select the normalization layer.
    """
    if norm_type == 'batch':
        return nn.BatchNorm1d(dim)
    elif norm_type == 'layer':
        return nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Supported: 'batch', 'layer'")

class Sparsemax(nn.Module):
    """
    Core of TabNet: Sparsemax activation function.
    It is similar to Softmax but can output true zeros, enabling sparse feature selection.
    """
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        dim = self.dim
        number_of_logits = input.size(dim)
        
        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range_values = torch.arange(start=1, end=number_of_logits + 1, device=input.device).float().view(1, -1)
        bound = 1 + range_values * zs
        
        cumsum_zs = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumsum_zs
        k = torch.max(is_gt * range_values, dim, keepdim=True)[0]
        
        zs_sparse = is_gt * zs
        tau = (torch.sum(zs_sparse, dim=dim, keepdim=True) - 1) / k
        tau = tau.expand_as(input)
        
        self.output = torch.max(torch.zeros_like(input), input - tau)
        return self.output

class GatedLinearUnitBlock(nn.Module):
    """
    Basic building block of TabNet: Linear -> Norm -> GLU.
    """
    def __init__(self, input_dim, output_dim, norm_type='batch'):
        super(GatedLinearUnitBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.norm = get_normalization(norm_type, output_dim * 2)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return self.glu(x)

class FeatureTransformer(nn.Module):
    """
    Feature Transformer: Consists of shared layers (across steps) and step-dependent layers.
    """
    def __init__(self, input_dim, output_dim, shared_layers, n_step_layers, norm_type='batch'):
        super(FeatureTransformer, self).__init__()
        self.scale = torch.sqrt(torch.tensor(0.5))
        
        # Shared layers (reused across decision steps)
        self.shared = nn.ModuleList()
        for i in range(shared_layers):
            in_d = input_dim if i == 0 else output_dim
            self.shared.append(GatedLinearUnitBlock(in_d, output_dim, norm_type))
            
        # Independent layers (specific to each decision step)
        self.step_dependent = nn.ModuleList()
        for i in range(n_step_layers):
            self.step_dependent.append(GatedLinearUnitBlock(output_dim, output_dim, norm_type))

    def forward(self, x):
        # Pass through shared layers
        for layer in self.shared:
            # Residual connection with scaling (except for the first layer)
            x = (x + layer(x)) * self.scale if layer != self.shared[0] else layer(x)
            
        # Pass through step-dependent layers
        for layer in self.step_dependent:
            x = (x + layer(x)) * self.scale
        return x

class AttentiveTransformer(nn.Module):
    """
    Attentive Transformer: Learns the mask for the current step.
    """
    def __init__(self, input_dim, output_dim, norm_type='batch'):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = get_normalization(norm_type, output_dim)
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.norm(x)
        x = x * priors # Apply prior knowledge (reduce attention on previously used features)
        mask = self.sparsemax(x)
        return mask

class TabNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        n_d=64,            # Dimension of the prediction layer (decision)
        n_a=64,            # Dimension of the attention layer
        n_steps=5,         # Number of decision steps
        gamma=1.5,         # Relaxation parameter for updating priors
        n_shared=2,        # Number of shared GLU layers
        n_independent=2,   # Number of independent GLU layers
        norm_type='batch'  # Normalization type: 'batch' or 'layer'
    ):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            norm_type (str): 'batch' for BatchNorm1d, 'layer' for LayerNorm.
        """
        super(TabNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.norm_type = norm_type
        
        # --- Core Components ---
        
        # 1. Initial Feature Transformer (Processes raw input)
        self.initial_splitter = FeatureTransformer(
            input_dim, n_d + n_a, n_shared, n_independent, norm_type
        )
        
        # 2. Feature Transformers (One for each step)
        self.feat_transformers = nn.ModuleList()
        # 3. Attentive Transformers (One for each step)
        self.att_transformers = nn.ModuleList()
        
        for _ in range(n_steps):
            self.feat_transformers.append(
                FeatureTransformer(input_dim, n_d + n_a, n_shared, n_independent, norm_type)
            )
            self.att_transformers.append(
                AttentiveTransformer(n_a, input_dim, norm_type)
            )
            
        # Final mapping layer
        self.final_mapping = nn.Linear(n_d, output_dim)
        # Note: The final BN is technically optional but often used. 
        # We switch it based on norm_type as well for consistency.
        self.final_norm = get_normalization(norm_type, output_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        
        # Initialize Priors to 1s (all features are initially available for selection)
        priors = torch.ones(x.shape).to(x.device)
        
        # Accumulator for step outputs
        out_accumulator = 0
        
        loss_masks = [] 
        
        # 0. Initial Feature Processing
        feat_x = self.initial_splitter(x) 
        
        # Split into decision (d) and attention (a) parts
        decision_x, att_x = feat_x[:, :self.n_d], feat_x[:, self.n_d:]
        
        # --- Decision Steps Loop ---
        for step in range(self.n_steps):
            # 1. Compute Mask (Attention)
            mask = self.att_transformers[step](priors, att_x)
            loss_masks.append(mask) 
            
            # 2. Update Priors (Decay based on gamma to enforce sparsity/diversity)
            priors = priors * (self.gamma - mask)
            
            # 3. Apply Mask to raw input
            masked_x = mask * x
            
            # 4. Feature Transformation
            feat_x = self.feat_transformers[step](masked_x)
            
            # 5. Split output
            decision_x, att_x = feat_x[:, :self.n_d], feat_x[:, self.n_d:]
            
            # 6. Accumulate decision output (ReLU activation)
            out_accumulator += F.relu(decision_x)
            
        # --- Final Output ---
        out = self.final_mapping(out_accumulator)
        
        return out, loss_masks

class Classifier(nn.Module):
    def __init__(self, emb_dim, mlp_hidden_dims, output_dim, norm_type='layer', act=nn.LeakyReLU(), 
                 mode='MLP', n_steps=1, gamma=1.5, pre_mlp_hidden_dim=256, pre_tabnet_hidden_dim=32):
        super().__init__()
        
        self.mode = mode
        if mode == 'MLP':
            self.mlp = MLP(input_dim=emb_dim, hidden_dims=mlp_hidden_dims, output_dim=output_dim, norm_type=norm_type, act=act)
        elif mode == 'TabNet':
            self.tabnet = TabNet(input_dim=emb_dim, output_dim=output_dim, n_d=mlp_hidden_dims[0], 
                                 n_a=mlp_hidden_dims[0], n_steps=n_steps, gamma=gamma, n_shared=2, 
                                 n_independent=2, norm_type=norm_type)
        elif mode == 'both':
            # use both dl feats and manual feats
            self.pre_mlp = MLP(input_dim=emb_dim, hidden_dims=mlp_hidden_dims, output_dim=pre_mlp_hidden_dim, norm_type=norm_type, act=act)
            self.pre_tabnet = TabNet(input_dim=128, output_dim=pre_tabnet_hidden_dim, n_d=pre_tabnet_hidden_dim, 
                                     n_a=pre_tabnet_hidden_dim, n_steps=n_steps, gamma=gamma, n_shared=2, 
                                     n_independent=2, norm_type='batch')
            self.fusion = FusionLayer(input_dim=pre_mlp_hidden_dim, 
                                      condition_dim=pre_tabnet_hidden_dim, method="gated")
            self.final_mlp = MLP(input_dim=pre_mlp_hidden_dim, hidden_dims=mlp_hidden_dims, output_dim=output_dim, norm_type='layer', act=act)
        
    def forward(self, x, manual_feat=None):
        if self.mode == 'MLP':
            pred = self.mlp(x)
            return pred
        elif self.mode == 'TabNet':
            pred, loss_masks = self.tabnet(x)
            return pred, loss_masks
        elif self.mode == 'both':
            dl_feat_emb = self.pre_mlp(x)
            manual_feat_emb, loss_masks = self.pre_tabnet(manual_feat)
            pred = self.final_mlp(self.fusion(dl_feat_emb, manual_feat_emb))
            return pred, loss_masks

class LogitMixer(nn.Module):
    def __init__(self, N):
        super().__init__()
        # Define two linear layers without bias
        self.linear1 = nn.Linear(N, N, bias=False)
        self.linear2 = nn.Linear(N, N, bias=False)
        
        # Define separate bias parameters
        self.bias = nn.Parameter(torch.zeros(N))
        
        # Initialize weights
        with torch.no_grad():
            # Identity for linear1
            self.linear1.weight.copy_(torch.eye(N))
            
            # Zero for linear2
            self.linear2.weight.zero_()
            
            

    def forward(self, x, y):
        out1 = self.linear1(x)
        
        out2 = self.linear2(y)
        
        logits = out1 + out2 + self.bias
        
        return logits
