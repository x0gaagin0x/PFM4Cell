import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ConfusionMatrixMeter:
    def __init__(self, label_to_one_hot: dict):
        """
        Args:
            label_to_one_hot: mapping {class_name: one_hot_vector (list or Tensor)}.
        """
        # build name→idx and idx→name once
        self.class_to_idx = {}
        for name, vec in label_to_one_hot.items():
            if isinstance(vec, torch.Tensor):
                idx = int(torch.argmax(vec).item())
            if isinstance(vec, np.ndarray):
                idx = int(np.argmax(vec).item())
            else:
                idx = vec.index(1)
            self.class_to_idx[name] = idx

        # invert and ensure ordering 0..C-1
        self.idx_to_class = [None] * len(label_to_one_hot)
        for name, idx in self.class_to_idx.items():
            self.idx_to_class[idx] = name

        self.C = len(self.idx_to_class)
        # initialize count matrix
        self.cm = np.zeros((self.C, self.C), dtype=int)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Accumulate counts from one batch.
        
        Args:
            y_true: (B, C) one-hot Tensor
            y_pred: (B, C) logits or prob Tensor
        """
        # to CPU numpy arrays of ints
        t_idx = torch.argmax(y_true, dim=1).cpu().numpy()
        p_idx = torch.argmax(y_pred, dim=1).cpu().numpy()

        # vectorized increment
        np.add.at(self.cm, (t_idx, p_idx), 1)

    def result(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame of shape (C, C) with index=Actual, columns=Predicted.
        """
        df = pd.DataFrame(
            self.cm,
            index=self.idx_to_class,
            columns=self.idx_to_class
        )
        df.index.name = 'Actual'
        df.columns.name = 'Predicted'
        return df

    def reset(self):
        """Zero out the accumulated counts."""
        self.cm[:] = 0

def macro_f1_from_confusion_matrix(
    cm: pd.DataFrame,
    eps: float = 1e-8
) -> float:
    """
    Compute the macro-averaged F1 score from a confusion matrix.

    Parameters:
        cm (pd.DataFrame): Confusion matrix with rows=actual classes,
                           columns=predicted classes, values=counts.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Macro-averaged F1 score.
    """
    # Extract the raw count matrix as a NumPy array
    mat = cm.values.astype(np.float64)
    C = mat.shape[0]
    f1_scores = []

    for i in range(C):
        TP = mat[i, i]
        FP = mat[:, i].sum() - TP
        FN = mat[i, :].sum() - TP

        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = 2 * (precision * recall) / (precision + recall + eps)

        f1_scores.append(f1)

    # macro F1 is the unweighted average of per-class F1s
    return float(np.mean(f1_scores))

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MultiTaskLossWrapper(nn.Module):
    """
    Generalized Multi-task loss wrapper using Uncertainty Weighting.
    Applicable for N tasks (N >= 2).
    
    Formula: Loss_total = sum( 0.5 * exp(-s_i) * loss_i + 0.5 * s_i )
    where s_i = log(sigma_i^2) are learnable parameters.
    """
    def __init__(self, num_tasks):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_tasks = num_tasks
        
        # Define 's' parameters for N tasks.
        # Initialized to 0.0 -> implies initial weight of 0.5 for all tasks.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses (list or tuple): A list containing raw loss tensors (scalars) for each task.
                                    e.g., [nb_loss, ce_loss, mse_loss, ...]
                                    Order must match the logical order of tasks.
        """
        # Validity check
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, but got {len(losses)}.")

        # Stack the individual scalar losses into a single tensor: [num_tasks]
        # This allows us to use vectorized operations instead of a loop.
        loss_tensor = torch.stack(losses)
        
        # --- Vectorized Implementation of the Formula ---
        # 1. torch.exp(-self.log_vars) computes the precision (weights) for all tasks at once.
        # 2. We multiply weights by the raw losses.
        # 3. We add the regularization term (log terms).
        weighted_losses = 0.5 * (torch.exp(-self.log_vars) * loss_tensor + self.log_vars)
        
        # Sum up to get the final back-propable loss
        total_loss = weighted_losses.sum()
        
        return total_loss

    def get_current_weights(self):
        """
        Helper method to retrieve current weights for monitoring/logging.
        Returns a list of float values representing 0.5 * exp(-s).
        """
        with torch.no_grad():
            # Calculate actual weights: w = 0.5 * exp(-s)
            weights = 0.5 * torch.exp(-self.log_vars)
        return weights.tolist()

def sparsity_loss(masks):
    sparsity_loss = 0
    for mask in masks:
        sparsity_loss += torch.mean(torch.sum(-mask * torch.log(mask + 1e-10), dim=1))

    return sparsity_loss