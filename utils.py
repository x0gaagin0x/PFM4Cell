import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import torch.nn.functional as F
from loss import FocalLoss, sparsity_loss
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, accuracy_score
from scipy.stats import entropy
# Set all relevant random seeds to ensure reproducibility across runs
def set_seed(seed: int = 2434):
    # Python built-in RNG
    random.seed(seed)
    
    # NumPy RNG
    np.random.seed(seed)
    
    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For reproducibility in dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

# Ensure each DataLoader worker has a reproducible but unique random seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def check_equal_str(a: str, b: str):
    if a != b:
        raise AssertionError(f"str not equal: '{a}' != '{b}'")

# normalize manual features
def normalize_manual_features(data_list):
    
    all_x = torch.cat([data.x for data in data_list], dim=0)

    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0)

    std[std == 0] = 1

    for data in data_list:
        data.x = (data.x - mean) / std

    return data_list, mean, std

# training
def train_one_epoch(model, loader, optimizer, scheduler, accelerator, loss_type, num_classes, class_weights=None, 
                    graph_type='patch_subgraph', mode='MLP', sparsity_loss_weight=0):
    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer.zero_grad()
    if class_weights:
        class_weights = torch.tensor(class_weights).to(accelerator.device)
    if loss_type == "focal_loss":
        focal_loss = FocalLoss(gamma=2, alpha=class_weights, reduction='mean', task_type='multi-class', num_classes=num_classes)
    
    for batch in tqdm(loader):
        with accelerator.accumulate(model):
            
            # forward
            data, tile, slide_id = batch
            if mode == 'MLP':
                pred = model(tile, data)
            elif mode == 'TabNet' or mode == 'both':
                pred, loss_masks = model(tile, data)
            elif mode == 'logitmixer':
                pred = model(data.strong_logits, data.weak_logits)

            if graph_type == 'patch_subgraph':
                # compute raw loss (mean over seeds)
                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(
                        input=pred,
                        target=data.y_class,
                        weight=class_weights,
                        reduction='mean'
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                elif loss_type == "focal_loss":
                    loss = focal_loss(
                        inputs=pred,
                        targets=data.y_class
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                else:
                    raise ValueError(f"No supported loss_type: {loss_type}")
                
                # scale loss for gradient accumulation
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # accumulate metrics on raw loss
                num_seeds = data.y_class.shape[0]
                total_loss += loss.item() * num_seeds
                total_examples += num_seeds
            
            elif graph_type == 'k_hop_subgraph':
                # extract central node's embeddings and prediction results
                global_mapping = data.ptr[:-1] + data.mapping
                pred = pred[global_mapping]
                
                # Compute loss
                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(
                        input=pred,
                        target=data.y_class[global_mapping],
                        weight=class_weights,
                        reduction='mean'
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                elif loss_type == "focal_loss":
                    loss = focal_loss(
                        inputs=pred,
                        targets=data.y_class[global_mapping]
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                else:
                    raise ValueError(f"No supported loss_type: {loss_type}")
                    
                # scale loss for gradient accumulation
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # accumulate metrics on raw loss
                num_seeds = data.y_class[global_mapping].shape[0]
                total_loss += loss.item() * num_seeds
                total_examples += num_seeds
                
            
    avg_loss = total_loss / total_examples
    
    # Collect loss of all processes and compute mean
    avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
    avg_loss_all = accelerator.gather(avg_loss_tensor).mean().item()

    return avg_loss_all

# validation
def evaluate(model, loader, accelerator, loss_type, num_classes, class_weights=None, 
             graph_type='patch_subgraph', mode='MLP', sparsity_loss_weight=0):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_list = []
    labels_list = []
    if class_weights:
        class_weights = torch.tensor(class_weights).to(accelerator.device)
    if loss_type == "focal_loss":
        focal_loss = FocalLoss(gamma=2, alpha=class_weights, reduction='mean', task_type='multi-class', num_classes=num_classes)
    
    with torch.no_grad():
        for batch in tqdm(loader):
            
            # forward
            data, tile, slide_id = batch
            if mode == 'MLP':
                pred = model(tile, data)
            elif mode == 'TabNet' or mode == 'both':
                pred, loss_masks = model(tile, data)
            elif mode == 'logitmixer':
                pred = model(data.strong_logits, data.weak_logits)

            if graph_type == 'patch_subgraph':
                # Compute loss
                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(
                        input=pred,
                        target=data.y_class,
                        weight=class_weights,
                        reduction='mean'
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                elif loss_type == "focal_loss":
                    loss = focal_loss(
                        inputs=pred,
                        targets=data.y_class
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                else:
                    raise ValueError(f"No supported loss_type: {loss_type}")
                
                total_loss += loss.item() * data.y_class.shape[0]
                total_examples += data.y_class.shape[0]
                logits_list.append(pred)
                labels_list.append(data.y_class)
                
            elif graph_type == 'k_hop_subgraph':
                # extract central node's embeddings and prediction results
                global_mapping = data.ptr[:-1] + data.mapping
                pred = pred[global_mapping]
                
                # Compute loss
                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(
                        input=pred,
                        target=data.y_class[global_mapping],
                        weight=class_weights,
                        reduction='mean'
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                elif loss_type == "focal_loss":
                    loss = focal_loss(
                        inputs=pred,
                        targets=data.y_class[global_mapping]
                    )
                    if mode == 'TabNet' or mode == 'both':
                        loss += sparsity_loss_weight*sparsity_loss(loss_masks)
                else:
                    raise ValueError(f"No supported loss_type: {loss_type}")
                
                total_loss += loss.item() * data.y_class[global_mapping].shape[0]
                total_examples += data.y_class[global_mapping].shape[0]
                logits_list.append(pred)
                labels_list.append(data.y_class[global_mapping])
            
    # Gather across devices
    logits_all = accelerator.gather_for_metrics(torch.cat(logits_list))
    y_true = accelerator.gather_for_metrics(torch.cat(labels_list))
    
    # Move to CPU
    logits_all = logits_all.cpu().detach()
    y_true = y_true.cpu().detach()
    
    # Compute metrics
    probs_all = F.softmax(logits_all, dim=-1)
    preds_all = probs_all.argmax(dim=-1)
    y_all = F.one_hot(y_true, num_classes=num_classes).float()
    
    # Identify present classes
    present_classes = torch.unique(y_true).tolist()

    # Macro-F1
    macro_f1 = f1_score(y_true, preds_all, labels=present_classes, average='macro', zero_division=0)

    # AUPRC (1-vs-rest)
    num_classes = probs_all.shape[1]
    auprcs = []
    aurocs = []
    for i in range(num_classes):
        if y_all[:, i].sum() == 0:
            continue  # Skip class with no positives
        
        auprc = average_precision_score(y_all[:, i], probs_all[:, i])
        auprcs.append(auprc)
        auroc = roc_auc_score(y_all[:, i], probs_all[:, i])
        aurocs.append(auroc)
    avg_auprc = sum(auprcs)/len(auprcs)
    avg_auroc = sum(aurocs)/len(aurocs)

    # Final loss and accuracy
    avg_loss = total_loss / total_examples
    avg_loss_all = accelerator.gather(torch.tensor(avg_loss, device=accelerator.device)).mean().item()
    accuracy = (preds_all == y_true).sum().item() / probs_all.shape[0]

    return avg_loss_all, accuracy, macro_f1, avg_auprc, avg_auroc

# testing
def test(model, loader, graph_type='patch_subgraph', mode='MLP'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    # Set up a dict to store predictions
    test_results_dict = {}
    for slide_id in set(loader.dataset.slide_id_list):
        test_results_dict[slide_id] = []
    
    # Inference
    with torch.no_grad():
        for batch in tqdm(loader):
            
            # forward
            data, tile, slide_id = batch
            data = data.to(device)
            tile = tile.to(device)
            if mode == 'MLP' or mode == 'ensemble':
                pred = model(tile, data)
            elif mode == 'TabNet' or mode == 'both':
                pred, loss_masks = model(tile, data)
            
            prob = F.softmax(pred, dim=-1)
            if graph_type == 'patch_subgraph':
                # Store inference results into dict
                for i in range(prob.shape[0]):
                    test_results_dict[slide_id[data.batch[i].item()]].append({"centroid" : data.x_centroid[i].tolist(),
                                                                              "type_prob" : prob[i][torch.argmax(prob[i])].item(),
                                                                              "type": torch.argmax(prob[i]).item(),
                                                                              "prob_vector": prob[i].tolist()})
            elif graph_type == 'k_hop_subgraph':
                global_mapping = data.ptr[:-1] + data.mapping
                
                for i in global_mapping.tolist():
                    test_results_dict[slide_id[data.batch[i].item()]].append({"centroid" : data.x_centroid[i].tolist(),
                                                                              "type_prob" : prob[i][torch.argmax(prob[i])].item(),
                                                                              "type": torch.argmax(prob[i]).item(),
                                                                              "prob_vector": prob[i].tolist()})
    
    # Transform: list -> dict with string indices to map the output of CellViT++
    final_test_results_dict = {
        key: {str(i + 1): val for i, val in enumerate(lst)}
        for key, lst in test_results_dict.items()
    }
    
    return final_test_results_dict

def compute_classification_metrics(gt_directory, test_results_dict, gt_scale_factor=2, pairing_radius_um=3, tile_res=40,
                                   mat_suffix=''):
    """
    Evaluate segmentation/classification results against ground truth.

    Args:
        gt_directory (str): Path to ground truth .mat files.
        test_results_dict (dict): Predictions keyed by sample_id.
        pairing_radius (float): Max distance to consider a GT/pred pair.

    Returns:
        confusion_df (pd.DataFrame): Confusion matrix as DataFrame.
        f1_dict (dict): Per-class F1 scores.
        auprc_dict (dict): Per-class AUPRC scores.
        auroc_dict (dict): Per-class AUROC scores.
        accuracy (float): Overall accuracy.
    """
    
    pairing_radius_px = pairing_radius_um*tile_res/10
    all_true, all_pred, all_probs = [], [], []
    all_unpaired_pred, all_unpaired_true = [], []

    # ---------- Step 1: Pairing ----------
    for sample_id, pred_instances in test_results_dict.items():
        # Load ground truth
        mat = sio.loadmat(os.path.join(gt_directory, f"{sample_id}{mat_suffix}.mat"))
        gt_classes = mat['class'].flatten() - 1
        gt_centroids = mat['centroid']*gt_scale_factor

        # Track which GTs are paired
        gt_paired = np.zeros(len(gt_classes), dtype=bool)

        # Build arrays for preds
        pred_ids = list(pred_instances.keys())
        pred_centroids = np.array([pred_instances[i]["centroid"] for i in pred_ids])
        pred_types = np.array([pred_instances[i]["type"] for i in pred_ids])
        pred_probs = np.array([pred_instances[i]["prob_vector"] for i in pred_ids])

        if len(gt_centroids) == 0 or len(pred_centroids) == 0:
            continue

        # Compute distance matrix
        dist_matrix = cdist(pred_centroids, gt_centroids)

        # For each pred, find closest GT
        for i, pred_id in enumerate(pred_ids):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]
            if min_dist <= pairing_radius_px and not gt_paired[min_idx]:
                # Pair found
                gt_paired[min_idx] = True
                all_true.append(gt_classes[min_idx])
                all_pred.append(pred_types[i])
                all_probs.append(pred_probs[i])
            else:
                # Unpaired Prediction (False Positive for Detection)
                all_unpaired_pred.append(pred_types[i])

        # Capture GTs that were never paired (False Negatives)
        unpaired_gt_mask = ~gt_paired
        if np.any(unpaired_gt_mask):
            all_unpaired_true.extend(gt_classes[unpaired_gt_mask])

    if not all_true:
        raise ValueError("No paired instances found. Check pairing_radius or data.")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)

    classes = np.unique(all_true)
    n_classes = len(classes)

    # ---------- Step 2: Metrics ----------
    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=classes)
    confusion_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                                   columns=[f"pred_{c}" for c in classes])

    # Accuracy
    accuracy = accuracy_score(all_true, all_pred)

    # Per-class metrics
    f1_dict, f1_joint_dict, auprc_dict, auroc_dict = {}, {}, {}, {}
    for c in classes:
        # Binary ground truth for class c
        y_true_c = (all_true == c).astype(int)
        y_pred_c = (all_pred == c).astype(int)
        y_prob_c = all_probs[:, c] if all_probs.shape[1] > c else None

        f1_dict[str(c)] = f1_score(y_true_c, y_pred_c, zero_division=0)
        
        # --- FOR F1 JOINT ---
        # TP: Paired and correct
        tp = np.sum((all_true == c) & (all_pred == c))
        
        # FP: Paired wrong + Unpaired predictions (Ghosts)
        fp_paired = np.sum((all_true != c) & (all_pred == c))
        fp_unpaired = np.sum(np.array(all_unpaired_pred) == c)
        
        # FN: Paired wrong + Unpaired GT (Missed)
        fn_paired = np.sum((all_true == c) & (all_pred != c))
        fn_unpaired = np.sum(np.array(all_unpaired_true) == c)
        
        denom = (2 * tp) + (fp_paired + fp_unpaired) + (fn_paired + fn_unpaired)
        f1_joint_dict[str(c)] = (2 * tp) / denom if denom > 0 else 0.0

        if y_prob_c is not None:
            try:
                auprc_dict[str(c)] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                auprc_dict[str(c)] = np.nan
            try:
                auroc_dict[str(c)] = roc_auc_score(y_true_c, y_prob_c)
            except ValueError:
                auroc_dict[str(c)] = np.nan
        else:
            auprc_dict[str(c)] = np.nan
            auroc_dict[str(c)] = np.nan

    return confusion_df, f1_dict, f1_joint_dict, auprc_dict, auroc_dict, accuracy

# validation
def test_segpath(model, loader, num_classes, graph_type='patch_subgraph', target_class_idx=None, mode='MLP'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            
            # forward
            data, tile, slide_id = batch
            data = data.to(device)
            tile = tile.to(device)
            if mode == 'MLP':
                pred = model(tile, data)
            elif mode == 'TabNet' or mode == 'both':
                pred, loss_masks = model(tile, data)

            if graph_type == 'patch_subgraph':
                logits_list.append(pred)
                labels_list.append(data.y_class)
                
            elif graph_type == 'k_hop_subgraph':
                # extract central node's embeddings and prediction results
                global_mapping = data.ptr[:-1] + data.mapping
                pred = pred[global_mapping]
                
                logits_list.append(pred)
                labels_list.append(data.y_class[global_mapping])
            
    # Gather across devices
    logits_all = torch.cat(logits_list)
    y_true = torch.cat(labels_list)
    
    # Move to CPU
    logits_all = logits_all.cpu().detach()
    y_true = y_true.cpu().detach()
    
    # Get Predictions
    probs_all = F.softmax(logits_all, dim=-1)
    preds_all = probs_all.argmax(dim=-1)

    # ============================================
    # BINARY SETUP (A vs Not-A)
    # ============================================
    
    # Ground Truth: 1 if it is A, 0 if it is Not A
    y_binary = y_true.float()
    
    # Prediction: 1 if model chose A, 0 if model chose something else
    preds_binary = (preds_all == target_class_idx).float()
    
    # Probability of being A (needed for AUROC/AUPRC)
    probs_target = probs_all[:, target_class_idx].numpy()
    
    # ============================================
    # COMPUTE BINARY METRICS
    # ============================================

    # 1. Binary Accuracy
    # (True Positives + True Negatives) / Total Samples
    # This checks: Did the binary prediction match the binary label?
    accuracy = (preds_binary == y_binary).float().mean().item()

    # 2. Other Metrics
    y_binary_np = y_binary.numpy()
    preds_binary_np = preds_binary.numpy()
    
    if y_binary.sum() > 0:
        f1 = f1_score(y_binary_np, preds_binary_np, average='binary')
        auprc = average_precision_score(y_binary_np, probs_target)
        # Handle single-class edge case for AUROC
        if len(set(y_binary_np)) > 1:
            auroc = roc_auc_score(y_binary_np, probs_target)
        else:
            auroc = 0.0
    else:
        raise ValueError(
            f"Expected both positive and negative samples for target_class_idx={target_class_idx}, but not found..."
        )
    
    return accuracy, f1, auprc, auroc

def sanity_check_comparison(gt_directory, strong_dict, weak_dict, gt_scale_factor=2, pairing_radius_um=3, tile_res=40, mat_suffix=''):
    """
    Compares two models to see if the Weak model recovers errors made by the Strong model.
    """
    
    pairing_radius_px = pairing_radius_um * tile_res / 10
    
    # Counters
    total_overlapping_instances = 0
    strong_correct = 0
    strong_wrong = 0
    recovered_by_weak = 0 # Strong was wrong, but Weak was right
    both_wrong = 0
    
    # Identify common slides
    common_slides = set(strong_dict.keys()) & set(weak_dict.keys())
    
    print(f"Analyzing {len(common_slides)} common slides...")

    for sample_id in common_slides:
        # --- 1. Load Ground Truth ---
        try:
            mat_path = os.path.join(gt_directory, f"{sample_id}{mat_suffix}.mat")
            mat = sio.loadmat(mat_path)
        except FileNotFoundError:
            continue
            
        gt_classes = mat['class'].flatten() - 1
        gt_centroids = mat['centroid'] * gt_scale_factor
        
        if len(gt_centroids) == 0:
            continue

        # Helper to get pairing map (GT_Index -> Predicted_Class)
        def get_gt_to_pred_map(pred_instances):
            pred_ids = list(pred_instances.keys())
            if len(pred_ids) == 0: return {}
            
            p_centroids = np.array([pred_instances[i]["centroid"] for i in pred_ids])
            p_types = np.array([pred_instances[i]["type"] for i in pred_ids])
            
            # Distance matrix: Rows=Preds, Cols=GT
            dists = cdist(p_centroids, gt_centroids)
            
            # Map GT index to Pred Class
            gt_map = {}
            gt_paired_mask = np.zeros(len(gt_centroids), dtype=bool)
            
            # Greedy matching (iterating preds)
            for i, pred_id in enumerate(pred_ids):
                min_gt_idx = np.argmin(dists[i])
                min_dist = dists[i, min_gt_idx]
                
                if min_dist <= pairing_radius_px and not gt_paired_mask[min_gt_idx]:
                    gt_paired_mask[min_gt_idx] = True
                    gt_map[min_gt_idx] = p_types[i]
            
            return gt_map

        # --- 2. Pair Both Models to GT ---
        strong_map = get_gt_to_pred_map(strong_dict[sample_id])
        weak_map = get_gt_to_pred_map(weak_dict[sample_id])
        
        # --- 3. Compare Intersection ---
        # We only care about GT instances detected by BOTH models to ensure fair comparison
        common_gt_indices = set(strong_map.keys()) & set(weak_map.keys())
        
        for idx in common_gt_indices:
            total_overlapping_instances += 1
            true_cls = gt_classes[idx]
            s_pred = strong_map[idx]
            w_pred = weak_map[idx]
            
            if s_pred == true_cls:
                strong_correct += 1
            else:
                # Strong is WRONG
                strong_wrong += 1
                if w_pred == true_cls:
                    # Gold in the Mud!
                    recovered_by_weak += 1
                else:
                    both_wrong += 1

    # --- 4. Report Results ---
    print("\n" + "="*40)
    print("SANITY CHECK RESULTS")
    print("="*40)
    print(f"Total Common Instances Evaluated: {total_overlapping_instances}")
    print(f"Strong Model Accuracy (on subset): {strong_correct/total_overlapping_instances:.4f}")
    print("-" * 20)
    print(f"Instances where Strong Model FAILED: {strong_wrong}")
    print(f"Of those failures, Weak Model was CORRECT: {recovered_by_weak}")
    
    if strong_wrong > 0:
        recovery_rate = recovered_by_weak / strong_wrong
        print(f"Recovery Rate (Gold in Mud): {recovery_rate:.2%}")
        
        print("\nDECISION:")
        if recovery_rate > 0.1: # 10% threshold
            print("✅ PASS: The weak model has unique information. Use Strategy 2 (Stacking).")
        else:
            print("❌ FAIL: The weak model is just noise. Focus on improving the Strong model.")
    else:
        print("Strong model is perfect? No errors to analyze.")
    print("="*40 + "\n")

    return recovered_by_weak, strong_wrong

def calculate_entropy(prob_vector):
    """
    Calculates Shannon Entropy for a probability vector.
    H(p) = - sum(p * log(p))
    """
    p = np.array(prob_vector)
    # Add epsilon to avoid log(0)
    epsilon = 1e-9
    return -np.sum(p * np.log(p + epsilon))

def apply_ensemble_strategy(strong_dict, weak_dict, entropy_threshold):
    """
    Merges two dictionaries based on Uncertainty Gating.
    Returns a new 'ensembled_dict'.
    """
    ensembled_dict = {}
    
    # Iterate over slides
    for slide_id in strong_dict.keys():
        ensembled_dict[slide_id] = {}
        
        # We assume strict index alignment between Strong and Weak results 
        # (i.e., they were generated by the same dataloader order)
        strong_items = strong_dict[slide_id]
        
        # If weak_dict is missing this slide (edge case), just copy strong
        if slide_id not in weak_dict:
            ensembled_dict[slide_id] = strong_items
            continue
            
        weak_items = weak_dict[slide_id]
        
        # Iterate over instances in the slide
        for idx_key, s_data in strong_items.items():
            # Get corresponding weak data
            if idx_key not in weak_items:
                # Fallback if alignment fails
                ensembled_dict[slide_id][idx_key] = s_data
                continue
                
            w_data = weak_items[idx_key]
            
            # --- STRATEGY 1: ENTROPY GATING ---
            prob_s = np.array(s_data["prob_vector"])
            entropy = calculate_entropy(prob_s)
            
            new_entry = s_data.copy()
            
            # If Strong model is CONFUSED (High Entropy) -> Consult Weak Model
            if entropy > entropy_threshold:
                prob_w = np.array(w_data["prob_vector"])
                
                # Average the probabilities
                fused_prob = (prob_s + prob_w) / 2.0
                
                # Update the entry
                new_entry["prob_vector"] = fused_prob.tolist()
                new_entry["type"] = int(np.argmax(fused_prob))
                new_entry["type_prob"] = float(np.max(fused_prob))
                
            # Else: Keep Strong model as is (already copied)
            
            ensembled_dict[slide_id][idx_key] = new_entry
            
    return ensembled_dict

def cond_ensemble(gt_directory, strong_dict, weak_dict=None, entropy_threshold=0.5, 
                  gt_scale_factor=2, pairing_radius_um=3, tile_res=40, mat_suffix=''):
    """
    Upgraded evaluation function.
    
    Args:
        strong_dict: Predictions from the Strong model.
        weak_dict: Predictions from the Weak model (Optional).
        entropy_threshold: Threshold to trigger ensembling.
    """
    
    # --- Step 0: Handle Ensembling ---
    if weak_dict is not None:
        print(f"⚡ Ensembling enabled with Entropy Threshold: {entropy_threshold}")
        test_results_dict = apply_ensemble_strategy(strong_dict, weak_dict, entropy_threshold)
    else:
        print("⚡ No weak dictionary provided. Evaluating Strong model only.")
        test_results_dict = strong_dict

    # --- Step 1: Standard Evaluation (Same as before) ---
    pairing_radius_px = pairing_radius_um * tile_res / 10
    all_true, all_pred, all_probs = [], [], []
    all_unpaired_pred, all_unpaired_true = [], []

    for sample_id, pred_instances in test_results_dict.items():
        try:
            mat = sio.loadmat(os.path.join(gt_directory, f"{sample_id}{mat_suffix}.mat"))
        except FileNotFoundError:
            continue
            
        gt_classes = mat['class'].flatten() - 1
        gt_centroids = mat['centroid'] * gt_scale_factor
        gt_paired = np.zeros(len(gt_classes), dtype=bool)

        pred_ids = list(pred_instances.keys())
        if not pred_ids: continue
        
        pred_centroids = np.array([pred_instances[i]["centroid"] for i in pred_ids])
        pred_types = np.array([pred_instances[i]["type"] for i in pred_ids])
        pred_probs = np.array([pred_instances[i]["prob_vector"] for i in pred_ids])

        if len(gt_centroids) == 0: continue

        dist_matrix = cdist(pred_centroids, gt_centroids)

        for i, pred_id in enumerate(pred_ids):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]
            if min_dist <= pairing_radius_px and not gt_paired[min_idx]:
                gt_paired[min_idx] = True
                all_true.append(gt_classes[min_idx])
                all_pred.append(pred_types[i])
                all_probs.append(pred_probs[i])
            else:
                all_unpaired_pred.append(pred_types[i])

        unpaired_gt_mask = ~gt_paired
        if np.any(unpaired_gt_mask):
            all_unpaired_true.extend(gt_classes[unpaired_gt_mask])

    if not all_true:
        print("No paired instances found.")
        return None

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)
    classes = np.unique(all_true)

    # --- Step 2: Metrics & Printing ---
    accuracy = accuracy_score(all_true, all_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    f1_dict = {}
    print("\n--- Per-Class F1 Scores ---")
    for c in classes:
        y_true_c = (all_true == c).astype(int)
        y_pred_c = (all_pred == c).astype(int)
        
        f1 = f1_score(y_true_c, y_pred_c, zero_division=0)
        f1_dict[str(c)] = f1
        print(f"Class {c}: {f1:.4f}")

    return f1_dict, accuracy