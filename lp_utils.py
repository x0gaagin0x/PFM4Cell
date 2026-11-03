import torch
from tqdm import tqdm
import torch.nn.functional as F
from loss import FocalLoss, sparsity_loss
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

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
                pred = model(data.cell_emb)
            elif mode == 'TabNet':
                pred, loss_masks = model(data.x)
            elif mode == 'both':
                pred, loss_masks = model(data.cell_emb, data.x)

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
                pred = model(data.cell_emb)
            elif mode == 'TabNet':
                pred, loss_masks = model(data.x)
            elif mode == 'both':
                pred, loss_masks = model(data.cell_emb, data.x)

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