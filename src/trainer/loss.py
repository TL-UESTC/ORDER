import torch
import torch.nn.functional as F
import torch.nn as nn


def cliploss(rep0, rep1, t=0.1):
    batch_size = rep0.shape[0]
    out_joint_mod = torch.cat([rep0, rep1], dim=0)   # batch*2, 128
    sim_matrix_joint_mod = torch.exp(torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / t)     # batch*2, batch*2
    mask_joint_mod = (torch.ones_like(sim_matrix_joint_mod) - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)).bool()
    sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

    pos_sim_joint_mod = torch.exp(torch.sum(rep0 * rep1, dim=-1) / t)  # batch (matched similarity)
    pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)    # [2*B]
    loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))

    return loss_joint_mod


class OrderLoss(nn.Module):
    def __init__(self, mean, std, total_epoch, label_dis='l1', feature_dis='l2', label_norm='zscore', label_min=None, label_max=None, alpha=0.5, device='cuda:0'):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)
        if label_min is not None and label_max is not None:
            self.label_min = label_min.to(device)
            self.label_max = label_max.to(device)
        self.label_diff_fn = self._get_label_diff_fn(label_dis)
        self.feature_diff_fn = self._get_feature_sim_fn(feature_dis)
        self.t = 0.1
        self.t_rnc = torch.arange(1, 0.1, -0.9/total_epoch)
        self.alpha = alpha
        self.label_norm = label_norm
        self.device = device

    def _get_label_diff_fn(self, label_diff):
        if label_diff == 'l1':
            return lambda labels: torch.cdist(labels, labels, p=1)
        elif label_diff == 'l2':
            return lambda labels: torch.cdist(labels, labels, p=2)
        else:
            raise ValueError(f"Unsupported label_diff: {label_diff}")
    
    def _get_feature_sim_fn(self, feature_sim):
        if feature_sim == 'l2':
            return lambda features: -torch.cdist(features, features, p=2)
        elif feature_sim == 'l1':
            return lambda features: -torch.cdist(features, features, p=1)
        elif feature_sim == 'cosine':
            return lambda features: F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        elif feature_sim == 'product':
            return lambda features: features @ features.T    
        else:
            raise ValueError(f"Unsupported feature_sim: {feature_sim}")
        
    def normalize_labels(self, labels):
        if self.label_norm == 'zscore':
            return (labels - self.mean) / (self.std + 1e-8)
        elif self.label_norm == 'minmax':
            return (labels - self.label_min) / (self.label_max - self.label_min + 1e-8)
        else:
            return labels
        
    def compute_RNCloss(self, rep0, origin_labels, epoch=0):        
        features = rep0
        label_diffs = self.label_diff_fn(origin_labels)  # [2*batch_size, 2*batch_size]
        
        logits = self.feature_diff_fn(features) / self.t_rnc[epoch]  # [2*batch_size, 2*batch_size]
        
        exp_logits = logits.exp()
        
        n = logits.shape[0]  # n = 2*batch_size
        
        mask = (1 - torch.eye(n, device=logits.device)).bool()
        logits = logits.masked_select(mask).view(n, n - 1)
        exp_logits = exp_logits.masked_select(mask).view(n, n - 1)
        label_diffs = label_diffs.masked_select(mask).view(n, n - 1)
        
        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # [2*batch_size]
            pos_label_diffs = label_diffs[:, k]  # [2*batch_size]
            
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2*batch_size, 2*batch_size-1]

            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))            
            loss += -(pos_log_probs / (n * (n - 1))).sum()
        
        return loss

    def compute_CLIPloss(self, rep0, rep1):
        batch_size = rep0.shape[0]
        out_joint_mod = torch.cat([rep0, rep1], dim=0)   # batch*2, 128
        sim_matrix_joint_mod = torch.exp(torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.t)     # batch*2, batch*2
        mask_joint_mod = (torch.ones_like(sim_matrix_joint_mod) - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)).bool()
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

        pos_sim_joint_mod = torch.exp(torch.sum(rep0 * rep1, dim=-1) / self.t)  # batch (matched similarity)
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)    # [2*B]
        loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))

        return loss_joint_mod

    def forward(self, rep0, rep1, origin_label, epoch_idx=0):
        label = self.normalize_labels(origin_label)
        loss_rnc_tab = self.compute_RNCloss(rep0, label, epoch=epoch_idx)
        loss_rnc_img = self.compute_RNCloss(rep1, label, epoch=epoch_idx)
        loss_CLIP = (self.compute_CLIPloss(rep0, rep1) + self.compute_CLIPloss(rep1, rep0)).mean()
        return loss_rnc_tab, loss_rnc_img, loss_CLIP


class RnCLoss(nn.Module):
    def __init__(self, mean, std, device, total_epoch, temperature=2, label_diff='l1', feature_sim='product', label_norm='zscore'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.label_diff_fn = self._get_label_diff_fn(label_diff)
        self.feature_sim_fn = self._get_feature_sim_fn(feature_sim)
        self.t_rnc = torch.arange(1, 0.1, -0.9/total_epoch)
        self.label_norm = label_norm

    def _get_label_diff_fn(self, label_diff):
        if label_diff == 'l1':
            return lambda labels: torch.cdist(labels, labels, p=1)
        elif label_diff == 'l2':
            return lambda labels: torch.cdist(labels, labels, p=2)
        else:
            raise ValueError(f"Unsupported label_diff: {label_diff}")
    
    def _get_feature_sim_fn(self, feature_sim):
        if feature_sim == 'l2':
            return lambda features: -torch.cdist(features, features, p=2)
        elif feature_sim == 'l1':
            return lambda features: -torch.cdist(features, features, p=1)
        elif feature_sim == 'cosine':
            return lambda features: F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        elif feature_sim == 'product':
            return lambda features: features @ features.T    
        else:
            raise ValueError(f"Unsupported feature_sim: {feature_sim}")
        
    def normalize_labels(self, labels):
        if self.label_norm == 'zscore':
            return (labels - self.mean) / (self.std + 1e-8)
        elif self.label_norm == 'minmax':
            return (labels - self.label_min) / (self.label_max - self.label_min + 1e-8)
        else:
            raise NotImplementedError()

    def forward(self, features, origin_label, epoch):
        labels = self.normalize_labels(origin_label)
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features) / self.t_rnc[epoch]
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


if __name__ == "__main__":
    pass
