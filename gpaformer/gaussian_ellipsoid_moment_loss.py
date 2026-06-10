import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianEllipsoidMomentLoss(nn.Module):
    """
    Regularizes predicted organ masks to have Gaussian-like moments (mean and covariance)
    similar to ground-truth per class. Operates on logits (pre-softmax) and integer labels.

    Args:
        voxel_spacing: tuple (sz, sy, sx) in mm
        classes_to_regularize: iterable of class ids to regularize
        w_mu: weight for mean difference term
        w_sigma: weight for covariance difference term
    """

    def __init__(
        self,
        voxel_spacing=(2.0, 1.5, 1.5),
        classes_to_regularize=(1, 2, 3, 12, 13),
        w_mu: float = 1.0,
        w_sigma: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spacing = torch.tensor(list(voxel_spacing), dtype=torch.float32)
        self.classes = tuple(int(c) for c in classes_to_regularize)
        self.w_mu = float(w_mu)
        self.w_sigma = float(w_sigma)
        self.eps = float(eps)

    @staticmethod
    def _compute_coords(D: int, H: int, W: int, spacing: torch.Tensor, device: torch.device):
        z = torch.arange(D, device=device, dtype=torch.float32) * spacing[0]
        y = torch.arange(H, device=device, dtype=torch.float32) * spacing[1]
        x = torch.arange(W, device=device, dtype=torch.float32) * spacing[2]
        try:
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        except TypeError:
            zz, yy, xx = torch.meshgrid(z, y, x)
        coords = torch.stack([zz, yy, xx], dim=0)  # [3, D, H, W]
        return coords

    @staticmethod
    def _soft_one_hot(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, D, H, W] -> softmax across classes
        return F.softmax(logits, dim=1)

    @staticmethod
    def _moments(prob: torch.Tensor, coords: torch.Tensor):
        # prob: [B, 1, D, H, W] or [B, 1, N]; coords: [3, D, H, W]
        B = prob.shape[0]
        prob_flat = prob.view(B, -1)  # [B,N]
        coords_flat = coords.view(3, -1)  # [3,N]

        mass = prob_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,1]
        mu = (prob_flat @ coords_flat.t()) / mass  # [B,3]

        # E[xx^T] - mu mu^T, where E is w-normalized expectation
        norm_w = (prob_flat / mass).unsqueeze(1)  # [B,1,N]
        weighted_coords = coords_flat.unsqueeze(0) * norm_w  # [B,3,N]
        second_moment = torch.matmul(weighted_coords, coords_flat.t())  # [B,3,3]
        mu_outer = mu.unsqueeze(-1) @ mu.unsqueeze(1)  # [B,3,3]
        cov = second_moment - mu_outer  # [B,3,3]
        return mu, cov

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, D, H, W]; labels: [B, 1, D, H, W] int
        device = logits.device
        B, C, D, H, W = logits.shape
        coords = self._compute_coords(D, H, W, self.spacing.to(device), device)
        probs = self._soft_one_hot(logits)  # [B, C, D, H, W]

        loss_total = logits.new_zeros(())
        participated = 0

        for cls in self.classes:
            if cls < 0 or cls >= C:
                continue

            prob_c = probs[:, cls:cls + 1, ...]  # [B,1,D,H,W]
            # hard gt mask for class (strip MetaTensor to plain Tensor)
            gt_mask = (labels == cls)
            gt_c = torch.as_tensor(gt_mask, dtype=probs.dtype, device=device)  # [B,1,D,H,W]

            # Skip moment terms if the class is entirely absent in GT for the batch
            if gt_c.sum() < 1:
                # tiny mass penalty to discourage spurious predictions without forcing moments
                mass_penalty = prob_c.sum(dim=(1, 2, 3, 4))  # [B]
                loss_total = loss_total + 0.001 * mass_penalty.mean()
                continue

            mu_p, cov_p = self._moments(prob_c, coords)
            mu_g, cov_g = self._moments(gt_c, coords)

            # mean term (L2)
            mu_term = (mu_p - mu_g).pow(2).sum(dim=-1)  # [B]

            # covariance term (Frobenius)
            cov_diff = cov_p - cov_g
            sigma_term = (cov_diff.pow(2).sum(dim=(-2, -1)))  # [B]

            loss_c = self.w_mu * mu_term + self.w_sigma * sigma_term  # [B]
            loss_total = loss_total + loss_c.mean()
            participated += 1

        # average over selected classes (avoid div by zero)
        denom = max(participated, 1)
        return loss_total / float(denom)


