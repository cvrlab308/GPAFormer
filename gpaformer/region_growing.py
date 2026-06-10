
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionGrowing(nn.Module):
    """
    基於局部圖的 patch 合併；此版本加入：
    - 26 鄰連邊（可調整半徑）
    - 高斯橢球 gating/權重（使用 voxel spacing 與 Σ）
    - 特徵相似(余弦) 與 空間高斯權重結合
    - 【重點改動 | Approach B】：只有「幾何建圖」不需要梯度；assignment_net 與池化保留梯度。
    """
    def __init__(
        self,
        in_channels: int,
        num_clusters: int,
        alpha: float = 0.5,
        # --- 高斯橢球與鄰域設定 ---
        voxel_spacing=(2.0, 1.5, 1.5),   # (sz, sy, sx) in mm；請依實際 token 尺度設定
        use_gaussian_ellipsoid: bool = True,
        sigma_mm=(2.5, 2.5, 3.0),        # (σz, σy, σx) in mm，z 可略放寬
        chi2_thresh: float = 7.815,      # 自由度 df=3, 95% 門檻 ≈ 7.815；99%≈11.34
        neighbor_radius: int = 1,        # r=1 → 26 鄰；r=2 → 124 鄰
        combine: str = "prod",           # 'prod' 或 'blend'（權重結合方式）
        feat_temp: float = 1.0           # 特徵相似溫度
    ):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.alpha = float(alpha)

        # 分配矩陣網路（需保留梯度）
        self.assignment_net = nn.Linear(in_channels, num_clusters)

        # 幾何/權重參數
        self.use_ge = bool(use_gaussian_ellipsoid)
        self.voxel_spacing = torch.tensor(voxel_spacing, dtype=torch.float32)  # (sz, sy, sx)
        self.sigma_mm = torch.tensor(sigma_mm, dtype=torch.float32)            # (σz, σy, σx)
        inv = 1.0 / (self.sigma_mm ** 2)
        self.Sigma_inv = torch.diag(inv)                                       # (3,3)
        self.chi2_thresh = float(chi2_thresh)
        self.neighbor_radius = int(neighbor_radius)
        assert combine in ["prod", "blend"]
        self.combine = combine
        self.feat_temp = float(feat_temp)

    # 允許在外部（trainer/GPAFormer）動態改參數
    def set_config(
        self,
        voxel_spacing=None, sigma_mm=None, chi2_thresh=None,
        neighbor_radius=None, use_gaussian_ellipsoid=None,
        combine=None, feat_temp=None
    ):
        if voxel_spacing is not None:
            self.voxel_spacing = torch.tensor(voxel_spacing, dtype=torch.float32)
        if sigma_mm is not None:
            self.sigma_mm = torch.tensor(sigma_mm, dtype=torch.float32)
            inv = 1.0 / (self.sigma_mm ** 2)
            self.Sigma_inv = torch.diag(inv)
        if chi2_thresh is not None:
            self.chi2_thresh = float(chi2_thresh)
        if neighbor_radius is not None:
            self.neighbor_radius = int(neighbor_radius)
        if use_gaussian_ellipsoid is not None:
            self.use_ge = bool(use_gaussian_ellipsoid)
        if combine is not None:
            assert combine in ["prod", "blend"]
            self.combine = combine
        if feat_temp is not None:
            self.feat_temp = float(feat_temp)

    @staticmethod
    def _make_neighbor_offsets(r=1, device="cpu"):
        rng = torch.arange(-r, r + 1, device=device)
        try:
            dz, dy, dx = torch.meshgrid(rng, rng, rng, indexing="ij")
        except TypeError:  # 兼容舊版 torch
            dz, dy, dx = torch.meshgrid(rng, rng, rng)
        offs = torch.stack([dz.flatten(), dy.flatten(), dx.flatten()], dim=1)  # (M,3)
        keep = torch.any(offs != 0, dim=1)  # 移除 (0,0,0)
        return offs[keep]                   # (M-1,3)

    def build_graph(self, x):
        """
        幾何圖構建（與 batch 無關的部分）——**不需要梯度**：
        - 產生 N 個節點的 r-鄰（r=1 為 26 鄰）
        - 計算 mm 座標與馬氏距離
        - 回傳 (src, dst, w_space, N)，其中 w_space 是空間權重
        """
        B, N, C = x.shape
        device = x.device
        n = round(N ** (1 / 3))
        assert n ** 3 == N, f"N={N} 必須是完美立方，現在 n^3={n**3}"

        # 建立索引座標 (z,y,x) 與 mm 座標
        z = torch.arange(n, device=device)
        y = torch.arange(n, device=device)
        xg = torch.arange(n, device=device)
        try:
            zz, yy, xx = torch.meshgrid(z, y, xg, indexing="ij")
        except TypeError:
            zz, yy, xx = torch.meshgrid(z, y, xg)
        grid_idx = torch.stack([zz.flatten(), yy.flatten(), xx.flatten()], dim=1)  # [N,3]
        spacing = self.voxel_spacing.to(device)  # (sz, sy, sx)
        pos_mm = grid_idx * spacing  # [N,3] in mm

        # r-鄰
        offs = self._make_neighbor_offsets(self.neighbor_radius, device)  # [M,3]
        nbr_idx = grid_idx.unsqueeze(1) + offs.unsqueeze(0)               # [N,M,3]
        valid = (
            (nbr_idx[..., 0] >= 0) & (nbr_idx[..., 0] < n) &
            (nbr_idx[..., 1] >= 0) & (nbr_idx[..., 1] < n) &
            (nbr_idx[..., 2] >= 0) & (nbr_idx[..., 2] < n)
        )
        center_lin = (grid_idx[:, 0] * n + grid_idx[:, 1]) * n + grid_idx[:, 2]     # [N]
        nbr_lin = (nbr_idx[..., 0] * n + nbr_idx[..., 1]) * n + nbr_idx[..., 2]     # [N,M]
        src = center_lin.unsqueeze(1).expand_as(nbr_lin)[valid]  # [E]
        dst = nbr_lin[valid]                                     # [E]

        # 空間高斯權重（馬氏距離或各向同性退化）
        dpos = (pos_mm[dst] - pos_mm[src])  # [E,3] (mm)
        if self.use_ge:
            Sinv = self.Sigma_inv.to(device)                       # (3,3)
            d2 = torch.sum((dpos @ Sinv) * dpos, dim=-1)           # [E]
            mask = d2 <= self.chi2_thresh
            src, dst, d2 = src[mask], dst[mask], d2[mask]
            w_space = torch.exp(-0.5 * d2)                         # [E]
        else:
            # 各向同性退化：半徑以 max(σ) 近似
            d2 = torch.sum(dpos ** 2, dim=-1)
            r2 = (self.sigma_mm.max().to(device) ** 2)
            mask = d2 <= r2
            src, dst, d2 = src[mask], dst[mask], d2[mask]
            w_space = torch.exp(-0.5 * d2 / (r2 + 1e-6))

        return (src, dst, w_space, N)

    def learn_assignments(self, x, geom):
        """
        x: [B, N, C]
        geom: (src, dst, w_space, N)
        - 合成 batch 專屬的稀疏鄰接矩陣（加入特徵相似）
        - 做 A @ x（稀疏乘法），與原特徵融合後，預測 assignment S
        """
        src, dst, w_space, N = geom
        device = x.device
        B, N_check, C = x.shape
        assert N_check == N

        # 特徵相似
        x_norm = F.normalize(x, p=2, dim=-1)  # [B,N,C]
        adj_x_list = []
        for b in range(B):
            cos = (x_norm[b, src] * x_norm[b, dst]).sum(-1)  # [E]
            cos = torch.clamp(cos / max(self.feat_temp, 1e-6), -1.0, 1.0)
            if self.combine == "prod":
                w = w_space * (0.5 * (cos + 1.0))            # [0,1]
            else:  # blend
                w = 0.5 * w_space + 0.5 * (0.5 * (cos + 1.0))

            A = torch.sparse_coo_tensor(
                indices=torch.stack([src, dst], dim=0),  # [2,E]
                values=w,
                size=(N, N),
                device=device
            ).coalesce()

            # 稀疏乘法：A @ x[b]
            adj_x_b = torch.sparse.mm(A, x[b])  # [N,C]
            adj_x_list.append(adj_x_b)

        adj_x = torch.stack(adj_x_list, dim=0)  # [B,N,C]
        fused_x = self.alpha * adj_x + (1.0 - self.alpha) * x  # [B,N,C]

        logits = self.assignment_net(fused_x)   # [B,N,K]
        S = F.softmax(logits, dim=-1)          # [B,N,K]
        return S

    @staticmethod
    def _cube_root_int(n):
        return round(n ** (1 / 3))

    def pool(self, x, S):
        # 合併節點：S^T x
        return torch.bmm(S.transpose(1, 2), x)  # [B, K, C]

    # --------- 主要改動：幾何建圖 no-grad，分配/池化可學習 ---------
    def forward(self, x, ori_x=None):
        """
        x: [B, N, C] tokens
        return: [B, num_clusters, C]
        """
        # 幾何建圖不需要梯度（減少圖的記憶體占用與不必要的反向）
        with torch.no_grad():
            geom = self.build_graph(x)

        # 以下保留梯度，assignment_net 與池化能學習
        S = self.learn_assignments(x, geom)
        x_pooled = self.pool(x, S)
        return x_pooled
