"""
BracketFormer v6 — NCAA March Madness 深度学习预测器
=====================================================
相对 v5 的改进:
  [特征] 使用 v2 增强特征 (55 维 vs 37 维), 含 Four Factors / SOS / 调整效率
  [架构] GatedFusion 门控融合 (取代简单拼接), 参数效率更高
  [优化] SAM (Sharpness-Aware Minimization) 优化器, 提升泛化能力
  [校准] Platt Scaling + 温度缩放双校准, 选择最优
  [集成] 8 个 snapshot (v5 为 7), 更宽 clip [0.01, 0.99]
  [正则] 增强 DropPath=0.12, Dropout=0.22, Label Smooth=0.02

v5 基线: OOF Brier = 0.17082, AUC = 0.8259
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn

warnings.filterwarnings('ignore')

# ── 中文字体配置 ──
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).resolve().parent
FEAT_DIR = DATA_DIR / 'features'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {
    'seed':             42,
    'n_folds':          5,
    'epochs':           420,
    'batch_size':       512,
    'lr':               2.0e-4,
    'weight_decay':     1e-3,
    # 架构
    'embed_dim':        128,
    'n_heads':          8,
    'n_cross_layers':   3,
    'n_self_layers':    2,
    'se_ratio':         4,
    'dropout':          0.22,
    'drop_path_rate':   0.12,
    'diff_hidden':      96,       # v5=64 → v6=96 (适配更多 diff 特征)
    'fusion_dim':       384,      # GatedFusion 输出维度
    # 损失 & 正则
    'clip_grad':        1.0,
    'mixup_alpha':      0.4,
    'label_smooth':     0.020,    # v5=0.015 → v6=0.020
    'feature_noise':    0.02,
    'feat_mask_rate':   0.05,
    # 学习率调度
    'warmup_steps':     200,
    'cosine_T0':        50,
    'cosine_Tmult':     2,
    # SWA & 集成
    'swa_start_ratio':  0.58,
    'swa_lr':           4e-5,
    'n_snapshots':      8,        # v5=7 → v6=8
    'patience':         90,       # v5=80 → v6=90
    # SAM
    'sam_rho':          0.05,
}

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG['seed'])
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════
# 1. 数据加载: 使用 v2 增强特征
# ══════════════════════════════════════════════════════════════
def load_team_features(path):
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns
                if c not in ('Season', 'TeamID', 'Gender', 'ConfAbbrev')
                and pd.api.types.is_numeric_dtype(df[c])]
    df[num_cols] = df[num_cols].fillna(0)
    return df, num_cols


def build_feat_index(feat_df, cols):
    idx = {}
    for row in feat_df.itertuples():
        idx[(row.Season, row.TeamID)] = np.array(
            [getattr(row, c) for c in cols], dtype=np.float32)
    return idx


def get_team_vec(feat_df, feat_idx, cols, season, tid):
    k = (season, tid)
    if k in feat_idx:
        return feat_idx[k]
    sub = feat_df[(feat_df['TeamID'] == tid) & (feat_df['Season'] <= season)]
    if not sub.empty:
        return sub.sort_values('Season').iloc[-1][cols].values.astype(np.float32)
    sub2 = feat_df[feat_df['TeamID'] == tid]
    if not sub2.empty:
        return sub2.sort_values('Season').iloc[-1][cols].values.astype(np.float32)
    return np.zeros(len(cols), np.float32)


def build_training_data(hist_df, feat_df, feat_idx, team_cols, diff_cols):
    fA_list, fB_list, diff_list, y_list, season_list = [], [], [], [], []
    for row in hist_df.itertuples():
        fA = get_team_vec(feat_df, feat_idx, team_cols, row.Season, row.TeamA)
        fB = get_team_vec(feat_df, feat_idx, team_cols, row.Season, row.TeamB)
        d  = np.array([getattr(row, c, 0.0) for c in diff_cols], np.float32)
        fA_list.append(fA); fB_list.append(fB)
        diff_list.append(d); y_list.append(row.Label)
        season_list.append(row.Season)
    return (np.array(fA_list, np.float32), np.array(fB_list, np.float32),
            np.array(diff_list, np.float32), np.array(y_list, np.float32),
            np.array(season_list, np.int32))


# ── 优先加载 scientific v3 特征, 若不存在则回退到 v2/v1 ──
v3_m_path = FEAT_DIR / 'M_team_features_v3_scientific.csv'
v3_w_path = FEAT_DIR / 'W_team_features_v3_scientific.csv'
v3_mh_path = FEAT_DIR / 'M_train_full_history_v3_scientific.csv'
v3_wh_path = FEAT_DIR / 'W_train_full_history_v3_scientific.csv'

v2_m_path = FEAT_DIR / 'M_team_features_v2.csv'
v2_w_path = FEAT_DIR / 'W_team_features_v2.csv'
v2_mh_path = FEAT_DIR / 'M_train_full_history_v2.csv'
v2_wh_path = FEAT_DIR / 'W_train_full_history_v2.csv'

if v3_m_path.exists() and v3_w_path.exists() and v3_mh_path.exists() and v3_wh_path.exists():
    print("加载 scientific v3 特征文件...")
    m_feat, m_cols = load_team_features(v3_m_path)
    w_feat, w_cols = load_team_features(v3_w_path)
    m_hist = pd.read_csv(v3_mh_path)
    w_hist = pd.read_csv(v3_wh_path)
    feat_version = 'v3_scientific'
elif v2_m_path.exists() and v2_w_path.exists() and v2_mh_path.exists() and v2_wh_path.exists():
    print("加载 v2 增强特征文件...")
    m_feat, m_cols = load_team_features(v2_m_path)
    w_feat, w_cols = load_team_features(v2_w_path)
    m_hist = pd.read_csv(v2_mh_path)
    w_hist = pd.read_csv(v2_wh_path)
    feat_version = 'v2'
else:
    print("⚠ 未检测到 v3/v2 完整特征资产, 回退到 v1 特征")
    m_feat, m_cols = load_team_features(FEAT_DIR / 'M_team_features_final.csv')
    w_feat, w_cols = load_team_features(FEAT_DIR / 'W_team_features_final.csv')
    m_hist = pd.read_csv(FEAT_DIR / 'M_train_full_history.csv')
    w_hist = pd.read_csv(FEAT_DIR / 'W_train_full_history.csv')
    feat_version = 'v1'

common_team_cols = [c for c in m_cols if c in w_cols]
m_diff_cols = [c for c in m_hist.columns if c.startswith('Diff_')]
w_diff_cols = [c for c in w_hist.columns if c.startswith('Diff_')]
common_diff_cols = [c for c in m_diff_cols if c in w_diff_cols]

print(f"  特征版本: {feat_version}")
print(f"  公共球队特征: {len(common_team_cols)}  公共 Diff 特征: {len(common_diff_cols)}")

m_idx = build_feat_index(m_feat, common_team_cols)
w_idx = build_feat_index(w_feat, common_team_cols)

m_fA, m_fB, m_diff, m_y, m_seasons = build_training_data(
    m_hist, m_feat, m_idx, common_team_cols, common_diff_cols)
w_fA, w_fB, w_diff, w_y, w_seasons = build_training_data(
    w_hist, w_feat, w_idx, common_team_cols, common_diff_cols)

feat_A      = np.concatenate([m_fA, w_fA], axis=0)
feat_B      = np.concatenate([m_fB, w_fB], axis=0)
diffs       = np.concatenate([m_diff, w_diff], axis=0)
y_all       = np.concatenate([m_y,  w_y],  axis=0)
seasons_all = np.concatenate([m_seasons, w_seasons], axis=0)

# NaN/Inf → 0
np.nan_to_num(feat_A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.nan_to_num(feat_B, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.nan_to_num(diffs,  nan=0.0, posinf=0.0, neginf=0.0, copy=False)

print(f"  训练样本: {len(y_all)}  (M={len(m_y)}, W={len(w_y)})")

scaler_team = StandardScaler().fit(np.concatenate([feat_A, feat_B], axis=0))
scaler_diff = StandardScaler().fit(diffs)

feat_A_n = np.nan_to_num(scaler_team.transform(feat_A).astype(np.float32))
feat_B_n = np.nan_to_num(scaler_team.transform(feat_B).astype(np.float32))
diffs_n  = np.nan_to_num(scaler_diff.transform(diffs).astype(np.float32))

TEAM_DIM = feat_A_n.shape[1]
DIFF_DIM = diffs_n.shape[1]
print(f"  归一化完成 (team_dim={TEAM_DIM}, diff_dim={DIFF_DIM})")


# ══════════════════════════════════════════════════════════════
# 2. Dataset + Augmentation
# ══════════════════════════════════════════════════════════════
class MatchupDataset(Dataset):
    def __init__(self, fA, fB, diff, y):
        self.fA   = torch.from_numpy(fA)
        self.fB   = torch.from_numpy(fB)
        self.diff = torch.from_numpy(diff)
        self.y    = torch.from_numpy(y).float()

    def __len__(self):  return len(self.y)

    def __getitem__(self, i):
        return self.fA[i], self.fB[i], self.diff[i], self.y[i]


def mixup_batch(fA, fB, diff, y, alpha):
    if alpha <= 0:
        return fA, fB, diff, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(len(y), device=y.device)
    return (lam*fA  + (1-lam)*fA[idx],
            lam*fB  + (1-lam)*fB[idx],
            lam*diff + (1-lam)*diff[idx],
            lam*y   + (1-lam)*y[idx])


def add_noise(x, std, mask_rate):
    if std > 0:
        x = x + torch.randn_like(x) * std
    if mask_rate > 0:
        x = x * torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
    return x


# ══════════════════════════════════════════════════════════════
# 3. BracketFormer v6 架构
# ══════════════════════════════════════════════════════════════
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        rand = torch.bernoulli(
            torch.full((x.shape[0],) + (1,)*(x.ndim-1), keep,
                       dtype=x.dtype, device=x.device)) / keep
        return x * rand


class SEBlock(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        hidden = max(dim // ratio, 8)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, dim), nn.Sigmoid())

    def forward(self, x): return x * self.gate(x)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout, se_ratio=4, drop_path=0.0):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.net       = nn.Sequential(
            nn.Linear(dim, dim*2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*2, dim),
            SEBlock(dim, se_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.net(self.norm(x)))


class TeamEncoder(nn.Module):
    def __init__(self, feat_dim, embed_dim, dropout, se_ratio=4, dp_rate=0.12):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim))
        dp_rates = [dp_rate * i / 3 for i in range(4)]
        self.blocks = nn.ModuleList(
            [ResBlock(embed_dim, dropout, se_ratio, dp) for dp in dp_rates])

    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks: h = blk(h)
        return h


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn_AB = nn.MultiheadAttention(embed_dim, n_heads,
                                              dropout=dropout, batch_first=True)
        self.attn_BA = nn.MultiheadAttention(embed_dim, n_heads,
                                              dropout=dropout, batch_first=True)
        self.norm_A1 = nn.LayerNorm(embed_dim)
        self.norm_B1 = nn.LayerNorm(embed_dim)

        def ffn(d, dr):
            return nn.Sequential(
                nn.Linear(d, d*4), nn.GELU(), nn.Dropout(dr),
                nn.Linear(d*4, d), nn.Dropout(dr))

        self.ffn_A   = ffn(embed_dim, dropout)
        self.ffn_B   = ffn(embed_dim, dropout)
        self.norm_A2 = nn.LayerNorm(embed_dim)
        self.norm_B2 = nn.LayerNorm(embed_dim)

    def forward(self, eA, eB):
        A, B = eA.unsqueeze(1), eB.unsqueeze(1)
        cAB, _ = self.attn_AB(A, B, B)
        cBA, _ = self.attn_BA(B, A, A)
        A = self.norm_A1(eA + cAB.squeeze(1))
        B = self.norm_B1(eB + cBA.squeeze(1))
        A = self.norm_A2(A + self.ffn_A(A))
        B = self.norm_B2(B + self.ffn_B(B))
        return A, B


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, eA, eB):
        seq = torch.stack([eA, eB], dim=1)
        out, _ = self.attn(seq, seq, seq)
        seq = self.norm1(seq + out)
        seq = self.norm2(seq + self.ffn(seq))
        return seq[:, 0], seq[:, 1]


class DiffMLP(nn.Module):
    """增强版 Diff 特征编码器 (v6: 更深、带残差连接)"""
    def __init__(self, diff_dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(diff_dim, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden*2))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.LayerNorm(hidden))
        # 残差快捷连接
        self.shortcut = nn.Linear(diff_dim, hidden) if diff_dim != hidden else nn.Identity()

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return h + self.shortcut(x)


class GatedFusion(nn.Module):
    """门控融合模块: 学习性地选择和加权不同信息源."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gate      = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU())
        self.norm      = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.gate(x) * self.transform(x))


class BracketFormer(nn.Module):
    """
    v6 架构:
      TeamBranch: Siamese TeamEncoder → 3×CrossAttn → 2×SelfAttn
      DiffBranch: DiffMLP (增强 v2 diff 特征) → 96dim (含残差)
      Fusion:     GatedFusion([eA, eB, eA-eB, eA*eB, d_diff]) → 384dim
      Head:       384 → 256 → 128 → 1
    """
    def __init__(self, team_dim, diff_dim, cfg):
        super().__init__()
        ed   = cfg['embed_dim']
        nh   = cfg['n_heads']
        drop = cfg['dropout']
        dp   = cfg.get('drop_path_rate', 0.0)
        dh   = cfg['diff_hidden']
        fd   = cfg['fusion_dim']

        self.encoder = TeamEncoder(team_dim, ed, drop, cfg['se_ratio'], dp)

        self.cross_layers = nn.ModuleList(
            [CrossAttentionLayer(ed, nh, drop)
             for _ in range(cfg['n_cross_layers'])])
        self.self_layers = nn.ModuleList(
            [SelfAttentionLayer(ed, nh, drop)
             for _ in range(cfg['n_self_layers'])])

        self.diff_mlp = DiffMLP(diff_dim, dh, drop)

        fuse_in = ed * 4 + dh      # eA, eB, eA-eB, eA*eB, d_diff
        self.fusion = GatedFusion(fuse_in, fd)

        self.head = nn.Sequential(
            nn.Linear(fd, 256), nn.GELU(), nn.Dropout(drop),
            nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(128, 1))

    def encode_teams(self, fA, fB):
        eA = self.encoder(fA)
        eB = self.encoder(fB)
        for l in self.cross_layers: eA, eB = l(eA, eB)
        for l in self.self_layers:  eA, eB = l(eA, eB)
        return eA, eB

    def forward(self, fA, fB, diff):
        """返回 logit (未经 sigmoid)"""
        eA, eB  = self.encode_teams(fA, fB)
        d       = self.diff_mlp(diff)
        raw     = torch.cat([eA, eB, eA - eB, eA * eB, d], dim=-1)
        fused   = self.fusion(raw)
        return self.head(fused).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 4. SAM 优化器 (Sharpness-Aware Minimization)
# ══════════════════════════════════════════════════════════════
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization: 寻找平坦最小值以提升泛化."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group.get("adaptive", False)
                       else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.param_groups[0].get("adaptive") else 1.0)
                 * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2)
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM uses first_step/second_step, not step()")


# ══════════════════════════════════════════════════════════════
# 5. 损失函数
# ══════════════════════════════════════════════════════════════
def brier_loss(logit, target, smooth=0.02):
    """MSE(sigmoid(logit), smoothed_target) = 直接优化 Brier Score"""
    t    = target * (1 - smooth) + 0.5 * smooth
    pred = torch.sigmoid(logit)
    return F.mse_loss(pred, t)


# ══════════════════════════════════════════════════════════════
# 6. 学习率调度
# ══════════════════════════════════════════════════════════════
def build_warmup_scheduler(optimizer, warmup_steps):
    def lr_fn(step):
        return min(1.0, step / max(warmup_steps, 1))
    return LambdaLR(optimizer, lr_fn)


# ══════════════════════════════════════════════════════════════
# 7. 训练 / 评估
# ══════════════════════════════════════════════════════════════
def train_epoch_sam(model, loader, optimizer, warmup_sched, cosine_sched,
                    cfg, warmup_done):
    """使用 SAM 优化器的训练循环 (两步前向-反向)"""
    model.train()
    total_loss = 0.0
    ns = cfg.get('feature_noise', 0.0)
    nm = cfg.get('feat_mask_rate', 0.0)

    for fA, fB, diff, y in loader:
        fA, fB, diff, y = (fA.to(DEVICE), fB.to(DEVICE),
                           diff.to(DEVICE), y.to(DEVICE))
        fA = add_noise(fA, ns, nm)
        fB = add_noise(fB, ns, nm)

        fA_m, fB_m, diff_m, y_m = mixup_batch(fA, fB, diff, y, cfg['mixup_alpha'])

        # SAM 第一步: 计算梯度 → 扰动参数
        logit = model(fA_m, fB_m, diff_m)
        loss  = brier_loss(logit, y_m, cfg['label_smooth'])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        optimizer.first_step(zero_grad=True)

        # SAM 第二步: 在扰动后重新计算梯度 → 更新参数
        logit2 = model(fA_m, fB_m, diff_m)
        loss2  = brier_loss(logit2, y_m, cfg['label_smooth'])
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        optimizer.second_step(zero_grad=True)

        if not warmup_done[0]:
            warmup_sched.step()
            if warmup_sched.get_last_lr()[0] >= cfg['lr']:
                warmup_done[0] = True

        total_loss += loss.item() * len(y)

    if warmup_done[0]:
        cosine_sched.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, temperature=1.0):
    model.eval()
    probs, logits, targets = [], [], []
    for fA, fB, diff, y in loader:
        fA, fB, diff = fA.to(DEVICE), fB.to(DEVICE), diff.to(DEVICE)
        logit_ab = model(fA, fB, diff)
        logit_ba = model(fB, fA, -diff)
        avg_logit = (logit_ab - logit_ba) / 2.0
        p = torch.sigmoid(avg_logit / temperature)
        logits.extend(avg_logit.cpu().numpy())
        probs.extend(p.cpu().numpy())
        targets.extend(y.numpy())
    probs   = np.array(probs)
    logits  = np.array(logits)
    targets = np.array(targets)
    return (brier_score_loss(targets, probs),
            roc_auc_score(targets, probs),
            probs, logits)


# ══════════════════════════════════════════════════════════════
# 8. 双校准: 温度缩放 + Platt Scaling
# ══════════════════════════════════════════════════════════════
def fit_temperature(oof_logits, y_true):
    def brier_fn(T):
        if T <= 0: return 1.0
        p = 1.0 / (1.0 + np.exp(-oof_logits / T))
        return np.mean((p - y_true) ** 2)
    result = minimize_scalar(brier_fn, bounds=(0.1, 5.0), method='bounded')
    T_star = float(result.x)
    p_cal  = 1.0 / (1.0 + np.exp(-oof_logits / T_star))
    brier  = brier_score_loss(y_true, p_cal)
    return T_star, brier, p_cal


def fit_platt(oof_logits, y_true):
    """Platt Scaling: sigmoid(a * logit + b), 2 参数"""
    lr = LogisticRegression(C=1e6, max_iter=1000, solver='lbfgs')
    lr.fit(oof_logits.reshape(-1, 1), y_true.astype(int))
    probs = lr.predict_proba(oof_logits.reshape(-1, 1))[:, 1]
    brier = brier_score_loss(y_true, probs)
    return lr, brier, probs


def choose_best_calibration(oof_logits, y_true):
    """比较温度缩放和 Platt 缩放, 选择最优"""
    T_star, brier_temp, p_temp = fit_temperature(oof_logits, y_true)
    platt_model, brier_platt, p_platt = fit_platt(oof_logits, y_true)

    brier_raw = brier_score_loss(y_true, 1.0/(1.0+np.exp(-oof_logits)))

    print(f"\n  [校准比较]")
    print(f"    原始 Brier:      {brier_raw:.5f}")
    print(f"    温度缩放 (T={T_star:.3f}): {brier_temp:.5f}")
    print(f"    Platt 缩放:      {brier_platt:.5f}")

    if brier_temp <= brier_platt:
        print(f"    ✓ 选择: 温度缩放")
        return 'temperature', T_star, brier_temp, p_temp
    else:
        print(f"    ✓ 选择: Platt 缩放")
        return 'platt', platt_model, brier_platt, p_platt


# ══════════════════════════════════════════════════════════════
# 9. 5 折交叉验证训练
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"开始 5 折交叉验证训练 (BracketFormer v6, feats={feat_version})")
print("=" * 65)

# GroupKFold 按赛季分组: 同一赛季的比赛全部在同一折内，避免时序泄漏
gkf   = GroupKFold(n_splits=CFG['n_folds'])
y_int = y_all.astype(int)

oof_probs      = np.zeros(len(y_all))
oof_logits_all = np.zeros(len(y_all))
fold_briers    = []
fold_aucs      = []
fold_snapshots = []

t0 = time.time()

for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat_A_n, y_int, groups=seasons_all)):
    print(f"\n  Fold {fold+1}/{CFG['n_folds']}  "
          f"(train={len(tr_idx)}x2={len(tr_idx)*2}, val={len(val_idx)})")

    # 翻转增强
    tr_fA   = np.concatenate([feat_A_n[tr_idx], feat_B_n[tr_idx]])
    tr_fB   = np.concatenate([feat_B_n[tr_idx], feat_A_n[tr_idx]])
    tr_diff = np.concatenate([diffs_n[tr_idx], -diffs_n[tr_idx]])
    tr_y    = np.concatenate([y_all[tr_idx], 1.0 - y_all[tr_idx]])

    tr_ds   = MatchupDataset(tr_fA, tr_fB, tr_diff, tr_y)
    val_ds  = MatchupDataset(feat_A_n[val_idx], feat_B_n[val_idx],
                             diffs_n[val_idx], y_all[val_idx])

    tr_loader  = DataLoader(tr_ds,  CFG['batch_size'], shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, CFG['batch_size'], shuffle=False,
                            num_workers=0, pin_memory=True)

    model = BracketFormer(TEAM_DIM, DIFF_DIM, CFG).to(DEVICE)

    # SAM 优化器
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=CFG['sam_rho'],
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay'])

    warmup_sched = build_warmup_scheduler(optimizer.base_optimizer, CFG['warmup_steps'])
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer.base_optimizer, T_0=CFG['cosine_T0'],
        T_mult=CFG['cosine_Tmult'], eta_min=1e-6)
    warmup_done = [False]

    swa_model   = AveragedModel(model)
    swa_start   = int(CFG['epochs'] * CFG['swa_start_ratio'])
    swa_started = False

    snapshots  = []
    best_brier = 1.0
    no_improve = 0

    for ep in range(1, CFG['epochs'] + 1):
        tr_loss = train_epoch_sam(model, tr_loader, optimizer,
                                  warmup_sched, cosine_sched, CFG, warmup_done)

        if ep >= swa_start and not swa_started:
            for pg in optimizer.base_optimizer.param_groups:
                pg['lr'] = CFG['swa_lr']
            swa_started = True
        if swa_started:
            swa_model.update_parameters(model)

        val_brier, val_auc, _, _ = evaluate(model, val_loader)

        snapshots.append((val_brier, {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}))
        snapshots.sort(key=lambda x: x[0])
        snapshots = snapshots[:CFG['n_snapshots']]

        if val_brier < best_brier:
            best_brier = val_brier
            no_improve = 0
        else:
            no_improve += 1

        if ep % 40 == 0 or ep == 1:
            swa_tag = "[SWA]" if swa_started else ""
            print(f"    Ep{ep:3d}{swa_tag}  loss={tr_loss:.5f}  "
                  f"Brier={val_brier:.4f}  AUC={val_auc:.4f}  "
                  f"best={best_brier:.4f}")

        if no_improve >= CFG['patience']:
            print(f"    早停 ep={ep}")
            break

    # SWA 评估
    if swa_started:
        update_bn(tr_loader, swa_model, device=DEVICE)
        swa_b, swa_auc, _, _ = evaluate(swa_model, val_loader)
        print(f"    SWA  Brier={swa_b:.4f}  AUC={swa_auc:.4f}")
        if swa_b < snapshots[0][0]:
            swa_st = {k: v.cpu().clone()
                      for k, v in swa_model.module.state_dict().items()}
            snapshots.insert(0, (swa_b, swa_st))
            snapshots = snapshots[:CFG['n_snapshots']]

    # Snapshot 集成
    snap_probs, snap_logits_list = [], []
    for _, state in snapshots:
        m = BracketFormer(TEAM_DIM, DIFF_DIM, CFG).to(DEVICE)
        m.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
        _, _, p, lg = evaluate(m, val_loader)
        snap_probs.append(p)
        snap_logits_list.append(lg)

    avg_prob  = np.mean(snap_probs, axis=0)
    avg_logit = np.mean(snap_logits_list, axis=0)
    ens_brier = brier_score_loss(y_all[val_idx], avg_prob)
    ens_auc   = roc_auc_score(y_all[val_idx], avg_prob)

    oof_probs[val_idx]      = avg_prob
    oof_logits_all[val_idx] = avg_logit
    fold_briers.append(ens_brier)
    fold_aucs.append(ens_auc)
    fold_snapshots.append(snapshots)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Fold {fold+1}  Ens-Brier={ens_brier:.4f}  "
          f"AUC={ens_auc:.4f}  params={total_p:,}  snaps={len(snapshots)}")

oof_brier = brier_score_loss(y_all, oof_probs)
oof_auc   = roc_auc_score(y_all, oof_probs)
print(f"\n{'='*65}")
print(f"OOF Brier (raw)  = {oof_brier:.5f}   OOF AUC = {oof_auc:.4f}")
print(f"各折 Brier: {[f'{b:.4f}' for b in fold_briers]}")
print(f"耗时: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════
# 10. 双校准
# ══════════════════════════════════════════════════════════════
cal_type, cal_obj, cal_brier, oof_probs_cal = choose_best_calibration(
    oof_logits_all, y_all)
print(f"OOF Brier (校准) = {cal_brier:.5f}")


# ══════════════════════════════════════════════════════════════
# 11. 生成提交文件
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
n_ens = sum(len(s) for s in fold_snapshots)
print(f"生成提交文件（{n_ens} 模型集成 × 校准 × 对称 TTA）...")

sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmissionStage1.csv')
sub_df     = sample_sub.copy()
sub_df[['Season', 'TeamA', 'TeamB']] = (
    sub_df['ID'].str.split('_', expand=True).astype(int))
sub_df['Gender'] = np.where(sub_df['TeamA'] < 3000, 'M', 'W')


def build_cache(feat_df, cols):
    cache = {}
    for row in feat_df.itertuples():
        cache[(row.Season, row.TeamID)] = np.array(
            [getattr(row, c) for c in cols], dtype=np.float32)
    return cache


def look(cache, feat_df, cols, season, tid):
    k = (season, tid)
    if k in cache: return cache[k]
    sub = feat_df[(feat_df['TeamID']==tid) & (feat_df['Season']<=season)]
    if sub.empty: sub = feat_df[feat_df['TeamID']==tid]
    if sub.empty: return np.zeros(len(cols), np.float32)
    return sub.sort_values('Season').iloc[-1][cols].values.astype(np.float32)


m_cache  = build_cache(m_feat, common_team_cols)
w_cache  = build_cache(w_feat, common_team_cols)

BATCH   = 4096
all_sub = []

for (season, gender), grp in sub_df.groupby(['Season', 'Gender']):
    cache    = m_cache  if gender=='M' else w_cache
    feat_df_ = m_feat   if gender=='M' else w_feat

    n = len(grp)
    fA_raw = np.zeros((n, TEAM_DIM), np.float32)
    fB_raw = np.zeros((n, TEAM_DIM), np.float32)
    df_arr = np.zeros((n, DIFF_DIM), np.float32)

    grp_list = list(grp.itertuples())
    for i, r in enumerate(grp_list):
        fA_raw[i] = look(cache, feat_df_, common_team_cols, season, r.TeamA)
        fB_raw[i] = look(cache, feat_df_, common_team_cols, season, r.TeamB)
        for j, dc in enumerate(common_diff_cols):
            feat_name = dc[5:]  # strip "Diff_"
            if feat_name in common_team_cols:
                fi = common_team_cols.index(feat_name)
                df_arr[i, j] = fA_raw[i, fi] - fB_raw[i, fi]

    fA_n = np.nan_to_num(scaler_team.transform(fA_raw).astype(np.float32))
    fB_n = np.nan_to_num(scaler_team.transform(fB_raw).astype(np.float32))
    df_n = np.nan_to_num(scaler_diff.transform(df_arr).astype(np.float32))

    batch_preds_logit = []
    for start in range(0, n, BATCH):
        end_   = min(start + BATCH, n)
        fAt    = torch.from_numpy(fA_n[start:end_]).to(DEVICE)
        fBt    = torch.from_numpy(fB_n[start:end_]).to(DEVICE)
        dt     = torch.from_numpy(df_n[start:end_]).to(DEVICE)

        fold_logits = []
        for snapshots in fold_snapshots:
            for _, state in snapshots:
                m = BracketFormer(TEAM_DIM, DIFF_DIM, CFG).to(DEVICE)
                m.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
                m.eval()
                with torch.no_grad():
                    logit_ab = m(fAt, fBt,  dt)
                    logit_ba = m(fBt, fAt, -dt)
                    avg_logit = (logit_ab - logit_ba) / 2.0
                fold_logits.append(avg_logit.cpu().numpy())
        batch_preds_logit.extend(np.mean(fold_logits, axis=0).tolist())

    # 校准
    logits_arr = np.array(batch_preds_logit)
    if cal_type == 'temperature':
        T_star = cal_obj
        preds = 1.0 / (1.0 + np.exp(-logits_arr / T_star))
    else:  # platt
        platt_model = cal_obj
        preds = platt_model.predict_proba(logits_arr.reshape(-1, 1))[:, 1]

    r_df        = grp[['ID']].copy()
    r_df['Pred'] = preds
    all_sub.append(r_df)
    print(f"  Season={season} {gender}  rows={n}  "
          f"mean={np.mean(preds):.4f}")

sub_final = pd.concat(all_sub).sort_index().reset_index(drop=True)
sub_final['Pred'] = sub_final['Pred'].clip(0.01, 0.99).round(6)
assert len(sub_final) == 519144, f"行数错误: {len(sub_final)}"

out_path = DATA_DIR / 'submission_dl.csv'
sub_final.to_csv(out_path, index=False)
print(f"\n  提交文件: {out_path}")
print(f"  行数: {len(sub_final):,}  "
      f"min={sub_final['Pred'].min():.4f}  "
      f"max={sub_final['Pred'].max():.4f}  "
      f"mean={sub_final['Pred'].mean():.4f}")


# ══════════════════════════════════════════════════════════════
# 12. 可视化
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('BracketFormer v6 — 训练结果汇总', fontsize=14, fontweight='bold')

axes[0].bar(range(1, CFG['n_folds']+1), fold_briers,
            color='#3498db', alpha=0.85, edgecolor='white')
axes[0].axhline(oof_brier,  color='red',   ls='--', lw=2,
                label=f'OOF(raw)={oof_brier:.4f}')
axes[0].axhline(cal_brier,  color='green', ls='--', lw=2,
                label=f'OOF(cal)={cal_brier:.4f}')
axes[0].set_xlabel('Fold'); axes[0].set_ylabel('Brier Score')
axes[0].set_title('各折 Brier Score'); axes[0].legend(fontsize=8)
axes[0].set_ylim(min(fold_briers)*0.97, max(fold_briers)*1.02)
for i, v in enumerate(fold_briers):
    axes[0].text(i+1, v+0.0003, f'{v:.4f}', ha='center', fontsize=8)

axes[1].hist(oof_probs_cal[y_all==1], bins=40, alpha=0.6,
             color='#2ecc71', label='实际胜方')
axes[1].hist(oof_probs_cal[y_all==0], bins=40, alpha=0.6,
             color='#e74c3c', label='实际负方')
axes[1].set_xlabel('校准后预测概率'); axes[1].set_ylabel('频次')
axes[1].set_title('OOF 预测分布 (校准后)'); axes[1].legend()

tmp   = BracketFormer(TEAM_DIM, DIFF_DIM, CFG)
npar  = sum(p.numel() for p in tmp.parameters())
info  = (
    f"BracketFormer v6\n\n"
    f"team_dim:  {TEAM_DIM}\n"
    f"diff_dim:  {DIFF_DIM}  (v2 FE)\n"
    f"embed_dim: {CFG['embed_dim']}\n"
    f"cross x{CFG['n_cross_layers']} + self x{CFG['n_self_layers']}\n"
    f"GatedFusion → {CFG['fusion_dim']}\n"
    f"DropPath={CFG['drop_path_rate']}\n"
    f"params: {npar:,}\n"
    f"ens:    {n_ens} models\n\n"
    f"[NEW] SAM rho={CFG['sam_rho']}\n"
    f"[NEW] v2 feats (Four Factors+SOS)\n"
    f"[NEW] GatedFusion\n"
    f"[NEW] {cal_type} calibration\n\n"
    f"OOF Brier (raw) = {oof_brier:.4f}\n"
    f"OOF Brier (cal) = {cal_brier:.4f}\n"
    f"OOF AUC         = {oof_auc:.4f}\n"
    f"device: {DEVICE}"
)
axes[2].text(0.05, 0.97, info, transform=axes[2].transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.85))
axes[2].axis('off'); axes[2].set_title('模型信息')

plt.tight_layout()
plot_path = DATA_DIR / 'bracketformer_result.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  图表: {plot_path}")
print("=" * 65)
print("全部完成!")
print(f"  submission_dl.csv        -> {out_path}")
print(f"  bracketformer_result.png -> {plot_path}")
