"""
NCAA March Madness — BracketFormer v5
======================================
深度利用 notebook 特征工程数据 + 三大核心优化：

  [优化1] 直接优化 Brier Score（MSE loss），与 Kaggle 评测完全一致
  [优化2] 温度缩放校准（Temperature Scaling）：在 OOF logit 上拟合 T*，
          使预测分布更锋利、校准更准确
  [优化4] 充分利用种子(SeedNum)、Elo、Massey、教练胜率等强信号，
          已在 notebook 特征工程中预计算

  [新架构] 双分支 Transformer + Diff 分支融合：
    - TeamEncoder (Siamese, 37维) x 4×ResBlock(SE+DropPath) → 128dim
    - CrossAttention x 3 + SelfAttention x 2
    - DiffMLP (37维预计算差分，直接来自 M/W_train_full_history) → 64dim
    - Fusion: [eA, eB, eA-eB, eA*eB, d_diff]  (512+64=576)
    - Head: 576→512→256→128→1  (logit，不加 sigmoid → 方便温度缩放)

  [训练] 翻转增强 × 特征噪声 × Mixup × Warmup+CosineWarmRestarts
         SWA + 快照集成(最多35模型) × 对称 TTA × 温度缩放

Output:
  submission_dl.csv      (519,144 行，Stage1 格式)
  bracketformer_result.png
"""

import os, time, warnings, pathlib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize_scalar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, update_bn

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

# ── 中文字体修复 ────────────────────────────────────────────────────
for _font in ['Microsoft YaHei', 'SimHei', 'STSong', 'WenQuanYi Micro Hei']:
    try:
        rcParams['font.sans-serif'] = [_font] + rcParams.get('font.sans-serif', [])
        plt.rcParams['font.sans-serif'] = [_font]
        break
    except Exception:
        pass
rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────────────────────────
# 0. 全局配置  (v5)
# ──────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).resolve().parent
FEAT_DIR = DATA_DIR / 'features'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = dict(
    # 模型结构
    embed_dim      = 128,
    n_heads        = 8,
    n_cross_layers = 3,
    n_self_layers  = 2,
    se_ratio       = 4,
    dropout        = 0.20,
    drop_path_rate = 0.08,
    diff_hidden    = 64,    # DiffMLP 隐藏层

    # 训练
    lr             = 1.5e-4,
    weight_decay   = 2e-4,
    epochs         = 400,
    warmup_steps   = 100,
    cosine_T0      = 60,
    cosine_Tmult   = 2,
    batch_size     = 512,
    n_folds        = 5,
    seed           = 42,
    clip_grad      = 1.0,

    # 正则化 & 损失
    mixup_alpha    = 0.4,
    label_smooth   = 0.015,
    feature_noise  = 0.04,
    feat_mask_rate = 0.04,

    # SWA
    swa_start_ratio = 0.65,
    swa_lr          = 3e-5,

    # 快照 & 早停
    n_snapshots    = 7,
    patience       = 80,
)

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
print(f"设备: {DEVICE}  (PyTorch {torch.__version__})")
print("=" * 65)


# ──────────────────────────────────────────────────────────────────
# 1. 数据加载：充分利用 notebook 特征工程结果
# ──────────────────────────────────────────────────────────────────
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


print("加载特征文件...")
m_feat, m_cols = load_team_features(FEAT_DIR / 'M_team_features_final.csv')
w_feat, w_cols = load_team_features(FEAT_DIR / 'W_team_features_final.csv')

m_hist = pd.read_csv(FEAT_DIR / 'M_train_full_history.csv')
w_hist = pd.read_csv(FEAT_DIR / 'W_train_full_history.csv')

common_team_cols = [c for c in m_cols if c in w_cols]
m_diff_cols = [c for c in m_hist.columns if c.startswith('Diff_')]
w_diff_cols = [c for c in w_hist.columns if c.startswith('Diff_')]
common_diff_cols = [c for c in m_diff_cols if c in w_diff_cols]

print(f"  公共球队特征: {len(common_team_cols)}  公共 Diff 特征: {len(common_diff_cols)}")
print(f"  Diff 特征示例: {common_diff_cols[:5]} ...")

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

# 原始数据清洗：NaN/Inf → 0
np.nan_to_num(feat_A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.nan_to_num(feat_B, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.nan_to_num(diffs,  nan=0.0, posinf=0.0, neginf=0.0, copy=False)

print(f"  训练样本: {len(y_all)}  (M={len(m_y)}, W={len(w_y)})")
print(f"  球队特征: {feat_A.shape[1]}  Diff 特征: {diffs.shape[1]}")

scaler_team = StandardScaler().fit(np.concatenate([feat_A, feat_B], axis=0))
scaler_diff = StandardScaler().fit(diffs)

feat_A_n = scaler_team.transform(feat_A).astype(np.float32)
feat_B_n = scaler_team.transform(feat_B).astype(np.float32)
diffs_n  = scaler_diff.transform(diffs).astype(np.float32)

# 缩放后再次填充（常数列 std=0 → NaN）
np.nan_to_num(feat_A_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.nan_to_num(feat_B_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.nan_to_num(diffs_n,  nan=0.0, posinf=0.0, neginf=0.0, copy=False)

TEAM_DIM = feat_A_n.shape[1]
DIFF_DIM = diffs_n.shape[1]
print(f"  归一化完成 (team_dim={TEAM_DIM}, diff_dim={DIFF_DIM})")


# ──────────────────────────────────────────────────────────────────
# 2. Dataset + Mixup
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# 3. BracketFormer v5 架构
# ──────────────────────────────────────────────────────────────────
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
    def __init__(self, feat_dim, embed_dim, dropout, se_ratio=4, dp_rate=0.08):
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
    """专用 Diff 特征编码器（notebook 预计算 Diff_* 特征）"""
    def __init__(self, diff_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(diff_dim, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.LayerNorm(hidden))

    def forward(self, x): return self.net(x)


class BracketFormer(nn.Module):
    """
    v5 双分支架构：
      TeamBranch: Siamese TeamEncoder → 3×CrossAttn → 2×SelfAttn
      DiffBranch:  DiffMLP(notebook预计算Diff) → 64dim
      Fusion: [eA, eB, eA-eB, eA*eB, d_diff] → Head → logit
    """
    def __init__(self, team_dim, diff_dim, cfg):
        super().__init__()
        ed   = cfg['embed_dim']
        nh   = cfg['n_heads']
        drop = cfg['dropout']
        dp   = cfg.get('drop_path_rate', 0.0)
        dh   = cfg['diff_hidden']

        self.encoder = TeamEncoder(team_dim, ed, drop, cfg['se_ratio'], dp)

        self.cross_layers = nn.ModuleList(
            [CrossAttentionLayer(ed, nh, drop)
             for _ in range(cfg['n_cross_layers'])])
        self.self_layers = nn.ModuleList(
            [SelfAttentionLayer(ed, nh, drop)
             for _ in range(cfg['n_self_layers'])])

        self.diff_mlp = DiffMLP(diff_dim, dh, drop)

        fuse_dim = ed * 4 + dh      # eA, eB, eA-eB, eA*eB, d_diff
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, 512), nn.GELU(), nn.Dropout(drop),
            nn.LayerNorm(512),
            nn.Linear(512, 256),      nn.GELU(), nn.Dropout(drop),
            nn.LayerNorm(256),
            nn.Linear(256, 128),      nn.GELU(), nn.Dropout(drop*0.5),
            nn.Linear(128, 1))

    def encode_teams(self, fA, fB):
        eA = self.encoder(fA)
        eB = self.encoder(fB)
        for l in self.cross_layers: eA, eB = l(eA, eB)
        for l in self.self_layers:  eA, eB = l(eA, eB)
        return eA, eB

    def forward(self, fA, fB, diff):
        """返回 logit（未经 sigmoid），便于温度缩放"""
        eA, eB  = self.encode_teams(fA, fB)
        d       = self.diff_mlp(diff)
        fused   = torch.cat([eA, eB, eA - eB, eA * eB, d], dim=-1)
        return self.head(fused).squeeze(-1)


# ──────────────────────────────────────────────────────────────────
# 4. [优化1] 直接 Brier Score 作为训练损失
# ──────────────────────────────────────────────────────────────────
def brier_loss(logit, target, smooth=0.015):
    """MSE(sigmoid(logit), smoothed_target) = 直接优化 Brier Score"""
    t    = target * (1 - smooth) + 0.5 * smooth
    pred = torch.sigmoid(logit)
    return F.mse_loss(pred, t)


# ──────────────────────────────────────────────────────────────────
# 5. 学习率调度：线性 warmup → CosineWarmRestarts
# ──────────────────────────────────────────────────────────────────
def build_warmup_scheduler(optimizer, warmup_steps):
    def lr_fn(step):
        return min(1.0, step / max(warmup_steps, 1))
    return LambdaLR(optimizer, lr_fn)


# ──────────────────────────────────────────────────────────────────
# 6. 训练 / 评估
# ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, warmup_sched, cosine_sched,
                cfg, warmup_done):
    model.train()
    total_loss = 0.0
    ns = cfg.get('feature_noise', 0.0)
    nm = cfg.get('feat_mask_rate', 0.0)

    for fA, fB, diff, y in loader:
        fA, fB, diff, y = (fA.to(DEVICE), fB.to(DEVICE),
                           diff.to(DEVICE), y.to(DEVICE))
        fA   = add_noise(fA,   ns, nm)
        fB   = add_noise(fB,   ns, nm)

        fA_m, fB_m, diff_m, y_m = mixup_batch(fA, fB, diff, y, cfg['mixup_alpha'])

        optimizer.zero_grad()
        logit = model(fA_m, fB_m, diff_m)
        loss  = brier_loss(logit, y_m, cfg['label_smooth'])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        optimizer.step()

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


# ──────────────────────────────────────────────────────────────────
# 7. [优化2] 温度缩放
# ──────────────────────────────────────────────────────────────────
def fit_temperature(oof_logits, y_true):
    def brier_fn(T):
        if T <= 0: return 1.0
        p = 1.0 / (1.0 + np.exp(-oof_logits / T))
        return np.mean((p - y_true) ** 2)

    result = minimize_scalar(brier_fn, bounds=(0.1, 5.0), method='bounded')
    T_star = float(result.x)
    p_cal  = 1.0 / (1.0 + np.exp(-oof_logits / T_star))
    brier_before = brier_score_loss(y_true, 1.0/(1.0+np.exp(-oof_logits)))
    brier_after  = brier_score_loss(y_true, p_cal)
    print(f"\n  [温度缩放] T* = {T_star:.4f}")
    print(f"  OOF Brier 校准前: {brier_before:.5f}  校准后: {brier_after:.5f}")
    return T_star


# ──────────────────────────────────────────────────────────────────
# 8. 5 折交叉验证训练
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("开始 5 折交叉验证训练 (BracketFormer v5)")
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

    model     = BracketFormer(TEAM_DIM, DIFF_DIM, CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CFG['lr'],
                                   weight_decay=CFG['weight_decay'])

    warmup_sched = build_warmup_scheduler(optimizer, CFG['warmup_steps'])
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG['cosine_T0'], T_mult=CFG['cosine_Tmult'], eta_min=1e-6)
    warmup_done = [False]

    swa_model   = AveragedModel(model)
    swa_start   = int(CFG['epochs'] * CFG['swa_start_ratio'])
    swa_started = False

    snapshots  = []
    best_brier = 1.0
    no_improve = 0

    for ep in range(1, CFG['epochs'] + 1):
        tr_loss = train_epoch(model, tr_loader, optimizer,
                              warmup_sched, cosine_sched, CFG, warmup_done)

        if ep >= swa_start and not swa_started:
            for pg in optimizer.param_groups: pg['lr'] = CFG['swa_lr']
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

    if swa_started:
        update_bn(tr_loader, swa_model, device=DEVICE)
        swa_b, swa_auc, _, _ = evaluate(swa_model, val_loader)
        print(f"    SWA  Brier={swa_b:.4f}  AUC={swa_auc:.4f}")
        if swa_b < snapshots[0][0]:
            swa_st = {k: v.cpu().clone()
                      for k, v in swa_model.module.state_dict().items()}
            snapshots.insert(0, (swa_b, swa_st))
            snapshots = snapshots[:CFG['n_snapshots']]

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


# ──────────────────────────────────────────────────────────────────
# 9. [优化2] 温度缩放校准
# ──────────────────────────────────────────────────────────────────
T_STAR = fit_temperature(oof_logits_all, y_all)

oof_probs_cal = 1.0 / (1.0 + np.exp(-oof_logits_all / T_STAR))
oof_brier_cal = brier_score_loss(y_all, oof_probs_cal)
print(f"OOF Brier (校准) = {oof_brier_cal:.5f}")


# ──────────────────────────────────────────────────────────────────
# 10. 生成提交文件（35 模型集成 + 温度缩放 + 对称 TTA）
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("生成提交文件（35 模型集成 x 温度缩放 x 对称 TTA）...")

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
        # diff 近似：用队伍特征差值
        for j, dc in enumerate(common_diff_cols):
            feat_name = dc[5:]  # strip "Diff_"
            if feat_name in common_team_cols:
                fi = common_team_cols.index(feat_name)
                df_arr[i, j] = fA_raw[i, fi] - fB_raw[i, fi]

    fA_n = scaler_team.transform(fA_raw).astype(np.float32)
    fB_n = scaler_team.transform(fB_raw).astype(np.float32)
    df_n = scaler_diff.transform(df_arr).astype(np.float32)
    np.nan_to_num(fA_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(fB_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(df_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    batch_preds = []
    for start in range(0, n, BATCH):
        end_   = min(start + BATCH, n)
        fAt    = torch.from_numpy(fA_n[start:end_]).to(DEVICE)
        fBt    = torch.from_numpy(fB_n[start:end_]).to(DEVICE)
        dt     = torch.from_numpy(df_n[start:end_]).to(DEVICE)

        fold_p = []
        for snapshots in fold_snapshots:
            for _, state in snapshots:
                m = BracketFormer(TEAM_DIM, DIFF_DIM, CFG).to(DEVICE)
                m.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
                m.eval()
                with torch.no_grad():
                    logit_ab = m(fAt, fBt,  dt)
                    logit_ba = m(fBt, fAt, -dt)
                    avg_logit = (logit_ab - logit_ba) / 2.0
                    p = torch.sigmoid(avg_logit / T_STAR).cpu().numpy()
                fold_p.append(p)
        batch_preds.extend(np.mean(fold_p, axis=0).tolist())

    r_df        = grp[['ID']].copy()
    r_df['Pred'] = batch_preds
    all_sub.append(r_df)
    print(f"  Season={season} {gender}  rows={n}  "
          f"mean={np.mean(batch_preds):.4f}")

sub_final = pd.concat(all_sub).sort_index().reset_index(drop=True)
sub_final['Pred'] = sub_final['Pred'].clip(0.02, 0.98).round(6)
assert len(sub_final) == 519144, f"行数错误: {len(sub_final)}"

out_path = DATA_DIR / 'submission_dl.csv'
sub_final.to_csv(out_path, index=False)
print(f"\n  提交文件: {out_path}")
print(f"  行数: {len(sub_final):,}  "
      f"min={sub_final['Pred'].min():.4f}  "
      f"max={sub_final['Pred'].max():.4f}  "
      f"mean={sub_final['Pred'].mean():.4f}")


# ──────────────────────────────────────────────────────────────────
# 11. 可视化
# ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('BracketFormer v5 -- 训练结果汇总', fontsize=14, fontweight='bold')

axes[0].bar(range(1, CFG['n_folds']+1), fold_briers,
            color='#3498db', alpha=0.85, edgecolor='white')
axes[0].axhline(oof_brier,     color='red',   ls='--', lw=2,
                label=f'OOF(raw)={oof_brier:.4f}')
axes[0].axhline(oof_brier_cal, color='green', ls='--', lw=2,
                label=f'OOF(cal)={oof_brier_cal:.4f}')
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
axes[1].set_title('OOF 预测分布 (温度缩放后)'); axes[1].legend()

tmp   = BracketFormer(TEAM_DIM, DIFF_DIM, CFG)
npar  = sum(p.numel() for p in tmp.parameters())
n_ens = sum(len(s) for s in fold_snapshots)
info  = (
    f"BracketFormer v5\n\n"
    f"team_dim:  {TEAM_DIM}\n"
    f"diff_dim:  {DIFF_DIM}  (notebook FE)\n"
    f"embed_dim: {CFG['embed_dim']}\n"
    f"cross x{CFG['n_cross_layers']} + self x{CFG['n_self_layers']}\n"
    f"DropPath={CFG['drop_path_rate']}\n"
    f"params: {npar:,}\n"
    f"ens:    {n_ens} models\n\n"
    f"[1] Loss = Brier Score\n"
    f"[2] Temp T* = {T_STAR:.4f}\n"
    f"[4] SeedNum+Elo+Coach\n\n"
    f"OOF Brier (raw) = {oof_brier:.4f}\n"
    f"OOF Brier (cal) = {oof_brier_cal:.4f}\n"
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
