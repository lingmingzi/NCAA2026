"""
feature_enhancement.py — NCAA March Madness 高级特征工程 v2
==========================================================
在 notebook 现有特征基础上，计算 18 个高级篮球分析特征：

[Dean Oliver's Four Factors]
  1. eFGPct      — 有效投篮命中率 = (FGM + 0.5×FGM3) / FGA
  2. TOVPct      — 失误率 = TO / (FGA + 0.44×FTA + TO)
  3. ORBPct      — 进攻篮板率 = OR / (OR + OppDR)
  4. FTRate      — 罚球率 = FTA / FGA
  5. Opp_eFGPct  — 对手有效投篮命中率（防守质量）
  6. DRBPct      — 防守篮板率 = DR / (DR + OppOR)
  7. Opp_FTRate  — 对手罚球率（犯规控制）

[Opponent-Adjusted Metrics]
  8. SOS         — 赛程强度 = mean(对手 Elo)
  9. AdjOffEff   — Pomeroy 式调整进攻效率
 10. AdjDefEff   — Pomeroy 式调整防守效率

[Consistency & Clutch]
 11. ScoreStd    — 得分标准差（一致性）
 12. CloseWinRate— 接近比赛胜率（|分差| ≤ 5）
 13. PythagWin   — 毕达哥拉斯期望胜率
 14. AstTORatio  — 助攻/失误比

[Tempo]
 15. Pace        — 平均每场回合数

[Recent Form (Last 5 Games)]
 16. Last5_eFGPct — 最近 5 场有效投篮%
 17. Last5_OffEff — 最近 5 场进攻效率
 18. Last5_DefEff — 最近 5 场防守效率

用法: python feature_enhancement.py
输出: features/*_v2.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

DATA_DIR = Path(__file__).resolve().parent
FEAT_DIR = DATA_DIR / 'features'


# ═══════════════════════════════════════════════════════════════
# 1. 从紧凑结果计算基础高级特征 (覆盖所有赛季)
# ═══════════════════════════════════════════════════════════════
def compute_compact_features(compact_df, elo_df):
    """
    从 CompactResults (全赛季) 计算:
      SOS, PythagWin, CloseWinRate, ScoreStd
    """
    # 拆分为每支球队每场比赛一行
    w = compact_df[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']].copy()
    w.columns = ['Season', 'TeamID', 'OppID', 'Score', 'OppScore']
    w['Win'] = 1

    l = compact_df[['Season', 'LTeamID', 'WTeamID', 'LScore', 'WScore']].copy()
    l.columns = ['Season', 'TeamID', 'OppID', 'Score', 'OppScore']
    l['Win'] = 0

    games = pd.concat([w, l], ignore_index=True)
    games['Margin'] = games['Score'] - games['OppScore']
    games['IsClose'] = games['Margin'].abs() <= 5

    # SOS: merge 对手赛季末 Elo
    games = games.merge(
        elo_df.rename(columns={'TeamID': 'OppID', 'Elo': 'OppElo'}),
        on=['Season', 'OppID'], how='left')
    games['OppElo'] = games['OppElo'].fillna(1500)

    # 聚合到 (Season, TeamID)
    agg = games.groupby(['Season', 'TeamID']).agg(
        SOS=('OppElo', 'mean'),
        SumScore=('Score', 'sum'),
        SumOppScore=('OppScore', 'sum'),
        ScoreStd=('Score', 'std'),
    ).reset_index()

    # Pythagorean expected win%: 1 / (1 + (OppPts/Pts)^10)
    ratio = (agg['SumOppScore'] / agg['SumScore'].clip(lower=1))
    agg['PythagWin'] = 1.0 / (1.0 + ratio ** 10)

    # Close game win rate
    close = games[games['IsClose']].groupby(['Season', 'TeamID']).agg(
        CloseWins=('Win', 'sum'),
        CloseGames=('Win', 'count')
    ).reset_index()
    close['CloseWinRate'] = close['CloseWins'] / close['CloseGames']

    agg = agg.merge(close[['Season', 'TeamID', 'CloseWinRate']],
                    on=['Season', 'TeamID'], how='left')
    agg['CloseWinRate'] = agg['CloseWinRate'].fillna(0.5)
    agg['ScoreStd'] = agg['ScoreStd'].fillna(0)

    return agg[['Season', 'TeamID', 'SOS', 'PythagWin', 'CloseWinRate', 'ScoreStd']]


# ═══════════════════════════════════════════════════════════════
# 2. 从详细结果计算 Four Factors 及高级统计
# ═══════════════════════════════════════════════════════════════
def build_game_stats(detailed_df):
    """向量化计算每场比赛每支球队的高级统计 (Four Factors + Efficiency)."""
    d = detailed_df

    # 回合数估算 (两队平均)
    w_poss = d['WFGA'] - d['WOR'] + d['WTO'] + 0.44 * d['WFTA']
    l_poss = d['LFGA'] - d['LOR'] + d['LTO'] + 0.44 * d['LFTA']
    poss = ((w_poss + l_poss) / 2).clip(lower=1)

    # ── 胜方记录 ──
    winner = pd.DataFrame({
        'Season': d['Season'], 'DayNum': d['DayNum'],
        'TeamID': d['WTeamID'], 'OppID': d['LTeamID'],
        'Win': 1,
        'Score': d['WScore'], 'OppScore': d['LScore'],
        'eFGPct':      (d['WFGM'] + 0.5 * d['WFGM3']) / d['WFGA'].clip(lower=1),
        'TOVPct':      d['WTO'] / (d['WFGA'] + 0.44 * d['WFTA'] + d['WTO']).clip(lower=1),
        'ORBPct':      d['WOR'] / (d['WOR'] + d['LDR']).clip(lower=1),
        'FTRate':      d['WFTA'] / d['WFGA'].clip(lower=1),
        'DRBPct':      d['WDR'] / (d['WDR'] + d['LOR']).clip(lower=1),
        'Opp_eFGPct':  (d['LFGM'] + 0.5 * d['LFGM3']) / d['LFGA'].clip(lower=1),
        'Opp_FTRate':  d['LFTA'] / d['LFGA'].clip(lower=1),
        'AstTORatio':  d['WAst'] / d['WTO'].clip(lower=1),
        'Poss':        poss,
        'OffEff_g':    d['WScore'] / poss * 100,
        'DefEff_g':    d['LScore'] / poss * 100,
    })

    # ── 负方记录 ──
    loser = pd.DataFrame({
        'Season': d['Season'], 'DayNum': d['DayNum'],
        'TeamID': d['LTeamID'], 'OppID': d['WTeamID'],
        'Win': 0,
        'Score': d['LScore'], 'OppScore': d['WScore'],
        'eFGPct':      (d['LFGM'] + 0.5 * d['LFGM3']) / d['LFGA'].clip(lower=1),
        'TOVPct':      d['LTO'] / (d['LFGA'] + 0.44 * d['LFTA'] + d['LTO']).clip(lower=1),
        'ORBPct':      d['LOR'] / (d['LOR'] + d['WDR']).clip(lower=1),
        'FTRate':      d['LFTA'] / d['LFGA'].clip(lower=1),
        'DRBPct':      d['LDR'] / (d['LDR'] + d['WOR']).clip(lower=1),
        'Opp_eFGPct':  (d['WFGM'] + 0.5 * d['WFGM3']) / d['WFGA'].clip(lower=1),
        'Opp_FTRate':  d['WFTA'] / d['WFGA'].clip(lower=1),
        'AstTORatio':  d['LAst'] / d['LTO'].clip(lower=1),
        'Poss':        poss,
        'OffEff_g':    d['LScore'] / poss * 100,
        'DefEff_g':    d['WScore'] / poss * 100,
    })

    return pd.concat([winner, loser], ignore_index=True)


def aggregate_advanced_stats(game_df):
    """聚合比赛级别统计到 (Season, TeamID) 级别."""
    agg = game_df.groupby(['Season', 'TeamID']).agg(
        eFGPct=('eFGPct', 'mean'),
        TOVPct=('TOVPct', 'mean'),
        ORBPct=('ORBPct', 'mean'),
        FTRate=('FTRate', 'mean'),
        DRBPct=('DRBPct', 'mean'),
        Opp_eFGPct=('Opp_eFGPct', 'mean'),
        Opp_FTRate=('Opp_FTRate', 'mean'),
        AstTORatio=('AstTORatio', 'mean'),
        Pace=('Poss', 'mean'),
        OffEff_adv=('OffEff_g', 'mean'),
        DefEff_adv=('DefEff_g', 'mean'),
    ).reset_index()

    return agg


# ═══════════════════════════════════════════════════════════════
# 3. Pomeroy 式对手调整效率
# ═══════════════════════════════════════════════════════════════
def compute_adjusted_efficiency(adv_stats, game_df):
    """
    AdjOffEff = OffEff × (league_avg_DefEff / mean_opp_DefEff)
    AdjDefEff = DefEff × (league_avg_OffEff / mean_opp_OffEff)
    """
    # 联赛平均效率 (per season)
    league = adv_stats.groupby('Season').agg(
        AvgOff=('OffEff_adv', 'mean'),
        AvgDef=('DefEff_adv', 'mean')
    ).reset_index()

    # 每支球队所遇对手的平均效率
    opp_eff = game_df[['Season', 'TeamID', 'OppID']].merge(
        adv_stats[['Season', 'TeamID', 'OffEff_adv', 'DefEff_adv']].rename(
            columns={'TeamID': 'OppID',
                     'OffEff_adv': 'OppOff',
                     'DefEff_adv': 'OppDef'}),
        on=['Season', 'OppID'], how='left')

    mean_opp = opp_eff.groupby(['Season', 'TeamID']).agg(
        MeanOppOff=('OppOff', 'mean'),
        MeanOppDef=('OppDef', 'mean')
    ).reset_index()

    adj = (adv_stats[['Season', 'TeamID', 'OffEff_adv', 'DefEff_adv']]
           .merge(league, on='Season')
           .merge(mean_opp, on=['Season', 'TeamID']))

    adj['AdjOffEff'] = adj['OffEff_adv'] * (adj['AvgDef'] / adj['MeanOppDef'].clip(lower=1))
    adj['AdjDefEff'] = adj['DefEff_adv'] * (adj['AvgOff'] / adj['MeanOppOff'].clip(lower=1))

    return adj[['Season', 'TeamID', 'AdjOffEff', 'AdjDefEff']]


# ═══════════════════════════════════════════════════════════════
# 4. 近期状态 (Last N Games)
# ═══════════════════════════════════════════════════════════════
def compute_last_n(game_df, n=5):
    """最近 N 场比赛的高级统计趋势."""
    df = game_df.sort_values(['Season', 'TeamID', 'DayNum'])
    last_n = df.groupby(['Season', 'TeamID']).tail(n)

    result = last_n.groupby(['Season', 'TeamID']).agg(
        Last5_eFGPct=('eFGPct', 'mean'),
        Last5_OffEff=('OffEff_g', 'mean'),
        Last5_DefEff=('DefEff_g', 'mean'),
    ).reset_index()

    return result


# ═══════════════════════════════════════════════════════════════
# 5. 合并增强特征
# ═══════════════════════════════════════════════════════════════
def merge_enhanced(existing_df, compact_feats, adv_stats, adj_eff, last_n):
    """将所有新特征合并到现有特征表."""
    df = existing_df.copy()

    # Compact features: SOS, PythagWin, CloseWinRate, ScoreStd
    df = df.merge(compact_feats, on=['Season', 'TeamID'], how='left')

    # Detailed features: Four Factors, AstTORatio, Pace
    adv_cols = ['Season', 'TeamID', 'eFGPct', 'TOVPct', 'ORBPct', 'FTRate',
                'DRBPct', 'Opp_eFGPct', 'Opp_FTRate', 'AstTORatio', 'Pace']
    df = df.merge(adv_stats[adv_cols], on=['Season', 'TeamID'], how='left')

    # Adjusted efficiency
    df = df.merge(adj_eff, on=['Season', 'TeamID'], how='left')

    # Last 5 game stats
    df = df.merge(last_n, on=['Season', 'TeamID'], how='left')

    # 填充 NaN (详细结果不可用的早期赛季)
    new_cols = [c for c in df.columns if c not in existing_df.columns]
    for c in new_cols:
        df[c] = df[c].fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════
# 6. 重建训练历史 (差分特征)
# ═══════════════════════════════════════════════════════════════
def build_history(tourney_compact, team_feat, common_cols):
    """从锦标赛结果重建差分训练数据."""
    # 构建索引
    feat_idx = {}
    for row in team_feat.itertuples():
        feat_idx[(row.Season, row.TeamID)] = np.array(
            [getattr(row, c) for c in common_cols], dtype=np.float32)

    def get_feat(season, tid):
        k = (season, tid)
        if k in feat_idx:
            return feat_idx[k]
        # 回退：最近的历史赛季
        candidates = [(s, t) for (s, t) in feat_idx if t == tid and s <= season]
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            return feat_idx[candidates[0]]
        return np.zeros(len(common_cols), dtype=np.float32)

    records = []
    for _, row in tourney_compact.iterrows():
        season = row['Season']
        w, l = row['WTeamID'], row['LTeamID']

        if w < l:
            a, b, label = w, l, 1
        else:
            a, b, label = l, w, 0

        fa = get_feat(season, a)
        fb = get_feat(season, b)

        rec = {'Season': season, 'TeamA': a, 'TeamB': b, 'Label': label}
        for i, c in enumerate(common_cols):
            rec[f'Diff_{c}'] = float(fa[i] - fb[i])
        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 65)
    print("NCAA March Madness 2026 — 高级特征工程 v2")
    print("=" * 65)

    # ── 加载数据 ──
    print("\n[1/8] 加载原始数据...")
    m_det = pd.read_csv(DATA_DIR / 'MRegularSeasonDetailedResults.csv')
    w_det = pd.read_csv(DATA_DIR / 'WRegularSeasonDetailedResults.csv')
    m_cpt = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
    w_cpt = pd.read_csv(DATA_DIR / 'WRegularSeasonCompactResults.csv')
    m_trn = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
    w_trn = pd.read_csv(DATA_DIR / 'WNCAATourneyCompactResults.csv')
    m_elo = pd.read_csv(FEAT_DIR / 'M_elo_features.csv')
    w_elo = pd.read_csv(FEAT_DIR / 'W_elo_features.csv')
    m_feat_old = pd.read_csv(FEAT_DIR / 'M_team_features_final.csv')
    w_feat_old = pd.read_csv(FEAT_DIR / 'W_team_features_final.csv')
    print(f"  男子详细记录: {len(m_det):,}  女子: {len(w_det):,}")
    print(f"  男子紧凑记录: {len(m_cpt):,}  女子: {len(w_cpt):,}")
    print(f"  男子 Elo 记录: {len(m_elo):,}  女子: {len(w_elo):,}")

    # ── 紧凑结果特征 (SOS, PythagWin, CloseWinRate, ScoreStd) ──
    print("\n[2/8] 计算紧凑结果高级特征...")
    m_cpt_feat = compute_compact_features(m_cpt, m_elo)
    w_cpt_feat = compute_compact_features(w_cpt, w_elo)
    print(f"  男子: {len(m_cpt_feat)} 条  女子: {len(w_cpt_feat)} 条")

    # ── 详细结果: 比赛级别统计 ──
    print("\n[3/8] 构建比赛级别高级统计 (Four Factors)...")
    m_game = build_game_stats(m_det)
    w_game = build_game_stats(w_det)
    print(f"  男子比赛记录: {len(m_game):,}  女子: {len(w_game):,}")

    # ── 详细结果: 赛季聚合 ──
    print("\n[4/8] 聚合赛季高级统计...")
    m_adv = aggregate_advanced_stats(m_game)
    w_adv = aggregate_advanced_stats(w_game)
    print(f"  男子: {len(m_adv)} 条  女子: {len(w_adv)} 条")

    # ── 对手调整效率 ──
    print("\n[5/8] 计算 Pomeroy 式调整效率...")
    m_adj = compute_adjusted_efficiency(m_adv, m_game)
    w_adj = compute_adjusted_efficiency(w_adv, w_game)
    print(f"  男子: {len(m_adj)} 条  女子: {len(w_adj)} 条")

    # ── 近期状态 ──
    print("\n[6/8] 计算近期状态 (Last 5 games)...")
    m_last5 = compute_last_n(m_game, n=5)
    w_last5 = compute_last_n(w_game, n=5)
    print(f"  男子: {len(m_last5)} 条  女子: {len(w_last5)} 条")

    # ── 合并 ──
    print("\n[7/8] 合并增强特征...")
    m_v2 = merge_enhanced(m_feat_old, m_cpt_feat, m_adv, m_adj, m_last5)
    w_v2 = merge_enhanced(w_feat_old, w_cpt_feat, w_adv, w_adj, w_last5)

    # 检查公共特征
    exclude = {'Season', 'TeamID', 'Gender', 'ConfAbbrev'}
    m_num = [c for c in m_v2.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(m_v2[c])]
    w_num = [c for c in w_v2.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(w_v2[c])]
    common = [c for c in m_num if c in w_num]

    print(f"  男子总特征: {len(m_num)}  女子: {len(w_num)}  公共: {len(common)}")

    m_v2.to_csv(FEAT_DIR / 'M_team_features_v2.csv', index=False)
    w_v2.to_csv(FEAT_DIR / 'W_team_features_v2.csv', index=False)
    print(f"  已保存 M_team_features_v2.csv ({m_v2.shape})")
    print(f"  已保存 W_team_features_v2.csv ({w_v2.shape})")

    # ── 重建训练历史 ──
    print("\n[8/8] 重建 Diff 训练历史...")
    m_hist = build_history(m_trn, m_v2, common)
    w_hist = build_history(w_trn, w_v2, common)

    m_hist.to_csv(FEAT_DIR / 'M_train_full_history_v2.csv', index=False)
    w_hist.to_csv(FEAT_DIR / 'W_train_full_history_v2.csv', index=False)
    print(f"  已保存 M_train_full_history_v2.csv ({m_hist.shape})")
    print(f"  已保存 W_train_full_history_v2.csv ({w_hist.shape})")

    dt = time.time() - t0
    print(f"\n{'='*65}")
    print(f"✅ 完成! 耗时: {dt:.1f}s")
    print(f"  新增特征 ({len(common) - 37} 个):")
    new_feats = [c for c in common if c not in [
        'Games', 'FGA', 'FGA3', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk',
        'PF', 'NumOT', 'WinRate', 'FGPct', 'FG3Pct', 'FTPct', 'TotalReb',
        'OffEff', 'DefEff', 'NetEff', 'TourneyApps', 'TourneyWinRate',
        'TourneyMaxRound', 'TourneyChampions', 'WinRate_H', 'WinRate_A',
        'WinRate_N', 'HomeAdvantage', 'ConfStrengthIndex', 'ConfAvgScoreDiff',
        'LateWinRate', 'LastNWinRate', 'LastNScoreDiff', 'WinRateTrend',
        'RecentWinRate', 'SeedNum', 'Elo']]
    for f in new_feats:
        print(f"    + {f}")
    print(f"  公共特征总数: {len(common)}")
    print("=" * 65)


if __name__ == '__main__':
    main()
