"""第六章：购买用户分层分析（RFM 框架）—— 可复用工具函数。

抽取自 03_analysis_visualization.ipynb 第六章 6.2~6.7
（cell 111/116/117/118/123/126/130/131/133/139），计算逻辑与 notebook
逐字一致，并已在 cleaned_data.csv 上全量复测通过。

小节与口径：
- 6.2 买家级 RFM 宽表：R（最近购买距截止日的天数，浮点）/ F（购买次数）/
      M_item（购买去重商品数）/ M_cat（购买去重品类数）/ buy_days（购买天数）/
      pv_cnt（浏览次数）。**M 不是金额**，本数据集无价格字段，M 是商品广度代理。
- 6.3 维度诊断：六个候选维度的 Spearman/Pearson 相关矩阵 + 重合度检验，
      用于证明 F 与 M_item 高度共线（ρ≈0.985，90.3% 用户两值完全相等）。
- 6.4 分层：R 三档（≤1 / 1–4 / >4 天）× F 三档（单次 / 复购 / 高频复购）= 9 层。
- 6.5 各层画像：补齐 fav/cart 次数、活跃天数、每单浏览；购买/行为两维画像；
      稳健性检验（以 M_item 替换 F 重跑分层）。
- 6.7 未转化用户池：Buy > Cart > Fav > PV 互斥分组（口径同 user_funnel 5.3），
      补三组未购买用户的行为画像。

⚠️ 相对 notebook 的一处**修正**（6.5 效率图）：
    notebook cell #131 引用了变量 `COLORS`，但该变量从未在 notebook 中定义
    （原「人数条形图」cell 被删除时带走了定义，重启内核后 cell #131 会 NameError）。
    本模块以常量 `F_SEG_COLORS` 补上三档配色，`plot_segment_efficiency` 不再依赖
    外部未定义变量。

口径说明：
- R 用浮点天（除以 86400 而非 .dt.days），避免同日购买者大量并列——与 6.2 一致。
- pv_cnt / fav_cnt / cart_cnt 均为**行为次数**口径（一人可计多次），
  与 5.2 节渗透率的**去重人数**口径不同，两者不可互相引用。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 常量（单一事实来源）
# ============================================================

# 分析截止时刻：2017-12-04 00:00:00（= 12-03 24:00，与第二章清洗口径一致）
ANALYSIS_END = pd.Timestamp('2017-12-04 00:00:00')
SECONDS_PER_DAY = 24 * 60 * 60  # 86,400 秒 = 1 天

# 6.3 参与维度诊断的六个候选指标
DIM_COLS = ['Recency_days', 'Frequency', 'buy_days', 'M_item', 'M_cat', 'pv_cnt']

# 6.4 分层切点（R 三分位取整、F 自然断层 + 均值整数化，见 6.4 口径）
R_LABELS = ['1 天以内', '1–4 天', '4 天以上']   # R ≤1 / 1–4 / >4 天
F_LABELS = ['单次', '复购', '高频复购']          # F = 1 / 2–3 / ≥4
R_BINS = [-0.01, 1, 4, 99]
F_BINS = [0, 1, 3, 10 ** 9]

# 6.5 效率图三档配色（补 notebook cell #131 缺失的 COLORS 定义）
F_SEG_COLORS = {'单次': '#85B7EB', '复购': '#5DCAA5', '高频复购': '#EF9F27'}


# ============================================================
# 6.2 RFM 基础指标计算
# ============================================================


def build_buyer_rfm(df, end_time=ANALYSIS_END):
    """构建买家级 RFM 基础表（一人一行，仅购买用户）。

    参数
    ----
    df : DataFrame，清洗后的全量行为数据
         需含列 user_id / item_id / category_id / behavior_type / datetime
    end_time : Timestamp，分析截止时刻，默认 2017-12-04 00:00:00

    返回
    ----
    DataFrame，索引 = user_id，列 =
        last_buy      : 最近一次购买时间
        Recency_days  : 最近购买距 end_time 的天数（浮点，保留日内精度）
        Frequency     : 窗口内购买次数
        buy_days      : 有购买行为的去重天数
        M_item        : 购买过的去重商品数
        M_cat         : 购买过的去重品类数
        pv_cnt        : 窗口内浏览次数（全行为口径，供 6.5 效率分析）

    说明
    ----
    - 使用 buy.copy() 派生临时列 _date，不修改传入的原始 df
    - pv_cnt 是唯一使用全量行为的字段，口径与 5.2 的「去重人数」不同
    - 未产生 pv 行为的买家，pv_cnt 以 0 填充（而非 NaN）
    """
    buy = df[df['behavior_type'] == 'buy'].copy()
    buy['_date'] = buy['datetime'].dt.date

    rfm = (
        buy.groupby('user_id')
           .agg(
               last_buy=('datetime', 'max'),
               Frequency=('behavior_type', 'size'),
               buy_days=('_date', 'nunique'),
               M_item=('item_id', 'nunique'),
               M_cat=('category_id', 'nunique'),
           )
    )

    # R：时间差 → 总秒数 → 天。用浮点保留日内精度，避免同日购买者出现大量并列
    rfm['Recency_days'] = (
        (end_time - rfm['last_buy']).dt.total_seconds() / SECONDS_PER_DAY
    )

    # pv_cnt：唯一取自全量行为的字段（口径说明见 6.1 第六节）
    pv_cnt = (df[df['behavior_type'] == 'pv']
              .groupby('user_id').size().rename('pv_cnt'))
    rfm = rfm.join(pv_cnt).fillna({'pv_cnt': 0})

    return rfm[['last_buy', 'Recency_days', 'Frequency',
                'buy_days', 'M_item', 'M_cat', 'pv_cnt']]


# ============================================================
# 6.3 维度诊断
# ============================================================


def compute_dimension_correlations(rfm):
    """计算六个候选维度的 Spearman 与 Pearson 相关矩阵。

    用 Spearman 秩相关为主（F / M_item / M_cat / buy_days 均为离散计数且
    严重右偏，不满足 Pearson 的线性与双正态假设），Pearson 作对照。

    返回
    ----
    (corr_spearman, corr_pearson) : 两个 DataFrame，均为 6×6 相关矩阵
    """
    corr_spearman = rfm[DIM_COLS].corr(method='spearman')
    corr_pearson = rfm[DIM_COLS].corr(method='pearson')
    return corr_spearman, corr_pearson


def compute_dimension_overlap(rfm):
    """计算维度重合度：统计有多少用户的两个指标取值完全相同。

    逻辑：若两个指标在测同一件事，大量用户的两个取值会完全相同。
    这是比相关系数更直观的「共线」证据（6.3 第二节）。

    返回
    ----
    DataFrame，列 = [指标对, 相同用户数, 占比]，按重合度从高到低排列。
    """
    pairs = [('Frequency', 'M_item'), ('M_item', 'M_cat'),
             ('Frequency', 'M_cat'), ('M_item', 'buy_days'),
             ('Frequency', 'buy_days'), ('Frequency', 'pv_cnt')]
    n_buyers = len(rfm)

    rows = []
    for a, b in pairs:
        n_same = int((rfm[a] == rfm[b]).sum())
        rows.append({'指标对': f'{a} == {b}',
                     '相同用户数': n_same,
                     '占比': n_same / n_buyers})
    return pd.DataFrame(rows)


# ============================================================
# 6.4 分层
# ============================================================


def segment_buyers(rfm):
    """按 R 三档 × F 三档给买家打 9 层标签（原地修改并返回）。

    在 rfm 上新增三列：
        R_seg   : R 段位（1 天以内 / 1–4 天 / 4 天以上）
        F_seg   : F 档位（单次 / 复购 / 高频复购）
        segment : 组合层名，形如「高频复购 · 1 天以内」

    返回
    ----
    rfm : 原地新增三列后的同一个 DataFrame（便于链式使用）。
    """
    rfm['R_seg'] = pd.cut(rfm['Recency_days'], bins=R_BINS, labels=R_LABELS)
    rfm['F_seg'] = pd.cut(rfm['Frequency'], bins=F_BINS, labels=F_LABELS)
    rfm['segment'] = rfm['F_seg'].astype(str) + ' · ' + rfm['R_seg'].astype(str)
    return rfm


def compute_segment_grid(rfm):
    """返回 9 层交叉表（行 = R 段位，列 = F 档位，含合计）。

    返回
    ----
    DataFrame：pd.crosstab(R_seg, F_seg, margins=True)。
    """
    return pd.crosstab(rfm['R_seg'], rfm['F_seg'],
                       margins=True, margins_name='合计')


# ============================================================
# 6.5 各层画像
# ============================================================


def add_behavior_features(rfm, df):
    """在买家级 rfm 表上补齐行为次数 / 活跃天数 / 每单浏览（原地修改并返回）。

    新增列：
        fav_cnt     : 收藏次数（行为次数口径）
        cart_cnt    : 加购次数（行为次数口径）
        active_days : 有任意行为的去重天数
        pv_per_buy  : pv_cnt / Frequency，每成交 1 单所需浏览次数

    说明：pv / fav / cart 均为「行为次数」口径（一人可计多次），与 5.2 节
    渗透率的「去重人数」口径不同，两者不可互相引用（见 6.1 第六节）。

    返回
    ----
    rfm : 原地新增四列后的同一个 DataFrame。
    """
    for bt in ['fav', 'cart']:
        rfm[f'{bt}_cnt'] = df[df['behavior_type'] == bt].groupby('user_id').size()
    rfm[['fav_cnt', 'cart_cnt']] = rfm[['fav_cnt', 'cart_cnt']].fillna(0)

    rfm['active_days'] = (df.assign(_d=df['datetime'].dt.date)
                            .groupby('user_id')['_d'].nunique())
    rfm['pv_per_buy'] = rfm['pv_cnt'] / rfm['Frequency']
    return rfm


def build_segment_profiles(rfm):
    """构建各层的购买维度 + 行为维度画像。

    前置：rfm 需已执行 segment_buyers() 与 add_behavior_features()。

    返回
    ----
    (profile_buy, profile_beh) :
        profile_buy : DataFrame，购买维度画像（人数/占比/平均R/平均购买/
                      平均去重商品/平均去重品类/平均购买天数），按人数降序。
        profile_beh : DataFrame，行为维度画像（平均浏览/平均收藏/平均加购/
                      平均活跃天数/每单所需浏览均值与中位），行序同 profile_buy。
    """
    n_buyers = len(rfm)

    profile_buy = (rfm.groupby('segment', observed=True)
                      .agg(人数=('segment', 'size'),
                           平均R天数=('Recency_days', 'mean'),
                           平均购买次数=('Frequency', 'mean'),
                           平均去重商品=('M_item', 'mean'),
                           平均去重品类=('M_cat', 'mean'),
                           平均购买天数=('buy_days', 'mean')))
    profile_buy['占比%'] = (profile_buy['人数'] / n_buyers * 100).round(1)
    profile_buy = profile_buy.sort_values('人数', ascending=False).round(2)

    profile_beh = (rfm.groupby('segment', observed=True)
                      .agg(平均浏览=('pv_cnt', 'mean'),
                           平均收藏=('fav_cnt', 'mean'),
                           平均加购=('cart_cnt', 'mean'),
                           平均活跃天数=('active_days', 'mean'),
                           每单所需浏览_均值=('pv_per_buy', 'mean'),
                           每单所需浏览_中位=('pv_per_buy', 'median')))
    profile_beh = profile_beh.reindex(profile_buy.index).round(2)

    return profile_buy, profile_beh


def compute_segment_robustness(rfm):
    """稳健性检验：以 M_item 替换 F 重跑 9 层分层，比较结果。

    前置：rfm 需已执行 segment_buyers()。

    返回
    ----
    dict，键：
        grid_F   : 用 F 分层的 9 层交叉表
        grid_M   : 用 M_item 分层的 9 层交叉表
        diff     : 逐格人数差（M_item − F）
        n_same   : 分层结果完全不变的用户数
        n_diff   : 分层结果发生变化的用户数
    """
    m_seg = pd.cut(rfm['M_item'], bins=F_BINS, labels=F_LABELS)
    segment_m = m_seg.astype(str) + ' · ' + rfm['R_seg'].astype(str)

    grid_f = pd.crosstab(rfm['R_seg'], rfm['F_seg'])
    grid_m = pd.crosstab(rfm['R_seg'], m_seg)

    n_same = int((rfm['segment'] == segment_m).sum())
    return {
        'grid_F': grid_f,
        'grid_M': grid_m,
        'diff': grid_m - grid_f,
        'n_same': n_same,
        'n_diff': len(rfm) - n_same,
    }


# ============================================================
# 6.7 未转化用户池
# ============================================================


def classify_user(row):
    """按最深行为归组（与 5.3 / user_funnel.compute_behavior_combination 口径一致）。

    优先级 Buy > Cart > Fav > PV，逐层互斥。
    """
    if row['buy']:
        return '购买'
    if row['cart']:
        return '加购未购买'
    if row['fav']:
        return '收藏未购买'
    return '仅浏览'


def build_nonbuyer_profile(df):
    """构建全平台用户的最深行为分组 + 未购买三组画像。

    返回
    ----
    (has, prof) :
        has  : DataFrame，索引 user_id，含 buy/cart/fav/pv（是否发生过）、
               group（最深行为分组）、pv_cnt/cart_cnt/fav_cnt（行为次数）、
               active_days（活跃天数）。
        prof : DataFrame，未购买三组画像（人数/平均浏览/平均加购/平均收藏/
               平均活跃天数/占未购买比例%/占全平台比例%）。
    """
    has = df.groupby('user_id').agg(
        buy=('behavior_type', lambda s: (s == 'buy').any()),
        cart=('behavior_type', lambda s: (s == 'cart').any()),
        fav=('behavior_type', lambda s: (s == 'fav').any()),
        pv=('behavior_type', lambda s: (s == 'pv').any()),
    )
    has['group'] = has.apply(classify_user, axis=1)

    for bt in ['pv', 'cart', 'fav']:
        has[f'{bt}_cnt'] = df[df['behavior_type'] == bt].groupby('user_id').size()
    has = has.fillna({'pv_cnt': 0, 'cart_cnt': 0, 'fav_cnt': 0})
    has['active_days'] = df.groupby('user_id')['datetime'].apply(
        lambda s: s.dt.date.nunique())

    n_total = len(has)
    nb = has[has['group'] != '购买']
    prof = (nb.groupby('group', observed=True)
              .agg(人数=('group', 'size'),
                   平均浏览=('pv_cnt', 'mean'),
                   平均加购=('cart_cnt', 'mean'),
                   平均收藏=('fav_cnt', 'mean'),
                   平均活跃天数=('active_days', 'mean')))
    prof['占未购买比例%'] = (prof['人数'] / len(nb) * 100).round(1)
    prof['占全平台比例%'] = (prof['人数'] / n_total * 100).round(1)

    return has, prof


# ============================================================
# 可视化（依赖 notebook 全局中文字体设置，见 notebook cell #2）
# ============================================================


def plot_corr_heatmap(corr_spearman):
    """6.3 可视化：六个候选维度的 Spearman 相关系数热力图。"""
    labels = ['R', 'F', 'buy_days', 'M_item', 'M_cat', 'pv_cnt']

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr_spearman.values, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title('六个候选维度的 Spearman 相关系数', fontsize=13, pad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr_spearman.values[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if abs(v) > 0.6 else '#2C2C2A')

    plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_segment_heatmap(rfm):
    """6.4 可视化：九层人数热力图（行 = R 段位，列 = F 档位）。

    前置：rfm 需已执行 segment_buyers()。
    """
    row_lbl = ['1 天以内\n(12-03)',
               '1–4 天\n(11-30 ~ 12-02)',
               '4 天以上\n(11-25 ~ 11-29)']
    col_lbl = ['单次\nF = 1', '复购\nF = 2–3', '高频复购\nF ≥ 4']
    note = ('注：R = 最近一次购买距 2017-12-04 00:00 的天数；F = 窗口内购买次数。'
            'R 三档 × F 三档 交叉得 9 层。')

    counts = pd.crosstab(rfm['R_seg'], rfm['F_seg'])
    total = counts.values.sum()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(counts.values, cmap='Blues')

    ax.set_xticks(range(len(col_lbl)))
    ax.set_xticklabels(col_lbl, fontsize=11)
    ax.set_yticks(range(len(row_lbl)))
    ax.set_yticklabels(row_lbl, fontsize=10)
    ax.set_title('九层人数分布（行 = R 段位，列 = F 档位）', fontsize=13, pad=12)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts.values[i, j]
            ax.text(j, i, f'{v:,}\n{v / total:.1%}', ha='center', va='center',
                    fontsize=12, color='white' if v > 900 else '#2C2C2A')

    plt.colorbar(im, ax=ax, label='人数', shrink=0.8)
    plt.tight_layout(rect=[0, 0.10, 1, 1])        # 底部预留 10% 给脚注
    fig.text(0.5, 0.03, note, ha='center', va='bottom',
             fontsize=9.5, color='#5F5E5A')
    plt.show()


def plot_segment_efficiency(rfm):
    """6.5 可视化：各层「每成交 1 单所需浏览次数」中位数（越低越高效）。

    前置：rfm 需已执行 segment_buyers() 与 add_behavior_features()。
    """
    eff = (rfm.groupby('segment', observed=True)['pv_per_buy']
              .median().sort_values())          # 升序：效率最高的在顶部

    colors = [F_SEG_COLORS[s.split(' · ')[0]] for s in eff.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(eff)), eff.values, color=colors)
    ax.set_yticks(range(len(eff)))
    ax.set_yticklabels([s.replace(' · ', ' ·\n') for s in eff.index], fontsize=10)
    ax.set_xlabel('每成交 1 单所需浏览次数（中位数）', fontsize=11)
    ax.set_title('各层浏览转化效率对比（越低越高效）', fontsize=13, pad=12)

    for i, v in enumerate(eff.values):
        ax.text(v + 1, i, f'{v:.1f}', va='center', fontsize=10, color='#2C2C2A')

    handles = [plt.Rectangle((0, 0), 1, 1, color=F_SEG_COLORS[k])
               for k in F_SEG_COLORS]
    ax.legend(handles, F_SEG_COLORS.keys(), title='F 档位',
              loc='lower right', fontsize=10)
    ax.set_xlim(0, max(eff.values) * 1.15)
    plt.tight_layout()
    plt.show()
