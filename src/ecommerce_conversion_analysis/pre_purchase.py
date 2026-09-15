"""购买前行为路径与转化分析 —— 可复用工具函数。

抽取自 03_analysis_visualization.ipynb 第四章，逻辑与 notebook 逐字一致。

口径约定：
- 4.1  用户级·跨商品（since-last-purchase，不锁 item_id）
- 4.2  商品级·单商品（item-scoped，锁 (user_id, item_id)）
- 4.3  商品级·单商品，按品类汇总（路径锁 item_id，结果按 category_id 对比）
- 4.4  商品级·单商品（item-scoped，含耗时）

本模块为纯抽取：notebook 内联版本暂未删除，后续再切换为 import 复用。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

# ============================================================
# 常量（单一事实来源）
# ============================================================
# 行为编码与路径标签
BEHAVIOR_CODE = {'pv': 0, 'fav': 1, 'cart': 2, 'buy': 3}
PRE_LABELS   = ['pv', 'fav', 'cart']   # code -> 行为标签（不含 buy）
EMPTY_LABEL  = '（空）'
TOP_K        = 20

# 路径分桶：互斥且完备
BUCKET_LABELS  = {'empty': '空路径', 'pv_only': '仅浏览', 'cart': '含加购', 'fav': '含收藏'}
BUCKET_ORDER   = ['empty', 'pv_only', 'cart', 'fav']
BUCKET_COLORS  = {'empty': '#cbd5e0', 'pv_only': '#2b6cb0', 'cart': '#c05621', 'fav': '#2f855a'}

# 购买类型（4.1）：颜色 / 中文标签
TYPE_ORDER  = ['first', 'same_item_rebuy', 'diff_item_rebuy']
TYPE_LABELS = {'first': '首购', 'same_item_rebuy': '同品复购', 'diff_item_rebuy': '换品复购'}
TYPE_COLORS = {'first': '#2b6cb0', 'same_item_rebuy': '#2f855a', 'diff_item_rebuy': '#c05621'}
SEGMENT_COLORS = ['#2b6cb0', '#c05621']  # 购买1次 / 购买≥2次


# ============================================================
# 4.1 购买前行为路径长度（用户级·跨商品）
# ============================================================


def compute_pre_purchase_gaps(df, user_col='user_id',
                              behavior_col='behavior_type', ts_col='timestamp'):
    """返回每位用户每次购买前的非购买行为次数（口径·since-last-purchase）。

    口径：第 N 次购买计入 = 仅「上一次购买之后 ~ 本次购买之前」的
          pv/fav/cart 次数（首次购买：从样本起点 ~ 首次购买之前）。
    同秒事件：buy 排在非购买行为之后，保证计数正确。
    返回 tidy DataFrame: [user_id, purchase_seq, gap_before_purchase]
    """
    BEHAVIOR_ORDER = {'pv': 0, 'fav': 1, 'cart': 2, 'buy': 3}  # 同秒时 buy 排在非购买之后
    events = (
        df[[user_col, behavior_col, ts_col]].copy()
        .assign(_order=lambda d: d[behavior_col].map(BEHAVIOR_ORDER))
        .sort_values([user_col, ts_col, '_order'])
        .reset_index(drop=True)
    )
    is_buy = events[behavior_col].eq('buy')
    # 每个用户从起点累计的非购买次数（buy 行本身贡献 0）
    events['_cum_nonbuy'] = (~is_buy).astype(int).groupby(events[user_col]).cumsum()
    # 每个用户的第几次购买（1,2,3...）
    events['purchase_seq'] = is_buy.astype(int).groupby(events[user_col]).cumsum()

    buy_rows = events[is_buy]
    prev_cum = buy_rows.groupby(user_col)['_cum_nonbuy'].shift(1)
    gaps = (buy_rows['_cum_nonbuy'] - prev_cum.fillna(0)).astype(int)

    return pd.DataFrame({
        'user_id': buy_rows[user_col].values,
        'purchase_seq': buy_rows['purchase_seq'].values,
        'gap_before_purchase': gaps.values,
    })


def classify_pre_purchase_gaps(df, user_col='user_id', item_col='item_id',
                               behavior_col='behavior_type', ts_col='timestamp'):
    """返回每次购买事件: [user_id, purchase_seq, gap_before_purchase, purchase_type]。"""
    BEHAVIOR_ORDER = {'pv': 0, 'fav': 1, 'cart': 2, 'buy': 3}
    events = (
        df[[user_col, item_col, behavior_col, ts_col]].copy()
        .assign(_order=lambda d: d[behavior_col].map(BEHAVIOR_ORDER))
        .sort_values([user_col, ts_col, '_order'])
        .reset_index(drop=True)
    )
    is_buy = events[behavior_col].eq('buy')
    events['_cum_nonbuy'] = (~is_buy).astype(int).groupby(events[user_col]).cumsum()
    events['purchase_seq'] = is_buy.astype(int).groupby(events[user_col]).cumsum()

    buy_rows = events[is_buy]
    prev_cum = buy_rows.groupby(user_col)['_cum_nonbuy'].shift(1)
    gaps = (buy_rows['_cum_nonbuy'] - prev_cum.fillna(0)).astype(int)

    types, seen = [], {}
    pairs = buy_rows[[user_col, item_col]].itertuples(index=False, name=None)
    for (u, it), seq in zip(pairs, buy_rows['purchase_seq']):
        s = seen.setdefault(u, set())
        if seq == 1:
            types.append('first')
        else:
            types.append('same_item_rebuy' if it in s else 'diff_item_rebuy')
        s.add(it)

    return pd.DataFrame({
        'user_id': buy_rows[user_col].values,
        'purchase_seq': buy_rows['purchase_seq'].values,
        'gap_before_purchase': gaps.values,
        'purchase_type': types,
    })


# ============================================================
# 4.2 购买前行为路径模式（商品级·单商品）
# ============================================================


def extract_pre_purchase_paths(df):
    """对每条购买事件，提取其所属 (user_id, item_id) 在该购买之前的非购买行为序列，
    折叠为有序阶段路径。返回 pd.Series（每条购买事件一条路径模式字符串）。

    实现：排序后取 numpy 数组，单次遍历 O(N) 维护「已折叠序列」状态，
    遇 buy 快照路径。避免在 groupby 上做 Python 层逐组迭代（极慢）。
    """
    work = (df[['user_id', 'item_id', 'behavior_type', 'timestamp']]
            .assign(_o=lambda d: d['behavior_type'].map(BEHAVIOR_CODE))
            .sort_values(['user_id', 'item_id', 'timestamp', '_o'])
            .reset_index(drop=True))

    uid = work['user_id'].to_numpy()
    iid = work['item_id'].to_numpy()
    code = work['_o'].to_numpy()
    n = len(work)

    paths = []
    cur_u = None
    cur_i = None
    stages = []        # 已折叠的非购买行为 code 列表
    prev_code = -1
    for i in range(n):
        u, it = uid[i], iid[i]
        if u != cur_u or it != cur_i:          # 进入新的 (user, item)：重置状态
            cur_u, cur_i = u, it
            stages = []
            prev_code = -1
        c = code[i]
        if c == 3:                             # buy：快照当前已折叠前缀
            paths.append('→'.join(PRE_LABELS[s] for s in stages) if stages else EMPTY_LABEL)
        elif c != prev_code:                   # 非 buy：折叠连续重复
            stages.append(c)
            prev_code = c
    return pd.Series(paths, name='pre_purchase_path')


def summarize_paths(path_series, topn=15):
    counts = path_series.value_counts()
    total = int(counts.sum())
    with_fav = int(counts[counts.index.str.contains('fav')].sum())
    print(f"总购买事件数：{total:,}")
    print(f"去重路径模式数：{len(counts):,}")
    print(f"含收藏(fav)环节路径占比：{with_fav / total * 100:.1f}%")
    print(f"不含收藏环节路径占比：{(total - with_fav) / total * 100:.1f}%\n")
    print(f"TOP {topn} 路径模式：")
    for i, (p, c) in enumerate(counts.head(topn).items(), 1):
        print(f"{i:>2}. {p:<22} {c:>6,}  ({c / total * 100:5.1f}%)")
    return counts, total


# ============================================================
# 共享辅助：路径分桶 / 阶段计数
# ============================================================


def classify_bucket(path):
    """将路径归入互斥且完备的桶：空路径 / 仅浏览 / 含加购 / 含收藏。"""
    if path == EMPTY_LABEL:
        return 'empty'
    if 'fav' in path:
        return 'fav'
    if 'cart' in path:
        return 'cart'
    return 'pv_only'


def path_stage_count(path):
    """路径阶段数（复杂度）：空路径=0，否则 '→' 数量 + 1。"""
    return 0 if path == EMPTY_LABEL else path.count('→') + 1


# ============================================================
# 4.3 路径的品类差异（单商品·按品类汇总）
# ============================================================


def extract_paths_with_category(df):
    """O(N) 单次遍历：对每条购买事件，提取其所属 (user_id, item_id) 购买前的
    非购买行为序列（折叠去重），并记录该购买事件的 category_id。

    返回 DataFrame: [category_id, pre_purchase_path]（每条购买事件一行）。
    复用于 4.2 的提取逻辑，仅在 buy 快照时额外记录 category_id。
    """
    work = (df[['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']]
            .assign(_o=lambda d: d['behavior_type'].map(BEHAVIOR_CODE))
            .sort_values(['user_id', 'item_id', 'timestamp', '_o'])
            .reset_index(drop=True))
    uid  = work['user_id'].to_numpy()
    iid  = work['item_id'].to_numpy()
    cid  = work['category_id'].to_numpy()
    code = work['_o'].to_numpy()
    n = len(work)

    paths, cats = [], []
    cur_u = cur_i = None
    stages, prev_code = [], -1
    for i in range(n):
        u, it = uid[i], iid[i]
        if u != cur_u or it != cur_i:          # 进入新的 (user, item)：重置状态
            cur_u, cur_i = u, it
            stages, prev_code = [], -1
        c = code[i]
        if c == 3:                             # buy：快照路径 + 记录品类
            paths.append('→'.join(PRE_LABELS[s] for s in stages) if stages else EMPTY_LABEL)
            cats.append(cid[i])
        elif c != prev_code:                   # 非 buy：折叠连续重复
            stages.append(c)
            prev_code = c
    return pd.DataFrame({'category_id': cats, 'pre_purchase_path': paths})


def build_category_summary(df_paths, top_k=TOP_K):
    """构建 Top-K 品类（按购买事件数）的路径结构摘要。"""
    top_cats = df_paths['category_id'].value_counts().head(top_k).index.tolist()
    rows = []
    for cid in top_cats:
        sub = df_paths[df_paths['category_id'] == cid]
        n = len(sub)
        buckets = sub['pre_purchase_path'].map(classify_bucket).value_counts()
        rows.append({
            'category_id': cid,
            'n_buy': n,
            'empty_rate':   buckets.get('empty', 0) / n * 100,
            'pv_only_rate': buckets.get('pv_only', 0) / n * 100,
            'cart_rate':    buckets.get('cart', 0) / n * 100,
            'fav_rate':     buckets.get('fav', 0) / n * 100,
            'avg_stages':   sub['pre_purchase_path'].map(path_stage_count).mean(),
            'dominant_path': sub['pre_purchase_path'].value_counts().index[0],
        })
    return (pd.DataFrame(rows)
            .sort_values('n_buy', ascending=False)
            .reset_index(drop=True))


# ============================================================
# 4.4 购买路径耗时（商品级·单商品）
# ============================================================


def extract_paths_with_duration(df):
    """O(N) 单次遍历：对每条购买事件，提取其所属 (user_id, item_id) 购买前的
    非购买行为序列（折叠去重路径）+ 该商品窗口内首次非购买行为到本次购买的耗时（秒）。

    返回 DataFrame: [category_id, pre_purchase_path, duration_sec]。
    与 4.2 同口径（stages 自 (user, item) 起始累积，不在 buy 处重置）→ 路径与耗时窗口一致；
    空路径（窗口内无前缀接触）duration_sec = NaN。
    """
    work = (df[['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']]
            .assign(_o=lambda d: d['behavior_type'].map(BEHAVIOR_CODE))
            .sort_values(['user_id', 'item_id', 'timestamp', '_o'])
            .reset_index(drop=True))
    uid = work['user_id'].to_numpy()
    iid = work['item_id'].to_numpy()
    cid = work['category_id'].to_numpy()
    code = work['_o'].to_numpy()
    ts = work['timestamp'].to_numpy()
    n = len(work)

    rows = []
    cur_u = cur_i = None
    stages, prev_code = [], -1
    first_nonbuy_ts = None
    for i in range(n):
        u, it, c = uid[i], iid[i], code[i]
        if u != cur_u or it != cur_i:           # 进入新的 (user, item)：重置状态
            cur_u, cur_i = u, it
            stages, prev_code = [], -1
            first_nonbuy_ts = None
        if c == 3:                              # buy：快照路径 + 耗时
            path = '→'.join(PRE_LABELS[s] for s in stages) if stages else EMPTY_LABEL
            dur = (ts[i] - first_nonbuy_ts) if first_nonbuy_ts is not None else np.nan
            rows.append((cid[i], path, float(dur)))
        elif c != prev_code:                    # 非 buy：折叠连续重复，记录首次非购买 ts
            stages.append(c)
            if first_nonbuy_ts is None:
                first_nonbuy_ts = ts[i]
            prev_code = c
    return pd.DataFrame(rows, columns=['category_id', 'pre_purchase_path', 'duration_sec'])


def summarize_duration(df_paths, topn=15):
    """打印整体可观测耗时分布 + TOP 路径模式的中位耗时。"""
    d = df_paths.dropna(subset=['duration_sec'])
    hrs = d['duration_sec'] / 3600.0
    print(f"有可观测耗时的购买事件：{len(d):,}"
          f"（空路径 {len(df_paths) - len(d):,} 笔无窗口内前缀接触，已排除）")
    print(f"耗时中位数：{hrs.median():.2f} 小时（≈ {hrs.median() / 24:.2f} 天）")
    print(f"P75：{hrs.quantile(.75):.2f} 小时 ｜ P90：{hrs.quantile(.90):.2f} 小时")
    print("注：右删失于 ~9 天窗口；以下为窗口内下界，非真实 time-to-purchase\n")
    g = d.assign(hrs=hrs).groupby('pre_purchase_path')['hrs']
    stat = g.agg(['size', 'median']).sort_values('size', ascending=False).head(topn)
    print(f"TOP {topn} 路径模式的可观测耗时（中位数，小时）：")
    for p, row in stat.iterrows():
        print(f"  {p:<22} n={int(row['size']):>6}  median={row['median']:.2f}h")
    return d


# ============================================================
# 绘图函数
# ============================================================


def plot_purchase_type_composition(classified, figsize=(8.2, 3.6)):
    """购买事件按类型（首购 / 同品复购 / 换品复购）的构成（数量 + 占比）。"""
    counts = (classified['purchase_type'].value_counts()
              .reindex(TYPE_ORDER).fillna(0).astype(int))
    total = int(counts.sum())
    pct = counts / total * 100

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(counts))[::-1]            # 首购置于顶部
    ax.barh(y, counts.values, color=[TYPE_COLORS[k] for k in TYPE_ORDER])
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABELS[k] for k in TYPE_ORDER])
    for yi, v, p in zip(y, counts.values, pct):
        ax.text(v, yi, f"  {v:,}（{p:.1f}%）", va='center', fontsize=9)
    ax.set_xlabel('购买事件数')
    ax.set_title('【用户级·全平台商品】购买事件类型构成（不区分商品ID）', fontweight='semibold'
)
    ax.margins(x=0.14)
    plt.tight_layout(); plt.show()


def plot_buyer_segmentation(pre_purchase, figsize=(6.5, 3.6)):
    """买家（用户）构成——按购买次数分为「仅 1 次」与「≥2 次」。

    注：以购买事件次数度量（pre_purchase 每行即一次购买），非商品件数。
    """
    purchases_per_user = pre_purchase.groupby('user_id').size()
    n_only = int((purchases_per_user == 1).sum())
    n_multi = int((purchases_per_user >= 2).sum())
    total_users = n_only + n_multi
    pct_multi = n_multi / total_users * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(['购买 1 次', '购买 ≥2 次'], [n_only, n_multi],
                  color=SEGMENT_COLORS, width=0.5)
    for b, v in zip(bars, [n_only, n_multi]):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:,}\n({v / total_users * 100:.1f}%)",
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('买家人数')
    ax.set_title(f'【用户级·全平台商品】买家构成：{pct_multi:.1f}% 买家购买≥2次（含跨品类/商品）', fontweight='semibold'
)
    ax.margins(y=0.15)
    plt.tight_layout(); plt.show()


def plot_top_path_patterns(counts, total, topn=12, figsize=(9.5, 4.6)):
    top = counts.head(topn)
    labels = [p if p != EMPTY_LABEL else '空路径' for p in top.index]
    colors = ['#cbd5e0' if p == EMPTY_LABEL else '#2b6cb0' for p in top.index]
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for yi, v in zip(y, top.values):
        ax.text(v, yi, f"  {v:,}（{v / total * 100:.1f}%）", va='center', fontsize=8.5)
    ax.set_xlabel('购买事件数')
    ax.set_title(f'【单商品 item-scoped】购买前路径模式 TOP {topn}（每条=同一商品的购前序列）', fontweight='semibold'
)
    ax.margins(x=0.16)
    fig.text(0.5, 0.015,
             "注：空路径含左删失（11-25 前行为不可见）与换品直购，非「未浏览即购买」；"
             "口径为商品绑定，与 4.1 全局行为量不同。",
             ha='center', fontsize=7.5, color='#718096')
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


def plot_category_buckets(summary, figsize=(10, 7)):
    """百分比堆叠条形图：各 Top 品类路径分桶构成。"""
    s = summary.sort_values('n_buy', ascending=True)   # 底部=购买最多
    cats = s['category_id'].astype(str).tolist()
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(cats))
    for b in BUCKET_ORDER:
        vals = s[f'{b}_rate'].values
        ax.barh(cats, vals, left=left, color=BUCKET_COLORS[b], label=BUCKET_LABELS[b])
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel('购买事件占比（%）')
    ax.set_ylabel('品类编号 (category_id)')
    ax.set_title(f'【单商品·按品类汇总】Top {len(cats)} 品类路径结构（Y轴=品类编号category_id）', fontweight='semibold'
)
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout(); plt.show()


def plot_duration_by_bucket(df_paths, figsize=(8.5, 3.8)):
    """各非空路径桶的可观测耗时中位数对比（回答：含 cart 是否更久）。"""
    d = df_paths.dropna(subset=['duration_sec']).copy()
    d['bucket'] = d['pre_purchase_path'].map(classify_bucket)
    d = d[d['bucket'] != 'empty']
    d['hrs'] = d['duration_sec'] / 3600.0
    med = d.groupby('bucket')['hrs'].median().reindex(['pv_only', 'cart', 'fav'])
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(med))
    ax.bar(x, med.values, color=[BUCKET_COLORS[b] for b in med.index], width=0.5)
    for xi, v in zip(x, med.values):
        ax.text(xi, v, f"{v:.2f}h", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[b] for b in med.index])
    ax.set_ylabel('可观测耗时中位数（小时）')
    ax.set_title('【单商品 item-scoped】不同路径模式的可观测耗时（窗口内下界）', fontweight='semibold'
)
    ax.margins(y=0.15)
    plt.tight_layout(); plt.show()

# ============================================================
# 4.2 购买前路径构成桑基图（事件级占比）


def first_touch(p):
    """购买前路径的首触行为（L1 维度）：空路径 → '空路径'，否则取路径首段。"""
    return '空路径' if p == EMPTY_LABEL else p.split('→')[0]


def _stack_ys(vals):
    """返回每个分段在 y 轴上的起始坐标（自底向上堆叠）。"""
    ys, acc = [], 0.0
    for v in vals:
        ys.append(acc)
        acc += v
    return ys


def _bezier_band(ax, x1, x2, ys, yt, v, color, alpha=0.45):
    """绘制一条贝塞尔流带（桑基图带子）。

    (x1, ys) → (x2, yt) 之间宽度为 v 的带子；
    edgecolor 与面同色 + lw=0.3，消除相邻带子间的抗锯齿白缝。
    """
    xm = (x1 + x2) / 2
    verts = [(x1, ys), (xm, ys), (xm, yt), (x2, yt),
             (x2, yt+v), (xm, yt+v), (xm, ys+v), (x1, ys+v), (x1, ys)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes),
                           facecolor=color, edgecolor=color, lw=0.3,
                           alpha=alpha, zorder=2))


def plot_pre_purchase_sankey(paths, figsize=(13, 5.8)):
    """【单商品 item-scoped】购买前路径构成桑基图（事件级占比）。

    三层：首触行为(L1) → 路径大类(L2) → 购买(L3)。
    - L1 首触：路径首段；空路径 → '空路径'
    - L2 路径大类：复用 classify_bucket 的 4 个互斥桶
      （判定顺序 empty → fav → cart → pv_only，fav 优先于 cart，MECE）
    - L3 购买：全部事件 100% 收口（样本框为已购事件，非转化率）
    """
    # ---- 数据准备 ----
    touch = paths.map(first_touch)
    bucket = paths.map(classify_bucket)          # 复用 src 既有分类器
    # 英文 key → 桑基图展示标签（语境与 BUCKET_LABELS 不同：empty→无前置行为、pv_only→纯直购）
    L2_LABEL = {'empty': '无前置行为', 'pv_only': '纯直购',
                'cart': '含加购', 'fav': '含收藏'}
    cls = bucket.map(L2_LABEL)
    total = len(paths)

    L1_labels = ['空路径', 'pv', 'cart', 'fav']
    L2_labels = ['无前置行为', '纯直购', '含加购', '含收藏']
    L1_vals = [int((touch == l).sum()) for l in L1_labels]
    L2_vals = [int((cls   == l).sum()) for l in L2_labels]
    cross = (pd.crosstab(touch, cls)
             .reindex(index=L1_labels, columns=L2_labels)
             .fillna(0).astype(int))

    # ---- 颜色（按源节点着色：空/无前置灰、pv/纯直购蓝、cart/含加购橙、fav/含收藏绿）----
    COL_L1 = {'空路径': '#cbd5e0', 'pv': '#2b6cb0', 'cart': '#c05621', 'fav': '#2f855a'}
    COL_L2 = {'无前置行为': '#cbd5e0', '纯直购': '#2b6cb0',
              '含加购': '#c05621', '含收藏': '#2f855a'}
    COL_L3 = '#e53e3e'

    # ---- 布局 ----
    fig, ax = plt.subplots(figsize=figsize)
    node_w = 0.18
    xL1, xL2, xL3 = 0.0, 1.3, 2.6
    L1_top, L2_top = _stack_ys(L1_vals), _stack_ys(L2_vals)

    # ---- L1→L2 流带（按源节点上色）----
    src_off = {l: 0.0 for l in L1_labels}
    dst_off = {l: 0.0 for l in L2_labels}
    for i, s in enumerate(L1_labels):
        for j, t in enumerate(L2_labels):
            v = int(cross.loc[s, t])
            if v == 0:
                continue
            _bezier_band(ax, xL1, xL2, L1_top[i] + src_off[s],
                         L2_top[j] + dst_off[t], v, COL_L1[s])
            src_off[s] += v
            dst_off[t] += v

    # ---- L2→L3 全流向购买（仍按 L2 源节点上色）----
    for j, t in enumerate(L2_labels):
        _bezier_band(ax, xL2, xL3, L2_top[j], 0.0, L2_vals[j], COL_L2[t], alpha=0.55)

    # ---- L1 节点 + 文字（节点外左侧）----
    for l, v, top in zip(L1_labels, L1_vals, L1_top):
        ax.add_patch(Rectangle((xL1 - node_w/2, top), node_w, v,
                               facecolor=COL_L1[l], edgecolor='white', lw=1.5, zorder=5))
        ax.text(xL1 - node_w/2 - 0.05, top + v/2, f"{l}\n{v:,} ({v/total*100:.1f}%)",
                ha='right', va='center', fontsize=9, color='#333', zorder=6)

    # ---- L2 节点 + 文字（节点外右侧，避免小节点文字被流带切断）----
    for l, v, top in zip(L2_labels, L2_vals, L2_top):
        ax.add_patch(Rectangle((xL2 - node_w/2, top), node_w, v,
                               facecolor=COL_L2[l], edgecolor='white', lw=1.5, zorder=5))
        ax.text(xL2 + node_w/2 + 0.05, top + v/2, f"{l}\n{v:,} ({v/total*100:.1f}%)",
                ha='left', va='center', fontsize=9, fontweight='semibold',
                color='#333', zorder=6)

    # ---- L3 节点 + 文字（节点外右侧）----
    ax.add_patch(Rectangle((xL3 - node_w/2, 0), node_w, total,
                           facecolor=COL_L3, edgecolor='white', lw=1.5, zorder=5))
    ax.text(xL3 + node_w/2 + 0.05, total/2, f"购买\n{total:,} (100%)",
            ha='left', va='center', fontsize=9, fontweight='semibold', color='#333', zorder=6)

    # ---- 层标题 / 坐标 / 主标题 / 图注 ----
    for x, t in [(xL1, '首次接触行为(含无记录)'), (xL2, '路径大类'), (xL3, '结局')]:
        ax.text(x, total + total*0.025, t, ha='center', fontsize=11,
                fontweight='semibold', color='#333')
    ax.set_xlim(-1.0, 3.3)
    ax.set_ylim(0, total*1.08)
    ax.axis('off')
    ax.set_title('【单商品 item-scoped】购买前路径构成（多路径并发）',
                 fontsize=13, fontweight='semibold', pad=14)
    fig.text(0.5, 0.02,
             "本图为已购买家的路径构成（事件级占比）：仅统计已购事件的「首触→路径大类→购买」流转，"
             "不含未购买用户的流失流；图中百分比为占已购事件的比例，不是转化率。",
             ha='center', fontsize=8.2, color='#555')
    plt.show()

