"""第三章：商品维度分析 —— 可复用工具函数。

抽取自 03_analysis_visualization.ipynb 第三章 3.1 ~ 3.4.7
（cell 35/36/40/41/42/45/48/52/54/55/58/60/62/64/68），
计算逻辑与 notebook 逐字一致，并已在 cleaned_data.csv 上全量复测通过。

小节与口径：
- 3.1 热销品类与商品 TOP10：按**行为事件量**排序（非成交额），
      item_share = 该商品行为量 / 所属品类总行为量 × 100%。
- 3.2 热销商品多维画像：3.2.1 品类归属映射 + item_share；
      3.2.2 TOP10 商品所属品类的行为类型构成（100% 堆叠）。
- 3.3 品类内商品集中度：CR1（头部商品占比）/ CR5（前五累计）/ HHI（赫芬达尔指数），
      仅针对 TOP10 商品所涉及的品类（cat_order）。
- 3.4 各品类购买转化率：**用户级**口径——
      转化率 = 既浏览又购买该品类的去重用户数 / 浏览过该品类的去重用户数 × 100%。
      ⚠️ 与第五章的「有序流转率」不同：此处**不要求购买发生在浏览之后**，
      且不跨商品；两者分母不可混用。
      3.4.2~3.4.5 为阈值敏感性分析（小样本虚高噪声的识别与处理）。

⚠️ 相对 notebook 的两处**清理**（不影响任何输出）：
    1. cell 35 末尾的 `overlap`（品类 TOP10 与商品 TOP10 所属品类的交集）
       计算后从未被使用，属死代码，抽取时省略。
    2. cell 42 用同名变量 `category_total_behavior` 覆盖了 cell 35 的
       DataFrame（类型由 DataFrame 变为 int），易误读；
       抽取后 `query_item_category` 返回独立命名的标量，不再覆盖。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 常量（单一事实来源）
# ============================================================

# 行为类型配色（与 notebook cell 9 的 color_map 一致；
# 3.2.2 原代码中的 colors_bt 与之完全重复，抽取后统一使用本常量）
DEFAULT_COLOR_MAP = {'pv': '#3498db', 'fav': '#2ecc71',
                     'cart': '#f39c12', 'buy': '#e74c3c'}
BEHAVIOR_ORDER = ['pv', 'fav', 'cart', 'buy']

# 3.1 / 3.3 配色
CAT_BAR_COLOR = '#2b6cb0'
ITEM_BAR_COLOR = '#e74c3c'
CR1_COLOR = '#e74c3c'
CR5_COLOR = '#3498db'

# 3.4 默认阈值与过滤门槛
DEFAULT_THRESHOLDS = [10, 30, 50, 100, 200]
DEFAULT_MIN_BROWSER = 50       # 3.4.6 论证后选定的主阈值
CONV_CLIP_UPPER = 60           # 全品类分布图的横轴截断点（避免 100% 小样本拉飞）
HIST_BINS_ALL = 40
HIST_BINS_VALID = 30


# ============================================================
# 3.1 热销品类与商品 TOP10
# ============================================================


def compute_top_categories_items(df, top_n=10):
    """3.1 按行为量取 TOP-N 品类与商品，并计算商品的 item_share。

    item_share = 该商品行为量 / 所属品类总行为量 × 100%

    返回
    ----
    dict，键：
        top_categories               : DataFrame[category_id, behavior_count]
        top_items                    : DataFrame[item_id, behavior_count]
        top_items_with_category_total: 上述商品表 + category_id +
                                       category_total_behavior + item_share
        n_cat / n_item               : 全量品类数 / 商品数
    """
    category_behavior = df.groupby('category_id').size().reset_index(name='behavior_count')
    top_categories = category_behavior.sort_values('behavior_count', ascending=False).head(top_n)

    item_behavior = df.groupby('item_id').size().reset_index(name='behavior_count')
    top_items = item_behavior.sort_values('behavior_count', ascending=False).head(top_n)

    top_items_info = top_items.merge(
        df[['item_id', 'category_id']].drop_duplicates(), on='item_id', how='left')

    category_total_behavior = (df.groupby('category_id').size()
                                 .reset_index(name='category_total_behavior'))

    top_items_with_category_total = top_items_info.merge(
        category_total_behavior, on='category_id', how='left')
    top_items_with_category_total['item_share'] = (
        top_items_with_category_total['behavior_count']
        / top_items_with_category_total['category_total_behavior'] * 100)

    return {
        'top_categories': top_categories,
        'top_items': top_items,
        'top_items_with_category_total': top_items_with_category_total,
        'n_cat': df['category_id'].nunique(),
        'n_item': df['item_id'].nunique(),
    }


def plot_top_categories_items(top_categories, top_items, figsize=(14, 5)):
    """3.1 可视化：1×2 子图 —— 热销品类 TOP10 + 热销商品 TOP10。"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('热销品类与商品 TOP10（按行为量）', fontsize=16, fontweight='semibold')

    # 左：热销品类 TOP10
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(top_categories)), top_categories['behavior_count'],
                    color=CAT_BAR_COLOR)
    ax1.set_title('热销品类 TOP10', fontsize=12, fontweight='semibold')
    ax1.set_xlabel('品类 ID')
    ax1.set_ylabel('行为量')
    ax1.set_xticks(range(len(top_categories)))
    ax1.set_xticklabels(top_categories['category_id'].astype(str), rotation=45)
    ax1.set_ylim(0, top_categories['behavior_count'].max() * 1.15)
    for bar, val in zip(bars1, top_categories['behavior_count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val * 0.01,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='semibold')

    # 右：热销商品 TOP10
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(top_items)), top_items['behavior_count'],
                    color=ITEM_BAR_COLOR)
    ax2.set_title('热销商品 TOP10', fontsize=12, fontweight='semibold')
    ax2.set_xlabel('商品 ID')
    ax2.set_ylabel('行为量')
    ax2.set_xticks(range(len(top_items)))
    ax2.set_xticklabels(top_items['item_id'].astype(str), rotation=45)
    ax2.set_ylim(0, top_items['behavior_count'].max() * 1.15)
    for bar, val in zip(bars2, top_items['behavior_count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val * 0.01,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='semibold')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


# ============================================================
# 3.2 热销商品多维画像
# ============================================================


def prepare_item_category_view(df, top_items_with_category_total):
    """3.2.1 预备数据：品类→商品映射、统一配色、品类行为类型构成。

    返回
    ----
    (category_group, item_colors, bt_cat, bt_pct)
        category_group : DataFrame[category_id, item_id(list), count]，按商品数降序
        item_colors    : list，每个商品对应的品类颜色（与表格行序一致）
        bt_cat         : DataFrame，各品类各行为类型的绝对量
        bt_pct         : DataFrame，各品类各行为类型的占比（%）
    """
    top = (top_items_with_category_total
           .sort_values('behavior_count', ascending=False).reset_index(drop=True))

    # 品类 → 颜色 统一映射（tab10）
    category_ids = sorted(top['category_id'].unique())
    cmap = plt.cm.tab10
    cat_color = {c: cmap(i) for i, c in enumerate(category_ids)}
    item_colors = [cat_color[c] for c in top['category_id']]

    # 品类 → 商品 堆叠映射
    category_group = (top.groupby('category_id')['item_id'].apply(list).reset_index())
    category_group['count'] = category_group['item_id'].str.len()
    category_group = category_group.sort_values('count', ascending=False).reset_index(drop=True)

    # 品类按总行为量降序（小品类放最后，对比更直观）
    cat_order = (top.groupby('category_id')['behavior_count']
                    .sum().sort_values(ascending=False).index.tolist())
    df_cat = df[df['category_id'].isin(cat_order)]

    bt_cat = (df_cat.groupby(['category_id', 'behavior_type']).size()
                    .unstack(fill_value=0)
                    .reindex(columns=BEHAVIOR_ORDER, fill_value=0)
                    .reindex(cat_order))
    bt_pct = bt_cat.div(bt_cat.sum(axis=1), axis=0) * 100

    return category_group, item_colors, bt_cat, bt_pct


def plot_item_category_structure(top_items_with_category_total,
                                 category_group, item_colors, figsize=(16, 6)):
    """3.2.1 可视化：1×2 子图 —— 品类归属映射 + 商品 item_share。"""
    top = (top_items_with_category_total
           .sort_values('behavior_count', ascending=False).reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('TOP10 热销商品多维度分析', fontsize=18, fontweight='semibold')

    # 左：品类归属映射（堆叠方块，每块一个商品 ID）
    ax1 = axes[0]
    category_ids = sorted(top['category_id'].unique())
    cmap = plt.cm.tab10
    cat_color = {c: cmap(i) for i, c in enumerate(category_ids)}

    for x, row in category_group.iterrows():
        for y, item in enumerate(row['item_id']):
            ax1.bar(x, 1, bottom=y, width=0.6, color=cat_color[row['category_id']],
                    edgecolor='black', alpha=0.85)
            ax1.text(x, y + 0.5, str(item), ha='center', va='center',
                     fontsize=8, fontweight='semibold')
        ax1.text(x, row['count'] + 0.15, f'{row["count"]}个',
                 ha='center', va='bottom', fontsize=9, fontweight='semibold')

    ax1.set_title('TOP10 商品的品类归属映射', fontsize=12, fontweight='semibold')
    ax1.set_xlabel('品类 ID')
    ax1.set_ylabel('TOP10 商品数量')
    ax1.set_xticks(range(len(category_group)))
    ax1.set_xticklabels(category_group['category_id'].astype(str), rotation=45)
    ax1.set_ylim(0, category_group['count'].max() + 0.8)

    # 右：item_share
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(top)), top['item_share'], color=item_colors, edgecolor='black')
    ax2.set_title('商品占所属品类的行为比例 (item share %)', fontsize=12, fontweight='semibold')
    ax2.set_xlabel('商品 ID（按行为量降序）')
    ax2.set_ylabel('占比 (%)')
    ax2.set_xticks(range(len(top)))
    ax2.set_xticklabels(top['item_id'].astype(str), rotation=45)
    ax2.set_ylim(0, top['item_share'].max() * 1.15)
    for bar, val in zip(bars2, top['item_share']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='semibold')

    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% 阈值')
    ax2.legend(loc='upper right')

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def query_item_category(df, item_id):
    """3.2.1 查询单个商品所属品类及该品类的总行为数。

    返回
    ----
    (category_id, category_total_behavior)
    """
    cat_id = df.loc[df['item_id'] == item_id, 'category_id'].iloc[0]
    category_total_behavior = int((df['category_id'] == cat_id).sum())
    return cat_id, category_total_behavior


def plot_category_behavior_composition(bt_cat, bt_pct, color_map=None, figsize=(10, 5)):
    """3.2.2 可视化：TOP10 商品所属品类的行为类型构成（100% 堆叠柱状图）。

    堆叠图中 buy 在大体量品类里通常 <0.5%，肉眼不可见，
    故函数返回 buy 占比字典供调用方显式打印审计。
    """
    color_map = color_map or DEFAULT_COLOR_MAP

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(bt_pct))
    for bt in BEHAVIOR_ORDER:
        vals = bt_pct[bt].values
        ax.bar(bt_pct.index.astype(str), vals, bottom=bottom,
               color=color_map[bt], edgecolor='black', label=bt)
        for i, v in enumerate(vals):
            if v >= 2:
                ax.text(i, bottom[i] + v / 2, f'{v:.1f}%',
                        ha='center', va='center', fontsize=8)
        bottom += vals

    for i, c in enumerate(bt_cat.index):       # 柱顶标总量，体量差异一眼可见
        ax.text(i, 101, f'共{bt_cat.sum(axis=1)[c]:,}',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 112)
    ax.set_title('TOP10 商品所属品类的行为类型构成（100% 堆叠）', fontweight='semibold')
    ax.set_xlabel('品类 ID')
    ax.set_ylabel('占比 (%)')
    ax.legend(loc='upper right', title='行为类型')
    plt.tight_layout()
    plt.show()


# ============================================================
# 3.3 品类内商品集中度（CR1 / CR5 / HHI）
# ============================================================


def compute_category_concentration(df, cat_order):
    """3.3 计算各品类内商品的集中度指标。

    CR1 = 头部商品行为量占比；CR5 = 前五商品累计占比；HHI = 各商品占比平方和。

    返回
    ----
    conc : DataFrame[category_id, 商品数, 总行为量, CR1%, CR5%, HHI]
    """
    rows = []
    for c in cat_order:
        cnt = df[df['category_id'] == c].groupby('item_id').size().sort_values(ascending=False)
        s = cnt / cnt.sum()
        rows.append([c, len(cnt), cnt.sum(), round(s.iloc[0] * 100, 2),
                     round(s.head(5).sum() * 100, 2), round((s ** 2).sum(), 4)])
    return pd.DataFrame(rows, columns=['category_id', '商品数', '总行为量',
                                       'CR1%', 'CR5%', 'HHI'])


def plot_category_concentration(conc, figsize=(9, 5)):
    """3.3 可视化：CR1 / CR5 分组柱状图。"""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(conc))
    w = 0.35
    ax.bar(x - w/2, conc['CR1%'], w, color=CR1_COLOR, edgecolor='black',
           label='CR1（TOP1 商品占比）')
    ax.bar(x + w/2, conc['CR5%'], w, color=CR5_COLOR, edgecolor='black',
           label='CR5（Top1~Top5 累计商品占比）')
    for i, r in conc.iterrows():
        ax.text(i - w/2, r['CR1%'] + 1, f"{r['CR1%']:.1f}%", ha='center', fontsize=8)
        ax.text(i + w/2, r['CR5%'] + 1, f"{r['CR5%']:.1f}%", ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(conc['category_id'].astype(str), rotation=45)
    ax.set_ylim(0, max(conc['CR5%'].max(), conc['CR1%'].max()) * 1.15)
    ax.set_ylabel('集中度 (%)')
    ax.set_title('品类内商品集中度对比')
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)
    plt.show()


# ============================================================
# 3.4 各品类的购买转化率分析
# ============================================================


def compute_category_conversion(df):
    """3.4.1 计算每个品类的购买转化率（**用户级**口径）。

    转化率 = 既浏览又购买该品类的去重用户数 / 浏览过该品类的去重用户数 × 100%
    ⚠️ 不要求「购买发生在浏览之后」，与第五章的有序流转率口径不同，勿混用。

    返回
    ----
    conversion_df : DataFrame[category_id, total_browser, converted_users,
                    conversion_rate, conversion_rate_pct(展示字符串)]
    """
    browser_df = df[df['behavior_type'] == 'pv'][['user_id', 'category_id']].drop_duplicates()
    buyer_df = df[df['behavior_type'] == 'buy'][['user_id', 'category_id']].drop_duplicates()

    converted_df = browser_df.merge(buyer_df, on=['user_id', 'category_id'], how='inner')
    converted_count = (converted_df.groupby('category_id').size()
                                   .reset_index(name='converted_users'))
    total_browser_count = (browser_df.groupby('category_id').size()
                                     .reset_index(name='total_browser'))

    conversion_df = total_browser_count.merge(converted_count, on='category_id', how='left')
    conversion_df['converted_users'] = conversion_df['converted_users'].fillna(0)
    conversion_df['conversion_rate'] = (conversion_df['converted_users']
                                        / conversion_df['total_browser'] * 100).round(2)
    conversion_df['conversion_rate_pct'] = conversion_df['conversion_rate'].astype(str) + '%'
    return conversion_df


def plot_conversion_distribution(conversion_df, min_browser=DEFAULT_MIN_BROWSER,
                                 clip_upper=CONV_CLIP_UPPER, figsize=(12, 4.5)):
    """3.4.1 可视化：全品类转化率分布 vs 过滤小样本后的分布（1×2 直方图）。

    返回
    ----
    valid : DataFrame，过滤后的有效品类（供后续打印量化效果）
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左：全品类（截断上限，避免 100% 小样本拉飞横轴）
    all_conv = conversion_df['conversion_rate'].clip(upper=clip_upper)
    axes[0].hist(all_conv, bins=HIST_BINS_ALL, color='#2b6cb0', edgecolor='black')
    axes[0].axvline(conversion_df['conversion_rate'].median(), color='red', ls='--',
                    label=f"中位数 {conversion_df['conversion_rate'].median():.1f}%")
    axes[0].set_title(f'全品类转化率分布（截断 {clip_upper}%）')
    axes[0].set_xlabel('转化率 (%)')
    axes[0].set_ylabel('品类数')
    axes[0].legend()

    # 右：过滤后
    valid = conversion_df[conversion_df['total_browser'] >= min_browser]
    axes[1].hist(valid['conversion_rate'], bins=HIST_BINS_VALID,
                 color='#e67e22', edgecolor='black')
    axes[1].axvline(valid['conversion_rate'].median(), color='red', ls='--',
                    label=f"中位数 {valid['conversion_rate'].median():.1f}%")
    axes[1].set_title(f"过滤 total_browser≥{min_browser} 后（剩 {len(valid)} 个品类）")
    axes[1].set_xlabel('转化率 (%)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return valid


def compute_threshold_sensitivity(conversion_df, thresholds=DEFAULT_THRESHOLDS, top_n=10):
    """3.4.2 阈值敏感性：各阈值下的转化率 TOP-N 品类。

    返回
    ----
    (sensitivity_results, top10_ids)
        sensitivity_results : dict{threshold: DataFrame(含 rank 列)}
        top10_ids           : dict{threshold: set(category_id)}
    """
    sensitivity_results = {}
    top10_ids = {}
    for th in thresholds:
        filtered = conversion_df[conversion_df['total_browser'] >= th].copy()
        top = filtered.sort_values('conversion_rate', ascending=False).head(top_n)
        top = top[['category_id', 'total_browser', 'converted_users',
                   'conversion_rate']].copy()
        top['rank'] = range(1, len(top) + 1)
        sensitivity_results[th] = top
        top10_ids[th] = set(top['category_id'].tolist())
    return sensitivity_results, top10_ids


def compute_top10_overlap(top10_ids, thresholds=DEFAULT_THRESHOLDS):
    """3.4.3 各阈值之间 TOP10 品类的两两重合数。

    返回
    ----
    list of (th1, th2, overlap_count)
    """
    out = []
    for i, th1 in enumerate(thresholds):
        for th2 in thresholds[i+1:]:
            out.append((th1, th2, len(top10_ids[th1] & top10_ids[th2])))
    return out


def find_core_categories(top10_ids, thresholds=DEFAULT_THRESHOLDS):
    """3.4.4 交集法：在所有阈值下都进入 TOP10 的品类集合。"""
    core_set = set(top10_ids[thresholds[0]])
    for th in thresholds[1:]:
        core_set = core_set & top10_ids[th]
    return core_set


def plot_threshold_sensitivity(sensitivity_results, conversion_df,
                               thresholds=DEFAULT_THRESHOLDS, figsize=(12, 10)):
    """3.4.5 可视化：不同阈值下 TOP10 的转化率分布（**纵向分面柱状图**）。

    ⚠️ 本图为分面柱状图（每个阈值一个子图），**请勿改为折线图**——
    纵向分面便于逐阈值读出 TOP10 的绝对转化率与衰减形态。
    """
    fig, axes = plt.subplots(len(thresholds), 1, figsize=figsize)

    for idx, th in enumerate(thresholds):
        ax = axes[idx]
        data = sensitivity_results[th]
        bars = ax.bar(range(len(data)), data['conversion_rate'],
                      color='steelblue', edgecolor='black')
        ax.set_title(f'阈值 = {th}（品类数: '
                     f'{len(conversion_df[conversion_df["total_browser"] >= th])}）',
                     fontsize=12)
        ax.set_ylabel('转化率 (%)')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['category_id'].astype(str), rotation=45, fontsize=8)
        ax.set_ylim(0, max(data['conversion_rate']) * 1.15)

        for bar, val in zip(bars, data['conversion_rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.show()


def get_top_conversion_categories(conversion_df, min_browser=DEFAULT_MIN_BROWSER, top_n=10):
    """3.4.7 过滤小样本后的转化率 TOP-N 品类（展示表，中文列名）。"""
    filtered = conversion_df[conversion_df['total_browser'] >= min_browser].copy()
    top = filtered.sort_values('conversion_rate', ascending=False).head(top_n)

    display_df = top[['category_id', 'total_browser', 'converted_users',
                      'conversion_rate_pct']].copy()
    display_df.columns = ['品类ID', '浏览人数', '购买人数', '转化率(%)']
    return display_df
