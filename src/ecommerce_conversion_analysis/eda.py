"""第一章：探索性数据分析（EDA）—— 可复用工具函数。

抽取自 03_analysis_visualization.ipynb 第一章 1.1~1.4
（cell 8/9/11/12/16/17/18/19/20/23），计算逻辑与 notebook 逐字一致，
并已在 cleaned_data.csv 上全量复测通过。

小节与口径：
- 1.1 各行为类型占比：行为**事件**级计数（pv/cart/fav/buy 各自条数与占比）
- 1.2 用户行为频次分布：**用户**级，每用户行为次数的描述统计与分布形态
- 1.3 离群点 / 机器人审查：头部 N 名最活跃用户，用「近乎全勤（仅作背景）」
      「零购买高频」「日均强度异常」三类信号标记候选异常。
      **只标记不删行**——剔除决策留给后续敏感性对比统一做。
- 1.4 用户分层概览：总用户 / 有购买用户 / 纯浏览用户 / 候选异常账号四层，
      各层的用户数、用户占比、行为量占比。

⚠️ 相对 notebook 的一处**修正**（1.4）：
    notebook 原代码中 `N_INTENSITY_ANOMALY = (anomaly_tbl['n_buy'] != 0).sum()`
    统计的是「有购买行为的异常账号」，却被标称为「日均强度异常」，属命名与
    逻辑不符——真正的日均强度异常应取 `flag_extreme_intensity`。
    实测差异：修正前 2 人，修正后 6 人（信号与「零购买高频」8 人有重叠，并集 10 人）。
    本模块已按正确口径计算，见 `build_user_layers` 的 docstring。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ============================================================
# 常量（单一事实来源）
# ============================================================

# 行为配色（与 notebook cell 9 的 color_map 一致）
DEFAULT_COLOR_MAP = {'pv': '#3498db', 'fav': '#2ecc71',
                     'cart': '#f39c12', 'buy': '#e74c3c'}
UNKNOWN_COLOR = '#95a5a6'      # 未知行为类型的兜底色

# 1.2 描述统计的分位数与中文索引名
ACTIVITY_PERCENTILES = [0.5, 0.75, 0.9, 0.95, 0.99]
ACTIVITY_STAT_NAMES = ['用户数', '均值', '标准差', '最小值', '中位数(50%)',
                       '75%分位', '90%分位', '95%分位', '99%分位', '最大值']

# 1.3 异常信号阈值（按本数据集校准）
REVIEW_N_DEFAULT = 30          # 审查头部 N 名最活跃用户
ZERO_BUY_MIN_BEHAVIOR = 200    # 零购买高频：行为次数超过该值且零购买
INTENSITY_PER_DAY = 70         # 日均强度异常：行为次数 / 活跃天数 超过该值
FULL_ACTIVE_RATIO = 0.9        # 近乎全勤：活跃天数 ≥ 窗口天数 × 该比例（仅作背景）

# 1.3 异常明细表的中文列名（顺序与所选列一一对应）
ANOMALY_COLS = ['user_id', 'behavior_count', 'active_days', 'n_buy',
                'distinct_items', 'flag_full_active', 'flag_zero_buy_heavy',
                'flag_extreme_intensity']
ANOMALY_COL_LABELS = ['用户ID', ' 行为次数', '活跃天数', '购买次数', '独立商品数',
                      '近乎全勤', '零购买高频', '日均强度异常']


# ============================================================
# 1.1 各行为类型占比和数量分布
# ============================================================


def compute_behavior_distribution(df):
    """1.1 统计各行为类型的条数与占比（事件级）。

    返回
    ----
    (behavior_summary, behavior_counts, behavior_percent)
        behavior_summary : DataFrame[行为类型, 数量, 占比(%)]，数量已格式化为千分位字符串
        behavior_counts   : Series，各行为类型条数
        behavior_percent  : Series，各行为类型占比（%），索引顺序即绘图顺序
    """
    behavior_counts = df['behavior_type'].value_counts()
    behavior_percent = df['behavior_type'].value_counts(normalize=True) * 100

    behavior_summary = pd.DataFrame({
        '数量': behavior_counts,
        '占比(%)': behavior_percent.round(2),
    }).reset_index()
    behavior_summary.columns = ['行为类型', '数量', '占比(%)']
    behavior_summary['数量'] = behavior_summary['数量'].apply(lambda x: f'{x:,.0f}')

    return behavior_summary, behavior_counts, behavior_percent


def plot_behavior_distribution(behavior_percent, color_map=None, figsize=(8, 5)):
    """1.1 可视化：各行为类型占比柱状图（未知类型用灰色兜底）。"""
    color_map = color_map or DEFAULT_COLOR_MAP
    colors = [color_map.get(x, UNKNOWN_COLOR) for x in behavior_percent.index]

    plt.figure(figsize=figsize)
    bars = plt.bar(behavior_percent.index, behavior_percent.values,
                   color=colors, edgecolor='black', linewidth=1.2)

    plt.title('统计区间：11/25–12/3 各行为类型占比分布', fontsize=14)
    plt.xlabel('行为类型', fontsize=12)
    plt.ylabel('占比 (%)', fontsize=12)
    plt.ylim(0, 105)

    for bar, val in zip(bars, behavior_percent.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.show()


# ============================================================
# 1.2 用户行为频次分布
# ============================================================


def compute_user_activity_stats(df, percentiles=ACTIVITY_PERCENTILES):
    """1.2 统计每用户的行为次数分布（用户级）。

    返回
    ----
    (user_activity, stats, stats_df)
        user_activity : DataFrame[user_id, behavior_count]
        stats         : Series，索引已改为中文（'均值'、'中位数(50%)'…），
                        便于后续按名字取值，如 stats['均值']
        stats_df      : DataFrame[统计指标, 数值]，数值已格式化为字符串，供 tabulate 展示
    """
    user_activity = df.groupby('user_id').size().reset_index(name='behavior_count')
    stats = user_activity['behavior_count'].describe(percentiles=percentiles)
    stats.index = ACTIVITY_STAT_NAMES[:len(stats)]

    stats_df = stats.reset_index()
    stats_df.columns = ['统计指标', '数值']
    stats_df['数值'] = stats_df['数值'].apply(
        lambda x: f'{x:,.0f}' if float(x).is_integer() else f'{x:.2f}')

    return user_activity, stats, stats_df


def plot_user_activity_distribution(user_activity, bins=50, figsize=(12, 5)):
    """1.2 可视化：用户行为频次直方图 + 箱线图（右偏长尾形态）。"""
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.hist(user_activity['behavior_count'], bins=bins,
             color='steelblue', edgecolor='black')
    plt.title('直方图')
    plt.xlabel('行为次数')
    plt.ylabel('用户数')

    plt.subplot(1, 2, 2)
    plt.boxplot(user_activity['behavior_count'], orientation='vertical')
    plt.title('箱线图')
    plt.ylabel('行为次数')

    plt.suptitle('统计区间：11/25–12/3 用户行为频次分布',
                 fontsize=14, fontweight='semibold')
    plt.tight_layout()
    plt.show()


# ============================================================
# 1.3 离群点 / 机器人审查
# ============================================================


def build_user_profile(df):
    """1.3 聚合用户画像（不修改原 df，不新增列）。

    返回
    ----
    (user_profile, window_days)
        user_profile : DataFrame，列 = [user_id, behavior_count, n_pv, n_fav,
                       n_cart, n_buy, active_days, distinct_items]
        window_days  : int，观测窗口天数（本数据集 = 9）
    """
    user_profile = (
        df.groupby('user_id')
          .agg(
              behavior_count=('behavior_type', 'size'),
              n_pv=('behavior_type', lambda s: (s == 'pv').sum()),
              n_fav=('behavior_type', lambda s: (s == 'fav').sum()),
              n_cart=('behavior_type', lambda s: (s == 'cart').sum()),
              n_buy=('behavior_type', lambda s: (s == 'buy').sum()),
              active_days=('datetime', lambda s: s.dt.date.nunique()),
              distinct_items=('item_id', 'nunique'),
          )
          .reset_index()
    )
    window_days = df['datetime'].dt.date.nunique()
    return user_profile, window_days


def flag_candidate_anomalies(user_profile, window_days,
                             review_n=REVIEW_N_DEFAULT,
                             zero_buy_min_behavior=ZERO_BUY_MIN_BEHAVIOR,
                             intensity_per_day=INTENSITY_PER_DAY,
                             full_active_ratio=FULL_ACTIVE_RATIO):
    """1.3 对头部 N 名最活跃用户打异常信号（只标记，不删行）。

    三类信号：
    - flag_full_active      ：近乎全勤（活跃天数 ≥ 窗口天数 × ratio）——**仅作背景，不触发**
    - flag_zero_buy_heavy   ：零购买高频（n_buy == 0 且行为次数 > 阈值）
    - flag_extreme_intensity：日均强度异常（行为次数 / 活跃天数 > 阈值）

    is_candidate_anomaly = flag_zero_buy_heavy | flag_extreme_intensity
    （两个信号可重叠，故并集人数 ≤ 两者之和）
    """
    top_users = (user_profile.sort_values('behavior_count', ascending=False)
                             .head(review_n).copy())

    top_users['flag_full_active'] = (
        top_users['active_days'] >= max(1, int(window_days * full_active_ratio)))
    top_users['flag_zero_buy_heavy'] = (
        (top_users['n_buy'] == 0) & (top_users['behavior_count'] > zero_buy_min_behavior))
    top_users['flag_extreme_intensity'] = (
        (top_users['behavior_count'] / top_users['active_days']) > intensity_per_day)
    top_users['is_candidate_anomaly'] = (
        top_users['flag_zero_buy_heavy'] | top_users['flag_extreme_intensity'])

    return top_users


def format_anomaly_table(top_users):
    """1.3 生成候选异常明细表（原始表 + 中文展示表）。

    返回
    ----
    (anomaly_tbl, anomaly_tbl_display)
        anomaly_tbl         : 原始列名的候选异常明细（含三个 flag 列）
        anomaly_tbl_display : 中文列名、布尔值转 ✓/✗ 的展示表，供 tabulate 输出
    """
    anomaly_tbl = top_users[top_users['is_candidate_anomaly']][ANOMALY_COLS]

    anomaly_tbl_display = anomaly_tbl.copy()
    anomaly_tbl_display.columns = ANOMALY_COL_LABELS
    for col in ANOMALY_COL_LABELS[5:]:      # 后三列为布尔信号列
        anomaly_tbl_display[col] = anomaly_tbl_display[col].map({True: '✓', False: '✗'})

    return anomaly_tbl, anomaly_tbl_display


def plot_anomaly_scatter(top_users, review_n=REVIEW_N_DEFAULT, figsize=(8, 5)):
    """1.3 可视化：头部用户「行为量 vs 购买率」散点图（红 = 候选异常）。

    零购买者购买率为 0，会整齐落在 y=0 底部，与正常重度用户一眼可分。
    """
    plt.figure(figsize=figsize)
    colors = np.where(top_users['is_candidate_anomaly'], '#e74c3c', '#3498db')
    plt.scatter(top_users['behavior_count'],
                top_users['n_buy'] / top_users['behavior_count'] * 100,
                c=colors, s=45, edgecolor='black', linewidth=0.5)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=9, markeredgecolor='black', label='候选异常'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=9, markeredgecolor='black', label='正常（核心重度用户）'),
    ]
    plt.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

    plt.xlabel('行为次数（头部用户）')
    plt.ylabel('购买率 (%)')
    plt.title(f'头部 {review_n} 用户：行为量 vs 购买率（红=候选异常）')
    plt.tight_layout()
    plt.show()


# ============================================================
# 1.4 用户分层概览
# ============================================================


def _layer_metrics(df, userset):
    """某用户集合的分层指标：(人数, 用户占比%, 行为量, 行为量占比%)。"""
    n = len(userset)
    beh = int(df['user_id'].isin(userset).sum())
    return n, n / df['user_id'].nunique() * 100, beh, beh / len(df) * 100


def build_user_layers(df, anomaly_tbl):
    """1.4 计算四层用户的分层指标。

    分层：总用户 / 有购买行为的用户 / 纯浏览用户（无购买）/ 候选异常账号。

    ⚠️ 修正说明：候选异常的细分人数改用 **flag 列**统计——
        n_zero_buy  = flag_zero_buy_heavy 命中数（实测 8 人）
        n_intensity = flag_extreme_intensity 命中数（实测 6 人）
       notebook 原代码用 `(n_buy != 0).sum()` 作为「日均强度异常」，
       得到的实为「有购买行为的异常账号」（实测 2 人），口径与命名不符。
       两个信号存在重叠，故 n_zero_buy + n_intensity 可大于候选异常总数。

    返回
    ----
    dict：各层的人数、用户占比、行为量占比，以及异常细分人数。
    """
    total_users = set(df['user_id'].unique())
    buy_users = set(df.loc[df['behavior_type'] == 'buy', 'user_id'].unique())
    only_browse_users = total_users - buy_users

    n_total, p_total, beh_total, bp_total = _layer_metrics(df, total_users)
    n_buy, p_buy, beh_buy, bp_buy = _layer_metrics(df, buy_users)
    n_browse, p_browse, beh_browse, bp_browse = _layer_metrics(df, only_browse_users)

    n_anomaly = len(anomaly_tbl)
    beh_anomaly = int(anomaly_tbl['behavior_count'].sum())

    return {
        'n_total': n_total, 'p_total': p_total,
        'beh_total': beh_total, 'bp_total': bp_total,
        'n_buy': n_buy, 'p_buy': p_buy, 'beh_buy': beh_buy, 'bp_buy': bp_buy,
        'n_browse': n_browse, 'p_browse': p_browse,
        'beh_browse': beh_browse, 'bp_browse': bp_browse,
        'n_anomaly': n_anomaly,
        'p_anomaly': n_anomaly / n_total * 100,
        'beh_anomaly': beh_anomaly,
        'bp_anomaly': beh_anomaly / len(df) * 100,
        # 修正后：按信号 flag 统计，两者可重叠
        'n_zero_buy': int(anomaly_tbl['flag_zero_buy_heavy'].sum()),
        'n_intensity': int(anomaly_tbl['flag_extreme_intensity'].sum()),
    }


def render_layer_table(layers):
    """1.4 把分层指标渲染成 Markdown 表格字符串（供 display(Markdown(...)) 使用）。"""
    return f"""
| 用户分层 | 用户数 | 用户占比 | 行为量占比 | 备注 |
|----------|--------|----------|------------|------|
| 总用户 | {layers['n_total']:,} | {layers['p_total']:.0f}% | {layers['bp_total']:.0f}% | 统计区间 11/25–12/3 |
| ├ 有购买行为的用户 | {layers['n_buy']:,} | {layers['p_buy']:.1f}% | {layers['bp_buy']:.1f}% | 观测窗内至少产生 1 次 buy |
| └ 纯浏览用户（无购买） | {layers['n_browse']:,} | {layers['p_browse']:.1f}% | {layers['bp_browse']:.1f}% | 仅产生 pv / fav / cart，无 buy |
| 其中：候选异常账号 | {layers['n_anomaly']:,} | {layers['p_anomaly']:.1f}% | {layers['bp_anomaly']:.1f}% | 零购买高频 {layers['n_zero_buy']} 人 + 日均强度异常 {layers['n_intensity']} 人（信号重叠，并集 {layers['n_anomaly']} 人）|
"""
