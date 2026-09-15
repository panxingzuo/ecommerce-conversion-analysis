"""第二章：时间维度分析 —— 可复用工具函数。

抽取自 03_analysis_visualization.ipynb 第二章 2.1 / 2.2（cell 26 / 27 / 28 / 31），
计算逻辑与 notebook 逐字一致，并已在 cleaned_data.csv 上全量复测通过。

小节与口径：
- 2.1 每日趋势（**日粒度**）：每日行为量 = 当日行为事件数；DAU = 当日去重 user_id。
      横坐标标签为「月-日 + 星期几」，便于识别周末效应。
- 2.2 按小时分布（**小时粒度**）：总行为量 + pv / fav / cart / buy 四条序列。
      合并图为双 y 轴——pv 与总行为量同轴（量级接近），fav / cart / buy 走右轴
      （与总量相差约 20–60 倍，同轴会被压成贴地直线）。

⚠️ 相对 notebook 的一处**修复**（2.2）：
    notebook 原 cell 31 被整体替换为合并图代码后，`hourly_behavior`
    （按小时聚合的总行为量）的定义随之丢失，执行会 NameError。
    本模块 `compute_hourly_behavior` **同时返回** hourly_behavior 与 hourly_pivot，
    调用方无需再单独聚合，该问题在抽取后自动消除。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 常量（单一事实来源）
# ============================================================

WEEKDAY_MAP = {0: '周一', 1: '周二', 2: '周三', 3: '周四',
               4: '周五', 5: '周六', 6: '周日'}

# 2.2 合并图：柱体配色（中性浅灰蓝，刻意不与 pv 蓝线撞色）
BAR_COLOR = '#c3d0db'
BAR_EDGE = '#7f8c9a'
BAR_LABEL_COLOR = '#5a6b7d'

# 2.2 折线绘制顺序：pv 走左轴，其余三条走右轴
LEFT_AXIS_BEHAVIOR = 'pv'
RIGHT_AXIS_BEHAVIORS = ('fav', 'cart', 'buy')

DEFAULT_COLOR_MAP = {'pv': '#3498db', 'fav': '#2ecc71',
                     'cart': '#f39c12', 'buy': '#e74c3c'}


# ============================================================
# 2.1 每日趋势 —— 行为量与 DAU
# ============================================================


def compute_daily_trends(df):
    """2.1 计算每日行为量与每日活跃用户数（DAU）。

    参数
    ----
    df : DataFrame，需含 date、user_id 两列（date 会被就地转为 datetime）。

    返回
    ----
    daily_stats : DataFrame，列 = [date, behavior_count, user_count,
                  weekday_label, tick_label]
                  tick_label 为「月-日\\n星期几」，可直接用作 x 轴刻度。
    """
    df['date'] = pd.to_datetime(df['date'])

    daily_behavior = df.groupby('date').size().reset_index(name='behavior_count')
    daily_users = df.groupby('date')['user_id'].nunique().reset_index(name='user_count')
    daily_stats = daily_behavior.merge(daily_users, on='date')

    daily_stats['weekday_label'] = daily_stats['date'].dt.weekday.map(WEEKDAY_MAP)
    daily_stats['tick_label'] = (daily_stats['date'].dt.strftime('%m-%d')
                                 + '\n' + daily_stats['weekday_label'])
    return daily_stats


def format_daily_table(daily_stats):
    """2.1 生成展示用表格（日期格式化为 MM-DD、列名转中文）。"""
    print_df = daily_stats[['date', 'behavior_count', 'user_count', 'weekday_label']].copy()
    print_df['date'] = print_df['date'].dt.strftime('%m-%d')
    print_df.columns = ['日期', '行为量', '活跃用户数', '星期']
    return print_df


def plot_daily_trends(daily_stats, figsize=(14, 10)):
    """2.1 可视化：上下双子图（上=每日行为量，下=DAU），共享 x 轴。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # 上：行为量（Y 轴从 0 开始）
    ax1.bar(range(len(daily_stats)), daily_stats['behavior_count'],
            alpha=0.6, color='tab:blue')
    ax1.plot(range(len(daily_stats)), daily_stats['behavior_count'],
             color='darkblue', marker='o', linewidth=2, markersize=6)
    ax1.set_ylabel('行为量', fontsize=12)
    ax1.set_ylim(0, max(daily_stats['behavior_count']) * 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('每日行为量趋势', fontsize=14, fontweight='semibold')

    # 下：DAU（Y 轴从 0 开始）
    ax2.bar(range(len(daily_stats)), daily_stats['user_count'],
            alpha=0.6, color='tab:orange')
    ax2.plot(range(len(daily_stats)), daily_stats['user_count'],
             color='darkorange', marker='s', linewidth=2, markersize=6)
    ax2.set_ylabel('活跃用户数 (DAU)', fontsize=12)
    ax2.set_ylim(0, max(daily_stats['user_count']) * 1.1)
    ax2.set_xticks(range(len(daily_stats)))
    ax2.set_xticklabels(daily_stats['tick_label'], rotation=0, ha='center')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('每日活跃用户趋势', fontsize=14, fontweight='semibold')

    fig.suptitle('行为量与DAU趋势对比（真实量级）', fontsize=16, fontweight='semibold')
    fig.tight_layout()
    plt.show()


# ============================================================
# 2.2 按小时分布 —— 总行为量 + 四种行为
# ============================================================


def compute_hourly_behavior(df, behaviors=('pv', 'fav', 'cart', 'buy')):
    """2.2 按小时聚合总行为量，并给出各行为的小时透视表。

    返回
    ----
    (hourly_behavior, hourly_pivot)
        hourly_behavior : DataFrame[hour, behavior_count]，总行为量（作图柱体）
        hourly_pivot    : DataFrame，index=hour，列=四种行为，缺失填 0（作图折线）
    """
    hourly_behavior = df.groupby('hour').size().reset_index(name='behavior_count')

    hourly_behavior_type = (df.groupby(['hour', 'behavior_type'])
                              .size().reset_index(name='count'))
    hourly_pivot = (hourly_behavior_type
                    .pivot(index='hour', columns='behavior_type', values='count')
                    .reindex(columns=list(behaviors))
                    .fillna(0))

    return hourly_behavior, hourly_pivot


def plot_hourly_behavior(hourly_behavior, hourly_pivot,
                         color_map=None, figsize=(13, 6.8)):
    """2.2 可视化：总行为量柱 + pv / fav / cart / buy 四条折线（双 y 轴）。

    双轴原因：pv 与总行为量量级接近（pv 约占 89.6%），同左轴；
    fav / cart / buy 与总量相差数十倍，必须走右轴否则被压成贴地直线。
    代价是「视觉等高 ≠ 数量相等」，故 y 轴标签需标明各侧归属。
    """
    color_map = color_map or DEFAULT_COLOR_MAP

    fig, ax1 = plt.subplots(figsize=figsize)

    # ---- 左轴：总行为量柱状图（中性浅灰蓝，不与 pv 蓝线撞色）----
    bars = ax1.bar(hourly_behavior['hour'], hourly_behavior['behavior_count'],
                   color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.8,
                   label='总行为量（柱·左轴）')
    ax1.set_xlabel('小时')
    ax1.set_ylabel('总行为量 / pv 浏览（左轴）', fontsize=11)
    ax1.set_xticks(range(0, 24))
    ax1.set_ylim(0, max(hourly_behavior['behavior_count']) * 1.15)

    # 柱顶数字（只标注 > 10000 的，避免密集）
    for bar, val in zip(bars, hourly_behavior['behavior_count']):
        if val > 10000:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f'{val:,}', ha='center', va='bottom',
                     fontsize=8, color=BAR_LABEL_COLOR)

    # ---- 右轴 ----
    ax2 = ax1.twinx()

    ax1.plot(hourly_pivot.index, hourly_pivot[LEFT_AXIS_BEHAVIOR], marker='o',
             color=color_map[LEFT_AXIS_BEHAVIOR], linewidth=2.4, markersize=5,
             label=f'{LEFT_AXIS_BEHAVIOR}（线·左轴）', zorder=5)

    for bt in RIGHT_AXIS_BEHAVIORS:
        ax2.plot(hourly_pivot.index, hourly_pivot[bt], marker='o',
                 color=color_map[bt], linewidth=2.4, markersize=5,
                 label=f'{bt}（线·右轴）', zorder=5)

    ax2.set_ylabel('fav 收藏 / cart 加购 / buy 购买（右轴）', fontsize=11)
    ax2.set_ylim(0, hourly_pivot[list(RIGHT_AXIS_BEHAVIORS)].values.max() * 1.35)

    # ---- 图例：左上角一列（0–6 点为低谷区，不遮挡折线）----
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper left', ncol=1,
               fontsize=14, markerscale=1.5,
               handlelength=1.6, handletextpad=0.7,
               labelspacing=0.65, borderpad=0.8,
               frameon=True, framealpha=0.92, edgecolor='#d0d7de')

    ax1.set_title('11/25–12/3 用户行为按小时分布（柱 = 总行为量，线 = 各行为类型）',
                  fontsize=18, fontweight='semibold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
