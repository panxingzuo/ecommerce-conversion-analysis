"""第五章：全平台用户行为渗透与流转分析 —— 可复用工具函数。

抽取自 03_analysis_visualization.ipynb 第五章（cell 99/100/104/105/108/109），
计算逻辑与 notebook 内联版本逐字一致，并已在 cleaned_data.csv 上全量复测通过。

口径约定（三节彼此独立，勿混用）：
- 5.2 用户行为渗透率（并行漏斗）
      pv / fav / cart / buy **各自独立**统计去重用户数，四者并行计算、
      互不要求先后顺序（非递减，不是真漏斗）；分母统一 = 全平台总活跃用户数。
      四者之和 > 100%（同一用户可同时出现在多个分子里）。
- 5.3 用户行为组合分布（互斥分组 · Buy 封顶）
      按「9 天内达到的最深行为」把每个用户归入唯一分组，优先级
      Buy > Cart > Fav > PV；四组互斥且完备，之和 = 总活跃用户数。
- 5.4 用户行为流转分析（有序流转率 + 对照组 + Lift）
      流转率 P(Buy|X) = 「先做过 X、之后发生过购买」的去重用户 / 做过 X 的去重用户
        时间约束 first(X) < last(buy)——X 之后发生过**任意一笔**购买即计入
        （含老客复购），不限于首次购买。
      对照组 P(Buy|未X) = 从未做过 X 且买过的用户 / 从未做过 X 的用户
        ——对照组「未做过 X」无从定义时间先后，无法施加时序约束，
        故 Lift 为**保守估计**（两侧均不分顺序时：加购 1.26、收藏 1.08）。
      Lift = P(Buy|X) / P(Buy|未X)，> 1 表示该行为对购买有正向促进。

三节的关系：5.2 看「做过什么」（可叠加）、5.3 看「最深做到哪一步」（互斥）、
5.4 看「某行为带来的增量」（对照 + Lift）。口径不同，数字不可互相替代。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 常量（单一事实来源）
# ============================================================

# 行为展示顺序：5.2 用逻辑顺序（非数值降序，刻意不呈收口状）
BEHAVIOR_ORDER = ('pv', 'fav', 'cart', 'buy')

# 5.3 互斥分组：优先级 Buy > Cart > Fav > PV（顺序即展示顺序）
COMBO_SPEC = (
    ('buy',  '购买用户 Buy'),
    ('cart', '加购用户 Cart'),
    ('fav',  '收藏用户 Fav'),
    ('pv',   '仅浏览用户 PV'),
)

# 5.4 流转分析：展示顺序（加购 → 收藏 → 浏览）
CONVERSION_SPEC = (('cart', '加购'), ('fav', '收藏'), ('pv', '浏览'))
KEY_BY_ZH = {zh: key for key, zh in CONVERSION_SPEC}

# 默认配色（与 notebook 1.1 章节 color_map 一致）；调用方可传入自定义 color_map 覆盖
DEFAULT_COLOR_MAP = {'pv': '#3498db', 'fav': '#2ecc71',
                     'cart': '#f39c12', 'buy': '#e74c3c'}

# 5.4 数字标签深色版（柱体用亮色，标签用深色保证可读）
DARK_COLOR_MAP = {'pv': '#2b6cb0', 'fav': '#2f855a',
                  'cart': '#c05621', 'buy': '#c0392b'}

# 对照组样本量下限：低于此值时 Lift 不可靠，图上自动标注「样本不足」
MIN_CTRL_N = 100


# ============================================================
# 5.2 用户行为渗透率（并行漏斗）
# ============================================================


def compute_behavior_penetration(df, behaviors=BEHAVIOR_ORDER):
    """统计各行为的去重用户渗透率（并行漏斗）。

    口径：每种行为**独立**统计「有过该行为的去重用户数」，四者并行计算、
    互不要求先后顺序；分母统一 = 全平台总活跃用户数（df 中出现过的全部 user_id）。
    注意四者之和 > 100%——同一用户可同时计入多个行为的分子。

    参数
    ----
    df : DataFrame
        需含 user_id、behavior_type 两列。
    behaviors : sequence of str
        行为展示顺序，默认 ('pv', 'fav', 'cart', 'buy')（逻辑顺序，非数值降序）。

    返回
    ----
    (penetration, total_users)
        penetration : DataFrame，列 = [行为, 去重用户数, 渗透率(%)]
        total_users : int，全平台总活跃用户数（分母）
    """
    total_users = df['user_id'].nunique()

    rows = []
    for b in behaviors:
        n_users = df.loc[df['behavior_type'] == b, 'user_id'].nunique()
        rows.append({
            '行为': b,
            '去重用户数': n_users,
            '渗透率(%)': round(n_users / total_users * 100, 2),
        })

    return pd.DataFrame(rows), total_users


def plot_behavior_penetration(penetration, total_users,
                              color_map=None, figsize=(9.8, 4.9)):
    """5.2 可视化：并行漏斗（100% 灰底槽 + 居中条带）。

    灰底槽 = 分母 100%，条带宽度 = 渗透率；按逻辑顺序 pv→fav→cart→buy 排列
    （**不按数值降序**），刻意不呈收口状——本图为并行口径、互不递减，
    画成收口漏斗会误导读者以为存在级间流失。
    """
    color_map = color_map or DEFAULT_COLOR_MAP

    stages = list(penetration['行为'])
    ref = penetration.set_index('行为')
    vals = ref.loc[stages, '渗透率(%)'].values
    nums = ref.loc[stages, '去重用户数'].values

    fig, ax = plt.subplots(figsize=figsize)
    n = len(stages)
    y_pos = np.arange(n)[::-1]        # 首个行为在顶、末个在底
    band_h = 0.72
    max_w = 100.0                     # 分母 = 100%

    for yi, v, s, num in zip(y_pos, vals, stages, nums):
        # 灰色底槽 = 分母 100%
        ax.barh(yi, max_w, left=0, color='#e9edf1',
                edgecolor='none', height=band_h, zorder=2)
        # 居中条带 = 渗透率
        ax.barh(yi, v, left=(max_w - v) / 2, color=color_map[s],
                edgecolor='white', linewidth=1.3, height=band_h, zorder=3)
        # 数字标签：带宽 ≥10% 放带内白字，否则溢出到带外右侧深色
        label = f"{num:,} 人 · {v:.1f}%"
        if v / max_w >= 0.10:
            ax.text(max_w / 2, yi, label, ha='center', va='center',
                    fontsize=11, fontweight='semibold', color='white', zorder=4)
        else:
            ax.text((max_w + v) / 2 + 1.5, yi, label, ha='left', va='center',
                    fontsize=10, fontweight='semibold', color='#333', zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_xlabel(f'渗透率（分母 = 全平台总活跃用户 {total_users:,} 人）')
    ax.set_title('用户行为渗透率（并行漏斗 · 四指标同分母、互不递减）',
                 fontweight='semibold', pad=12)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.25, linestyle='--', zorder=0)

    plt.tight_layout()
    plt.show()


# ============================================================
# 5.3 用户行为组合分布（互斥分组 · Buy 封顶）
# ============================================================


def compute_behavior_combination(df):
    """按「9 天内达到的最深行为」互斥分组（Buy 封顶）。

    分组优先级 Buy > Cart > Fav > PV，逐层剔除已归入更高优先级的用户：
        购买用户 = 有过 Buy（无论有无 PV/Fav/Cart）
        加购用户 = 有过 Cart 但无 Buy
        收藏用户 = 有过 Fav 但无 Cart、无 Buy
        仅浏览用户 = 只有 PV

    返回
    ----
    (combo, total_users)
        combo : DataFrame，列 = [用户分组, 用户数, 占比(%)]
        total_users : int，全平台总活跃用户数（四组之和恒等于此值）
    """
    total_users = df['user_id'].nunique()
    all_users = set(df['user_id'].unique())
    user_sets = {b: set(df.loc[df['behavior_type'] == b, 'user_id'].unique())
                 for b in ('pv', 'fav', 'cart', 'buy')}

    # Buy 封顶：逐层做集合差，保证互斥
    buy_u = user_sets['buy']
    cart_u = user_sets['cart'] - buy_u
    fav_u = user_sets['fav'] - buy_u - user_sets['cart']
    pv_u = all_users - buy_u - user_sets['cart'] - user_sets['fav']

    groups = {'buy': buy_u, 'cart': cart_u, 'fav': fav_u, 'pv': pv_u}
    combo = pd.DataFrame([
        {'用户分组': zh, '用户数': len(groups[key]),
         '占比(%)': round(len(groups[key]) / total_users * 100, 2)}
        for key, zh in COMBO_SPEC
    ])

    # 互斥完备性防呆：四组之和必须等于总活跃用户数
    assert combo['用户数'].sum() == total_users, \
        '互斥分组校验失败：四组之和 ≠ 总活跃用户数'

    return combo, total_users


def plot_behavior_combination(combo, total_users,
                              color_map=None, figsize=(10.5, 4.4)):
    """5.3 可视化：100% 横向堆叠条（互斥构成，四组刚好填满）。

    占比悬殊时段内放不下标签（如 4.7% / 5.8%），自动用引出线标注到条外上方，
    并按出现的第 N 个窄段递增 y 高度错开，避免两个窄段标签横向重叠。
    """
    color_map = color_map or DEFAULT_COLOR_MAP

    pcts = combo['占比(%)'].values
    nums = combo['用户数'].values
    names = combo['用户分组'].values
    cols = [color_map[key] for key, _ in COMBO_SPEC]     # 顺序与 combo 行一致

    fig, ax = plt.subplots(figsize=figsize)
    left, narrow_i = 0, 0
    for p, n, nm, c in zip(pcts, nums, names, cols):
        ax.barh(0, p, left=left, color=c, edgecolor='white',
                linewidth=1.5, height=0.55, zorder=3)
        if p >= 10:                    # 段够宽 → 段内白字
            ax.text(left + p / 2, 0, f"{nm}\n{n:,} 人 · {p:.1f}%",
                    ha='center', va='center', fontsize=10.5,
                    fontweight='semibold', color='white', zorder=4)
        else:                          # 段太窄 → 引出到条外上方并错开高度
            ty = 1.25 + 0.85 * narrow_i
            ax.annotate(f"{nm}\n{n:,} 人 · {p:.1f}%",
                        xy=(left + p / 2, 0.30), xytext=(left + p / 2, ty),
                        ha='center', va='bottom', fontsize=9.5, color='#333',
                        arrowprops=dict(arrowstyle='-', color='#999', lw=1),
                        zorder=4)
            narrow_i += 1
        left += p

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 3.4)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_xlabel(f'占全平台总活跃用户 {total_users:,} 人的比例')
    ax.set_title('用户行为组合分布（互斥分组 · 四组之和 = 100%）',
                 fontweight='semibold', pad=12)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    plt.show()


# ============================================================
# 5.4 用户行为流转分析（有序流转率 + 对照组 + Lift）
# ============================================================


def compute_behavior_conversion(df, min_ctrl_n=MIN_CTRL_N,
                                spec=CONVERSION_SPEC):
    """有序流转率 P(Buy|X) + 对照组 P(Buy|未X) + Lift。

    口径：
    - 流转率 P(Buy|X) = 「先做过 X、之后发生过购买」的去重用户 / 做过 X 的去重用户
      时间约束 first(X) < last(buy)：X 之后发生过**任意一笔**购买即计入（含老客复购），
      不限于首次购买（若限定首购在 X 之后，加购流转率将从 64.09% 降至 49.20%）。
    - 对照组 P(Buy|未X) = 从未做过 X 且买过的用户 / 从未做过 X 的用户
      「未做过 X」无从定义时间先后，故对照组为整体购买率、无法施加时序约束。
    - Lift = P(Buy|X) / P(Buy|未X)，> 1 表示该行为对购买有正向促进
      （受上述口径不对称影响，Lift 为**保守估计**）。

    参数
    ----
    df : DataFrame
        需含 user_id、behavior_type、datetime 三列。
    min_ctrl_n : int
        对照组样本量下限，低于此值标记「样本不足」（Lift 不可靠）。
    spec : sequence of (key, zh)
        展示顺序，默认 (cart, 加购) → (fav, 收藏) → (pv, 浏览)。

    返回
    ----
    (flow, baseline)
        flow : DataFrame，列 = [行为, 暴露人数, 流转人数, 流转率(%),
                                对照人数, 对照率(%), Lift, 对照组可信]
        baseline : float，全平台购买渗透率（%），作为基准线
    """
    total_users = df['user_id'].nunique()
    df_sorted = df.sort_values('datetime')
    first_t = (df_sorted.groupby(['user_id', 'behavior_type'])['datetime']
               .min().unstack())
    last_t = (df_sorted.groupby(['user_id', 'behavior_type'])['datetime']
              .max().unstack())

    baseline = (df.loc[df['behavior_type'] == 'buy', 'user_id'].nunique()
                / total_users * 100)

    rows = []
    for key, zh in spec:
        exposed = first_t[key].notna()                       # 做过 X
        converted = exposed & (first_t[key] < last_t['buy'])  # 且 X 之后买过
        unexposed = first_t[key].isna()                       # 从未做过 X（对照）
        ctrl_buy = unexposed & last_t['buy'].notna()

        n_exp, n_conv = int(exposed.sum()), int(converted.sum())
        n_unexp, n_ctrl = int(unexposed.sum()), int(ctrl_buy.sum())
        p_treat, p_ctrl = n_conv / n_exp * 100, n_ctrl / n_unexp * 100

        rows.append({
            '行为': zh,
            '暴露人数': n_exp, '流转人数': n_conv, '流转率(%)': round(p_treat, 2),
            '对照人数': n_unexp, '对照率(%)': round(p_ctrl, 2),
            'Lift': round(p_treat / p_ctrl, 3),
            '对照组可信': '✓' if n_unexp >= min_ctrl_n else f'样本不足(n={n_unexp})',
        })

    return pd.DataFrame(rows), baseline


def plot_behavior_conversion(flow, baseline, min_ctrl_n=MIN_CTRL_N,
                             color_map=None, figsize=(14, 5.4)):
    """5.4 可视化：左 = 绝对水平对比（含基准线）/ 右 = Lift 提升度。

    左图横坐标只写行为名（加购/收藏/浏览），「→购买」这一共性交给 y 轴承载——
    因为两根柱的语义不同（流转率含时序约束、对照组为整体购买率），
    横坐标写「加购→购买」会让读者误以为灰柱也属于该流转路径。

    右图 Lift 以 1.0 为基准线：>1 绿（正向）/ <1 橙（负向）/ 对照组不足灰（不可信）。
    """
    color_map = color_map or DEFAULT_COLOR_MAP
    dark_map = DARK_COLOR_MAP

    keys = [KEY_BY_ZH[zh] for zh in flow['行为']]
    bar_cols = [color_map[k] for k in keys]
    txt_cols = [dark_map[k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': [1.3, 1]})

    # ============ 左图：流转率 vs 对照组（分组柱状 + 全平台基准线）============
    x = np.arange(len(flow))
    w = 0.36
    b1 = ax1.bar(x - w/2, flow['流转率(%)'], w, color=bar_cols,
                 edgecolor='white', linewidth=1.2, label='流转率 P(Buy|X)', zorder=3)
    b2 = ax1.bar(x + w/2, flow['对照率(%)'], w, color='#b0bec5',
                 edgecolor='white', linewidth=1.2, label='对照组 P(Buy|未X)', zorder=3)

    for bar, v, c in zip(b1, flow['流转率(%)'], txt_cols):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.2, f'{v:.1f}%',
                 ha='center', fontsize=9.5, fontweight='semibold', color=c, zorder=4)

    for bar, v, n in zip(b2, flow['对照率(%)'], flow['对照人数']):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.2, f'{v:.1f}%',
                 ha='center', fontsize=9.5, color='#555', zorder=4)
        if n < min_ctrl_n:                     # 对照组样本不足 → 红字警示
            ax1.text(bar.get_x() + bar.get_width()/2, v + 6.5,
                     f'n={n}\n样本不足', ha='center', fontsize=8.5,
                     color='#c0392b', fontweight='semibold', zorder=4)

    ax1.axhline(baseline, ls='--', lw=1.4, color='#e53e3e', alpha=0.85, zorder=2)
    ax1.annotate(f'全平台基准线 {baseline:.1f}%',
                 xy=(-0.45, baseline), xytext=(-0.45, 75),
                 fontsize=9, color='#e53e3e', ha='left', fontweight='semibold',
                 arrowprops=dict(arrowstyle='-', color='#e53e3e', lw=0.8, alpha=0.6))

    ax1.set_xticks(x)
    ax1.set_xticklabels(list(flow['行为']))     # 只写行为名，「→购买」交给 y 轴
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('购买率 (%)')
    ax1.set_title('各行为「做过 vs 没做过」的购买率对比',
                  fontweight='semibold', fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 0.98),
               ncol=1, fontsize=9, frameon=False, handlelength=1.2)
    ax1.grid(axis='y', alpha=0.25, ls='--', zorder=0)
    for s in ('top', 'right'):
        ax1.spines[s].set_visible(False)

    # ============ 右图：Lift（基准线 1.0）============
    y = np.arange(len(flow))[::-1]             # 首个行为在顶
    lifts = flow['Lift'].values
    ok = flow['对照人数'].values >= min_ctrl_n
    lift_cols = ['#2f855a' if (l > 1 and o) else ('#c05621' if o else '#a0aec0')
                 for l, o in zip(lifts, ok)]   # 绿=正向 / 橙=负向 / 灰=不可信

    ax2.barh(y, lifts, color=lift_cols, edgecolor='white', linewidth=1.2,
             height=0.6, zorder=3)
    ax2.axvline(1.0, color='#333', ls='--', lw=1.4, zorder=2)
    ax2.text(1.0, len(flow) - 1.38, ' Lift = 1.0（无影响）',
             fontsize=9, color='#333', va='top', zorder=4)

    for yi, l, o in zip(y, lifts, ok):
        tag = f'{l:.2f}' if o else f'{l:.2f}（不可信）'
        ax2.text(l + 0.025, yi, tag, va='center', fontsize=10,
                 fontweight='semibold',
                 color='#2f855a' if (l > 1 and o) else ('#c05621' if o else '#718096'),
                 zorder=4)

    ax2.set_yticks(y)
    ax2.set_yticklabels(list(flow['行为']))
    ax2.set_xlim(0, 1.4)
    ax2.set_xticks(np.arange(0, 1.41, 0.2))
    ax2.set_xlabel('Lift = 流转率 / 对照组购买率')
    ax2.set_title('提升度 Lift（相对对照组的增量贡献）',
                  fontweight='semibold', fontsize=12)
    ax2.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    for s in ('top', 'right', 'left'):
        ax2.spines[s].set_visible(False)

    fig.suptitle('用户行为流转分析（有序流转率 · 用户级 · 跨商品口径）',
                 fontsize=16.5, fontweight='semibold', y=1.02)
    plt.tight_layout()
    plt.show()
