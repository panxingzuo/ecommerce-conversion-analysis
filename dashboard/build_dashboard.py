"""生成交互式 HTML 看板（第 10.1 节）。

用法
----
    python dashboard/build_dashboard.py

输出
----
    dashboard/index.html —— 自包含的交互看板（Plotly.js 走 CDN），
    双击即可在浏览器打开，也可部署到 GitHub Pages。

设计原则
--------
1. 图表数据全部现算，不硬编码 —— 保证看板与 notebook / SQL 结果同源
2. 只放 5 张图，每张对应报告里的一条核心发现，不做图表大杂烩
3. 每张图配「结论式副标题」，让人不看正文也知道该记住什么

⚠️ 正文（副标题、KPI 备注）里引用的具体数字是**编辑文案**，取自
   reports/summary_report.md 中已经验证过的结论值。若要换数据集，
   这些文案需要同步核对 —— 图里的数字会变，文案不会。

口径注意
--------
「加购未购买」必须用**集合差集**算，不能用 加购人数 − 购买人数：
两群人有重叠（5,302 人既加购又购买），直接相减得到的 634 无意义。
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

# ---- 定位项目根目录（与 notebook 同一套逻辑）----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_conversion_analysis.pre_purchase import compute_pre_purchase_gaps  # noqa: E402
from ecommerce_conversion_analysis.rfm import (  # noqa: E402
    build_buyer_rfm,
    build_nonbuyer_profile,
    compute_segment_grid,
    segment_buyers,
)
from ecommerce_conversion_analysis.time_trends import compute_hourly_behavior  # noqa: E402
from ecommerce_conversion_analysis.user_funnel import compute_behavior_penetration  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
OUT_PATH = PROJECT_ROOT / "dashboard" / "index.html"
JS_PATH = PROJECT_ROOT / "dashboard" / "plotly.min.js"

# ============================================================
# 视觉常量
# ============================================================

# 中文字体栈：在浏览者的机器上取第一个可用的
FONT = "PingFang SC, Microsoft YaHei, Hiragino Sans GB, Noto Sans CJK SC, sans-serif"

# 与 03 notebook 的 color_map 保持一致
C_PV, C_FAV, C_CART, C_BUY = "#3498db", "#2ecc71", "#f39c12", "#e74c3c"
C_MUTED = "#94a3b8"
C_TEXT = "#1e293b"
C_GRID = "#e2e8f0"

# 报告里「网络状况良好」的统一底色
PLOT_BG = "#ffffff"


def base_layout(fig, height=380, **kwargs):
    """给所有图套一层统一的排版设置。"""
    fig.update_layout(
        height=height,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, size=13, color=C_TEXT),
        margin=dict(l=16, r=24, t=16, b=16),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
        **kwargs,
    )
    return fig


# ============================================================
# 数据计算
# ============================================================


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def build_kpis(df: pd.DataFrame) -> list[dict]:
    """顶部 KPI 卡片 —— 按「全量 → 用户 → 买家 → 加购 → 差集」递进，最后一张是行动落点。"""
    total_users = df["user_id"].nunique()
    buy_u = set(df.loc[df["behavior_type"] == "buy", "user_id"])
    cart_u = set(df.loc[df["behavior_type"] == "cart", "user_id"])

    # ⚠️ 「加购未购买」必须用集合差集算，不能用 加购人数 − 购买人数：
    #    两个人群有重叠（5,302 人既加购又购买），直接相减得到的 634 是无意义的。
    cart_not_bought = len(cart_u - buy_u)

    return [
        {"label": "行为记录", "value": f"{len(df):,}", "note": "9 天全量"},
        {"label": "活跃用户", "value": f"{total_users:,}", "note": "去重人数"},
        {"label": "购买用户", "value": f"{len(buy_u):,}",
         "note": f"购买渗透率 {len(buy_u) / total_users * 100:.1f}%"},
        {"label": "加购用户", "value": f"{len(cart_u):,}",
         "note": f"渗透率 {len(cart_u) / total_users * 100:.1f}%，高于购买"},
        {"label": "加购未购买", "value": f"{cart_not_bought:,}", "note": "P0 优先召回人群"},
    ]


def fig_penetration(df: pd.DataFrame) -> go.Figure:
    """① 行为并行渗透率。"""
    pen, total = compute_behavior_penetration(df)
    labels = {"pv": "浏览 pv", "fav": "收藏 fav", "cart": "加购 cart", "buy": "购买 buy"}
    colors = {"pv": C_PV, "fav": C_FAV, "cart": C_CART, "buy": C_BUY}
    pen = pen.copy()
    pen["名称"] = pen["行为"].map(labels)
    pen["颜色"] = pen["行为"].map(colors)

    fig = go.Figure(
        go.Bar(
            x=pen["渗透率(%)"],
            y=pen["名称"],
            orientation="h",
            marker=dict(color=pen["颜色"]),
            text=[f"{u:,} 人 · {p:.2f}%" for u, p in zip(pen["去重用户数"], pen["渗透率(%)"])],
            textposition="outside",
            textfont=dict(family=FONT, size=13, color=C_TEXT),
            hovertemplate="%{y}<br>渗透率 %{x:.2f}%<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(range=[0, 118], showgrid=True, gridcolor=C_GRID, ticksuffix="%",
                     zeroline=False, title=None)
    fig.update_yaxes(showgrid=False, autorange="reversed", title=None)
    return base_layout(fig, height=300,
                       xaxis_title=f"渗透率（分母 = 全平台活跃用户 {total:,} 人）")


def fig_hourly(df: pd.DataFrame) -> go.Figure:
    """② 小时节律：流量高峰 ≠ 转化高峰。"""
    _, piv = compute_hourly_behavior(df)
    pv_pct = piv["pv"] / piv["pv"].max() * 100
    buy_pct = piv["buy"] / piv["buy"].max() * 100
    hours = list(piv.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=pv_pct, name="浏览 pv", mode="lines",
        line=dict(color=C_PV, width=2.5),
        hovertemplate="%{x}:00<br>占峰值 %{y:.1f}%<extra>浏览</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=buy_pct, name="购买 buy", mode="lines",
        line=dict(color=C_BUY, width=2.5),
        hovertemplate="%{x}:00<br>占峰值 %{y:.1f}%<extra>购买</extra>",
    ))
    # 高亮 10–16 点：转化效率相对更高的时段
    fig.add_vrect(x0=9.5, x1=16.5, fillcolor=C_CART, opacity=0.10, line_width=0,
                  annotation_text="10–16 点", annotation_position="top left",
                  annotation_font=dict(family=FONT, size=12, color="#b45309"))
    # 标记峰值小时
    fig.add_vline(x=21, line=dict(color=C_MUTED, width=1, dash="dot"))

    fig.update_xaxes(showgrid=False, dtick=2, ticksuffix=":00", title=None)
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, ticksuffix="%", range=[0, 112],
                     title="占各自全天峰值的比例")
    return base_layout(fig, height=340, hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                   xanchor="left", x=0, font=dict(size=13)))


def fig_segment_heatmap(df: pd.DataFrame) -> go.Figure:
    """③ R×F 九层分层热力图。"""
    rfm = segment_buyers(build_buyer_rfm(df))
    grid = compute_segment_grid(rfm)
    core = grid.drop(index="合计").drop(columns="合计")

    z = core.values
    text = [[f"{v:,}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(core.columns),
        y=list(core.index),
        text=text,
        texttemplate="%{text}",
        textfont=dict(family=FONT, size=15, color=C_TEXT),
        colorscale=[[0, "#f1f5f9"], [0.5, "#fcd9a8"], [1, "#e88b3c"]],
        showscale=False,
        hovertemplate="R %{y} × F %{x}<br>%{z:,} 人<extra></extra>",
        xgap=3, ygap=3,
    ))
    fig.update_xaxes(title="F 购买次数", showgrid=False, side="bottom")
    fig.update_yaxes(title="R 最近购买距今", showgrid=False, autorange="reversed")
    return base_layout(fig, height=340)


def fig_zero_gap(df: pd.DataFrame) -> go.Figure:
    """④ 老客 vs 新客：零间隙率。"""
    pre = compute_pre_purchase_gaps(df)
    freq = pre.groupby("user_id")["purchase_seq"].max()
    first_buyers, repeat_buyers = freq[freq == 1].index, freq[freq >= 2].index

    n_first = len(first_buyers)
    x_first = int((pre[pre["user_id"].isin(first_buyers)]["gap_before_purchase"] == 0).sum())
    rep = pre[pre["user_id"].isin(repeat_buyers) & (pre["purchase_seq"] >= 2)]
    n_repeat = len(repeat_buyers)
    x_repeat = int(rep[rep["gap_before_purchase"] == 0]["user_id"].nunique())

    rate_first = x_first / n_first * 100
    rate_repeat = x_repeat / n_repeat * 100

    fig = go.Figure(go.Bar(
        x=["新客（只买 1 次）", "老客（买 ≥2 次）"],
        y=[rate_first, rate_repeat],
        marker=dict(color=[C_PV, C_BUY]),
        text=[f"{rate_first:.2f}%<br><span style='font-size:12px'>{x_first:,} / {n_first:,} 人</span>",
              f"{rate_repeat:.2f}%<br><span style='font-size:12px'>{x_repeat:,} / {n_repeat:,} 人</span>"],
        textposition="outside",
        textfont=dict(family=FONT, size=14, color=C_TEXT),
        hovertemplate="%{x}<br>零间隙率 %{y:.2f}%<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(range=[0, 52], showgrid=True, gridcolor=C_GRID, ticksuffix="%",
                     title="零间隙率（不浏览/收藏/加购，直接下单）")
    return base_layout(fig, height=340)


def fig_nonbuyers(df: pd.DataFrame) -> go.Figure:
    """⑤ 未转化用户池：3,050 人卡在哪一步。"""
    _, prof = build_nonbuyer_profile(df)
    # 与图 ① 保持语义一致的配色：加购=橙、收藏=绿、仅浏览=蓝
    group_colors = {"加购未购买": C_CART, "收藏未购买": C_FAV, "仅浏览": C_PV}
    # 显式指定从上到下的顺序（与报告一致），不依赖 plotly 的默认分类排序
    order = ["加购未购买", "收藏未购买", "仅浏览"]
    prof = prof.reindex(order)
    colors = [group_colors[g] for g in prof.index]

    fig = go.Figure(go.Bar(
        x=prof["人数"],
        y=prof.index,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{n:,} 人　平均浏览 {b:.1f} 次" for n, b in zip(prof["人数"], prof["平均浏览"])],
        textposition="outside",
        textfont=dict(family=FONT, size=13, color=C_TEXT),
        hovertemplate="%{y}<br>%{x:,} 人<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_xaxes(range=[0, 2750], showgrid=True, gridcolor=C_GRID, title="人数")
    fig.update_yaxes(showgrid=False, autorange="reversed", title=None)
    return base_layout(fig, height=260)


# ============================================================
# 页面组装
# ============================================================

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>电商用户转化分析 · 交互看板</title>
<script src="plotly.min.js" charset="utf-8"></script>
<style>
  :root {{
    --text: #1e293b; --muted: #64748b; --line: #e2e8f0;
    --bg: #f8fafc; --card: #ffffff; --accent: #185fa5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: {font};
    -webkit-font-smoothing: antialiased; line-height: 1.6;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px 64px; }}
  header h1 {{ font-size: 26px; font-weight: 600; margin: 0 0 10px; letter-spacing: -.01em; }}
  header p {{ color: var(--muted); margin: 0; font-size: 15px; max-width: 720px; }}
  header .meta {{ color: var(--muted); font-size: 13px; margin-top: 14px; }}
  .kpis {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin: 28px 0 32px;
  }}
  .kpi {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }}
  .kpi .v {{ font-size: 24px; font-weight: 600; letter-spacing: -.02em; }}
  .kpi .l {{ font-size: 13px; color: var(--muted); margin-top: 2px; }}
  .kpi .n {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px 24px 12px; margin-bottom: 18px;
  }}
  .card h2 {{ font-size: 17px; font-weight: 600; margin: 0 0 6px; }}
  .card .sub {{ color: var(--muted); font-size: 14px; margin: 0 0 4px; }}
  .card .sub b {{ color: var(--text); font-weight: 600; }}
  .tag {{
    display: inline-block; font-size: 12px; color: var(--accent);
    background: #e6f1fb; border-radius: 6px; padding: 2px 8px; margin-right: 8px;
  }}
  footer {{ color: var(--muted); font-size: 13px; margin-top: 36px;
            border-top: 1px solid var(--line); padding-top: 18px; }}
  footer a {{ color: var(--accent); }}
  @media (max-width: 640px) {{
    .wrap {{ padding: 24px 14px 48px; }}
    header h1 {{ font-size: 21px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>电商用户转化分析 · 交互看板</h1>
    <p>基于淘宝 9 天近百万条用户行为数据（双 12 预热期），回答四个问题：<b>谁在买、怎么买、什么时候买、能不能预测</b>。</p>
    <div class="meta">数据：阿里天池 UserBehavior（公开数据集）｜999,530 条行为记录 ｜ 9,739 名活跃用户 ｜ 2017-11-25 ~ 12-03</div>
  </header>

  <div class="kpis">{kpis}</div>

  <main>
    <section class="card">
      <h2><span class="tag">01</span>行为渗透率</h2>
      <p class="sub">四类行为各自独立统计，分母统一。加购渗透率（75.19%）比购买渗透率（68.68%）还高，但两者人群重叠——真正的缺口是 <b>2,021 人加购了却没有下单</b>。</p>
      {fig1}
    </section>

    <section class="card">
      <h2><span class="tag">02</span>时间节律</h2>
      <p class="sub">流量高峰 ≠ 转化高峰。全天行为量在 <b>21:00</b> 达峰，但 10–16 点的购买量能达到自身峰值的 <b>91%</b>，同期浏览只有 <b>60%</b>。</p>
      {fig2}
    </section>

    <section class="card">
      <h2><span class="tag">03</span>用户分层</h2>
      <p class="sub">R×F 九层（M 因与 F 高度共线 ρ=0.985 而剔除）。右上角「最近买过 × 高频复购」<b>736 人</b>最该做会员维护；左下角「久未购买 × 只买一次」<b>1,106 人</b>最该召回。</p>
      {fig3}
    </section>

    <section class="card">
      <h2><span class="tag">04</span>老客 vs 新客</h2>
      <p class="sub">老客复购时 <b>39.99%</b> 是「不浏览、不收藏、不加购，直接下单」，新客首购只有 <b>3.27%</b>——<b>差 12 倍</b>。老客靠惯性，新客靠比较。</p>
      {fig4}
    </section>

    <section class="card">
      <h2><span class="tag">05</span>未转化用户池</h2>
      <p class="sub">3,050 名用户 9 天内一单未下。其中 <b>2,021 人加了购物车却没买</b>（平均浏览 81.6 次），是全平台召回价值最高的一块人群。</p>
      {fig5}
    </section>
  </main>

  <footer>
    <p>📄 完整论证见 <a href="../reports/summary_report.md">分析报告</a>　｜　🔍 核心分析的 SQL 实现见 <a href="../sql/analysis.sql">sql/analysis.sql</a>　｜　📓 分析过程见 <a href="../notebooks/03_analysis_visualization.ipynb">notebooks/</a></p>
    <p>⚠️ 本数据为观测性快照，无随机分流与对照组，所有结论均为<b>关联性</b>而非因果性；任何「提升 X%」的量化收益承诺都超出数据支撑范围。</p>
  </footer>
</div>
</body>
</html>
"""


def render() -> None:
    print("📖 读取数据 …")
    df = load_data()
    print(f"   {len(df):,} 行 × {df.shape[1]} 列")

    print("📊 计算各图表数据 …")
    fig1 = fig_penetration(df)
    print("   ① 行为渗透率 ✓")
    fig2 = fig_hourly(df)
    print("   ② 小时节律 ✓")
    fig3 = fig_segment_heatmap(df)
    print("   ③ R×F 九层 ✓")
    fig4 = fig_zero_gap(df)
    print("   ④ 老客 vs 新客 ✓")
    fig5 = fig_nonbuyers(df)
    print("   ⑤ 未转化用户池 ✓")

    # plotly.js 已在 <head> 里本地引入，所有图都不再重复内联
    def div(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False, "responsive": True})

    kpis = build_kpis(df)
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{k["value"]}</div>'
        f'<div class="l">{k["label"]}</div><div class="n">{k["note"]}</div></div>'
        for k in kpis
    )

    html = TEMPLATE.format(
        font=FONT,
        kpis=kpi_html,
        fig1=div(fig1),
        fig2=div(fig2),
        fig3=div(fig3),
        fig4=div(fig4),
        fig5=div(fig5),
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # plotly.js 落成本地文件：不依赖 CDN，离线可开；且内容静态，
    # 重建看板时不会反复产生大体积的 git 历史
    JS_PATH.write_text(get_plotlyjs(), encoding="utf-8")
    OUT_PATH.write_text(html, encoding="utf-8")

    print(f"\n✅ 已生成 {OUT_PATH.relative_to(PROJECT_ROOT)}"
          f"（{OUT_PATH.stat().st_size / 1024:.0f} KB）")
    print(f"✅ 已生成 {JS_PATH.relative_to(PROJECT_ROOT)}"
          f"（{JS_PATH.stat().st_size / 1024 / 1024:.1f} MB，静态依赖）")
    print("   双击 index.html 即可在浏览器打开，无需联网")


if __name__ == "__main__":
    render()
