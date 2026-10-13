"""生成 Tableau 用的数据源（第 10.2 节）。

用法
----
    python tableau/build_tableau_data.py

输出
----
    tableau/users.csv   —— 用户级宽表（9,739 行，一人一行）
    tableau/hourly.csv  —— 小时 × 行为聚合（96 行）

为什么不用原始百万行数据
------------------------
cleaned_data.csv 有 75 MB / 999,530 行，直接喂 Tableau Public 会导致：
  ① 生成的 .hyper 提取体积大，上传慢且容易中断
  ② 漏斗 / 分层 / 零间隙率都要在 Tableau 里写复杂 LOD 公式，易错
改用「用户级宽表 + 小时聚合表」两张小表：
  ① 用户级 9,739 行仍是**真实粒度**（不是把结果预先揉成几行），
     漏斗渗透率、R×F 分层、未转化池、零间隙率都能在 Tableau 里直接聚合
  ② 两张表合计约 1 MB，发布秒级完成

口径与 notebooks/src 完全一致，不重新定义任何指标。
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_conversion_analysis.pre_purchase import compute_pre_purchase_gaps  # noqa: E402
from ecommerce_conversion_analysis.rfm import build_buyer_rfm, segment_buyers  # noqa: E402
from ecommerce_conversion_analysis.time_trends import compute_hourly_behavior  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
OUT_DIR = PROJECT_ROOT / "tableau"

# 未购买三分类的展示顺序（人数降序，也是报告里的顺序）
GROUP_ORDER = ["购买", "加购未购买", "收藏未购买", "仅浏览"]
BEHAVIOR_CN = {"pv": "浏览", "fav": "收藏", "cart": "加购", "buy": "购买"}


def build_users(df: pd.DataFrame) -> pd.DataFrame:
    """用户级宽表：一人一行，涵盖看板需要的全部维度。"""
    # ---- 1. 行为次数（全 9 天）----
    cnt = (df.groupby(["user_id", "behavior_type"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["pv", "fav", "cart", "buy"], fill_value=0))
    cnt.columns = ["浏览次数", "收藏次数", "加购次数", "购买次数"]

    # ---- 2. 活跃天数 / 浏览品类数 ----
    active_days = (df.assign(d=df["datetime"].dt.normalize())
                     .groupby("user_id")["d"].nunique().rename("活跃天数"))
    n_cats = (df.loc[df["behavior_type"] == "pv"]
                .groupby("user_id")["category_id"].nunique().rename("浏览品类数"))

    users = (df[["user_id"]].drop_duplicates().set_index("user_id")
             .join(cnt, how="left")
             .join(active_days, how="left")
             .join(n_cats, how="left")
             .fillna(0))
    users[["浏览次数", "收藏次数", "加购次数", "购买次数", "活跃天数", "浏览品类数"]] = \
        users[["浏览次数", "收藏次数", "加购次数", "购买次数", "活跃天数", "浏览品类数"]].astype(int)

    # ---- 3. 用户类型（Buy > Cart > Fav > PV 互斥分组，口径同 user_funnel 5.3）----
    has = {b: set(df.loc[df["behavior_type"] == b, "user_id"]) for b in BEHAVIOR_CN}
    users["用户类型"] = "仅浏览"
    users.loc[users.index.isin(has["fav"]), "用户类型"] = "收藏未购买"
    users.loc[users.index.isin(has["cart"]), "用户类型"] = "加购未购买"
    users.loc[users.index.isin(has["buy"]), "用户类型"] = "购买"
    users["用户类型"] = pd.Categorical(users["用户类型"], categories=GROUP_ORDER, ordered=True)

    # ---- 4. R×F 分层（仅买家有值，非买家标「非买家」）----
    rfm = segment_buyers(build_buyer_rfm(df))
    users["R段位"] = rfm["R_seg"].astype(str).reindex(users.index).fillna("非买家")
    users["F档位"] = rfm["F_seg"].astype(str).reindex(users.index).fillna("非买家")

    # ---- 5. 买家的首购/复购归类 ----
    pre = compute_pre_purchase_gaps(df)
    freq = pre.groupby("user_id")["purchase_seq"].max()
    first_u, repeat_u = set(freq[freq == 1].index), set(freq[freq >= 2].index)

    users["买家类型"] = "非买家"
    users.loc[users.index.isin(first_u), "买家类型"] = "首购用户"
    users.loc[users.index.isin(repeat_u), "买家类型"] = "复购用户"

    # ---- 6. 零间隙标记 ----
    # 新客口径：其唯一那次购买是否零间隙
    # 老客口径：复购订单（第 2 次及以后）中是否出现过零间隙
    zg_first = set(pre.loc[pre["user_id"].isin(first_u) & (pre["gap_before_purchase"] == 0), "user_id"])
    rep = pre[pre["user_id"].isin(repeat_u) & (pre["purchase_seq"] >= 2)]
    zg_repeat = set(rep.loc[rep["gap_before_purchase"] == 0, "user_id"])
    users["零间隙标记"] = users.index.isin(zg_first | zg_repeat).astype(int)

    # ---- 7. 收尾 ----
    users = users.reset_index()
    return users[["user_id", "浏览次数", "收藏次数", "加购次数", "购买次数",
                  "活跃天数", "浏览品类数", "用户类型", "买家类型",
                  "R段位", "F档位", "零间隙标记"]].rename(columns={"user_id": "用户ID"})


def build_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """小时 × 行为聚合，并预先算好「占该行为全天峰值的百分比」。"""
    _, piv = compute_hourly_behavior(df)
    long = (piv.reset_index()
               .melt(id_vars="hour", var_name="行为代码", value_name="行为条数"))
    long["行为类型"] = long["行为代码"].map(BEHAVIOR_CN)
    # 归一化到各自峰值 —— 这正是「流量高峰 ≠ 转化高峰」能看出来的关键
    peak = long.groupby("行为代码")["行为条数"].transform("max")
    long["占峰值百分比"] = (long["行为条数"] / peak * 100).round(2)
    long = long.rename(columns={"hour": "小时"})
    return long[["小时", "行为代码", "行为类型", "行为条数", "占峰值百分比"]]


def main() -> None:
    print("📖 读取数据 …")
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    print(f"   {len(df):,} 行")

    print("👥 构建用户级宽表 …")
    users = build_users(df)
    print(f"   {len(users):,} 行 × {users.shape[1]} 列")

    print("⏰ 构建小时聚合表 …")
    hourly = build_hourly(df)
    print(f"   {len(hourly):,} 行 × {hourly.shape[1]} 列")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    users.to_csv(OUT_DIR / "users.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(OUT_DIR / "hourly.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ {OUT_DIR.relative_to(PROJECT_ROOT)}/users.csv"
          f"（{(OUT_DIR / 'users.csv').stat().st_size / 1024:.0f} KB）")
    print(f"✅ {OUT_DIR.relative_to(PROJECT_ROOT)}/hourly.csv"
          f"（{(OUT_DIR / 'hourly.csv').stat().st_size / 1024:.0f} KB）")

    # ---- 自检：关键口径必须与报告一致 ----
    print("\n🔍 自检（应与报告数字一致）：")
    buyers = users[users["买家类型"] != "非买家"]
    for seg in ["首购用户", "复购用户"]:
        sub = buyers[buyers["买家类型"] == seg]
        print(f"   {seg:6s} {int(sub['零间隙标记'].sum()):>5,} / {len(sub):>5,} = "
              f"{sub['零间隙标记'].mean() * 100:.2f}%")
    print(f"   用户类型分布：{users['用户类型'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
