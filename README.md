# 电商用户转化分析

> 基于淘宝 9 天近百万条用户行为数据（双 12 预热期），分析用户的**购买路径、行为分层与转化瓶颈**，并构建购买预测模型。最终输出 **6 条核心发现与 5 项策略建议**（P0/P1/P2 优先级）。

---

## 核心发现

**1）有 2,021 人「加购了但没买」—— 全平台价值最高的一块可运营人群。**

加购渗透率 75.2%（7,323 人），比购买渗透率 68.7% 还高。这 2,021 人平均加购 6.1 次、浏览 81.6 次——不是"随手一点"，而是认真挑过却没下单。

**2）老客和新客是「两种模式」，差距是碾压级的。**

老客复购时 **39.99%** 是「不浏览、不收藏、不加购，直接下单」；新客首购只有 **3.27%**——**差 12 倍**。同样成交一单，高频买家平均浏览 18 次，低频买家 58 次，差 40 次。

**3）行为数据能预测的部分很有限 —— 但圈人是有效的。**

模型 AUC = **0.5862**，只比「直接看一个特征」高 0.007；9 个特征里只有「购买历史」真正起作用。但按预算取 Top10% 圈人，**命中率比随机发券高 44%**（Lift = 1.44）。

> 📄 完整论证见 **[reports/summary_report.md](reports/summary_report.md)** —— 含 6 条发现的证据链、5 项策略建议与优先级排序。

---

## 数据说明

| 项 | 值 |
| :--- | :--- |
| 数据来源 | [阿里天池 UserBehavior](https://tianchi.aliyun.com/dataset/649)（公开数据集） |
| 数据规模 | 999,530 条行为记录 / 9,739 名活跃用户 |
| 时间范围 | 2017-11-25 ~ 12-03（9 天，覆盖双 12 预热期） |
| 行为类型 | 浏览（pv）/ 收藏（fav）/ 加购（cart）/ 购买（buy） |

---

## 分析框架

| Notebook | 内容 | 方法 |
| :--- | :--- | :--- |
| `01_data_exploration` | 数据宏观了解 | 分布分析、字段理解 |
| `02_data_cleaning` | 数据清洗与质量校验 | 去重、时间窗过滤、异常审查 |
| `03_analysis_visualization` | **七章完整分析**（见下） | 描述统计 / 可视化 / 假设检验 |
| `04_purchase_prediction` | 购买预测建模 | 逻辑回归 / 随机森林 / XGBoost |
| `05_sql_analysis` | **SQL 版本**：核心分析用标准 SQL 重写 | CTE / 窗口函数 / 与 Python 逐项对账 |

**`03` 的七个章节：**

| # | 章节 | 核心内容 |
| :--- | :--- | :--- |
| 一 | 探索性数据分析（EDA） | 行为分布、用户活跃度、异常账号审查 |
| 二 | 时间维度分析 | 日趋势（DAU/行为量）、小时节律 |
| 三 | 商品维度分析 | 热销品类/SKU、集中度（CR1/CR5/HHI）、品类转化率 |
| 四 | 行为路径与转化 | 购买前路径模式、路径耗时、零间隙购买分解 |
| 五 | 行为渗透与流转（漏斗） | 并行渗透率 + 有序流转率 + Lift |
| 六 | 购买用户分层（RFM） | R×F 九层分层、切点敏感性、未转化用户池 |
| 七 | 群体行为差异的显著性检验 | 两比例 Z 检验、Mann-Whitney U、效应量 |

---

## 交互看板

同一套结论做了**两个版本**的交互看板，互为验证。

### ① Plotly 版 · 自包含 HTML（`dashboard/index.html`）

- **本地查看**：clone 后直接双击打开 —— `plotly.js` 已随仓库分发，**无需联网**
- **在线查看**：仓库启用 GitHub Pages 后，可通过 `https://panxingzuo.github.io/ecommerce-conversion-analysis/dashboard/` 访问

### ② Tableau Public 版 · 在线作品

🔗 **https://public.tableau.com/app/profile/xingzuo.pan/viz/-12_17918784363500/1**

数据源不喂原始百万行，而由 `tableau/build_tableau_data.py` 生成两张小表（用户级宽表 9,739 行 + 小时聚合 96 行），口径与 `src/` 完全一致；5 张工作表组装为一个仪表板。

**两个版本共同承载的内容**：KPI 概览 → ① 行为渗透率 → ② 时间节律（流量高峰 ≠ 转化高峰）→ ③ R×F 九层分层 → ④ 老客 vs 新客 → ⑤ 未转化用户池

```bash
# 重新生成（数据全部现算，不硬编码）
python dashboard/build_dashboard.py      # → dashboard/index.html
python tableau/build_tableau_data.py     # → tableau/users.csv + tableau/hourly.csv
```

---

## SQL 版本

核心分析除 Python 实现外，另用**标准 SQL**（通用写法，不绑定特定数据库方言）重写了一遍，位于 **[`sql/analysis.sql`](sql/analysis.sql)**：

| 段 | 查询 | 覆盖的 SQL 能力 |
| :--- | :--- | :--- |
| `01` | 漏斗：行为并行渗透率 | CTE、`COUNT(DISTINCT CASE WHEN)`、`CROSS JOIN` |
| `02` | 买家 R×F 九层分层 | 条件聚合、`CASE WHEN` 分桶、日期差换算 |
| `03` | 老客 vs 新客零间隙率 | **累计计数窗口 + `LAG()` 跨行取差**（同秒排序权重） |
| `04` | 观察窗特征表（9 个特征） | 时间窗过滤、多指标聚合并行、`LEFT JOIN` 保底 |

**每一段的结果都与 pandas 实现逐项对账至完全一致**——26 个汇总指标 + `9 × 9,739` 个特征值，零差异。执行与对账过程见 **[`notebooks/05_sql_analysis.ipynb`](notebooks/05_sql_analysis.ipynb)**。

---

## 技术亮点

- **建模防数据泄漏**：采用「观察窗 7 天 → 预测窗 2 天」的时间切分，并用 `assert` 硬校验窗口不重叠 + 特征-标签相关性排查（防 target leakage）
- **口径严格分离**：漏斗与路径分析明确区分「有序口径」（要求时序）与「集合口径」（不要求时序），避免同数字不同含义导致的误读
- **效应量优先于 p 值**：9,739 样本下 p 值必然显著，故改用 Cohen's h / Cliff's δ 做效应量排序，并区分符号（方向）与绝对值（强度）
- **维度共线性诊断**：RFM 分层前先做 Spearman 相关 + 重合度双重检验，剔除与 F 高度共线（ρ = 0.985）的 M 维度，避免重复计权
- **按数据形态选模型**：逻辑回归（CV AUC 0.6015）胜过随机森林与 XGBoost —— 本数据信号以线性/单调为主，复杂树模型反而过拟合
- **Python + SQL 双实现并对账**：核心分析另用标准 SQL 重写，并逐项比对至与 pandas **完全一致**。这不是把同一件事做两遍——**它同时是一次代码交叉校验**：SQL 的 `ORDER BY` 强制显式声明排序键，而 pandas 的同秒排序是隐式的，写 SQL 的过程会把这些隐藏假设逼出来
- **可视化双平台交付**：同一套结论分别用 Plotly（自包含 HTML，离线可开）与 Tableau Public（在线可交互）实现，说明结论不依赖特定工具

---

## 项目结构

```
ecommerce-conversion-analysis/
├── notebooks/
│   ├── 01_data_exploration.ipynb        # 数据宏观了解
│   ├── 02_data_cleaning.ipynb           # 数据清洗与质量校验
│   ├── 03_analysis_visualization.ipynb  # 七章完整分析
│   ├── 04_purchase_prediction.ipynb     # 购买预测建模
│   └── 05_sql_analysis.ipynb            # SQL 版本：建库、执行与逐项对账
├── sql/
│   └── analysis.sql                     # 核心分析的 SQL 实现（4 段查询）
├── dashboard/
│   ├── build_dashboard.py               # 看板生成脚本
│   ├── index.html                       # 交互看板（自包含，离线可开）
│   └── plotly.min.js                    # 绘图库本地副本（不依赖 CDN）
├── tableau/
│   ├── build_tableau_data.py            # 生成 Tableau 数据源
│   ├── users.csv                        # 用户级宽表（9,739 行 × 12 列）
│   └── hourly.csv                       # 小时 × 行为聚合（96 行）
├── reports/
│   └── summary_report.md                # 分析报告（结论 + 策略建议）
├── src/
│   └── ecommerce_conversion_analysis/   # 可复用分析模块
│       ├── eda.py                       # EDA 相关
│       ├── time_trends.py               # 时间维度
│       ├── item_category.py             # 商品与品类
│       ├── pre_purchase.py              # 购买前路径
│       ├── user_funnel.py               # 渗透与漏斗
│       └── rfm.py                       # RFM 分层
├── pyproject.toml                       # 依赖声明
├── uv.lock                              # 依赖版本锁定
└── README.md
```

---

## 如何运行

```bash
# 1. 安装依赖（需先安装 uv：https://docs.astral.sh/uv/）
uv sync

# 2. 准备数据
#    从阿里天池下载 UserBehavior 数据集，解压后置于 data/ 目录
#    依次运行 01 → 02，生成清洗后的 cleaned_data.csv

# 3. 运行分析
#    依次运行 notebooks/03 → 04

# 4. 运行 SQL 版本
#    运行 notebooks/05_sql_analysis.ipynb：自动建库 data/ecommerce.db，
#    逐段执行 sql/analysis.sql，并与 Python 结果逐项对账

# 5. 生成交互看板
#    python dashboard/build_dashboard.py  →  输出 dashboard/index.html

# 6. 生成 Tableau 数据源
#    python tableau/build_tableau_data.py  →  输出 tableau/users.csv + tableau/hourly.csv
#    上传至 Tableau Public / Web Authoring，即可复现上面的在线看板
```

> `data/`（数据文件，含 `ecommerce.db`）与 `.venv/`（虚拟环境）已在 `.gitignore` 中排除，不随仓库分发。

---

## 说明与局限

- **观测性数据，无因果结论**：本数据集没有随机分流、干预与对照组，所有结论均为「关联性」而非「因果性」；任何"提升 X%"的量化收益承诺都超出数据支撑范围
- **窗口仅 9 天，且在大促预热期**：用户行为受双 12 影响，结论外推到常规周期需重新验证
- **存在左删失**：11/25 之前的行为不可见，"首次购买前路径"可能被低估
- **无金额字段**：数据不含价格、订单金额与利润，无法计算 ROI / 客单价 / GMV，所有建议均为行为层面
- **要验证某个动作是否有效，必须另做 A/B 实验**（随机分组 + 留对照组）
