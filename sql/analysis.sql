-- ============================================================
-- 电商用户转化分析 · SQL 版本
-- ============================================================
-- 目的：把 pandas 实现的核心分析用标准 SQL 重写，验证「同一批业务问题
--       可以用 SQL 解决」。本文件是 SQL 的唯一事实来源——notebook 只负责
--       读取、执行与对账，不复制这里的代码。
--
-- 输入表：user_behavior
--   user_id            INT   用户 ID
--   item_id            INT   商品 ID
--   category_id        INT   品类 ID
--   behavior_type      TEXT  行为类型：pv 浏览 / fav 收藏 / cart 加购 / buy 购买
--   timestamp          INT   Unix 秒级时间戳
--   datetime           TEXT  北京时间（UTC+8），'YYYY-MM-DD HH:MM:SS'
--   date               TEXT  日期 'YYYY-MM-DD'
--   hour / weekday     INT   小时 / 星期
--   is_suspicious_bot  INT   可疑爬虫标记 0/1
--   行数 999,530 ｜ 唯一用户 9,739 ｜ 覆盖 2017-11-25 ~ 12-03
--
-- 方言：通用标准 SQL（CTE + 窗口函数 + CASE WHEN + 条件聚合），不依赖特定
--       数据库；已在 SQLite 上验证通过。
-- 执行：见 notebooks/05_sql_analysis.ipynb
-- ============================================================


-- ============================================================
-- 01  漏斗：用户行为并行渗透率（对应 03 notebook §5.2）
-- ============================================================
-- 口径
--   四类行为**各自独立**统计去重用户数，互不要求先后顺序——不是逐级递减的真漏斗。
--   分母统一 = 全平台总活跃用户数（9,739）。
--   因此四个渗透率之和 > 100%：同一用户可同时计入多个分子。
--   ⚠️ 不能写成「浏览 → 收藏 → 加购 → 购买」逐级筛选的口径，那会制造出
--      本不存在的级间流失（本平台 83% 的购买根本不经过加购）。
--
-- 对账基准（Python 版：src/ecommerce_conversion_analysis/user_funnel.py）
--   pv 9,706 / 99.66% ｜ fav 3,882 / 39.86% ｜ cart 7,323 / 75.19% ｜ buy 6,689 / 68.68%
-- ============================================================

-- @query: 01_funnel_penetration
WITH total AS (
    -- 分母：全平台总活跃用户数
    SELECT COUNT(DISTINCT user_id) AS total_users
    FROM user_behavior
),
by_behavior AS (
    -- 分子：每类行为的去重用户数
    SELECT behavior_type,
           COUNT(DISTINCT user_id) AS users
    FROM user_behavior
    GROUP BY behavior_type
)
SELECT b.behavior_type                           AS behavior,
       b.users                                   AS users,
       ROUND(b.users * 100.0 / t.total_users, 2) AS penetration_pct
FROM by_behavior b
CROSS JOIN total t
ORDER BY CASE b.behavior_type
             WHEN 'pv'   THEN 1
             WHEN 'fav'  THEN 2
             WHEN 'cart' THEN 3
             WHEN 'buy'  THEN 4
         END;



-- ============================================================
-- 02  用户分层：买家 R×F 九层（对应 03 notebook §6.4）
-- ============================================================
-- 口径
--   R（Recency）  = 最近一次购买距分析截止时刻 2017-12-04 00:00 的天数。
--                   **用浮点保留日内精度，不取整**——取整会让同日购买者大量并列，
--                   分档失去意义。
--   F（Frequency）= 窗口内购买次数。
--   M            = **已剔除**。本数据集无价格字段，M 只能用「购买商品数」代理，
--                   而它与 F 高度共线（Spearman ρ = 0.985，90.3% 的用户两值完全相同），
--                   保留等于把同一件事算两遍权重。
--
--   分层切点（与 Python 版 R_BINS / F_BINS 完全一致）
--     R：≤1 天 → 1 天以内 ｜ 1–4 天 → 1–4 天 ｜ >4 天 → 4 天以上
--     F：=1 → 单次 ｜ 2–3 → 复购 ｜ ≥4 → 高频复购
--   3 × 3 = 9 层，九层之和 = 购买用户总数 6,689
--
-- 对账基准（Python 版：src/ecommerce_conversion_analysis/rfm.py）
--   R 段位＼F 档位    单次    复购  高频复购    合计
--   1 天以内         323     700      736    1,759
--   1–4 天           831   1,158      843    2,832
--   4 天以上       1,106     757      235    2,098
--   合计           2,260   2,615    1,814    6,689
-- ============================================================

-- @query: 02_rf_segments
WITH buyer_rfm AS (
    -- 基础指标：一人一行，仅购买用户
    SELECT user_id,
           MAX(datetime) AS last_buy,
           COUNT(*)      AS frequency,
           -- R：截止时刻 − 最后一次购买，秒差 ÷ 86400 换算成天（浮点，不取整）
           (strftime('%s', '2017-12-04 00:00:00')
            - strftime('%s', MAX(datetime))) / 86400.0 AS recency_days
    FROM user_behavior
    WHERE behavior_type = 'buy'
    GROUP BY user_id
),
segmented AS (
    -- 打标签：R 三档 × F 三档
    SELECT user_id,
           CASE WHEN recency_days <= 1 THEN '1 天以内'
                WHEN recency_days <= 4 THEN '1–4 天'
                ELSE '4 天以上' END AS r_seg,
           CASE WHEN frequency = 1 THEN '单次'
                WHEN frequency <= 3 THEN '复购'
                ELSE '高频复购' END AS f_seg
    FROM buyer_rfm
)
SELECT r_seg                                            AS r_segment,
       SUM(CASE WHEN f_seg = '单次'     THEN 1 ELSE 0 END) AS single_buyers,
       SUM(CASE WHEN f_seg = '复购'     THEN 1 ELSE 0 END) AS repeat_buyers,
       SUM(CASE WHEN f_seg = '高频复购' THEN 1 ELSE 0 END) AS frequent_buyers,
       COUNT(*)                                         AS total
FROM segmented
GROUP BY r_seg
ORDER BY CASE r_seg WHEN '1 天以内' THEN 1 WHEN '1–4 天' THEN 2 ELSE 3 END;



-- ============================================================
-- 03  零间隙率：老客 vs 新客的决策路径差异（对应 03 notebook §7.3 H2）
-- ============================================================
-- 口径（关键：gap 是「次数」，不是「时间」）
--   gap_before_purchase = 这次购买之前、自上一次购买以来，用户做了多少次
--                         **非购买行为**（pv / fav / cart）。
--                         首次购买：从样本起点（11-25）算起。
--   gap = 0 → 自上次购买以来一次浏览/收藏/加购都没有 = **不比较、直接下单**。
--
--   同秒事件规则：同一秒内 buy 排在 pv/fav/cart 之后，保证计数正确。
--   （这正是 behavior_order 存在的意义，不是随便排的）
--
--   用户分组：
--     新客 = 窗口内只买过 1 次（n_purchases = 1）
--     老客 = 窗口内买过 ≥2 次（n_purchases >= 2）
--   分子口径两端不完全对称：
--     新客 → 其唯一那次购买是否零间隙
--     老客 → 复购订单（第 2 次及以后）中**至少有一次**零间隙的去重用户数
--
-- ⚠️ 左删失：11-25 之前的行为不可见，新客「首购前的真实路径」被截断，
--    故新客零间隙率是被**高估**的（上界），真实值只会更低。
--
-- 对账基准（Python 版：src/ecommerce_conversion_analysis/pre_purchase.py）
--   新客    74 / 2,260 = 3.27%
--   老客 1,771 / 4,429 = 39.99%
--   购买事件总数 20,359 ｜ 买家总数 6,689
-- ============================================================

-- @query: 03_zero_gap_rate
WITH ordered AS (
    SELECT user_id, behavior_type, timestamp, item_id, category_id,
           -- 同秒排序权重：buy 必须排在非购买行为之后
           CASE behavior_type
               WHEN 'pv'   THEN 0
               WHEN 'fav'  THEN 1
               WHEN 'cart' THEN 2
               WHEN 'buy'  THEN 3
           END AS behavior_order
    FROM user_behavior
),
running AS (
    SELECT user_id,
           behavior_type,
           -- 累计非购买行为次数（buy 行本身贡献 0）—— 必须显式写 ROWS
           SUM(CASE WHEN behavior_type = 'buy' THEN 0 ELSE 1 END)
               OVER (PARTITION BY user_id
                     ORDER BY timestamp, behavior_order, item_id, category_id
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_nonbuy,
           -- 这是该用户的第几次购买
           SUM(CASE WHEN behavior_type = 'buy' THEN 1 ELSE 0 END)
               OVER (PARTITION BY user_id
                     ORDER BY timestamp, behavior_order, item_id, category_id
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS purchase_seq
    FROM ordered
),
buys AS (
    -- 只看购买事件
    SELECT user_id, purchase_seq, cum_nonbuy
    FROM running
    WHERE behavior_type = 'buy'
),
gaps AS (
    -- 本次购买 与 上一次购买 之间的非购买行为次数
    SELECT user_id,
           purchase_seq,
           cum_nonbuy - COALESCE(
               LAG(cum_nonbuy) OVER (PARTITION BY user_id ORDER BY purchase_seq), 0
           ) AS gap_before_purchase
    FROM buys
),
buyer_stats AS (
    -- 每个买家一行：总共买几次 + 首购是否零间隙 + 复购是否出现过零间隙
    SELECT user_id,
           MAX(purchase_seq) AS n_purchases,
           MAX(CASE WHEN purchase_seq  = 1 AND gap_before_purchase = 0 THEN 1 ELSE 0 END) AS zero_gap_first,
           MAX(CASE WHEN purchase_seq >= 2 AND gap_before_purchase = 0 THEN 1 ELSE 0 END) AS zero_gap_repeat
    FROM gaps
    GROUP BY user_id
)
SELECT CASE WHEN n_purchases = 1 THEN 'first_time' ELSE 'repeat' END AS buyer_type,
       COUNT(*) AS n_users,
       SUM(CASE WHEN n_purchases = 1 THEN zero_gap_first
                ELSE zero_gap_repeat END)                                AS n_zero_gap,
       ROUND(100.0 * SUM(CASE WHEN n_purchases = 1 THEN zero_gap_first
                              ELSE zero_gap_repeat END) / COUNT(*), 2)   AS zero_gap_pct
FROM buyer_stats
GROUP BY buyer_type
ORDER BY CASE buyer_type WHEN 'first_time' THEN 1 ELSE 2 END;



-- ============================================================
-- 04  观察窗特征表：喂给购买预测模型的那 9 个特征（对应 04 notebook §1.4）
-- ============================================================
-- 窗口
--   观察窗：2017-11-25 ~ 12-01（7 天）——**所有特征只能用这段数据算**
--   预测窗：2017-12-02 ~ 12-03（2 天）——标签 y 的来源，本段不使用
--   ⚠️ 这条时间边界就是防标签泄漏的红线：特征表里绝不能出现预测窗的任何信息。
--
-- 9 个特征分五组
--   行为强度  pv_cnt / fav_cnt / cart_cnt   观察窗内浏览 / 收藏 / 加购次数
--   购买      obs_buy_cnt                   观察窗内购买次数（预期最强特征）
--   活跃      active_days                   有任意行为的去重天数
--             days_since_last_active        最后活跃距 12-01 的天数
--   广度      n_cats                        浏览过的去重品类数
--   效率      cart_depth                    加购转化率 = cart_cnt / pv_cnt
--   购买近期性 obs_recency_days             观察窗内最后购买距 12-01 天数（RFM 的 R）
--
-- ⚠️ 必须从「全量活跃用户」出发 LEFT JOIN 观察窗聚合
--    实测有 1 个用户在观察窗零行为、只在预测窗出现。若直接从观察窗出发建表，
--    样本会少 1 行（9,738 而非 9,739）。这个人的所有特征取 0，
--    obs_recency_days 取哨兵值 8。
--    （8 是哨兵值不是笔误：观察窗内购买最多只能"距 12-01 有 6 天"，
--      取 8 让"从没买过"和"6 天前买过"能被区分开。）
--
-- 对账基准（Python 版：04 notebook §1.4）
--   9,739 行 × 9 列 ｜ 缺失值 0 ｜ 9 个特征逐行逐列完全一致
-- ============================================================

-- @query: 04_observation_features
WITH all_users AS (
    -- 全量活跃用户：保证每人一行（含观察窗零行为的用户）
    SELECT DISTINCT user_id
    FROM user_behavior
),
obs AS (
    -- 观察窗：11-25 ~ 12-01 23:59:59
    SELECT user_id, behavior_type, category_id, date
    FROM user_behavior
    WHERE datetime <= '2017-12-01 23:59:59'
),
behavior_cnt AS (
    -- 一次扫描算出行为频次 / 活跃天数 / 浏览广度 / 最后活跃日
    SELECT user_id,
           SUM(CASE WHEN behavior_type = 'pv'   THEN 1 ELSE 0 END) AS pv_cnt,
           SUM(CASE WHEN behavior_type = 'fav'  THEN 1 ELSE 0 END) AS fav_cnt,
           SUM(CASE WHEN behavior_type = 'cart' THEN 1 ELSE 0 END) AS cart_cnt,
           SUM(CASE WHEN behavior_type = 'buy'  THEN 1 ELSE 0 END) AS obs_buy_cnt,
           COUNT(DISTINCT date)                                      AS active_days,
           COUNT(DISTINCT CASE WHEN behavior_type = 'pv'
                               THEN category_id END)                 AS n_cats,
           MAX(date)                                                 AS last_active_date
    FROM obs
    GROUP BY user_id
),
last_buy AS (
    -- 观察窗内最后一次购买
    SELECT user_id, MAX(date) AS last_buy_date
    FROM obs
    WHERE behavior_type = 'buy'
    GROUP BY user_id
)
SELECT u.user_id,
       COALESCE(b.pv_cnt,       0) AS pv_cnt,
       COALESCE(b.fav_cnt,      0) AS fav_cnt,
       COALESCE(b.cart_cnt,     0) AS cart_cnt,
       COALESCE(b.obs_buy_cnt,  0) AS obs_buy_cnt,
       COALESCE(b.active_days,  0) AS active_days,
       COALESCE(julianday('2017-12-01') - julianday(b.last_active_date), 0)
                                   AS days_since_last_active,
       COALESCE(b.n_cats,       0) AS n_cats,
       CASE WHEN COALESCE(b.pv_cnt, 0) = 0 THEN 0.0
            ELSE b.cart_cnt * 1.0 / b.pv_cnt
       END                         AS cart_depth,
       COALESCE(julianday('2017-12-01') - julianday(l.last_buy_date), 8)
                                   AS obs_recency_days
FROM all_users u
LEFT JOIN behavior_cnt b ON b.user_id = u.user_id
LEFT JOIN last_buy     l ON l.user_id = u.user_id
ORDER BY u.user_id;

