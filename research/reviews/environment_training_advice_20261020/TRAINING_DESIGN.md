# 既有训练设置、评价口径与下一方案边界

本文的第一至六节都是**已执行研究的证据说明**，第七节是**尚未执行、请外部模型回答的设计问题**。不从失败结果反推新训练授权。精确原件投影见 [EVIDENCE.json](EVIDENCE.json)，来源与SHA见 [PROVENANCE.json](PROVENANCE.json)。

## 1. 数据、候选与证据上限

量价面板来源身份为现有QFQ快照，6,453,109行、5,576代码，2021-05-06至2026-10-15；沪深300使用Futu恢复替代 `SH.000300 / K_DAY / AuType.NONE`，1,399行、2021-01-04至2026-10-15。源SHA绑定见EVIDENCE；本次未读或重新审计行情。原TDX官方指数文件身份、官方交易日历parity与完整历史全市场/ST/上市/交易状态认证仍CANNOT_CONFIRM，不能用当前资格替代历史资格。

正式M0–M4使用同日同代码、五臂特征共同finite的交集；没有用V192/V193触发信号或未来label筛候选。先做全历史时序特征与当日观察全市场/行业聚合，再取共同keys。Top10选出后才连接隔离未来标签；缺选中结果不补股。不是既定D−1 Top100/Top200召回实验，也未证明该类池的赢家Recall。行业源是当前分类回溯，保留UNKNOWN且不推断补行业；**数值年界并不认证行业历史PIT**。

20年行业描述包是2006–2025的包内同日Top100频率统计；严格稳健季节增量仅一行业×月份格（C40，11月），不是20年模型收益回测，也不证明未来季节因果。S1统计先验取自该既有聚合链。

## 2. M0–M4究竟加了什么

所有臂用同算法、超参、label、共同keys、训练窗与排序规则，依次嵌套。旧baseline名字含V192/V193，但实际M0模型输入是V193快照计算的36个core；V192 XG与公式触发比较是独立诊断，不是训练输入或硬过滤器。M0没有因此被认证为原始Formula193。

| 臂 | 新增于上一臂的特征 | 重要边界 |
|---|---|---|
| M0 | 36个量价core，名称见下 | 原公式快照/引擎身份与历史来源限制仍保留 |
| M1 | index_ma5_position、index_ma20_position、index_return_1、index_return_5、stock_rs5、market_advancer_share、market_amount_activity | RS5=个股原日历5日收益−指数5日收益；涨跌家数/成交额为观察面板代理 |
| M2 | month_01…month_12 one-hot + 单列AUDITM | 月份取信号日D；AUDITM来自原公式，合同称3/4/12输出；不是三列新特征或S1的D+1月份锚 |
| M3 | 行业类别one-hot、industry_mean_ret5/20、industry_advancer_share、industry_amount_activity、stock_relative_industry_ret5 | 行业均值/上涨比例为同日观察行业聚合；RS为股5日收益减行业均值；当前行业回溯 |
| M4 | amv_proxy_sma10、amv_proxy_change5、amv_proxy_vs_ma20、amv_proxy_ma5_vs_ma13 | 所有观察股票成交额CNY之和→递归0.1新值+0.9旧值；其5日变化、相对MA20−1、MA5/MA13−1；不是官方0AMV算法 |

M1的指数位置=`index_close/MA5或MA20−1`；market_amount_activity=`当日总额/总额MA20`，industry_amount_activity为行业相同口径。M4的MA20/MA5/MA13以平滑后的proxy为输入。前述全局同日量对各股票相同，其是否通过非线性交互改善截面排序仍需实验，不能把“有这个字段”当作已学到或已剔除环境。

M0 core名称：DISTW, DISTB, C, YEL0, LCOUNT, HIST0, AMTR, VOLR, VMA20R, VSPHHV, FCOUNT, SFCOUNT, RET20P, GAIN60, RANGE60, RANGE120, JVAL, BKRSI, CPOS, UPSH, WASHSCR, B1PRE, BKEXIST, FRESH, CONT0, B1SIG, B2SIG, B3SIG, BREAKOUT, BKRES, RTETYPE, RISK0, HEALTH, LOWEXT, KQUAL, LIQSTD。逐项计算请查 [旧包原M0源码](../g1_20261020/archive/frozen_m0/model_selection.py) 与formula_adapter；不凭名称猜金融含义。本文源码对照为S1保留的M0副本，原件SHA在provenance；不将该SHA当作一项新的历史fit-time认证。

## 3. 算法、窗、label、purge与选择

`sklearn.ensemble.HistGradientBoostingRegressor`，squared_error，max_iter=150、learning_rate=.05、max_depth=3、max_leaf_nodes=7、min_samples_leaf=200、l2_regularization=10、early_stopping=false、random_state=193。原研究没有超参搜索。每个信号日权重总和相等，再归一为样本权重均值1；它是回归损失，不是显式listwise/pairwise排行损失。每臂/每horizon独立模型，M0–M4正式18+72=90次fit；canary与正式不混计。

训练窗是`[fold.start−DateOffset(years=2), fold.start)`，不是每次都取两个完整自然年。训练标签必须finite且 `label_end_date < fold.start`，完整历史先warmup；未知test标签不进入特征或训练调参。

| fold | 名义训练信号窗 | 测试窗 |
|---|---|---|
| 2024H1 | [2022-01-01,2024-01-01) | 2024-01-01…06-30 |
| 2024H2 | [2022-07-01,2024-07-01) | 2024-07-01…12-31 |
| 2025H1 | [2023-01-01,2025-01-01) | 2025-01-01…06-30 |
| 2025H2 | [2023-07-01,2025-07-01) | 2025-07-01…12-31 |
| 2026H1 | [2024-01-01,2026-01-01) | 2026-01-01…06-30 |
| 2026H2部分期 | [2024-07-01,2026-07-01) | 2026-07-01…10-15 |

实际行仅含绑定观察日。label=`close(T+h)/close(T)−1`，h∈{1,2,5}取共同观察日历session，非个股行shift；无目标close记未知。每日分数降序、canonical代码升序tie-break，选10只后冻结keys，缺结果不补位。没有次日开盘、成交、滑点、费用、仓位、NAV或复投模型。净收益与可执行性CANNOT_CONFIRM。

## 4. 既有结果：支持集必须分开

PV原冻结门只验M4对M0：每horizon独立，配对完整日期平均差>0、至少4/6正fold、日Top10中位数不差、选中事件P10不差；六fold包含2026H2部分期。10日block/1000次/seed193为原辅助区间，不改变门。M1–M3仅探索比较，不是预冻结winner选择或组合。

| PV h2，六fold含部分期 | paired增量（百分点） | paired日期 | 正fold/6 |
|---|---:|---:|---:|
| M1−M0 | −0.6830 | 610 | 2 |
| M2−M0 | −0.6372 | 608 | 1 |
| M3−M0 | −0.7754 | 616 | 1 |
| M4−M0 | −1.4857 | 615 | 0 |

这些是独立收据四位舍入的percentage-point差，不是bp。EVIDENCE保留h1/h2/h5全部原结果与精确M4聚合门。M4三horizon均失败，保留M4_GATE_FAILED；不因某个P10子门通过选最佳horizon。每臂自己有效日期均值有不同支持集，不能直接相减替代上述配对差。

S1单独加到M0，不是M4或M4+S1：四列 `log(L_im/L_i), E_im/(E_im+100), support_years, missing`。`L_i=(H_i+100)/(E_i+100)`，`L_im=(H_im+100L_i)/(E_im+100)`。每行只用2006≤year<D.year的完整历史年度，月份为D+1原观察session；年末仍以D.year cutoff，不能偷用D+1年。UNKNOWN/无映射/零support编码[0,0,0,1]，保留旧keys。全六fold旧M0 identity精确绑定后复用6个M0 checkpoint，仅6次S1 h2正式fit。

S1 primary只用2024H1至2026H1五完整fold，564完整配对日/37缺失日：mean delta −13.05394 bp，原20日block区间[−34.96726,−5.39352] bp，1/5正；中位数/P10亦差，五门皆失败，状态S1_DEVELOPMENT_GATE_FAILED。冻结门为≥10bp、原区间LCB>0、≥4/5正、median/P10不差。h1/h5是同一h2选股列表的诊断；partial2026H2不投票。原可复现性PASS不是普适CI覆盖或科学突破；不得调κ、翻方向、换名复跑失败S1。

## 5. G1与归因不能回答什么

G1在原M0 Top50五rank层内按既有先验q30/q70分高低组，原样保留ties和缺失停门，统计每层 `mean(h2−M0pred|high)−mean(h2−M0pred|low)`，层/有效日等权。它不是新策略、S1收益改进或行业中性因果效应。

原点满足 **raw_h2_spread 30.12333442 bp − M0_score_gap 0.36684348 bp = residual_spread 29.75649094 bp**。547/601有效日；五fold有效数110/123/106/112/96；q*=2.9877439760955102区间[−6.01749,+65.53048] bp跨零，信息门INCONCLUSIVE。八指定DGP校准/验证通过仅支持指定模拟，不证明真实普适95%覆盖。早期bootstrap权重错误、CI修复零覆盖停门和HAC常数校准失败均在旧包保留。

归因的两fold（2024H2/2025H1）约87.46%原有符号点贡献、原第一rank1–10层约54.04%贡献是主控已给出的四舍五入描述；来源聚合表在旧归因包，本次未重算。不是87%方差解释、环境归因或只留赢家fold的理由。原第一层仅贡献五层中的1/5，不是新Top10策略lift。2026H2部分期64/71日点为−4.75409bp，仅描述。

同日高低差抵消的是相同加性市场项；高低组异质beta、行业构成、非线性市场/波动/流动性状态交互仍可能不同。现有残差没有实施beta/行业/环境条件调整，不能声称已经排除这些影响。先验在行业×D+1月份×信号年单元内为常数；缺乏同单元先验变化不能识别单元内因果效应。重复代码、单元或support不是独立N。

## 6. 资源与复现状态

原训练4线程。S1正式6次fit记录782.84秒，maxRSS=16,338,780,160 bytes，另一个peak memory footprint指标=62,939,529,480 bytes；两种指标不同，不当成同一峰值。S1冻结停止线maxRSS=80GiB、available≥8GiB；它不是任何未来实验的资源许可。PV完整90次fit的统一历时/峰值不在本包来源闭合范围，标CANNOT_CONFIRM，不按单次fit线性推算。

公开包无原行情/labels/predictions/mask/权重/完整公式与engine/本机registry环境；不能独立重跑真实CI、S1/PV训练或原mask480k模拟。原有PASS按原范围复用，本次仅投影JSON、哈希/链接/隐私检查，不导入或执行业务代码。不新增全量数据审计，已认证未变输入今后可FAST_BIND；新增依赖或原资格/PIT缺口必须明确。

## 7. 尚未执行：请外部模型设计一项可否证实验

请先区分目标：是固定Top10可执行收益、固定候选池未来赢家Recall、还是行业/环境条件下的信息增量；当前证据不能把它们互换。需设计诊断时，列所需字段、支持集、可识别条件与最小门；不把归因集中度直接写成交易规则。

候选讨论可以比较显式排行损失、独立相对收益目标、环境×个股/行业交互、冻结M0后的单一透明加分、行业内排序；**它们目前均未按本包新合同执行或获批**。外部建议必须从中择优一个主要机制、冻结对照和信息预算，不要求都跑。换label/候选池/TopN/损失主目标须新独立合同，不能改旧失败门。

设计要写清历时outer验证与仅训练窗内的inner调参、purge/embargo、已暴露历史的DIRECTION_ONLY上限、真正未来封存/不可窥视条件、换入换出配对、行业/beta/规模/流动性/风险暴露、费用/换手/可买性、缺失覆盖、日期依赖与多重比较、canary/资源上限和停止条件。阈值可建议但必须说明依据、单位/分母、训练前冻结与失败不救火；不许保证一定解决。完整请求见 [CHATGPT_PROMPT.md](CHATGPT_PROMPT.md)。
