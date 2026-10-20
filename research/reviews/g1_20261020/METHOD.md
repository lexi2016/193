# 方法与冻结评价

本文解释既有实现；没有新增 fit、预测、评价、模拟或假设检验。数值源为 [聚合结果](evidence/aggregate_results.json)，精确配置见 [冻结配置](evidence/frozen_configs.json)。收益使用小数单位，1 bp=0.0001。

## 1. 20年描述与先验

2006–2025 的包内行情按原复权代理计算同日涨幅，得到每日 Top100。对行业 i、时间格 c，实际命中 H 为入选次数；期望 E 为逐日 `min(100, all_rankable) × industry_rankable/all_rankable` 之和。`L_ic=H_ic/E_ic`，`S_ic=L_ic/L_i`。分年、半期和剔除最强三年的 pooled 分子/分母使用同一年度集合，不平均年度 lift。MAIN 与独立剔除复牌间隔人口均需过门。报告中的 1 格严格季节增量是冻结描述门的交集，未执行跨行业显著性或因果认证。

S1 对信号日 D 仅用 `2006 <= year < D.year` 的完整历史年度；月份锚为原观察日历的 D+1。κ=100：

```
L_i  = (H_i + κ) / (E_i + κ)
L_im = (H_im + κ L_i) / (E_im + κ)
prior = log(L_im/L_i)
support = E_im/(E_im+κ)
```

四列为 log relative lift、support、support_years、missing。UNKNOWN、无映射或 E<=0 中性编码，保留 keys；缺 D+1 锚 fail closed。严格年度 cutoff 只保证数值使用年界，不能消除当前行业分类回溯的 PIT/存续偏差。

## 2. S1 开发比较

原 M0 输入、特征及六 fold checkpoint 身份复用；S1 唯一机制为新增四列先验。正式六次 S1 fit，完整投票期为 2024H1、2024H2、2025H1、2025H2、2026H1；2026H2_PARTIAL_TO_2026_10_15 不投票。h2 标签=`close(T+2)/close(T)-1`，不是次日开盘可执行收益。训练 split 清除尚未成熟标签，原实现保存在 archive。

每臂按 score 降序、代码升序取 Top10，任一选中收益缺失就排除整对日期，不补股。564 完整配对日，37 缺失。冻结门为平均增量≥10 bp、原 20 日 block 区间下界>0、至少4/5完整 fold 正增量、中位数不差、event P10不差；五门全部失败。原 block 法按合法非循环起点重采样，不跨 fold/缺失日。此包保留当时方法及独立算术重现，不另行认证其一般置信覆盖。

## 3. G1 条件残差诊断

在 M0 原全池保存预测中按分数/代码确定 Top50，再分成 1–10、11–20、21–30、31–40、41–50 五层。每层排除 missing prior；至少6个已知值，numpy linear 的 q30/q70；高组 `prior>=q70`、低组 `prior<=q30`，并列全部保留。组重叠、每组不足2、任一选中 h2 缺失均使该层无效。五层全有效才保留整日，不用标签决定分组，不补选。

```
x_tb = mean(h2 − M0pred | high,t,b) − mean(h2 − M0pred | low,t,b)
x_t  = (1/5) sum_b x_tb
point = (1/N) sum_valid_t x_t
```

重要恒等式：**residual_spread = raw_h2_spread − M0_score_gap**。原聚合中 raw=0.003012333442049523，score_gap=0.00003668434809245033，residual=0.002975649093957072。不能把高低原收益差当成全部残差，也不能漏掉预测差的扣除。

primary N=547，原日历601日。每 fold 有效日为110/123/106/112/96，分母117/125/117/126/116；主点按有效日等权，不按 fold 等权。冻结信息门：point≥5 bp、区间下界>0、正完整 fold≥4、总体 coverage≥80%、各 fold≥70%。上界<5 bp 则低于资源线，其余 INCONCLUSIVE。coverage 是诊断有效日期比例，不是赢家召回率。

原 bootstrap 把170个合法起点混合抽样，与等有效日点估计权重不一致；原机器 PASS 和 [15.38312,64.03277] bp 区间未获统计接受。后续固定 fold 均值/逆覆盖权重修复遇到187日零 block覆盖，真实 bootstrap 为0并停止；没有丢日期、缩短 block 或改点估计救门。

## 4. HAC20 与 q*（固定点、fold 均值及 mask）

按原始日历位置 t_i，保留跨 fold 协方差：

```
K_ij = max(1 − abs(t_i−t_j)/20, 0)
X = five fold dummy columns; M = I − X (X'X)^−1 X'
z_i = x_i − accepted_fold_mean[fold_i]
D = tr(KM) = 460.30270684044433
c = N/D = 1.1883484321755418
V = c (z'Kz)/N² = 1.4336657135501758e−6
interval = accepted_point ± q* sqrt(V)
```

该区间的代码字段保留历史名称 `ci95`；其实际覆盖的外推边界见下段。常数1.959963984540054在原四种模拟中的覆盖为94.40%、94.26%、92.84%、92.06%，后两种 Wilson 下界<93%，正确停门且未读取真实残差。

q* 批次只改变临界值。八种预冻结 DGP：IID；MA1；MA5；MA5×fold标准差(1,2,1,2,1)；Gaussian AR(0.9)；t5 AR(0.9)；MA5×(2,1,2,1,2)；t5 AR(0.9)×`1+.75 sin(2πt/600)`。AR创新标准化、初始0并丢512步，MA前缀5。全601日生成后按原547 mask选择；每次重新估计fold均值/等日点，真均值按 n_f/N 加权。固定非零fold均值为(0,1,−1,2,−2)。

每情形校准50,000次：`Y=abs((point−truth)/SE)`，取最小k满足 `BinomCDF(k−1;50000,.95) >= 1−.01/8`；k=47,647，U_s为第k小值，无插值。`q*=max(1.95996,U_1,...,U_8)=2.9877439760955102`。99%族置信说明只针对这八个指定DGP的 Monte Carlo 分位数上界。

锁定 q* 和校准 SHA 后，每情形独立验证10,000次，全部 Wilson95 下界≥93%才执行一次真实CI。PCG64 SeedSequence(19320261021,spawn_key=(phase,scenario))，phase0/1区分校准/验证。累计校准400,000、验证80,000。八验证命中9951/9942/9932/9903/9567/9600/9908/9542。

最终区间 [−6.0174947561,+65.5304766352] bp 跨零，信息门 **INCONCLUSIVE**。模拟验证通过不是普适真实数据95%覆盖保证；没有证明候选策略收益、Recall@K、ranking lift、封存集或生产门。
