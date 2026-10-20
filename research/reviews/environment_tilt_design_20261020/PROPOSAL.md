# 冻结M0的环境调制行业月份加分：设计草案

状态：`CONTRACT_DRAFT_NOT_EXECUTION_READY / CONTRACT_ONLY_NOT_COUNTED / NOT_EXECUTED`。本轮只修订方案、定位元数据和准备材料；未授权训练、预测、诊断、回测、pretruth、数据认证、模型加载或新数据采集。历史最高仍为DIRECTION_ONLY。研究专用，production_allowed=false、buy_gate_updated=false、orders_sent=false。

## 1. 目标和唯一机制

唯一问题：在原M0共同finite候选池上，冻结M0分数和树结构，仅在原M0 Top50内，按事前观察市场广度调制同一个行业月份相对先验，能否改善最终等权Top10的h2毛收益？原模型、label、行业月份先验、Top10、并列和缺失规则保持其既有定义。Top50限制是本路线新增且显式冻结的重排policy，不能描述成原M0池本身。冻结权重保护基础预测函数，但不能保证组合性能零损失。

采用附件A的冻结M0透明加分骨架，移除其前置Beta/波动率正交化诊断、附加硬过滤和互相矛盾的Gate0阈值。环境门与先验的乘积是本路线唯一新评分机制；只学习一个非负标量λ。附件B的显式交互仅保留备选，收到的正文在HYPOTH处截断，未读取完整合同、不实施、不宣称比较验证。默认重点行业仍指“该月相对该行业自身全年更强”，并非长期绝对行业优势；不新增label或ranking loss。

S1失败对应四列先验进入HGB重新训练，保留原失败决定；它不是本新机制的评价结果。G1是原Top50分层残差信息统计量，区间跨零表示INCONCLUSIVE，不能当作新Top10收益改善或无效证明。分裂预算挤占、过拟合、beta混淆均只是解释假设，尚非已确认原因；support_years没有进入新评分，不因此声称旧列就是未来泄漏。

## 2. 评分的完整定义（所有新数值为待最终冻结的政策提议）

信号日T收盘后，在原共同finite的(date, canonical_code)集合U_T上，使用对应原M0 h2外层模型的已保存分数p。先按p降序、代码升序稳定排序，固定C_T=前K只，K=min(50,|U_T|)，原Top10为前min(10,|U_T|)。同分的第50名按代码截断，不能扩大到所有并列股；后续不扩大C_T，不用label改变U_T或C_T。

仅使用原producer `_calendar_daily_market` 的 `market_advancer_share` b_T：全观察面板同日有有效一观察session收益的股票中，收益>0的比例（flat留在分母），在切候选池前计算，非Top50内广度。b只依赖≤T价格及既有共同观察日历，不是官方完整全市场上涨率。建议g_T=max(0,2b_T−1)，因此g∈[0,1]；0.5表示上涨股票过半，线性增加避免额外分箱/阈值搜索。此结构选择未看新结果、不承诺强市一定最优。不再并用指数趋势、成交活跃度、Beta/vol gate。

原S1先验P_iT=log(L_im/L_i)，κ=100，L_i=(ΣH_i+100)/(ΣE_i+100)，L_im=(ΣH_im+100L_i)/(ΣE_im+100)。仅聚合2006≤year<T.year的完整年；m为T+1共同观察session的月份，年末仍以T.year截断。当前行业分类回溯的限制保留。定义z_iT=tanh(max(P_iT,0))，故0≤z≤1；无正先验不扣分。UNKNOWN/未映射/零support沿原missing标记令z=0，保留原股票，不用support_years或support另做权重。非finite且未被原missing规则解释的先验是合同错误，应停止；不能默默把Inf当大加分。

s_T=min(population_std(p over C_T, ddof=0),0.01)，单位为收益小数；0.01是100bp尺度上限提议，不是估计最优值。评分S_iT=p_iT+λ_k·s_T·g_T·z_iT，仅C_T参与最终Top10。0≤λ≤0.25，因此0≤S−p≤0.0025（25bp的预测分数上界），同时≤0.25·原TopK分数标准差；这是分数改变量，不是收益、换股数、风险或跟踪误差上界。tanh去除先验极值，尺度cap防止标准差极端放大；不需要再估计标准化参数。少于50只照K计算；少于10只保留不足名单并标为不可评价完整Top10，绝不补选。

λ=0分支直接返回原M0 Top10和原分数，避免0×NaN；名单需逐key、rank精确相同。λ>0时s=0返回原名单；合法b缺失令g=0并记录deactivated_missing_environment（不移除日期或股票），有限但不在[0,1]或同日冲突值则停止。M0分数非finite/重复key/歧义行业join、缺T+1日历锚或非finite算术结果均停止、保留证据，不过滤坏行重造池。未知prior按上述原规则中性；p原精度保存、不提前四舍五入，S并列仍代码升序。环境fallback和zero-scale的日期保留并报覆盖。

## 3. λ训练推荐：严格历时的一维拟合，当前被元数据依赖阻塞

主推荐保留每个外层fold一次、事前两年窗内的历时λ拟合，不在缺缓存时静默改固定λ，也不允许外层收益选λ。固定政策λ是另一种可行设计，但并非本草案主方法，若改用须显式重新裁决及冻结；不能把它当缺依赖时的自动fallback。

外层k使用[f_k.start−DateOffset(years=2),f_k.start)的信号日，λ训练label必须满足label_end<f_k.start。每一个训练日期t的M0分数必须来自时点上先于t训练完的历史模型，满足max(model_train_label_end)<t，模型训练信号窗/参数/36列及原共同finite policy可追溯。禁止用当前外层fold模型对其自身训练窗预测后拟合λ。历史已暴露不等于执行时点泄漏可忽略；回溯研究也必须守这个历时不变量。

每个t先按历时p取C_t、算u_i= s_t g_t z_i，再接入仅用于二层训练的成熟h2。仅当该日全部K个选定训练结果finite才进入λ拟合，不按label删股/换股，不把零填充当真值；其训练有效日mask与外层评价mask分别保存。训练标签在这些过去日期是已知的，但所用p必须严格out-of-time。报告窗内所有可评分日、完整标签日、缺失日、g激活日、u非零日；零有效日/全部u截面无变化则λ=0（NO_IDENTIFIABLE_TILT，非效果通过）；无法确认整个两年评分窗的源覆盖时STOP，不缩窗救火。

对每个完整训练日，以C_t等权去均值：a_it=(u_it−mean_C u)/0.01，r_it=((y_it−mean_C y)−(p_it−mean_C p))/0.01。目标为date-equal mean_t[mean_i (r_it−λa_it)^2]+ρλ²，建议ρ=1固定，λ=clip(mean_t mean_i(a r)/(mean_t mean_i(a²)+ρ),0,0.25)。ρ=1是归一单位下强收缩的政策提议，未调优，不允许网格、seed重试或外层选择。这个去均值残差只是二层标量拟合目标；最终唯一primary仍是原绝对h2 Top10毛收益差，未换label或收益定义。

因为ρ及gate/clip/Top50均事前固定，本路线不做inner超参搜索，不能同时声称“nested CV选择λ”又只列每fold一个闭式fit。内层历时分数产生需自己的上游模型时点合同；只能复用已合格分数，缺失时新生成需要另行明确授权和预算。本轮不生成。两年OOF/forward分数、训练label可用时点、coverage passport绑定、producer版本等未闭合，因此不能保证零基础fit、总计仅六次计算、15秒或某内存峰值。保留原六个外层M0仅说明外层基础模型可复用。

## 4. 当前元数据依赖结论

已定位：原M0/PV合同与输出manifest、六个M0 h2 checkpoint sidecar；原producer只对外层test矩阵xt predict并保存test predictions。2024H1 sidecar training signals为2022-01-04…2023-12-27、test为2024-01-02…2024-06-28；这证明记录的train/test身份，不能证明两年训练窗OOF分数。原S1 准备收据保存panel/calendar/m0/s1四个缓存的记录SHA和六fold身份parity；M0特征缓存是特征，先验特征缓存是追加先验后的特征，都不是历时分数缓存。

本次限定在已知A股模型目录、原S1 worktree、原M0 run的文件导航及小元数据内检索，没有定位覆盖每个两年窗的严格历时OOF/forward score manifest或该新用途的认证passport，记NOT_LOCATED；记录SHA的payload未打开/未重hash，实际缓存值、完整性及可读性仍CANNOT_CONFIRM。这不是不存在的证明，也不授权扩大数据审计。原producer环境字段定义可确认；独立持久化的事前广度窄缓存未定位，仅在源码可重建不等于缓存已存在。

执行前blocking：完整历时分数manifest及模型训练label_end边界、score/feature/prior/environment/calendar按key一对一绑定、训练与外层label隔离、现有passport/FAST_BIND或有范围的新依赖认证、输入/执行代码/配置SHA、实际资源canary合同。仅沿原producer恢复缺依赖，不另建数据底板；不以purge字段替代模型训练结束metadata。详见DEPENDENCY_SUMMARY.json；来源仅以逻辑ID引用。

## 5. 外层评价：有效日由本候选名单决定

保留原半年度外层窗：2024H1、2024H2、2025H1、2025H2、2026H1五完整fold作历史开发描述；2026H2至2026-10-15仅partial诊断不投票。不得从训练窗/partial反选参数。所有已暴露历史最高DIRECTION_ONLY；未来sealed来源、观察截止、期间长度和功效均未冻结，不能许诺223/618日足够，也不虚构已有未见样本。

先冻结两臂名单、分数、排序及hash，再由隔离评价器读取h2=close(T+2)/close(T)−1，共同观察日历、原复权与来源限制保留。仅两臂均为完整10只且各10个h2全部finite的date进入paired mask；任一缺一个选中label就排除整对date，记录每臂selected/known/unknown slots及缺失原因；不除以10_valid、不补选、不用label重造池。新候选paired日期数未知，不继承S1的564、G1的547或PV各臂支持集；基准parity只比较同日期、同keys、同排序，不要求候选结果复现旧别的候选paired分母。

primary μ=完整paired日期上mean_t[mean_10(y_tilt)−mean_10(y_M0)]，收益小数乘10000为bp；pool/fold/date权重不得混。报告每fold均值、全日期等权均值、median、选中事件P10、有效日与日历覆盖、正fold、partial单列。h1/h5若未来授权仅同一h2名单诊断，不新优化。拟议历史筛选线为μ≥8bp、正完整fold≥4/5，并且paired日Top10收益CVaR5%候选≥M0；是待最终冻结的有用效果/稳定性/尾部政策，不是数据推导最优值、旧门改写或统计突破。任一未过只关闭这个精确冻结路线；推断证据未定不能改写成所有行业先验或算法家族无效。

## 6. 不确定性：只定义一个工作估计，不挪用旧q*

G1的q*=2.9877439760955102来自Bartlett20 HAC、特定八DGP和547有效日mask的冻结校准，既不是stationary/moving-block bootstrap，也不可直接迁移给本候选Top10差序列。旧bootstrap权重问题、覆盖停门和历史结论原样保留。

本草案选择单一calendar-aware Bartlett20 HAC作为历史描述性工作SE。顺序必须是：固定评分参数与名单→隔离接label→固定paired mask→求等有效日μ→计算SE。以五完整fold的原共同观察session连续序列j计时，a_j为paired indicator，n=Σa，e_j=a_j(δ_j−μ)，缺失日e=0仅表示无估计贡献而非零收益；γ_l=Σ_{j=l+1..N}e_j e_{j−l}，V=[γ_0+2Σ_{l=1..20}(1−l/21)γ_l]/n²，SE=√V。原日历间距和缺失mask保留，不压缩缺失日；允许相邻fold边界的lag乘积，不错误假定fold独立。n=0、非finite、非正V均标推断CANNOT_CONFIRM（n=0也无μ）；严禁将异常SE设0制造确定性。若N不足21则仅报μ/coverage和CANNOT_CONFIRM，不扩展调lag。

20-session窗口取月度量级作为预先指定工作尺度，覆盖h2重叠和部分月度相关，不声称覆盖所有非平稳/长依赖。可同时展示μ±1.96SE，但名称必须为“未校准HAC正态参考区间”；不得标普适95%置信区间、用其LCB当确认门或做反复CI校准直到PASS。本轮不计算、不新增模拟。真正未见验证需在开封前另行冻结样本、estimand、缺失处理、一次性检验方法/阈值及资源；此处未具备，confirmation gate=null。既有历史多臂选择的研究者自由度如实披露，不能把14项机械计数当完整多重检验族，也不把历史显著性洗成前瞻证据。

## 7. 换股、风险和经济解释

在完整等权10只的paired date，A为M0、B为候选，D=A\B、I=B\A，k=|I|=|D|。严格恒等式δ=(Σ_I y−Σ_D y)/10；k>0时等于(k/10)(mean_I y−mean_D y)，k=0时δ=0。不保证换出都在M0末尾或换入只在11–30；按真实rank记录，允许11–50。缺label日不使用该等式的10_valid变体。

报告保留/换出/换入、原rank、正先验、环境值与加分分布、平均/分位数换股数、行业权重和HHI、总增量及fold集中度；λ上界不保证≤2.5次换股、≤0.05 beta或阶段贡献≤70%。beta及行业贡献分别报告，不强制先行业中性化；行业轮动允许贡献目标收益，同行业超额转负不自动杀整个路线。依赖字段如事前beta/规模/流动性/行业同口径收益未定位时标CANNOT_CONFIRM，未来单独绑定原source后才能算，不能静默新增risk producer。无需为了人为5%分类错配线重开全量PIT审计；当前分类回溯仅支持DIRECTION_ONLY，未来真实性部署/PIT仍阻塞。

CVaR5%是真尾部均值而非5%分位数：将同一paired日Top10收益排序，m=0.05n，取floor(m)个最差值加(m−floor(m))倍下一值，除以m；n>0有效。候选和M0各自的lower-tail CVaR差作拟议保守guard；δ序列CVaR另作诊断，不混对象。最差日、去最好5日敏感性、beta/行业收益分解仅诊断，不允许据此删日救主结果。重复股票/行业月单元不是独立N。

h2是重叠收盘区间毛收益，未定义资金占用、重叠持仓、成交/可买性、T+1交收、费用、滑点或净值；不能画日复利h2 NAV/最大回撤、用名单变化推出确定费率或把毛收益称净收益。未来可执行收益另立合同，不把通用25bp费用线或价限比例假定为已认证事实。

## 8. 后续阶段与停止规则

本次仅设计文档和元数据材料，登记状态CONTRACT_ONLY_NOT_COUNTED，不写registry/中央状态。独立Reviewer审查本草案及公开材料后，主控可按既有授权同步文档；这不授权训练，也不消除blocking依赖。

未来执行顺序：先明确授权及补齐上述元数据→冻结最终参数/验收/执行hash和passport binding→在任何真实检查、canary、推理/训练或评价启动前沿既有registry登记→小范围工程canary（synthetic先验边界、lambda0名单exact parity、时点/keys、资源）→获独立范围PASS后才有界正式执行。设计不预设资源限额：沿当前宿主负载与canary制定，最多4线程提议，RSS与footprint分别记录；不把footprint随意解释成虚拟内存/页缓存，也不把旧S1耗时线性外推。

身份/时点/label隔离/key冲突停止；无法定位依赖如实BLOCKED；覆盖未知不造PASS。科学筛选失败保留全部结果并关闭本精确组合，不换名翻向、放大λ、调gate/ρ/Top50或删fold重跑。此规则不封禁算法家族。任何部署、生产、BUY、服务/计划任务和订单均不在授权范围。

## 9. 有界OOF恢复设计与需要裁决的历史缺口（不执行）

最小历时score恢复方案沿原半年更新cadence，所需h2段为2022H1、2022H2、2023H1、2023H2、2024H1、2024H2、2025H1、2025H2、2026H1，共9段；这是需要覆盖的段数，绝不是新增fit预算。每段内的训练结束必须先于该段第一评分日，对每一条score记录仍验证max(model_train_label_end)<score_date。模型超参、两年名义训练窗、purge、warmup和原共同finite规则照原producer，不靠缩窗填空。每段所需key为该段所有信号日原共同finite U_t上的(date,canonical_code)，先有全池分数才可锁Top50；不能只用历史Top10/旧G1高低组代替。若只有已经完整、身份一致的Top50窄缓存，也须有全池Top50 selection provenance及boundaries才能采用，不能默认为足够。

| 外层λ用途 | 所需过去信号窗（右开） | 六外层checkpoint的test元数据可覆盖部分 | 尚未定位的早期score段 |
|---|---|---|---|
| 2024H1 | 2022-01-01…2024-01-01 | 无 | 2022H1/H2、2023H1/H2 |
| 2024H2 | 2022-07-01…2024-07-01 | 2024H1 | 2022H2、2023H1/H2 |
| 2025H1 | 2023-01-01…2025-01-01 | 2024H1/H2 | 2023H1/H2 |
| 2025H2 | 2023-07-01…2025-07-01 | 2024H1/H2、2025H1 | 2023H2 |
| 2026H1 | 2024-01-01…2026-01-01 | 2024H1/H2、2025H1/H2 | 无早期段，但payload与label-end绑定待确认 |
| 2026H2部分期 | 2024-07-01…2026-07-01 | 2024H2、2025H1/H2、2026H1 | 无早期段，但payload与label-end绑定待确认 |

表中范围是名义半开窗，实际共同观察日由原日历决定，训练label_end还需对外层start purge。现有2026H2外层test不是任何上述λ训练的必要分数段，只作partial外层评价。记录过test keys/模型SHA说明元数据可复用，不等于本次读过预测payload或验证了score值。早期score状态NOT_LOCATED不能写为“从未生成”。

更早的基础模型训练历史是另一种缺口：2022H1、2022H2、2023H1内层M0分别要求名义两年窗起于2020-01-01、2020-07-01、2021-01-01；原绑定面板始于2021-05-06，common finite始于2021-06-04，无法由这一个已绑定快照证明那些名义训练窗完整覆盖。这是BOUND_INPUT_SPAN_SHORTFALL，不只是“再predict一次”能解决。2023H2的名义窗始于2021-07-01，处于现有common-span内，但实际日覆盖、各特征warmup与模型训练标签成熟性仍待同producer metadata绑定，不升级为完整认证。

最小后续顺序：先由既有owner定位早期合格score/模型的manifest；已有且身份时点合格的优先复用。若无可证明的早期scores，先定位是否有原同源、同复权、同共同finite且含所需2020起训练历史的既有producer产物/manifest；找到以前不得承诺生成OOF。不得在本轮另抓历史、拼异源数据、改基准或以较短历史训练。是否允许补齐原链更早输入并生成缺失score段，或仅在新独立合同中改变研究窗，是主控/用户需要裁决的事项；本草案保持原外层窗和两年要求，状态BLOCKED_OOF_METADATA_AND_EARLY_INPUT_SPAN。2026H1/partial即便有覆盖也不得被挑出单跑当成原五fold验收替代。OOF实际fit次数、复用、耗时、峰值、canary及registry预算均须在依赖确认后另冻结，当前均CANNOT_CONFIRM。

阻塞范围精确限定为：**按原两年基础训练窗，为全部五完整外层fold学习λ**。它不意味着所有透明加分政策被禁止，也不意味着已经证明透明加分无效。本主方案不改成固定λ。需要一次明确裁决的是：若既有早期合格缓存/原链输入仍未能定位，是否另开有界范围，将同一权威输入和原producer延展到所需早期历史；这会改变input/version，须新输入身份和范围。不得自动借用另一个20年描述面板，不造源、不缩窗、不用外层test调λ。未裁决前本草案仅交付设计与缺口。
