# 环境 × 行业 × 月份：训练方案咨询材料

这是一份**向外部 ChatGPT 请求下一步研究方案的材料包**。现阶段仅准备证据和 prompt，没有新训练、预测、诊断、回测或科学评价。既有研究为 DIRECTION_ONLY：M4_GATE_FAILED、S1_DEVELOPMENT_GATE_FAILED、G1 信息门 INCONCLUSIVE，S2 未启动。下一实验仍须独立冻结、审批与登记。

阅读顺序：

1. [TRAINING_DESIGN.md](TRAINING_DESIGN.md)：实际做过什么、特征/训练/支持集/资源，以及尚待回答的设计问题。
2. [EVIDENCE.json](EVIDENCE.json)：原合同、M0–M4最终独立核对、精确M4聚合门、S1与G1/归因投影。原单位、历史快照与限制均保留。
3. [S1/G1/q*旧包](../g1_20261020/README.md) 和 [归因旧包](../g1_attribution_20261020/README.md)：先看各自README、方法及acceptance，再按需要读原代码/聚合表。
4. [CHATGPT_PROMPT.md](CHATGPT_PROMPT.md)：复制其中分隔线之间的完整中文请求给ChatGPT。
5. [PROVENANCE.json](PROVENANCE.json) 与 [SHA256SUMS](SHA256SUMS)：查来源身份、脱敏范围和发布字节。

旧两包固定基线：[fcda5cb15c2235f954e68105f4ff95a15b456dbe 的 G1 包](https://github.com/lexi2016/193/tree/fcda5cb15c2235f954e68105f4ff95a15b456dbe/research/reviews/g1_20261020)、[同一固定基线的归因包](https://github.com/lexi2016/193/tree/fcda5cb15c2235f954e68105f4ff95a15b456dbe/research/reviews/g1_attribution_20261020)。[本新入口的分支地址](https://github.com/lexi2016/193/blob/codex/g1-research-review-20261020/research/reviews/environment_training_advice_20261020/README.md)会随本分支更新；发布后以主控提供的新固定commit入口为准，不能把分支地址叫immutable。

已经研究过环境特征：M1市场、M2月份、M3行业、M4成交额代理均已进入正式比较；这并未证明任何其他环境机制都无效。PV使用六fold且部分期参与原投票，S1/G1使用五完整fold主评价；不能混为一套证据。G1两fold约87.46%/第一rank层约54.04%是原有符号点贡献的描述比例，不是因果解释率。共同大盘加项在同日高低差中抵消，也不能排除异质beta或环境交互。

不公开私人行情、逐日标签/预测/精确mask、权重、凭证或账户信息。已公开源码足够审阅算法，无需再复制业务代码；完整真实数据复现、历史行业PIT/完整全市场/官方日历、可执行净收益、封存验证、普适真实95%覆盖均仍CANNOT_CONFIRM。安全状态：research_only=true、production_allowed=false、buy_gate_updated=false、orders_sent=false。
