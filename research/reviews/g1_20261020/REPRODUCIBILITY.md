# 复现、依赖与公开范围

本包是已完成研究的审阅快照。原件位于未公开本地工件；`local-evidence/...` 是不可下载逻辑引用。`/LOCAL_ONLY/...` 是脱敏源码的不可用路径占位符。[PROVENANCE.json](PROVENANCE.json) 记录每个复制源的原始 SHA、公开 SHA及变换；正文/投影在生成时仅选取既有结果，没有重算研究。

## 可以检查什么

`src/` 的六个核心包含先验、Top50分组/残差恒等式、CI支持停门、HAC20、q*和checkpoint实现。`src/tests/` 是原有合成fixtures源码。仅 `g1_qstar_core.py` 的 HAC_SOURCE 改为同目录文件；该 HAC 副本与原冻结字节一致。源码个人路径脱敏发生在 archive 中，数学与控制流不改。合成测试mask是人为fixture，不是真实有效日身份。

原有测试及独立 review PASS 依据历史receipt复用，本次没有再次运行pytest、fit、真值评价或模拟。新导入路径仅经过包内 parse/import 校验；这不是新一次研究 TESTED 或科学验收。拥有常规Python依赖者可以在隔离目录自行检查小型fixtures，例如（本次未执行）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q src/tests/test_g1_qstar.py src/tests/test_g1_hac.py src/tests/test_g1_ci_weight.py src/tests/test_g1_diagnostic.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q src/tests/test_s1_industry_month_prior.py -k 'not frozen_route'
```

第二条排除唯一依赖完整旧M0运行环境的 `test_frozen_route_split_purges_label_maturity_before_any_fit`。这些指令不是48万次校准/验证，也不能复现真实CI。`requirements.txt` 仅列直接包名，不是原环境锁；历史运行的确切平台/依赖与二进制闭环未由公开包证明。

## 无法从本包重跑什么

- S1的原训练、六fold模型或原全池预测：未公开原行情/标签/公式文本/权重/完整动态engine、原绑定数据及本机registry adapter；M0部分Python源码仅供审阅。
- 真实G1点和区间：未公开逐日x、原日历有效mask和原选择keys。fold计数、D和聚合方差不唯一确定mask，也不能恢复原x。
- 原mask下480,000次模拟和恢复轨迹：缺精确mask及RNG checkpoint数组。公开核心、配置及aggregate结果足以审阅算法与报告，不能独立证明原模拟执行。
- 原20年描述全量：未公开TDX源ZIP/公司行动、14m行情、行业roster、全量年度计数CSV、bound index。archive保留原代码，不是一个新的行情producer。
- 历史registry完整链、远端源码历史和原始签名：仅投影/原件SHA；本次发布不推送原worktree的巨大混合V3 ancestry，不公开全registry。

`archive/orchestration/` 原runner依赖冻结绑定及原registry，可能具有写入和执行动作。**它们是审阅源码，不是本包可移植运行入口；不要把占位路径改为真实运行根后直接执行。** 依赖详情见 [DEPENDENCIES.json](DEPENDENCIES.json)。

## 科学缺口与核验层级

CANNOT_CONFIRM保留：历史行业PIT、官方历史日历/完整全市场/存续偏差、完整复权公司行动、未来赢家Recall@K、Top100/200截断、完整OOF ranking、概率校准、成本后可执行收益、封存forward及授权日推理预算。原静态展示PASS不证明浏览器UI；历史UI仍UI_NOT_VERIFIED。

历史review receipt的PASS范围是其冻结实现、算术、正确停门及指定展示；不是研究成功或本公开包最终验收。公开SHA只是字节身份。公开包会另行接受独立范围审阅；真正GitHub提交/推送与远端读回由主控单独完成。

## 字节检查

在本目录运行 `shasum -a 256 -c SHA256SUMS` 可检查公开payload；SHA256SUMS不自列自身。本地交付还包含未上传的src→repo-path清单，主控只上传该清单allowlist。原始数据、daily diagnostics、stock predictions、凭证/.env、持仓/订单、模型权重、日志、运行时目录及完整HTML未进入包。
