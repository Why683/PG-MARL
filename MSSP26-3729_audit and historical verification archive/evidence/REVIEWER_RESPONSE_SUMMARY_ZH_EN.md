# Reviewer response summary for code and data reproducibility

## 可直接用于回复信的中文版本

我们衷心感谢审稿人对实验可核查性、数据访问边界以及统计结果来源提出的细致且富有建设性的意见。为提高修订工作的透明度与可复现性，我们已将代码和证据文件按功能重新整理，并将历史开发档案与修订稿所对应的正式实现明确分开。正式代码仓库只保留修订稿中描述的 MMPS、三层四头 GATv2 学生网络、200 步 GRU 形式递推、支持集特征与概率蒸馏、查询集特征引导以及条件多核 MMD；探索过程中产生但未进入修订稿方法的试验性分支仅保存在独立历史档案中，不作为论文复现入口。

修订后的数据协议在试验级划分完成后生成不可变的 2.0 s 非重叠窗口。轨迹 A、B 和 C 是归一化拟合、教师与学生训练、超参数确定、验证和检查点选择所能访问的全部轨迹；轨迹 D 和 E 仅由独立的冻结模型最终评价程序注册。每个检查点均保存随机种子、配置文件 SHA-256、窗口清单 SHA-256、源域验证选择历史和选中轮次。最终评价程序只接受五个预先声明种子的冻结检查点，导出逐窗口 logits、预测标签、逐次混淆矩阵、逐类别指标以及固定等权 logit 汇总结果，不提供目标域校准、种子筛选、检查点重选或集成权重学习接口。

我们同时提供固定窗口清单、协议审计记录、配置文件、代码—论文对应矩阵、运行环境说明和文件 SHA-256 清单。对本地现有 A-D 文件树的独立审计得到 18,000 个 A/B/C 窗口和 5,700 个 D 域窗口；A/B/C 的训练、验证和测试窗口数分别为每条轨迹 3,000、1,500 和 1,500，D 域全部 5,700 个窗口仅标记为最终测试。该审计通过了试验互斥、窗口身份唯一以及目标域不可用于训练或验证的检查。

## Suggested English response

We sincerely thank the reviewer for the detailed and constructive comments on
experimental auditability, data-access boundaries, and the provenance of the
reported statistics. To improve transparency and reproducibility, we have
reorganized the code and evidence files by function and clearly separated the
historical development archive from the formal implementation corresponding
to the revised manuscript. The manuscript-facing repository contains only the
MMPS, the three-layer four-head GATv2 student, the 200-step GRU-form
recurrence, support-set feature and probability distillation, query-set
feature guidance, and conditional multi-kernel MMD described in the revision.
Experimental branches that were explored during development but are absent
from the revised method are retained only in a separate historical archive and
are not presented as reproduction entry points.

The revised protocol generates immutable, non-overlapping 2.0-s windows after
the trial-level split. Trajectories A, B, and C are the only trajectories
accessible during normalization fitting, teacher/student training,
hyperparameter determination, validation, and checkpoint selection.
Trajectories D and E are registered only by an independent final evaluator
after model freezing. Each checkpoint stores the random seed, configuration
SHA-256, manifest SHA-256, source-validation selection history, and selected
epoch. The final evaluator accepts only the frozen checkpoints of the five
predeclared seeds and exports per-window logits, predicted labels, per-run
confusion matrices, per-class metrics, and the fixed equal-weight logit
aggregate. It provides no interface for target-domain calibration, seed
screening, checkpoint reselection, or learned ensemble weighting.

We also provide the fixed-window manifest, protocol-audit records,
configuration file, manuscript-to-code evidence matrix, environment
description, and file SHA-256 ledger. An independent audit of the locally
available A-D folder tree produced 18,000 A/B/C windows and 5,700 D windows.
For each source trajectory, 3,000, 1,500, and 1,500 windows were assigned to
training, validation, and testing, respectively, whereas all 5,700 D windows
were marked exclusively as final target test. The audit passed the checks for
trial disjointness, unique window identity, and exclusion of the target domain
from training and validation.

## 作者内部提交前核对（不要原样放入回复信）

当前本机快照尚缺轨迹 E 原始/处理记录、五个最终检查点以及支撑修订稿最终表格的 A-E 逐窗口预测文件。因此，包内 `manuscript_claims/` 的数值是从修订稿转录的论文主张，不是本机重新计算的结果。在向审稿人声称“完整复现档案”或上传 GitHub 前，必须从服务器补入最终 A-E 清单、五个检查点、逐窗口 logits/预测、逐次矩阵、源域选择历史、2,000 次试验/运行级 bootstrap 记录、配置冻结/数据采集/首次访问时间以及作者选定的开源许可证，并重新运行 `scripts/verify_release.py`。

The local workstation snapshot still lacks trajectory E, the five final
checkpoints, and the final A-E per-window prediction exports supporting the
revised tables. Values under `manuscript_claims/` are therefore manuscript
transcriptions rather than locally regenerated results. Before describing the
release as evidence-complete or publishing it on GitHub, export the final A-E
manifest, checkpoints, per-window logits/predictions, per-run matrices,
source-selection histories, 2,000-replicate trial/run bootstrap records,
configuration-freeze/acquisition/first-access timestamps, and the authors'
chosen open-source license, then rerun `scripts/verify_release.py`.
