# VOC 调试报告 v2 (debug)

## 1. 最新观察（基于 voc_logs 与当前可视化）
- 第 50 轮日志显示整体 mAP≈0.00015，仅在人类类别（id=14）有极少量命中，说明模型几乎未学到其他类别。【F:logs_voc/voc_debug_epoch_050.json†L2-L52】
- 验证集预测置信度均值仅 0.23，小于此前默认阈值 0.35，导致解码时大部分框被过滤，难以在可视化中看到红框。【F:logs_voc/voc_debug_epoch_050.json†L53-L58】
- 绿色框仍然是 GT，红框是预测；若没有红框，多半是阈值过高或模型未收敛所致。【F:infer_voc_visualize.py†L36-L78】

## 2. 代码调整概要
- **评估/可视化默认阈值下调至 0.20**：`infer_voc_visualize.py` 与 `eval_voc.py` 默认使用 `VAL_CONF_THRESHOLD`，便于观察弱预测并定位训练问题。【F:infer_voc_visualize.py†L55-L78】【F:eval_voc.py†L57-L96】
- **测试集缺标注的兜底支持**：`PascalVOC2012` 在 test split 允许缺失 Annotations，会生成空 GT，必要时自动用全部图像代替缺失的 `test.txt`，避免 FileNotFoundError。【F:datasets/voc2012.py†L132-L189】
- **无 GT 的 eval 不再报错**：当数据集缺标注时，`eval_voc.py` 跳过指标计算，仅导出预测结果并附带提示说明。【F:eval_voc.py†L18-L96】

## 3. 推荐使用方式
- **可视化检查**：运行 `python infer_voc_visualize.py --checkpoint checkpoints_voc/voc_best.pt --split val --conf 0.2`，绿色=GT，红色=预测；若仍无红框，请确认模型权重与数据路径。
- **评估与导出预测**：
  - 有 GT（train/val 或自带标注的 test）：`python eval_voc.py --split val --checkpoint checkpoints_voc/voc_best.pt --conf 0.2`
  - 无 GT 的 test 仅推理：`python eval_voc.py --split test --checkpoint checkpoints_voc/voc_best.pt --conf 0.2 --save-json debug_eval/voc_test_preds.json`
- **路径确认**：确保 `VOC_ROOT` 指向包含 `VOC2012_train_val`/`VOC2012_test` 的目录；若缺少 `ImageSets/Main/test.txt`，程序会自动以 `JPEGImages` 下全部图片作为 test 列表。【F:config_voc2012.py†L9-L35】【F:datasets/voc2012.py†L132-L173】

---
以下保留 v1 内容，便于对照历史说明。

# VOC 调试报告 v1 (debug)

## 1. inference 可视化框色说明
- 在 `infer_voc_visualize.py` 中，首先对原图绘制绿色框，这些框来自 GT 标注（`meta["boxes_orig"]`）并转换成原始像素坐标，所以绿色=真值。随后才会把模型预测（decode 后 unletterbox）的结果用红色框叠加到同一张图上。若某张图没有红框，说明该图片在当前置信度阈值下没有预测到目标。【F:infer_voc_visualize.py†L36-L78】

## 2. mAP 偏低的常见原因排查
1) **数据路径/拆分是否正确**：
   - 代码默认从 `VOC_ROOT` 环境变量或 `E:/VOC` 读取数据，目录期望存在 `VOC2012_train_val` 与 `VOC2012_test`，且内部有 `JPEGImages/` 与 `Annotations/` 及 `ImageSets/Main/*.txt`。如果路径不一致或缺少标注，会导致训练/验证使用的样本不足或错误，从而 mAP 很低。【F:config_voc2012.py†L9-L35】【F:datasets/voc2012.py†L132-L175】
2) **测试集无标注**：官方 VOC2012 test 没公开标注；若你把无标注的 test 当成验证/eval，mAP 会异常。确保评估使用带 GT 的 train/val，或提供自定义带标注的 test 列表。【F:config_voc2012.py†L29-L35】
3) **置信度阈值过高**：推理与评估解码都使用 `CONF_THRESHOLD`（默认 0.35）。若模型尚未收敛，建议在可视化或 eval 时调低阈值（如 0.2）以检查是否存在预测。【F:config_voc2012.py†L84-L88】【F:infer_voc_visualize.py†L55-L78】
4) **训练轮次/数据均衡**：当前 VOC 配置仅 50 epoch，batch=4，若数据量大或类间不平衡，可能不足以收敛。可在 `config_voc2012.py` 提高 `EPOCHS`、适当增大 batch、或调整学习率与 warmup。【F:config_voc2012.py†L36-L52】
5) **标注质量/空框**：数据加载时会过滤无效框；若标注空缺或存在大量无效框，真实正样本变少也会影响 mAP。训练时可留意加载日志中的 `[WARN] Dropped ... invalid boxes` 输出。【F:datasets/voc2012.py†L33-L68】

## 3. eval 报错（Annotation not found）的定位与修复
- 报错来自 `PascalVOC2012._load_annotation`。当按 split 读取 `ImageSets/Main/test.txt` 时，会去 `VOC2012_test/Annotations/<id>.xml` 找标注，文件缺失就触发 FileNotFoundError。【F:datasets/voc2012.py†L132-L175】
- 默认路径由 `VOC_ROOT` 控制，当前写死为 `E:/VOC`。如果实际数据不在该盘符，需要：
  1. 在运行前导出正确路径，例如：`set VOC_ROOT=D:/your/VOC_root`（Windows cmd）或 `export VOC_ROOT=/data/VOC`。
  2. 确认目录下存在 `VOC2012_train_val/Annotations/` 与 `VOC2012_test/Annotations/`，并且 `ImageSets/Main/test.txt` 中的文件名（如 `2008_000001`）确实有对应 XML。
  3. 若手头没有 test 标注，评估时改用 `--split val`（有 GT），或自行准备带标注的测试集和对应 `test.txt`。

## 4. 建议的操作步骤
1) 设置或修改 `VOC_ROOT` 使其指向包含 `VOC2012_train_val`/`VOC2012_test` 的根目录，并确保两套子目录下的 `JPEGImages`、`Annotations`、`ImageSets/Main/*.txt` 完整。【F:config_voc2012.py†L9-L35】
2) 若只拥有官方 VOC2012（train/val 有标注，test 无标注），评估命令请使用 `--split val`，并确保 `config_voc2012.py` 的路径指向 train/val 数据。
3) 若要评估自建 test 集，先把 XML 标注放入 `VOC2012_test/Annotations`，补齐 `ImageSets/Main/test.txt`，再运行 `python eval_voc.py --split test --checkpoint ... --conf 0.2`。
4) 调低可视化与评估的阈值（如 `--conf 0.2`），确认红框是否出现；如仍无红框，检查模型是否正确加载（`voc_best.pt`）和数据是否与训练集一致。

## 5. 关于当前可视化结果的解读
- 图像上仅有绿色框时，表示模型在该阈值下没有预测；绿色框就是 GT，红框才是预测。若你看到的只有绿色，需调低阈值或检查 checkpoint 是否正确加载。【F:infer_voc_visualize.py†L36-L78】

---
如需继续排查，可先验证 `VOC_ROOT` 与数据结构，再用 `--split val --conf 0.2` 运行 `eval_voc.py` 观察是否仍有缺失文件与低 mAP 的情况。
