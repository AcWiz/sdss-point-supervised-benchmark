# SDSS 点监督源表生成研究计划

## 摘要

本项目目标是一篇 **Benchmark + Method** 论文：面向 SDSS DR17 原生
corrected-frame 图像，建立一个可复现的点监督源表生成基准，并提出一个
**PSF-constrained point-supervised cataloger** 作为主要方法。

基准部分定义无泄漏的数据划分、源表 schema、预测 schema、匹配规则和
stress tests。方法部分使用弱监督 PhotoObj 中心点、STAR/GALAXY 粗类别、
以及 noisy photometry，在多波段 cutout 上直接生成天体源表。

第一批真实数据固定为：

```text
/Data/sdss/sdss_dr17_l1735_1865_b30_40
```

Headline 实验使用 SDSS 原生 corrected-frame `u/g/r/i/z` 图像，不做 WCS
reprojection。重投影会引入插值选择、相关噪声和 PSF 改变，因此只作为单独
ablation，不作为主结果。

## 论文主张与贡献

- 建立一个面向 **source catalog generation** 的 SDSS DR17 点监督 benchmark，
  包含 region-safe split 和固定 CSV/JSON 合约。
- 构建 native-frame pilot dataset builder，输出 NPZ cutout shards，并用 ragged
  arrays 表示每个 cutout 内数量可变的源标签。
- 提供可运行的 PyTorch training / prediction loop，当前主模型为
  `AstronomyAwareBaseline`。
- 提出 PSF-constrained 学习目标：联合 point heatmap、稀疏类别监督、稀疏测光
  监督、多波段一致性和 Gaussian PSF proxy reconstruction。
- 评估设计重点关注 faint sources、close pairs、crowded fields 和 synthetic
  injection truth，而不是只报告与 SDSS PhotoObj 的一致性。

## 方法与数据流

完整 pilot 闭环如下：

1. 使用 `make dataset-pilot` 构建 100-field pilot dataset。
2. 使用 `make train-pilot` 训练 compact point-supervised baseline。
3. 使用 `make predict-pilot` 将 center heatmap 解码为 prediction catalog。
4. 使用 benchmark matcher 和 metrics 评估检测、定位、分类、测光和 deblending。

准备好的 dataset 将图像保存为 `(N, 5, H, W)`，标签保存为 ragged arrays。
训练时，dataloader 会把源点转换成 Gaussian heatmap、稀疏 class map 和稀疏
photometry target。当前 PSF 项使用 Gaussian kernel proxy；真实 SDSS psField
读取和空间变化 PSF 是 paper-scale 阶段的下一步。

## 模型设计

输入是 `5 x 128 x 128` 的 native `ugriz` cutout。模型输出四类 head：

- `center_heatmap`：源中心概率图，用于 detection 和 NMS decode。
- `class_logits`：STAR/GALAXY 像素级类别 logits，只在标注点监督。
- `flux`：非负多波段 flux proxy map，用于稀疏测光监督和 PSF reconstruction。
- `shape_params`：size / ellipticity 风格的 morphology proxy。

训练目标为：

```text
L = L_center + lambda_cls L_cls + lambda_phot L_phot
    + lambda_mb L_multiband + lambda_psf L_psf_recon
```

默认权重：

```text
lambda_cls = 0.5
lambda_phot = 1.0
lambda_mb = 0.05
lambda_psf = 0.2
```

其中 `L_psf_recon` 是方法贡献的核心：模型预测的 flux map 经过 PSF 卷积后，
应该能够解释 observed multiband image。这把弱点监督和天文成像物理联系起来。

## 实验矩阵

- **E0 Data integrity**：统计 ready/partial fields、缺失 frames/catalogs、每 field
  object counts、pilot QA quantiles。
- **E1 Benchmark contract**：固定 sky-region split、source catalog 转换、prediction
  catalog 合约和 matching radius policy。
- **E2 Baselines**：PhotoObj sanity check、SEP/SExtractor-style detector、center-only
  heatmap model、无 PSF loss 的 astronomy-aware baseline。
- **E3 Main method**：PSF-constrained model；阈值和 NMS 参数只在 validation set 选择，
  test set 只报告一次最终结果。
- **E4 Stress tests**：按 `mag_r`、SNR、seeing、nearest-neighbor distance 分桶报告。
- **E5 Synthetic injection**：在真实 SDSS background 上注入 PSF stars 和 simple galaxy
  profiles，获得已知位置、flux、separation 的可控真值。
- **E6 Ablations**：移除 PSF reconstruction、inverse-variance photometry、valid
  mask、多波段一致性、class head；比较 Gaussian PSF proxy vs real psField；比较
  native-frame vs reprojection。
- **E7 Efficiency**：报告 inference throughput、decode time、GPU memory 和 CPU
  classical baseline runtime。

## 主要评估指标

- **Detection**：precision、recall、F1、average precision。
- **Astrometry**：centroid MAE / RMSE，单位 arcsec。
- **Classification**：STAR/GALAXY accuracy、macro-F1、confusion matrix。
- **Photometry**：magnitude bias、scatter、outlier rate。
- **Deblending**：close-pair recall、missed companion rate、flux attribution error。
- **Stratified metrics**：按 faintness、SNR、seeing、crowding、nearest-neighbor distance
  分桶。

论文中的关键表格不应只展示 overall metric。主 claim 应主要由 faint 和 blended
regimes 支撑，因为这些区域最能体现 PSF-constrained learning 的价值。

## 可复现运行命令

创建/更新环境：

```bash
make env-create
make env-update
```

构建 pilot dataset：

```bash
make dataset-pilot-dry-run
make dataset-pilot
```

训练 pilot baseline：

```bash
make train-pilot
```

生成预测源表：

```bash
make predict-pilot
```

基础验证：

```bash
make test
make smoke
```

## 验收标准

- 在任何完成声明前，`make test` 和 `make smoke` 必须通过。
- Pilot shards 能被 `NpzCutoutDataset` 读取，并产生 finite training loss。
- 非 dry-run `train` 能写出 `best.pt` 和 `training_report.json`。
- 非 dry-run `predict` 能写出 benchmark-compatible prediction CSV。
- 主论文 claim 必须由 stratified metrics 支撑，尤其是 faint 和 blended regimes。

## 当前边界与下一步

当前 v1 闭环是刻意保持 compact 的 pilot：已经有 NPZ 数据读取、训练、checkpoint
保存和预测解码，但还没有真实 psField 解析、固定 validation split 阈值调参、
full paper-scale jobs，以及 deeper external catalog cross-match。

下一阶段优先级：

1. 构建固定 train/val/test split，并让 dataset loader 支持按 split 读取。
2. 下载或解析真实 SDSS psField，替换 Gaussian PSF proxy。
3. 增加 validation threshold sweep，固定 test 只评估一次。
4. 实现 SEP/SExtractor-style baseline，形成第一版 baseline table。
5. 扩展 synthetic injection stress test，重点覆盖 close pairs 和 flux attribution。
