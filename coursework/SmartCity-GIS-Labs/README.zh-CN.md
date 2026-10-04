# SmartCity-GIS-Labs 智慧城市信息模型实验

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="figures/exp1_vectorization.gif"><img src="figures/exp1_vectorization.gif" alt="栅格清理与中心线矢量化" width="49%" /></a>
  <a href="figures/exp2_rubbersheet.gif"><img src="figures/exp2_rubbersheet.gif" alt="导入街道橡皮页匹配" width="49%" /></a>
</p>
<p align="center">
  <a href="figures/exp4_siting.gif"><img src="figures/exp4_siting.gif" alt="缓冲区叠置与选址求解" width="49%" /></a>
  <a href="figures/exp5_gas_trace.gif"><img src="figures/exp5_gas_trace.gif" alt="燃气管网公共祖先追踪" width="49%" /></a>
</p>
<p align="center"><b>四个动态演示：矢量化 · 橡皮页变换 · 选址分析 · 管网追踪</b></p>

**智慧城市信息模型课程五个 ArcGIS Desktop 10.2 实验：扫描矢量化、橡皮页变换、拓扑处理、矢量空间分析与燃气管网应急分析。**

管线覆盖：像元级栅格清理与 Zhang-Suen 细化提取中心线；位移连接驱动的薄板样条（TPS）变换；地理数据库拓扑验证与修剪/延伸/删除修复循环；缓冲区分析与叠置分析求解设施选址；图追踪定位燃气管网爆管点。每个实验均为独立脚本，路径配置集中于 `scripts/common.py`。

**状态。** 课程作业完成。处理成果、地图与报告保存在课程工作区；本仓库收录代码与全部图件。

## 我的职责

余博涵独立完成全部五个实验：扫描地块图的栅格清理、细化与中心线追踪管线；导入街道网的位移连接匹配与薄板样条变换；地块线拓扑规则建立与悬挂点迭代修复；商场选址与住宅用地任务的缓冲区、相交、擦除与联合工作流；燃气管网的图结构构建、公共祖先追踪与下溯追踪。图件、动图与文档同步完成。

## 实验一览

| # | 实验 | 方法 | 核心结果 |
|---|---|---|---|
| 1 | 扫描矢量化 | 栅格清理、细化、中心线追踪 | 地块边界线 2735 条，总长 7.6 km，抽样矢量线 100% 落在扫描线划上 |
| 2 | 橡皮页变换 | 位移连接、薄板样条变换 | 位移连接 479 条，控制点平均残差 0.016 m，19 条对应街道与现状匹配 |
| 3 | 拓扑处理 | 悬挂点规则、修剪/延伸/删除操作 | 检出错误 150 处，修复 124 处 |
| 4 | 矢量空间分析 | 缓冲区分析、叠置分析 | 最佳选址区 1.87 公顷，五级区位评价 |
| 5 | 管网应急分析 | 公共祖先追踪分析、下溯追踪分析 | 爆管点定位，关阀隔离 35 段管线 0.89 km |

## 成果预览

### 一 扫描矢量化

![矢量化动画](figures/exp1_vectorization.gif)

<p align="center">
  <a href="figures/exp1_cleaning_before_after.png"><img src="figures/exp1_cleaning_before_after.png" alt="清理前后" width="49%" /></a>
  <a href="figures/exp1_cleaning_full_map.png"><img src="figures/exp1_cleaning_full_map.png" alt="全图清理" width="49%" /></a>
</p>

栅格清理去除注记字符与噪声，2241 个连通栅格组件中清理 2234 个，地块线划完整保留。

<p align="center">
  <a href="figures/exp1_vectorized_result.png"><img src="figures/exp1_vectorized_result.png" alt="矢量化结果" width="49%" /></a>
  <a href="figures/exp1_vectorized_zoom.png"><img src="figures/exp1_vectorized_zoom.png" alt="矢量化局部" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp1_vector_preview.png"><img src="figures/exp1_vector_preview.png" alt="地块边界线" width="60%" /></a>
</p>

### 二 橡皮页变换

<p align="center">
  <a href="figures/exp2_before_zoom.png"><img src="figures/exp2_before_zoom.png" alt="校正前局部" width="49%" /></a>
  <a href="figures/exp2_after_zoom.png"><img src="figures/exp2_after_zoom.png" alt="校正后局部" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp2_displacement_links.png"><img src="figures/exp2_displacement_links.png" alt="位移连接" width="49%" /></a>
  <a href="figures/exp2_before_after_overlay.png"><img src="figures/exp2_before_after_overlay.png" alt="前后叠加" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp2_alignment_full.png"><img src="figures/exp2_alignment_full.png" alt="匹配全图" width="49%" /></a>
  <a href="figures/exp2_alignment_zoom.png"><img src="figures/exp2_alignment_zoom.png" alt="匹配局部" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp2_before_full.png"><img src="figures/exp2_before_full.png" alt="校正前全图" width="49%" /></a>
  <a href="figures/exp2_after_full.png"><img src="figures/exp2_after_full.png" alt="校正后全图" width="49%" /></a>
</p>

### 三 拓扑处理

<p align="center">
  <a href="figures/exp3_dangle_errors.png"><img src="figures/exp3_dangle_errors.png" alt="悬挂点错误" width="49%" /></a>
  <a href="figures/exp3_after_repair.png"><img src="figures/exp3_after_repair.png" alt="修复后" width="49%" /></a>
</p>

### 四 矢量空间分析

<p align="center">
  <a href="figures/exp4_buffers.png"><img src="figures/exp4_buffers.png" alt="缓冲区" width="49%" /></a>
  <a href="figures/exp4_site_selection.png"><img src="figures/exp4_site_selection.png" alt="选址结果" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp4_suitability_grades.png"><img src="figures/exp4_suitability_grades.png" alt="分级评价" width="49%" /></a>
  <a href="figures/exp4_residential_task.png"><img src="figures/exp4_residential_task.png" alt="住宅用地任务" width="49%" /></a>
</p>

### 五 管网应急分析

<p align="center">
  <a href="figures/exp5_gas_flow_faults.png"><img src="figures/exp5_gas_flow_faults.png" alt="流向与故障" width="49%" /></a>
  <a href="figures/exp5_gas_affected_area.png"><img src="figures/exp5_gas_affected_area.png" alt="影响范围" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp5_gas_arcmap_result.png"><img src="figures/exp5_gas_arcmap_result.png" alt="应急结果" width="60%" /></a>
</p>

## 快速开始

```powershell
& "C:\Python27\ArcGIS10.2\python.exe" scripts\run_all.py
```

将课程数据工作区与本仓库同级放置，脚本自动定位。单实验运行示例：

```powershell
& "C:\Python27\ArcGIS10.2\python.exe" scripts\exp3_topology.py
```

## 目录

```text
SmartCity-GIS-Labs/
  README.md
  README.zh-CN.md
  LICENSE
  scripts/
    common.py
    exp1_arcscan.py
    exp2_rubbersheet.py
    exp3_topology.py
    exp4_siting.py
    exp5_gas.py
    run_all.py
  figures/
```

## 许可

课程代码以公有领域方式发布，采用 [Unlicense](LICENSE)。ArcGIS 软件与课程数据集保留各自原始条款。

*余博涵（Bohan Yu），智慧城市信息模型课程作业。*
