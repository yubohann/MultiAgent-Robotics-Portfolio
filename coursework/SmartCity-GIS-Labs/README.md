# Smart City GIS Labs

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="figures/exp1_vectorization.gif"><img src="figures/exp1_vectorization.gif" alt="Raster cleaning and centerline vectorization" width="49%" /></a>
  <a href="figures/exp2_rubbersheet.gif"><img src="figures/exp2_rubbersheet.gif" alt="Rubbersheet alignment of imported streets" width="49%" /></a>
</p>
<p align="center">
  <a href="figures/exp4_siting.gif"><img src="figures/exp4_siting.gif" alt="Buffer overlay and site selection" width="49%" /></a>
  <a href="figures/exp5_gas_trace.gif"><img src="figures/exp5_gas_trace.gif" alt="Gas network common ancestor trace" width="49%" /></a>
</p>
<p align="center"><b>Four animated demos: vectorization · rubbersheet · siting · network tracing</b></p>

**Five ArcGIS Desktop 10.2 experiments on smart city information modeling: scan vectorization, rubbersheet adjustment, topology repair, vector spatial analysis and gas network emergency tracing.**

The pipelines cover pixel-level raster cleaning and Zhang-Suen thinning for centerline extraction, thin plate spline warping driven by displacement links, geodatabase topology validation with trim, extend and delete repair loops, buffer and overlay analysis for facility siting, and graph tracing for burst localization in a gas network. Each experiment runs as a standalone script, with shared path configuration in `scripts/common.py`.

**Status.** Coursework complete. Processed datasets, maps and reports live in the course workspace; this repository keeps the code and the full figure set.

## My Role

Bohan Yu completed all five experiments end to end: the raster cleaning, thinning and centerline tracing pipeline for the scanned parcel map; the displacement link matching and thin plate spline warp for the imported street network; the topology rule setup and iterative dangle repair on the parcel lines; the buffer, intersect, erase and union workflows for mall siting and the residential land task; and the graph build, common ancestor trace and downstream trace for the gas network. The figures, animated demos and documentation are part of the same effort.

## Experiments

| # | Experiment | Method | Key result |
|---|---|---|---|
| 1 | Scan Vectorization | Raster cleaning, thinning, centerline tracing | 2735 parcel boundary lines, 7.6 km total, sampled vectors 100% on scanned linework |
| 2 | Rubbersheet Adjustment | Displacement links, thin plate spline transform | 479 links, mean control point residual 0.016 m, 19 corresponding streets match the existing network |
| 3 | Topology Repair | Must-not-have-dangles rule, trim / extend / delete | 150 dangle errors detected, 124 repaired |
| 4 | Vector Spatial Analysis | Buffer and overlay analysis | Best siting area 1.87 ha, five-grade suitability rating |
| 5 | Gas Network Emergency Tracing | Common ancestor trace, downstream trace | Burst node located, valve closure isolates 35 pipe segments over 0.89 km |

## Results Preview

### 1 Scan Vectorization

<p align="center">
  <a href="figures/exp1_cleaning_before_after.png"><img src="figures/exp1_cleaning_before_after.png" alt="Cleaning before and after" width="49%" /></a>
  <a href="figures/exp1_cleaning_full_map.png"><img src="figures/exp1_cleaning_full_map.png" alt="Full map cleaning" width="49%" /></a>
</p>

Raster cleaning removes label characters and noise so that 2234 of 2241 connected raster components are cleared, keeping the parcel linework intact.

<p align="center">
  <a href="figures/exp1_vectorized_result.png"><img src="figures/exp1_vectorized_result.png" alt="Vectorized result over cleaned raster" width="49%" /></a>
  <a href="figures/exp1_vectorized_zoom.png"><img src="figures/exp1_vectorized_zoom.png" alt="Vectorized zoom" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp1_vector_preview.png"><img src="figures/exp1_vector_preview.png" alt="Parcel boundary lines" width="60%" /></a>
</p>

### 2 Rubbersheet Adjustment

<p align="center">
  <a href="figures/exp2_before_zoom.png"><img src="figures/exp2_before_zoom.png" alt="Before correction" width="49%" /></a>
  <a href="figures/exp2_after_zoom.png"><img src="figures/exp2_after_zoom.png" alt="After correction" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp2_displacement_links.png"><img src="figures/exp2_displacement_links.png" alt="Displacement links" width="49%" /></a>
  <a href="figures/exp2_before_after_overlay.png"><img src="figures/exp2_before_after_overlay.png" alt="Before and after overlay" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp2_alignment_full.png"><img src="figures/exp2_alignment_full.png" alt="Alignment overview" width="49%" /></a>
  <a href="figures/exp2_alignment_zoom.png"><img src="figures/exp2_alignment_zoom.png" alt="Alignment zoom" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp2_before_full.png"><img src="figures/exp2_before_full.png" alt="Full view before" width="49%" /></a>
  <a href="figures/exp2_after_full.png"><img src="figures/exp2_after_full.png" alt="Full view after" width="49%" /></a>
</p>

### 3 Topology Repair

<p align="center">
  <a href="figures/exp3_dangle_errors.png"><img src="figures/exp3_dangle_errors.png" alt="Dangle errors" width="49%" /></a>
  <a href="figures/exp3_after_repair.png"><img src="figures/exp3_after_repair.png" alt="After repair" width="49%" /></a>
</p>

### 4 Vector Spatial Analysis

<p align="center">
  <a href="figures/exp4_buffers.png"><img src="figures/exp4_buffers.png" alt="Buffers" width="49%" /></a>
  <a href="figures/exp4_site_selection.png"><img src="figures/exp4_site_selection.png" alt="Site selection" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp4_suitability_grades.png"><img src="figures/exp4_suitability_grades.png" alt="Suitability grades" width="49%" /></a>
  <a href="figures/exp4_residential_task.png"><img src="figures/exp4_residential_task.png" alt="Residential land task" width="49%" /></a>
</p>

### 5 Gas Network Emergency Tracing

<p align="center">
  <a href="figures/exp5_gas_flow_faults.png"><img src="figures/exp5_gas_flow_faults.png" alt="Flow and faults" width="49%" /></a>
  <a href="figures/exp5_gas_affected_area.png"><img src="figures/exp5_gas_affected_area.png" alt="Affected area" width="49%" /></a>
</p>

<p align="center">
  <a href="figures/exp5_gas_arcmap_result.png"><img src="figures/exp5_gas_arcmap_result.png" alt="ArcMap result" width="60%" /></a>
</p>

## Quick Start

```powershell
& "C:\Python27\ArcGIS10.2\python.exe" scripts\run_all.py
```

Place the course data workspace next to this repository, and the scripts locate it automatically. Run a single experiment with:

```powershell
& "C:\Python27\ArcGIS10.2\python.exe" scripts\exp3_topology.py
```

## Layout

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

## License

Coursework code is released into the public domain under the [Unlicense](LICENSE). ArcGIS and the course datasets keep their original terms.

*Bohan Yu, Smart City GIS coursework.*
