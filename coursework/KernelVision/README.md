# KernelVision

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="figures/exp1_filtering.gif"><img src="figures/exp1_filtering.gif" alt="Spatial filtering progression" width="49%" /></a>
  <a href="figures/exp2_edges.gif"><img src="figures/exp2_edges.gif" alt="Canny threshold and denoising progression" width="49%" /></a>
</p>
<p align="center">
  <a href="figures/exp2_harris.gif"><img src="figures/exp2_harris.gif" alt="Harris corner accumulation" width="49%" /></a>
  <a href="figures/exp3_detection.gif"><img src="figures/exp3_detection.gif" alt="HOG pedestrian detection sweep" width="49%" /></a>
</p>
<p align="center">
  <a href="figures/exp4_training.gif"><img src="figures/exp4_training.gif" alt="CIFAR-10 training curves" width="49%" /></a>
</p>
<p align="center"><b>Five animated demos: filtering · edge detection · corner detection · pedestrian detection · CNN training</b></p>

**Four computer vision experiments that follow one thread — the convolution kernel — from classical image filtering to learned representations: spatial filtering, edge and corner detection, HOG pedestrian detection, and a convolutional neural network on CIFAR-10.**

The filtering experiment evaluates mean, Gaussian, median and bilateral filters with a two-layer protocol: no-reference noise metrics on real noisy images, and a controlled study on a clean reference with known synthetic noise that reports PSNR, SSIM and the edge-preservation index. The feature experiment builds Canny edge detection and Harris corner detection pipelines, quantifies the Sobel aperture gradient gain, and shows that corners concentrate on edge structure. The pedestrian detector loads the OpenCV HOG plus linear SVM model, adds a Malisiewicz non-maximum suppression implementation, and maps recall, precision and runtime across scale, stride, padding and overlap thresholds against five annotated pedestrians. The network experiment trains a two-stage CNN with data augmentation on CIFAR-10 and compares five parameter groups.

**Status.** Coursework complete. The code, the full figure set, metric tables and the trained baseline model live in this repository.

## My Role

Bohan Yu completed all four experiments end to end: the two-layer evaluation for spatial filtering, including the controlled synthetic-noise study and the kernel and sigma parameter sweeps; the Canny and Harris pipelines with threshold sweeps, gradient-gain compensation and the joint edge-corner analysis; the HOG sliding-window detector with hand-written non-maximum suppression and ground-truth based parameter analysis; and the PyTorch CNN on CIFAR-10 with five ablation groups. The figures, animated demos and documentation are part of the same effort.

## Experiments

| # | Experiment | Method | Key result |
|---|---|---|---|
| 1 | Image Filtering | Mean, Gaussian, median and bilateral filters; controlled and no-reference evaluation | 3x3 median reaches PSNR 39.54 dB and SSIM 0.986 on salt-and-pepper noise; Gaussian filtering reaches 31.48 dB on Gaussian noise, with the sigma sweep peaking at 31.41 dB near sigma 1.0 |
| 2 | Feature Detection | Canny edges, Harris corners, parameter sweeps | Threshold pair 100/200 yields 150 connected edge components; 468 corners after non-maximum suppression, 100 percent within the edge neighborhood |
| 3 | HOG Pedestrian Detection | OpenCV HOG + linear SVM, image pyramid sliding window, Malisiewicz NMS | Best setting recall 0.6 and precision 1.0; CPU frame time from about 7 ms to 35 ms across the sweep |
| 4 | Convolutional Networks | Two-stage CNN on CIFAR-10, augmentation, five ablation groups | Test accuracy 76.9 percent; stable learning rates from 0.0001 to 0.001; confusion concentrated in cat-dog and car-truck pairs |

## Results Preview

### 1 Image Filtering

<p align="center">
  <a href="figures/exp1_noise_analysis.png"><img src="figures/exp1_noise_analysis.png" alt="Noise analysis" width="49%" /></a>
  <a href="figures/exp1_comparison_salt_pepper.png"><img src="figures/exp1_comparison_salt_pepper.png" alt="Salt-and-pepper comparison" width="49%" /></a>
</p>

The salt-and-pepper image carries 7.1 percent extreme pixels with a 4.2 percent estimated impulse density, and the Gaussian image carries an estimated sigma of 20.0. All three linear and rank filters remove the impulses, while the residual maps separate their behavior: mean and Gaussian filters spread the impulse energy into the neighborhood, and the median filter replaces each impulse with the local median.

<p align="center">
  <a href="figures/exp1_parameter_sweep.png"><img src="figures/exp1_parameter_sweep.png" alt="Parameter sweep" width="80%" /></a>
</p>

The controlled study ranks median filtering first on salt-and-pepper noise (33.17 dB with a 5x5 kernel) and Gaussian filtering first on Gaussian noise, with bilateral filtering holding the highest edge-preservation index at 0.916.

### 2 Feature Detection

<p align="center">
  <a href="figures/exp2_canny_thresholds.png"><img src="figures/exp2_canny_thresholds.png" alt="Canny threshold sweep" width="49%" /></a>
  <a href="figures/exp2_canny_variants.png"><img src="figures/exp2_canny_variants.png" alt="Aperture and gradient variants" width="49%" /></a>
</p>

Raising the threshold pair from 30/60 to 200/400 cuts the edge ratio from 10.2 percent to 2.2 percent and raises the mean gradient along edges from 136 to 336. Increasing the Sobel aperture raises the gradient magnitude scale by 14x and 203x; compensating the thresholds by the measured gain restores clean edges.

<p align="center">
  <a href="figures/exp2_harris_corners.png"><img src="figures/exp2_harris_corners.png" alt="Harris corners" width="49%" /></a>
  <a href="figures/exp2_edge_corner_relation.png"><img src="figures/exp2_edge_corner_relation.png" alt="Edge corner relation" width="49%" /></a>
</p>

Non-maximum suppression reduces 2024 raw response pixels to 468 stable corners, all of them within the Canny edge neighborhood.

### 3 HOG Pedestrian Detection

<p align="center">
  <a href="figures/exp3_hog_features.png"><img src="figures/exp3_hog_features.png" alt="HOG feature visualization" width="49%" /></a>
  <a href="figures/exp3_detection_nms.png"><img src="figures/exp3_detection_nms.png" alt="Detection and NMS" width="49%" /></a>
</p>

The 64x128 window produces a 3780-dimensional descriptor from 8x8 cells and 16x16 blocks. Non-maximum suppression merges the raw candidate boxes into one box per pedestrian.

<p align="center">
  <a href="figures/exp3_scale_sweep.png"><img src="figures/exp3_scale_sweep.png" alt="Scale sweep" width="49%" /></a>
  <a href="figures/exp3_stride_padding.png"><img src="figures/exp3_stride_padding.png" alt="Stride and padding" width="49%" /></a>
</p>

Scale 1.03 and 1.10 both reach recall 0.6 with precision 1.0; scale 1.50 misses every pedestrian because the pyramid step exceeds the size range of the targets. Finer strides raise recall and runtime together, which makes the precision-speed trade-off explicit.

### 4 Convolutional Networks

<p align="center">
  <a href="figures/exp4_network_architecture.png"><img src="figures/exp4_network_architecture.png" alt="Network architecture" width="49%" /></a>
  <a href="figures/exp4_training_curves.png"><img src="figures/exp4_training_curves.png" alt="Training curves" width="49%" /></a>
</p>

The network holds 1,070,794 parameters, 98 percent of them in the fully connected layers. The baseline trains for 20 epochs and reaches 76.9 percent test accuracy.

<p align="center">
  <a href="figures/exp4_confusion_accuracy.png"><img src="figures/exp4_confusion_accuracy.png" alt="Confusion matrix" width="49%" /></a>
  <a href="figures/exp4_predictions.png"><img src="figures/exp4_predictions.png" alt="Prediction samples" width="49%" /></a>
</p>

Per-class accuracy spans 47.7 percent (cat) to 90.0 percent (automobile). Errors concentrate in the cat-dog pair with 228 and 99 mutual confusions, the car-truck pair with 61 and 62, and the bird-airplane pair with 75 and 42.

## Repository Layout

```
KernelVision/
├── run_all.py                  one-command run of all four experiments
├── requirements.txt            pinned dependencies
├── figures/                    README assets: animated demos and preview figures
├── exp1_filtering/             experiment 1, spatial filtering
├── exp2_features/              experiment 2, edge and corner detection
├── exp3_hog/                   experiment 3, HOG pedestrian detection
└── exp4_cnn/                   experiment 4, convolutional networks
```

Each experiment folder holds `code/`, `images/` input data, `figures/` report plates, `results/` metric CSV files and `output/` process images. The trained baseline model lives in `exp4_cnn/output/model_baseline.pt`.

## How to Run

Python 3.11 with the pinned dependencies:

```bash
pip install -r requirements.txt
python run_all.py
```

Every experiment also runs standalone, for example:

```bash
python exp2_features/code/experiment2_features.py
```

The network experiment regenerates all figures from the saved model and metric tables without retraining:

```bash
python exp4_cnn/code/experiment4_cnn.py figures
```

A full run takes about forty minutes on a recent GPU, dominated by the thirteen training runs of experiment 4.

## Data

| Experiment | Data | Source |
|---|---|---|
| 1 | image_fft.png, noised1.png, noised2.png | course dataset |
| 2 | test.png | course dataset |
| 3 | people.jpg | course dataset |
| 4 | CIFAR-10 | automatic download via torchvision |

## References

- Dalal N, Triggs B. Histograms of Oriented Gradients for Human Detection. CVPR, 2005.
- Tomasi C, Manduchi R. Bilateral Filtering for Gray and Color Images. ICCV, 1998.
- Canny J. A Computational Approach to Edge Detection. IEEE TPAMI, 1986.

Released into the public domain under the Unlicense; see [LICENSE](LICENSE).
