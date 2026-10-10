# Satellite Tile Enhancement — Technical Write-up

## 1. Pipeline Design

### Classical Enhancement

The classical pipeline applies bilateral filtering for denoising, CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement, and unsharp masking for sharpening.

The order is denoising → contrast enhancement → sharpening. Denoising is applied first to reduce noise before contrast enhancement and sharpening potentially amplify it.

Ablation experiments showed that bilateral filtering improved PSNR under the tested synthetic Gaussian-noise condition, while CLAHE caused a substantial decrease in PSNR. Sharpening further reduced PSNR in this experiment. These results suggest that contrast and sharpening parameters need careful tuning, and that the complete pipeline should not be assumed to improve every image.

### Super-Resolution

A pretrained FSRCNN model was used for 2× super-resolution through OpenCV's `dnn_superres` module. No model training was performed. Bicubic interpolation was used as the baseline.

## 2. Evaluation Design

For super-resolution evaluation, each original tile was downscaled to half its width and height, then reconstructed using bicubic interpolation and FSRCNN. Both outputs were compared against the original reference.

Metrics:
- **PSNR:** measures pixel-level reconstruction error; higher is generally better.
- **SSIM:** measures structural similarity; values closer to 1 indicate greater similarity.

In the supplied synthetic reconstruction results, FSRCNN achieved higher PSNR on all reported images. SSIM did not improve on every image, demonstrating that conclusions depend on the metric used.

For classical enhancement, Gaussian noise with standard deviation 15 was added to each original image. The noisy and enhanced images were compared with the original reference. The denoising-only variant improved PSNR in the reported examples, whereas adding CLAHE substantially reduced PSNR.

## 3. Limitations and Failure Cases

1. **Contrast amplification:** CLAHE reduced PSNR in the synthetic-noise experiment, suggesting that local contrast changes can move pixel values away from the reference.
2. **Sharpening artifacts:** Unsharp masking further reduced PSNR after CLAHE in the ablation experiment. It may exaggerate edges and noise.
3. **Perceptual detail versus accuracy:** FSRCNN can generate visually convincing details that are not guaranteed to correspond to real-world structures. Higher PSNR alone does not establish geographic accuracy.

The evaluation uses synthetic degradation and original images as references. It does not establish performance under every real satellite degradation, such as atmospheric haze, compression artifacts, motion blur, or unknown sensor noise.

## 4. Scaling to 50,000 Tiles per Day

A production system processing 50,000 tiles daily would need approximately 0.58 tiles per second on average, before accounting for peaks, retries, and operational overhead.

A scalable design would:
- Use a queue to distribute tiles to CPU workers.
- Process tiles in bounded batches and limit concurrent memory use.
- Cache model weights in each worker instead of loading them for every tile.
- Track processing time, failures, image dimensions, and output quality.
- Use retries, idempotent output paths, and monitoring.
- Benchmark throughput and memory use on representative tiles before choosing worker counts.

## 5. Data Rights and Terms

Before storing, displaying, or distributing enhanced satellite tiles, verify the applicable Google Maps Platform Terms and the specific imagery provider's conditions. Enhancement does not remove the original imagery's licensing or attribution obligations. Do not assume that generated derivatives can be freely stored or redistributed.

## 6. Conclusion

The experiments show that FSRCNN outperformed bicubic interpolation on PSNR in the reported synthetic reconstruction results, while SSIM improvements were not universal. Classical denoising helped under the tested Gaussian-noise condition, but the complete contrast-and-sharpening pipeline reduced similarity to the reference. The findings support evaluating each stage independently and treating visual sharpness as distinct from factual reconstruction accuracy.