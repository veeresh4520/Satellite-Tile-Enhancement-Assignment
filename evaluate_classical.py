
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def calculate_metrics(reference, prediction):
    psnr = cv2.PSNR(reference, prediction)

    ref_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

    ssim = structural_similarity(
        ref_rgb, pred_rgb, channel_axis=2, data_range=255
    )
    return psnr, ssim


def denoise(image):
    return cv2.bilateralFilter(
        image, d=5, sigmaColor=30, sigmaSpace=30
    )


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    )
    lightness = clahe.apply(lightness)

    result = cv2.merge((lightness, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.2, blurred, -0.2, 0)


def main():
    input_dir = Path("data")
    output_dir = Path("out/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = [
        "# Classical Pipeline Ablation",
        "",
        "| Image | Noisy PSNR | Denoise PSNR | +CLAHE PSNR | Full PSNR |",
        "|---|---:|---:|---:|---:|"
    ]

    for path in sorted(input_dir.glob("*.png")):
        original = cv2.imread(str(path))
        if original is None:
            continue

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 15, original.shape)
        noisy = np.clip(
            original.astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)

        denoised = denoise(noisy)
        contrast = apply_clahe(denoised)
        full = sharpen(contrast)

        variants = {
            "noisy": noisy,
            "denoise": denoised,
            "clahe": contrast,
            "full": full
        }

        scores = {}
        for name, result in variants.items():
            scores[name] = calculate_metrics(original, result)
            cv2.imwrite(
                str(output_dir / f"{path.stem}_{name}.png"), result
            )

        report.append(
            f"| {path.name} | {scores['noisy'][0]:.3f} "
            f"| {scores['denoise'][0]:.3f} "
            f"| {scores['clahe'][0]:.3f} "
            f"| {scores['full'][0]:.3f} |"
        )

        print(
            f"{path.name}: noisy={scores['noisy'][0]:.2f}, "
            f"denoise={scores['denoise'][0]:.2f}, "
            f"+CLAHE={scores['clahe'][0]:.2f}, "
            f"full={scores['full'][0]:.2f}"
        )

    Path("classical_ablation.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\nReport saved to classical_ablation.md")


if __name__ == "__main__":
    main()
