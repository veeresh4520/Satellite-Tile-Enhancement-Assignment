from pathlib import Path
import argparse
# pyrefly: ignore [missing-import]
import cv2    
import numpy as np
# pyrefly: ignore [missing-import]
from skimage.metrics import structural_similarity

def calculate_metrics(reference, prediction):
    """Calculate PSNR and SSIM between two BGR images."""
    psnr_value = cv2.PSNR(reference, prediction)

    reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    prediction_rgb = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

    ssim_value = structural_similarity(
        reference_rgb,
        prediction_rgb,
        channel_axis=2,
        data_range=255
    )

    return psnr_value, ssim_value
    
def evaluate_image(image, model_path):
    """Compare bicubic upscaling and FSRCNN against the original."""

    height, width = image.shape[:2]

    # 1. Artificially degrade the image to half its dimensions.
    small = cv2.resize(
        image,
        (width // 2, height // 2),
        interpolation=cv2.INTER_AREA
    )

    # 2. Reconstruct using bicubic interpolation.
    bicubic = cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # 3. Reconstruct using the pretrained FSRCNN model.
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel("fsrcnn", 2)
    fsrcnn = sr.upsample(small)

    # 4. Compare both reconstructions with the original.
    bicubic_psnr, bicubic_ssim = calculate_metrics(image, bicubic)
    fsrcnn_psnr, fsrcnn_ssim = calculate_metrics(image, fsrcnn)

    return {
        "bicubic_psnr": bicubic_psnr,
        "bicubic_ssim": bicubic_ssim,
        "fsrcnn_psnr": fsrcnn_psnr,
        "fsrcnn_ssim": fsrcnn_ssim
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bicubic vs FSRCNN super-resolution."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", default="results.md")
    parser.add_argument("--model", default="Models/FSRCNN_x2.pb")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    report_path = Path(args.report)

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Input directory not found: {input_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    image_paths = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise FileNotFoundError("No supported images found.")

    report_lines = [
        "# Super-Resolution Evaluation",
        "",
        "Experiment: downscale each image by 2x, then reconstruct.",
        "Reference: original image.",
        "",
        "| Image | Bicubic PSNR | FSRCNN PSNR | Bicubic SSIM | FSRCNN SSIM |",
        "|---|---:|---:|---:|---:|"
    ]

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        # FSRCNN requires dimensions divisible by 2 for this experiment.
        height, width = image.shape[:2]
        cropped = image[:height - height % 2, :width - width % 2]

        if cropped.size == 0:
            print(f"Skipping image too small: {path.name}")
            continue

        metrics = evaluate_image(cropped, model_path)

        report_lines.append(
            f"| {path.name} "
            f"| {metrics['bicubic_psnr']:.3f} "
            f"| {metrics['fsrcnn_psnr']:.3f} "
            f"| {metrics['bicubic_ssim']:.4f} "
            f"| {metrics['fsrcnn_ssim']:.4f} |"
        )

        print(
            f"{path.name}: "
            f"Bicubic PSNR={metrics['bicubic_psnr']:.2f}, "
            f"FSRCNN PSNR={metrics['fsrcnn_psnr']:.2f}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

