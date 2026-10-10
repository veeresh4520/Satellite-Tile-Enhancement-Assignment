
from pathlib import Path
import argparse

# pyrefly: ignore [missing-import]
import cv2

def enhance_super_resolution(image, model_path):
    """Upscale a BGR image by 2x using pretrained FSRCNN."""
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "dnn_superres is unavailable. Check your OpenCV installation."
        )

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel("fsrcnn", 2)

    return sr.upsample(image)


def enhance_classical(image):
    denoised = cv2.bilateralFilter(
        image, d=5, sigmaColor=30, sigmaSpace=30
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    )
    lightness = clahe.apply(lightness)

    merged = cv2.merge((lightness, a, b))
    contrast = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(contrast, 1.2, blurred, -0.2, 0)
def main():
    parser = argparse.ArgumentParser(
        description="Enhance satellite images using classical CV or super-resolution."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=["classical", "sr"],
        default="classical"
    )
    parser.add_argument(
        "--model",
        default="Models/FSRCNN_x2.pb"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png"}

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
    else:
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found at: {input_path}"
        )

    if args.method == "sr" and not model_path.is_file():
        raise FileNotFoundError(
            f"Super-resolution model not found: {model_path}"
        )

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        if args.method == "classical":
            result = enhance_classical(image)
        else:
            result = enhance_super_resolution(image, model_path)

        output_file = output_path / f"{path.stem}_{args.method}.png"

        if cv2.imwrite(str(output_file), result):
            print(f"Enhanced: {path.name} -> {output_file}")
            print(f"Output dimensions: {result.shape[1]} x {result.shape[0]}")
        else:
            print(f"Failed to save: {output_file}")


if __name__ == "__main__":
    main()
