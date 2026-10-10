
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

DATA = Path("data")
SR = Path("out/final_sr")
OUTPUT = Path("out/failure_cases")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Three actual examples from the dataset
CASES = [
    "building_d87dbc_google.png",
    "building_aba1eb_google.png",
    "building_20d961_google.png",
]

for filename in CASES:
    original_path = DATA / filename
    sr_path = SR / f"{Path(filename).stem}_sr.png"

    original = cv2.imread(str(original_path))
    sr = cv2.imread(str(sr_path))

    if original is None or sr is None:
        print(f"Skipping {filename}: image not found")
        continue

    # Resize the SR image to the original size for visual comparison.
    # This is only for display, not for evaluating 2x output resolution.
    sr_at_original_size = cv2.resize(
        sr,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    h, w = original.shape[:2]
    side = min(150, h, w)
    y, x = (h - side) // 2, (w - side) // 2

    panels = []
    for label, image in [
        ("Original", original),
        ("FSRCNN 2x (preview)", sr_at_original_size),
    ]:
        crop = image[y:y + side, x:x + side].copy()
        crop = cv2.resize(crop, (300, 300))
        cv2.putText(
            crop, label, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2
        )
        panels.append(crop)

    result = np.hstack(panels)
    output_path = OUTPUT / f"{Path(filename).stem}_sr_comparison.png"
    cv2.imwrite(str(output_path), result)
    print(f"Saved: {output_path}")
