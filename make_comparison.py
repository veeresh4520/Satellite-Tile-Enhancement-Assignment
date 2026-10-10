
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

DATA = Path("data")
ABLATION = Path("out/ablation")
SR_OUTPUT = Path("out/final_sr")
OUTPUT = Path("comparison_grid.png")


def add_label(image, label):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        image, label, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )
    return image


def main():
    filename = "building_d87dbc_google"

    paths = [
        (DATA / f"{filename}.png", "Original"),
        (ABLATION / f"{filename}_noisy.png", "Noisy"),
        (ABLATION / f"{filename}_denoise.png", "Denoising"),
        (ABLATION / f"{filename}_clahe.png", "Denoise + CLAHE"),
        (ABLATION / f"{filename}_full.png", "Full pipeline"),
        (SR_OUTPUT / f"{filename}_sr.png", "FSRCNN 2x"),
    ]

    tiles = []
    for path, label in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Missing image: {path}")
            continue

        image = cv2.resize(image, (320, 220))
        tiles.append(add_label(image, label))

    if not tiles:
        print("No images were found. Check the paths.")
        return

    # Arrange two columns per row.
    tile_h, tile_w = tiles[0].shape[:2]
    blank = np.full((tile_h, tile_w, 3), 230, dtype=np.uint8)

    while len(tiles) % 2:
        tiles.append(add_label(blank, "Not available"))

    rows = []
    for i in range(0, len(tiles), 2):
        rows.append(np.hstack(tiles[i:i + 2]))

    grid = np.vstack(rows)
    cv2.imwrite(str(OUTPUT), grid)
    print(f"Comparison grid saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
