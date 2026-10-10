# Satellite Tile Enhancement

A CPU-based Python project comparing classical image enhancement with pretrained 2× super-resolution for satellite tiles.

## Included

- `enhance.py`: classical enhancement and FSRCNN 2× CLI.
- `evaluate.py`: bicubic-versus-FSRCNN PSNR/SSIM evaluation using synthetic downscaling.
- `evaluate_classical.py`: synthetic Gaussian-noise ablation experiment.
- `make_comparison.py`: creates the example comparison grid after the experiment outputs are generated.
- `make_sr_failure.py`: creates original-versus-FSRCNN preview crops.
- `data/`: the 41 supplied input tiles.
- `Models/FSRCNN_x2.pb`: pretrained FSRCNN model.
- `results.md`, `classical_results.md`, `classical_ablation.md`: experiment reports.
- `comparison_grid.png` and `out/failure_cases/`: visual evidence.

## Setup

Use Python 3 and install dependencies:

```bash
python -m pip install -r requirements.txt
```

The FSRCNN path is case-sensitive on some systems; the supplied model is at `Models/FSRCNN_x2.pb`.

## Usage

### Classical enhancement

```bash
python enhance.py --input data --output out/final_classical --method classical
```

### 2× super-resolution

```bash
python enhance.py --input data --output out/final_sr --method sr --model Models/FSRCNN_x2.pb
```

### Super-resolution evaluation

```bash
python evaluate.py --input data --report results.md --model Models/FSRCNN_x2.pb
```

This downsamples each input image and compares bicubic and FSRCNN reconstructions against the original reference using PSNR and SSIM.

### Classical ablation

```bash
python evaluate_classical.py
```

This writes the ablation images to `out/ablation/` and updates `classical_ablation.md`. The script expects to be run from the project root.

### Visual comparison grid

Run the classical ablation and FSRCNN enhancement commands first, then run `python make_comparison.py`. The grid script uses `out/ablation/` and `out/final_sr/`.

### FSRCNN preview crops

After generating `out/final_sr/`, run `python make_sr_failure.py`. These are visual previews, not proof that reconstructed details match real-world geography.

## Evaluation notes and limitations

PSNR measures pixel-level reconstruction error, while SSIM measures structural similarity. Neither metric proves geographic accuracy. Synthetic degradation is a controlled test and may not represent real satellite degradation. Results, including negative results, should be reported honestly.

## Data rights and usage restrictions

Verify the applicable Google Maps Platform Terms and each imagery provider's conditions before storing, displaying, or redistributing source or enhanced imagery. Enhancement does not remove licensing or attribution obligations.
