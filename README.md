<div align="center">
  
  <img src="squeakout.png " width="500">

  [![bioRxiv](https://img.shields.io/badge/bioRxiv-DOI:10.1101/2024.04.19.590368-blue?style=flat-square&color=%23cf222e)](https://doi.org/10.1101/2024.04.19.590368)
  [![OSF](https://img.shields.io/badge/dataset-DOI:10.17605/OSF.IO/F9SBT-blue?style=flat-square)](https://osf.io/f9sbt/)
  
  
</div>

### SqueakOut: Autoencoder-based segmentation of mouse ultrasonic vocalizations

- [Paper @ bioRxiv](https://www.biorxiv.org/content/10.1101/2024.04.19.590368)
- [USV segmentation dataset @ OSF](https://osf.io/f9sbt/)

---

Inference-first repo. Supported workflow: load the pretrained checkpoint and segment a directory of spectrogram images.

## Environment

Create or update the permanent `squeakout` micromamba environment:

```bash
micromamba env create -f environment.yml
micromamba env update -n squeakout -f environment.yml
micromamba run -n squeakout python -m ipykernel install --user --name squeakout --display-name "Python (squeakout)"
```

## Usage

Notebook entrypoint: `segmentation.ipynb`

CLI entrypoint:

```bash
micromamba run -n squeakout python -m squeakout ./dataset/test
```

Python API:

```python
from pathlib import Path

from squeakout import load_model, resolve_device, segment_directory

device = resolve_device()
model = load_model(Path("./squeakout_weights.ckpt"), device=device)
results = segment_directory(
    Path("./dataset/test"),
    mask_root=Path("./outputs/segmentation"),
    montage_root=Path("./outputs/montages"),
    model=model,
    device=device,
)
print(f"saved {len(results)} masks")
```

Tests:

```bash
micromamba run -n squeakout pytest
```

---

`BibTex` citation
```latex
@article{Santana2024SqueakOut,
  title={SqueakOut: Autoencoder-based segmentation of mouse ultrasonic vocalizations},
  author={Gustavo M. Santana and Marcelo O. Dietrich},
  journal={bioRxiv},
  year={2024},
  doi={https://doi.org/10.1101/2024.04.19.590368},
  url={https://www.biorxiv.org/content/10.1101/2024.04.19.590368v1}
}
```
