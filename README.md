# MSInsight

CUDA-accelerated source scanning (SSA/jSSA) and BFNet focal-mechanism inference.

## Current BFNet checkpoint

The default example uses **`model/bfnet2_ep46_260122a.pt`** with
**`model_bfnet2.py`**, the dip-constrained BFNet model used for the current
GRSL manuscript, *BFNet: Improving Focal Mechanism Resolution in jSSA With a
Brightness-Field Neural Network*. The implementation name `bfnet2` distinguishes
this checkpoint from the earlier BFNet implementation; the paper calls it BFNet.

- Input: a peak-time jSSA brightness field, shaped `(batch, 8, 8, 8, 24, 7, 24)`.
  Axes are `(batch, x, y, z, strike, dip, rake)`. Keep the mechanism-bin order
  used to generate the training brightness fields: strike `0, 15, ..., 345`,
  dip `0, 15, ..., 90`, and rake `-180, -165, ..., 165` degrees. The legacy
  example started the rake grid at `-165`; that order is incompatible with the
  current checkpoint and has been updated in `data.gen_fm_grid`.
- Each sample is min-max normalized independently.
- Spatial depthwise convolutions and mechanism convolutions are followed by
  linear layers with LayerNorm and dropout.
- Output order: `[sin(strike), cos(strike), sin(dip), cos(dip), sin(rake), cos(rake)]`.
  Each pair is L2-normalized; absolute values constrain the dip pair to the first
  quadrant, giving dip in `[0, 90]` degrees.
- This checkpoint uses **no tanh activation** before pair normalization.
- Trainable parameters: **16,979,030** (16.98 million).
- Checkpoint SHA-256:
  `48084368ccbbf49d61fb7b35f29aeb58370865a6a62a222663d2a77235145407`.

`model_bfnet.py` and `model/bfnet_251104a.pt` are retained as the **legacy**
implementation/checkpoint. Do not load the current checkpoint into that class.

## Setup and example

The full waveform workflow uses Windows DLLs and requires a CUDA-compatible
GPU. The BFNet PyTorch model itself also supports CPU inference. The model
update is validated with PyTorch 2.6.0+cu124 on an NVIDIA GeForce RTX 3080 Ti.

Install a suitable CUDA-enabled PyTorch build for your environment and the
dependencies in `requirements.txt`, then run `test.ipynb` from the repository
root. The notebook loads the current checkpoint in evaluation mode and runs
SSA, jSSA, and BFNet on the bundled waveform.

The bundled waveform covers approximately **30 seconds** and contains one
example event. Its filtering parameters are data-specific and may be adjusted
for other recordings. It is not the complete 170-minute Dataset C experiment.

For a model-only smoke check:

```bash
python model_bfnet2.py
```

For inference on an existing peak-time brightness field:

```python
from pathlib import Path
import numpy as np
import torch
from model_bfnet2 import StrikeDipRakeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StrikeDipRakeNet((8, 8, 8, 24, 7, 24))
model.load_state_dict(torch.load(
    Path("model/bfnet2_ep46_260122a.pt"), map_location="cpu", weights_only=True
))
model.to(device).eval()
field = np.load("brightness_field.npy", allow_pickle=False)
x = torch.from_numpy(field).float().reshape(1, 8, 8, 8, 24, 7, 24).to(device)
with torch.inference_mode():
    pairs = model(x).cpu().numpy()
angles = np.rad2deg(np.arctan2(pairs[:, 0::2], pairs[:, 1::2]))
# Columns: strike, dip, rake. atan2 returns signed strike and rake angles;
# an equivalent strike in [0, 360) can be obtained with angles[:, 0] %= 360.
```

## Validation against archived predictions

`scripts/validate_bfnet.py` reruns inference from the original `.npy` brightness
fields and checks predictions, sine-cosine pair norms, and the dip constraint.
It accepts six-dimensional fields or fields saved with the three mechanism
axes flattened as `(8, 8, 8, 4032)`; their ordering must match the current grid.
For synthetic data it also recomputes the original evaluator's angular errors
over four equivalent nodal-plane pairings using `utils.angle_diff_deg`.
It does not use reference predictions as model inputs.

```bash
python scripts/validate_bfnet.py --kind synthetic --data-dir PATH_TO_DATASET_A --reference PATH_TO_ARCHIVED_VALIDATION_RESULTS --output-dir validation_runs/dataset_a
python scripts/validate_bfnet.py --kind field --data-dir PATH_TO_FIELD_BRIGHTNESS_FIELDS --reference PATH_TO_ARCHIVED_FIELD_RESULTS --output-dir validation_runs/field
```

The scripts accept a 10-column synthetic reference
`idx strike dip rake ts td tr strike_diff dip_diff rake_diff` or a 4-column field
reference `idx strike dip rake`. For field data, IDs enumerate brightness files
sorted by the numeric event suffix. Outputs include fresh predictions and a
JSON report; the command fails if differences exceed the default 0.03-degree
tolerance. Original full datasets and archived reference files are external
inputs, not included in this repository. Field prediction agreement checks
implementation consistency, not accuracy against independent mechanism truth.

Validation on 2026-10-09 reran **all 1000 Dataset A validation events** and
**all 127 archived field brightness fields**. Predictions agreed with their
two-decimal archived references to within **0.005 degrees**, all output pairs
had unit norm within `1.2e-7`, and all predicted dips were in `[0, 90]` degrees.
The complete bundled waveform example also ran successfully with the updated
checkpoint. The released class's forward computation matches the original
trained model exactly; CPU inference and batch independence were checked.

| Dataset A error (degrees) | Strike | Dip | Rake |
| --- | ---: | ---: | ---: |
| Mean | 3.49 | 1.80 | 8.62 |
| Median | 2.07 | 1.23 | 5.61 |

Detailed environment, checkpoint/reference hashes, and verification results
are recorded in [`validation/bfnet2_ep46_260122a.json`](validation/bfnet2_ep46_260122a.json).
These checks do not reproduce model training or the paper's inference-time
and end-to-end workflow benchmarks.

## Project files

| Path | Purpose |
| --- | --- |
| `model_bfnet2.py` | Current dip-constrained BFNet model |
| `model/bfnet2_ep46_260122a.pt` | Current trained checkpoint |
| `test.ipynb` | Bundled waveform example with the current model |
| `scripts/validate_bfnet.py` | Recompute and compare model predictions |
| `validation/` | Aggregate verification records |
| `model_bfnet.py`, `model/bfnet_251104a.pt` | Legacy model and checkpoint |
| `conf/` | Station, velocity, and search-grid configuration |
| `3draytracing/` | CUDA travel-time implementation |
| `fasterStackCUDA/` | CUDA SSA/jSSA stacking implementation |
| `data.py`, `config.py` | Data processing and configuration |
| `stackCU.py`, `stackMechCU.py` | Python interfaces to SSA/jSSA DLLs |
| `draw.py`, `utils.py` | Visualization and moment-tensor utilities |
| `waveform/`, `result/` | Bundled example waveform and results |

See the README files in `3draytracing/` and `fasterStackCUDA/` for compilation
details. This update releases the inference model and checkpoint; the example
does not include the full training or paper benchmark experiments.
