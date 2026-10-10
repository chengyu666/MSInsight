# MSInsight

CUDA-accelerated source scanning (SSA/jSSA) and BFNet focal-mechanism inference.

## Setup and example

The waveform workflow requires Windows, a CUDA-compatible GPU, and the included
DLLs. Install a CUDA-enabled PyTorch build and the dependencies in
`requirements.txt`, then run `test.ipynb` from the repository root.

The notebook processes the bundled 30-second waveform through SSA, jSSA, and
BFNet. Results are saved to `result/`, which is created automatically and ignored
by Git. Preprocessing and search settings can be adjusted for other recordings.

For a model-only smoke check (CPU or GPU):

```bash
python model_bfnet2.py
```

## BFNet model

The example uses `model_bfnet2.py` with `model/bfnet2_ep46_260122a.pt`.

- Input shape: `(batch, 8, 8, 8, 24, 7, 24)`, ordered as
  `(batch, x, y, z, strike, dip, rake)`.
- Mechanism bins in degrees: strike `0, 15, ..., 345`; dip `0, 15, ..., 90`;
  rake `-180, -165, ..., 165`. Preserve this order when preparing inputs.
- Each input sample is min-max normalized independently.
- Output: `[sin(strike), cos(strike), sin(dip), cos(dip), sin(rake), cos(rake)]`.
  Each pair is L2-normalized; absolute values constrain the dip pair to the
  first quadrant, giving dip in `[0, 90]` degrees.
- Decode angles with `np.rad2deg(np.arctan2(sin, cos))`; convert strike to
  `[0, 360)` with `strike %= 360` if needed.

`model_bfnet.py` and `model/bfnet_251104a.pt` are the earlier model and checkpoint;
use each checkpoint with its matching model class.

## Project files

| Path | Purpose |
| --- | --- |
| `test.ipynb` | SSA--jSSA--BFNet example |
| `model_bfnet2.py`, `model/` | Model implementation and trained checkpoints |
| `waveform/` | Example waveform |
| `conf/` | Station, velocity, and search-grid configuration |
| `data.py`, `config.py` | Data processing and configuration |
| `stackCU.py`, `stackMechCU.py` | Python interfaces to SSA/jSSA DLLs |
| `draw.py`, `utils.py` | Visualization and moment-tensor utilities |
| `3draytracing/` | CUDA travel-time implementation |
| `fasterStackCUDA/` | CUDA SSA/jSSA stacking implementation |

See the [travel-time README](3draytracing/readme.md) and
[stacking README](fasterStackCUDA/README.md) for compilation details.
