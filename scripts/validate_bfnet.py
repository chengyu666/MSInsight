"""Validate the released BFNet checkpoint against archived event predictions.

Synthetic references contain: idx strike dip rake ts td tr strike_diff
dip_diff rake_diff. Field references contain: idx strike dip rake.
The original .npy brightness fields are external inputs, not bundled here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from obspy.imaging.beachball import aux_plane
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_bfnet2 import StrikeDipRakeNet
from utils import angle_diff_deg

SHAPE = (8, 8, 8, 24, 7, 24)


def angle_errors(prediction, truth):
    """Use the original evaluator's four nodal-plane pairings and angle metric."""
    pred_planes = [prediction, aux_plane(*prediction)]
    true_planes = [truth, aux_plane(*truth)]
    return [
        min(angle_diff_deg(p[k], t[k]) for p in pred_planes for t in true_planes)
        for k in range(3)
    ]


def circular_difference(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def read_field(path):
    x = np.load(path, allow_pickle=False)
    if x.shape not in (SHAPE, (*SHAPE[:3], int(np.prod(SHAPE[3:])))) or not np.isfinite(x).all():
        raise ValueError(f"Invalid brightness field: {path.name}, shape={x.shape}")
    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).reshape(1, *SHAPE)
    # Match the original evaluation notebook's input preprocessing.
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["synthetic", "field"], required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "model" / "bfnet2_ep46_260122a.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tolerance-deg", type=float, default=0.03)
    args = parser.parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    reference = np.loadtxt(args.reference, ndmin=2)
    expected_columns = 10 if args.kind == "synthetic" else 4
    if reference.shape[1] != expected_columns or not np.isfinite(reference).all():
        raise ValueError("Reference has invalid columns or nonfinite values.")
    ids = reference[:, 0].astype(int)
    if not np.array_equal(ids, reference[:, 0]) or len(set(ids)) != len(ids):
        raise ValueError("Reference event IDs must be unique integers.")
    files = sorted(args.data_dir.glob("*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    if args.kind == "synthetic":
        file_map = {int(p.stem.split("_")[-1]): p for p in files}
    else:
        if len(files) != len(reference):
            raise ValueError("Field reference must cover all sorted input events.")
        file_map = dict(enumerate(files))
    if not all(i in file_map for i in ids):
        raise ValueError("Some reference events have no input brightness field.")

    model = StrikeDipRakeNet(SHAPE)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    rows = []
    norm_error = 0.0
    smallest_dip_component = 1.0
    with torch.inference_mode():
        for i, ref in enumerate(reference):
            pairs = model(read_field(file_map[int(ref[0])]).to(device)).cpu().numpy()[0]
            if not np.isfinite(pairs).all():
                raise ValueError(f"Nonfinite prediction for event {int(ref[0])}.")
            norm_error = max(norm_error, float(np.max(np.abs(np.linalg.norm(pairs.reshape(3, 2), axis=1) - 1))))
            smallest_dip_component = min(smallest_dip_component, float(pairs[2:4].min()))
            angles = np.rad2deg(np.arctan2(pairs[0::2], pairs[1::2]))
            if not 0 <= angles[1] <= 90:
                raise ValueError(f"Dip outside [0, 90] for event {int(ref[0])}.")
            row = [int(ref[0]), *angles.tolist()]
            if args.kind == "synthetic":
                row.extend([*ref[4:7].tolist(), *angle_errors(angles, ref[4:7])])
            rows.append(row)
            if (i + 1) % 100 == 0 or i + 1 == len(reference):
                print(f"Validated {i + 1}/{len(reference)} events", flush=True)

    result = np.asarray(rows)
    differences = circular_difference(result[:, 1:4], reference[:, 1:4])
    differences[:, 1] = np.abs(result[:, 2] - reference[:, 2])
    max_difference = differences.max(axis=0)
    metrics = {
        "kind": args.kind,
        "events": len(result),
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "reference": args.reference.name,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "input_shape": list(SHAPE),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "angle_order": ["strike", "dip", "rake"],
        "max_prediction_difference_deg": max_difference.tolist(),
        "tolerance_deg": args.tolerance_deg,
        "max_pair_norm_error": norm_error,
        "min_dip_component": smallest_dip_component,
        "dip_range_deg": [float(result[:, 2].min()), float(result[:, 2].max())],
    }
    if args.kind == "synthetic":
        metrics["mean_errors_deg"] = result[:, 7:10].mean(axis=0).tolist()
        metrics["median_errors_deg"] = np.median(result[:, 7:10], axis=0).tolist()
        metrics["reference_mean_errors_deg"] = reference[:, 7:10].mean(axis=0).tolist()
        metrics["reference_median_errors_deg"] = np.median(reference[:, 7:10], axis=0).tolist()
        metrics["max_error_difference_deg"] = np.abs(result[:, 7:10] - reference[:, 7:10]).max(axis=0).tolist()
        metrics["error_definition"] = "Original evaluator: minimum over four nodal-plane pairings using utils.angle_diff_deg."
    metrics["passed"] = bool(
        np.all(max_difference <= args.tolerance_deg)
        and norm_error < 1e-5 and smallest_dip_component >= 0
        and (args.kind != "synthetic" or np.max(metrics["max_error_difference_deg"]) <= args.tolerance_deg)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = "idx strike dip rake"
    if args.kind == "synthetic":
        header += " ts td tr strike_diff dip_diff rake_diff"
    np.savetxt(args.output_dir / "predictions.txt", result, fmt="%.8f", header=header)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    if not metrics["passed"]:
        raise SystemExit("Validation failed: predictions or output constraints differ from the reference.")


if __name__ == "__main__":
    main()
