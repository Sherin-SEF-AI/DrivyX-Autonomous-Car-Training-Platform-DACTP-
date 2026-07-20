"""Torch to TensorRT parity checking (CLAUDE.md section 11).

"parity: run 200 val images through torch and the TRT engine; abort (exit 1) if seg mIoU delta
> 1.0 absolute or ctrl ADE delta > 0.05 m. Writes export/parity.json."

Section 10 says the accuracy numbers themselves are the deliverable with no hardcoded gates.
This is the exception: section 11 makes parity "the only pass/fail". The reasoning is that an
engine which disagrees with the model it was built from is broken regardless of how good either
one is, and that is a question with a right answer.

The comparison is deliberately end to end (same preprocessing, same postprocessing, same
metric) rather than a tensor-level allclose. Quantisation legitimately changes individual
logits; what must not change is the decision the pipeline makes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Section 11's thresholds.
SEG_MIOU_TOLERANCE = 1.0  # absolute percentage points
CTRL_ADE_TOLERANCE = 0.05  # metres
PARITY_SAMPLES = 200


@dataclass
class ParityResult:
    """One parity comparison."""

    model: str
    precision: str
    samples: int
    torch_metric: float
    engine_metric: float
    delta: float
    tolerance: float
    metric_name: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "precision": self.precision,
            "samples": self.samples,
            "metric": self.metric_name,
            "torch": self.torch_metric,
            "engine": self.engine_metric,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


class TrtRunner:
    """Minimal TensorRT execution wrapper.

    Uses the python bindings directly rather than trtexec because parity needs the actual
    output tensors, not a latency report. Allocation happens once and buffers are reused, so
    the per-sample cost is the inference itself.
    """

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise {engine_path}")
        self.context = self.engine.create_execution_context()
        self.trt = trt

    def __enter__(self) -> TrtRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        del self.context
        del self.engine

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one inference. Inputs and outputs are keyed by tensor name."""
        import torch

        outputs: dict[str, np.ndarray] = {}
        bindings: list[int] = []
        held: list[torch.Tensor] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == self.trt.TensorIOMode.INPUT:
                array = np.ascontiguousarray(inputs[name].astype(dtype))
                tensor = torch.from_numpy(array).cuda()
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                tensor = torch.empty(
                    shape, dtype=getattr(torch, np.dtype(dtype).name), device="cuda"
                )
                outputs[name] = tensor

            held.append(tensor)
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            bindings.append(int(tensor.data_ptr()))

        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        return {k: v.cpu().numpy() for k, v in outputs.items()}


def check_seg_parity(
    paths: Any,
    run: Any,
    engine_path: Path,
    precision: str,
    *,
    samples: int = PARITY_SAMPLES,
) -> ParityResult:
    """Compare torch and engine mIoU over the same val images (section 11)."""
    import torch

    from drivyx.data.lut import IGNORE_ID
    from drivyx.data.seg_dataset import SegShardDataset
    from drivyx.eval.seg_eval import metrics_from_confusion
    from drivyx.models.pidnet import PIDNet
    from drivyx.torch_setup import AUTOCAST_DTYPE

    device = torch.device("cuda")
    ckpt = run.best_ckpt if run.best_ckpt.is_file() else run.last_ckpt
    state = torch.load(ckpt, map_location=device, weights_only=False)
    config = state.get("config", {})
    num_classes = config.get("num_classes", 8)

    model = PIDNet(num_classes)
    model.load_state_dict(state["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()

    # The engine's input is fixed at 384x768 (section 11), so parity is measured at that size
    # for both sides. Comparing torch at 1024x512 against an engine at 384x768 would measure
    # the resolution difference, not the quantisation.
    dataset = SegShardDataset(paths, "val", train=False, val_w=768, val_h=384)
    total = min(samples, len(dataset))

    torch_confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    engine_confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    with TrtRunner(engine_path) as runner, torch.no_grad():
        input_name = runner.engine.get_tensor_name(0)
        output_name = next(
            runner.engine.get_tensor_name(i)
            for i in range(runner.engine.num_io_tensors)
            if runner.engine.get_tensor_mode(runner.engine.get_tensor_name(i))
            == runner.trt.TensorIOMode.OUTPUT
        )

        for index in range(total):
            image, target = dataset[index]
            batch = image.unsqueeze(0)
            target_dev = target.unsqueeze(0).to(device)

            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                torch_logits = model(batch.to(device).contiguous(memory_format=torch.channels_last))
            torch_pred = _upsample_argmax(torch_logits.float(), target_dev.shape[-2:])

            engine_out = runner.infer({input_name: batch.numpy()})[output_name]
            engine_pred = _upsample_argmax(
                torch.from_numpy(engine_out).to(device), target_dev.shape[-2:]
            )

            valid = target_dev != IGNORE_ID
            if valid.any():
                for confusion, prediction in (
                    (torch_confusion, torch_pred),
                    (engine_confusion, engine_pred),
                ):
                    pairs = target_dev[valid] * num_classes + prediction[valid]
                    confusion += torch.bincount(pairs, minlength=num_classes**2).reshape(
                        num_classes, num_classes
                    )

    names = [f"c{i}" for i in range(num_classes)]
    torch_miou = metrics_from_confusion(torch_confusion, names, total).miou * 100.0
    engine_miou = metrics_from_confusion(engine_confusion, names, total).miou * 100.0
    delta = abs(torch_miou - engine_miou)

    result = ParityResult(
        model="seg",
        precision=precision,
        samples=total,
        torch_metric=torch_miou,
        engine_metric=engine_miou,
        delta=delta,
        tolerance=SEG_MIOU_TOLERANCE,
        metric_name="mIoU (percentage points)",
        passed=delta <= SEG_MIOU_TOLERANCE,
    )
    logger.info(
        "seg parity %s: torch mIoU %.3f vs engine %.3f, delta %.3f (tolerance %.1f) -> %s",
        precision,
        torch_miou,
        engine_miou,
        delta,
        SEG_MIOU_TOLERANCE,
        "PASS" if result.passed else "FAIL",
    )
    return result


def _upsample_argmax(logits: Any, size: Any) -> Any:
    import torch

    lifted = torch.nn.functional.interpolate(
        logits, size=size, mode="bilinear", align_corners=False
    )
    return lifted.argmax(1)


def check_ctrl_parity(
    paths: Any,
    run: Any,
    engine_path: Path,
    precision: str,
    *,
    samples: int = PARITY_SAMPLES,
) -> ParityResult:
    """Compare torch and engine ADE over the same control frames (section 11)."""
    import torch

    from drivyx.models.ctrlnet import CtrlNet
    from drivyx.train.ctrl_trainer import CtrlDataset, cache_for

    device = torch.device("cuda")
    ckpt = run.best_ckpt if run.best_ckpt.is_file() else run.last_ckpt
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cache = cache_for(paths, state["seg_run"])
    manifest = json.loads(cache.manifest.read_text())

    model = CtrlNet(manifest["num_classes"])
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    dataset = CtrlDataset(cache, val=True)
    total = min(samples, len(dataset))

    torch_errors: list[float] = []
    engine_errors: list[float] = []

    with TrtRunner(engine_path) as runner, torch.no_grad():
        names = [runner.engine.get_tensor_name(i) for i in range(runner.engine.num_io_tensors)]
        inputs = [
            n for n in names if runner.engine.get_tensor_mode(n) == runner.trt.TensorIOMode.INPUT
        ]
        output_name = next(
            n for n in names if runner.engine.get_tensor_mode(n) == runner.trt.TensorIOMode.OUTPUT
        )
        logits_name = next(n for n in inputs if "speed" not in n.lower())
        speed_name = next(n for n in inputs if "speed" in n.lower())

        for index in range(total):
            logits, speed, target = dataset[index]
            batch = logits.unsqueeze(0)
            speed_batch = speed.reshape(1, 1)

            torch_pred = model(batch.to(device), speed_batch.to(device)).float().cpu().numpy()[0]
            engine_pred = runner.infer(
                {logits_name: batch.numpy(), speed_name: speed_batch.numpy()}
            )[output_name].reshape(torch_pred.shape)

            truth = target.numpy()
            torch_errors.append(float(np.linalg.norm(torch_pred - truth, axis=-1).mean()))
            engine_errors.append(float(np.linalg.norm(engine_pred - truth, axis=-1).mean()))

    torch_ade = float(np.mean(torch_errors))
    engine_ade = float(np.mean(engine_errors))
    delta = abs(torch_ade - engine_ade)

    result = ParityResult(
        model="ctrl",
        precision=precision,
        samples=total,
        torch_metric=torch_ade,
        engine_metric=engine_ade,
        delta=delta,
        tolerance=CTRL_ADE_TOLERANCE,
        metric_name="ADE (metres)",
        passed=delta <= CTRL_ADE_TOLERANCE,
    )
    logger.info(
        "ctrl parity %s: torch ADE %.4f m vs engine %.4f m, delta %.4f (tolerance %.2f) -> %s",
        precision,
        torch_ade,
        engine_ade,
        delta,
        CTRL_ADE_TOLERANCE,
        "PASS" if result.passed else "FAIL",
    )
    return result


def write_parity(paths: Any, results: list[ParityResult]) -> Path:
    """Write export/parity.json (section 11)."""
    payload = {
        "passed": all(r.passed for r in results),
        "checks": [r.to_dict() for r in results],
        "tolerances": {
            "seg_miou_points": SEG_MIOU_TOLERANCE,
            "ctrl_ade_metres": CTRL_ADE_TOLERANCE,
        },
    }
    paths.export.mkdir(parents=True, exist_ok=True)
    path = paths.export / "parity.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
