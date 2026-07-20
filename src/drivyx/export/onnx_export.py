"""Torch to ONNX export (CLAUDE.md section 11).

"export: torch -> ONNX opset 17, static batch 1, input 1x3x384x768 (seg) and 1x8x48x96 + 1x1
(ctrl). Run onnxsim if importable."

Static batch 1 rather than dynamic: the deployment target runs one camera frame at a time, and
a fixed shape lets TensorRT specialise every kernel instead of building a shape-agnostic plan.

The models are exported in eval mode, which for PIDNet means the single-tensor output path
(the auxiliary and boundary heads exist only for training) and for CtrlNet means GroupNorm
behaves identically to training, since it has no running statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Section 11's opset.
OPSET = 17

#: Section 11's input shapes.
SEG_INPUT = (1, 3, 384, 768)
CTRL_LOGITS_INPUT = (1, 8, 48, 96)
CTRL_SPEED_INPUT = (1, 1)


@dataclass
class ExportResult:
    """What an export produced."""

    model: str
    onnx: Path
    simplified: bool
    inputs: dict[str, list[int]]
    outputs: dict[str, list[int]]
    opset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "onnx": str(self.onnx),
            "simplified": self.simplified,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "opset": self.opset,
        }


def _maybe_simplify(path: Path) -> bool:
    """Run onnxsim if it imported.

    Section 3 makes onnxsim optional on aarch64, so the guarded import is the contract: the
    export must produce a valid engine either way, and simplification is an optimisation.
    """
    try:
        import onnx
        import onnxsim
    except ImportError as exc:
        logger.info("onnxsim unavailable (%s); exporting without simplification", exc)
        return False

    try:
        model = onnx.load(str(path))
        simplified, ok = onnxsim.simplify(model)
        if not ok:
            logger.warning("onnxsim reported a failed check; keeping the unsimplified graph")
            return False
        onnx.save(simplified, str(path))
        logger.info("simplified %s", path.name)
        return True
    except Exception as exc:
        logger.warning("onnxsim failed (%s); keeping the unsimplified graph", exc)
        return False


def export_seg(checkpoint: Path, output: Path, *, opset: int = OPSET) -> ExportResult:
    """Export a trained PIDNet-S to ONNX at static batch 1."""
    from drivyx.models.pidnet import PIDNet

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_classes = state.get("config", {}).get("num_classes", 8)

    model = PIDNet(num_classes)
    model.load_state_dict(state["model"])
    # eval() is load-bearing: in train mode forward returns three tensors and the export would
    # carry the auxiliary heads into the engine.
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*SEG_INPUT)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    simplified = _maybe_simplify(output)

    with torch.no_grad():
        out_shape = list(model(dummy).shape)

    logger.info("exported seg -> %s (%s)", output, "simplified" if simplified else "as traced")
    return ExportResult(
        model="seg",
        onnx=output,
        simplified=simplified,
        inputs={"image": list(SEG_INPUT)},
        outputs={"logits": out_shape},
        opset=opset,
    )


def export_ctrl(checkpoint: Path, output: Path, *, opset: int = OPSET) -> ExportResult:
    """Export a trained CtrlNet to ONNX with its two inputs (section 11)."""
    from drivyx.models.ctrlnet import CtrlNet

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    in_channels = CTRL_LOGITS_INPUT[1]
    model = CtrlNet(in_channels)
    model.load_state_dict(state["model"])
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    logits = torch.randn(*CTRL_LOGITS_INPUT)
    speed = torch.zeros(*CTRL_SPEED_INPUT)

    torch.onnx.export(
        model,
        (logits, speed),
        str(output),
        input_names=["logits", "speed"],
        output_names=["waypoints"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    simplified = _maybe_simplify(output)

    with torch.no_grad():
        out_shape = list(model(logits, speed).shape)

    logger.info("exported ctrl -> %s (%s)", output, "simplified" if simplified else "as traced")
    return ExportResult(
        model="ctrl",
        onnx=output,
        simplified=simplified,
        inputs={"logits": list(CTRL_LOGITS_INPUT), "speed": list(CTRL_SPEED_INPUT)},
        outputs={"waypoints": out_shape},
        opset=opset,
    )


def verify_onnx(path: Path) -> dict[str, Any]:
    """Check the graph loads and report its declared shapes.

    Exporting a file that torch wrote but onnx cannot read is a failure worth catching before
    trtexec spends minutes on it.
    """
    try:
        import onnx
    except ImportError:
        return {"checked": False, "reason": "onnx not installed"}

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def shape_of(value: Any) -> list[int | str]:
        return [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in value.type.tensor_type.shape.dim
        ]

    return {
        "checked": True,
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "inputs": {i.name: shape_of(i) for i in model.graph.input},
        "outputs": {o.name: shape_of(o) for o in model.graph.output},
        "nodes": len(model.graph.node),
    }
