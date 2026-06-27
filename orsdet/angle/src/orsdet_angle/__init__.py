"""V2 angle-head utilities for the CIANNA-New OBB migration."""

from .angle_codec import (
    angle_diff_le90_deg,
    decode_angle_vector,
    encode_theta_le90,
    normalize_theta_le90,
)
from .angle_loss import (
    AngleLossConfig,
    AspectWeightConfig,
    angle_loss_and_grad,
    angle_weight_from_aspect,
)
from .tables import (
    AngleTargetTable,
    build_angle_target_table,
    load_angle_target_table,
    load_v1_rotated_table,
    save_angle_target_table,
)
from .cianna_bridge import (
    V2AngleHeadSpec,
    decode_angle_prediction,
    make_angle_target_channels,
)
from .cianna_targets import (
    V2_NB_ANGLE,
    V2_NB_PARAM,
    V2TargetSpec,
    convert_legacy_targets_to_v2,
    decode_legacy_pa,
    v2_target_dim,
)
from .cianna_forward_eval import (
    angle_distribution_report,
    append_angle_history_row,
    metrics_from_valid_targets_and_forward,
    parse_yolo_forward,
    summarize_all_metrics,
    write_valid_angle_report,
)

__all__ = [
    "AngleLossConfig",
    "AngleTargetTable",
    "AspectWeightConfig",
    "angle_diff_le90_deg",
    "angle_distribution_report",
    "angle_loss_and_grad",
    "angle_weight_from_aspect",
    "append_angle_history_row",
    "build_angle_target_table",
    "convert_legacy_targets_to_v2",
    "decode_angle_prediction",
    "decode_angle_vector",
    "decode_legacy_pa",
    "encode_theta_le90",
    "load_angle_target_table",
    "load_v1_rotated_table",
    "make_angle_target_channels",
    "metrics_from_valid_targets_and_forward",
    "normalize_theta_le90",
    "parse_yolo_forward",
    "save_angle_target_table",
    "summarize_all_metrics",
    "V2_NB_ANGLE",
    "V2_NB_PARAM",
    "V2AngleHeadSpec",
    "V2TargetSpec",
    "v2_target_dim",
    "write_valid_angle_report",
]
