"""Kuka iiwa14 + left Sharpa: SimToolReal's robot (the Play task's default, and its prior)."""

from __future__ import annotations

from .spec import RobotSpec

# ----------------------------------------------------------------------------
# Kuka iiwa14 + left Sharpa: SimToolReal's robot. Values verified with the pretrained checkpoint;
# scene_utils / obs_utils re-export them under their original names.
# ----------------------------------------------------------------------------

_SHARPA_JOINTS = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

# SimToolReal's own list (isaacgymenvs/tasks/simtoolreal/adjacent_links.py,
# LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS), verbatim: the recipe, so the Kuka control runs it.
# Its hand pairs are each jointed pair, the palm with the thumb MC (touching at rest), and the
# palm (or MC) with each proximal phalanx across the two-joint MCP knuckle.
_SHARPA_SELF_COLLISION_FILTER = {
    "iiwa14_link_0": ("iiwa14_link_1",),
    "iiwa14_link_1": ("iiwa14_link_0", "iiwa14_link_2"),
    "iiwa14_link_2": ("iiwa14_link_1", "iiwa14_link_3"),
    "iiwa14_link_3": ("iiwa14_link_2", "iiwa14_link_4"),
    "iiwa14_link_4": ("iiwa14_link_3", "iiwa14_link_5"),
    "iiwa14_link_5": ("iiwa14_link_4", "iiwa14_link_6"),
    "iiwa14_link_6": ("iiwa14_link_5", "iiwa14_link_7"),
    "iiwa14_link_7": (
        "iiwa14_link_6", "left_thumb_CMC_VL", "left_thumb_MC", "left_index_MCP_VL", "left_index_PP",
        "left_middle_MCP_VL", "left_middle_PP", "left_ring_MCP_VL", "left_ring_PP", "left_pinky_MC",
    ),
    "left_index_MCP_VL": ("iiwa14_link_7", "left_index_PP"),
    "left_index_PP": ("iiwa14_link_7", "left_index_MCP_VL", "left_index_MP"),
    "left_index_MP": ("left_index_PP", "left_index_DP"),
    "left_index_DP": ("left_index_MP",),
    "left_middle_MCP_VL": ("iiwa14_link_7", "left_middle_PP"),
    "left_middle_PP": ("iiwa14_link_7", "left_middle_MCP_VL", "left_middle_MP"),
    "left_middle_MP": ("left_middle_PP", "left_middle_DP"),
    "left_middle_DP": ("left_middle_MP",),
    "left_pinky_MC": ("iiwa14_link_7", "left_pinky_MCP_VL", "left_pinky_PP"),
    "left_pinky_MCP_VL": ("left_pinky_MC", "left_pinky_PP"),
    "left_pinky_PP": ("left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_MP"),
    "left_pinky_MP": ("left_pinky_PP", "left_pinky_DP"),
    "left_pinky_DP": ("left_pinky_MP",),
    "left_ring_MCP_VL": ("iiwa14_link_7", "left_ring_PP"),
    "left_ring_PP": ("iiwa14_link_7", "left_ring_MCP_VL", "left_ring_MP"),
    "left_ring_MP": ("left_ring_PP", "left_ring_DP"),
    "left_ring_DP": ("left_ring_MP",),
    "left_thumb_CMC_VL": ("iiwa14_link_7", "left_thumb_MC"),
    "left_thumb_MC": ("iiwa14_link_7", "left_thumb_CMC_VL", "left_thumb_MCP_VL", "left_thumb_PP"),
    "left_thumb_MCP_VL": ("left_thumb_MC", "left_thumb_PP"),
    "left_thumb_PP": ("left_thumb_MC", "left_thumb_MCP_VL", "left_thumb_DP"),
    "left_thumb_DP": ("left_thumb_PP",),
}

KUKA_SHARPA = RobotSpec(
    name="kuka_sharpa",
    arm_joint_regex="iiwa14_joint_.*",
    hand_joint_regex="left_.*",
    joint_names_canonical=tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + _SHARPA_JOINTS,
    palm_body_name="iiwa14_link_7",
    # Policy was trained against the palm center, not the raw wrist body.
    palm_center_offset=(-0.0, -0.02, 0.16),
    # Merged fingertip bodies land on the DP links in both sims.
    fingertip_body_regex="left_(index|middle|ring|thumb|pinky)_DP",
    fingertip_link_names=(
        "left_index_DP", "left_middle_DP", "left_ring_DP",
        "left_thumb_DP", "left_pinky_DP",
    ),
    # Shift fingertip body origins to the approximate pad centers.
    fingertip_offset=(0.02, 0.002, 0.0),
    arm_stiffness={
        "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
        "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
        "iiwa14_joint_7": 200.0,
    },
    arm_damping={
        "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
        "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
        "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
        "iiwa14_joint_7": 9.147747263670984,
    },
    hand_stiffness={
        "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
        "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
        "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
        "left_index_PIP": 0.9, "left_index_DIP": 0.9,
        "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
        "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
        "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
        "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
        "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
        "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
    },
    hand_damping={
        "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
        "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
        "left_thumb_IP": 0.04190723,
        "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
        "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
        "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
        "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
        "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
        "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
        "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
        "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
        "left_pinky_DIP": 0.0350446,
    },
    hand_armature={
        "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
        "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
        "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
        "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
        "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
        "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
        "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
        "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
        "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
        "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
    },
    # Proven-working default arm pose.
    arm_default_joint_pos={
        "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
        "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
        "iiwa14_joint_7": 1.308,
    },
    self_collision_filter=_SHARPA_SELF_COLLISION_FILTER,
    finger_segment_regex={
        "distal": "left_(thumb|index|middle|ring|pinky)_DP",
        "middle": "left_(thumb|index|middle|ring|pinky)_MP",
        "proximal": "left_(thumb|index|middle|ring|pinky)_PP",
    },
    # The URDF's `left_hand_C_MC` is merged into the palm body by the importer.
    palm_side_link_names=("left_thumb_MC", "left_pinky_MC"),
)
