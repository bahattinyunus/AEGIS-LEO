"""
AEGIS-LEO Active Debris Removal (ADR) Module
============================================
Spacecraft capture dynamics, 6-DOF robotic manipulator kinematics,
harpoon tethered momentum exchange, and post-capture detumbling
of non-cooperative spinning upper stages.
"""

from aegis_leo.adr.debris_capture import (
    DHParameter,
    ForwardKinematicsResult,
    InverseKinematicsResult,
    RoboticArm6DOF,
    HarpoonParameters,
    TetherParameters,
    PenetrationResult,
    TetherState,
    HarpoonTetherSystem,
    SpacecraftMassProperties,
    ReactionWheelArray,
    MagnetorquerArray,
    EarthMagneticFieldDipole,
    CombinedBodyDynamics,
    DetumbleController,
    PostCaptureMomentumManager,
)

__all__ = [
    "DHParameter",
    "ForwardKinematicsResult",
    "InverseKinematicsResult",
    "RoboticArm6DOF",
    "HarpoonParameters",
    "TetherParameters",
    "PenetrationResult",
    "TetherState",
    "HarpoonTetherSystem",
    "SpacecraftMassProperties",
    "ReactionWheelArray",
    "MagnetorquerArray",
    "EarthMagneticFieldDipole",
    "CombinedBodyDynamics",
    "DetumbleController",
    "PostCaptureMomentumManager",
]
