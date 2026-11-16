"""
AEGIS-LEO Analytics Package
Comprehensive mission analytics, trend analysis, and performance reporting.
"""

from aegis_leo.analytics.mission_analytics import (
    MissionAnalyticsEngine,
    MissionHealthSnapshot,
    SubsystemTrendReport,
    analyze_mission_health,
)
from aegis_leo.analytics.debris_tracker import (
    DebrisTrackerEngine,
    TrackedObject,
    DebrisFleetRiskMatrix,
    build_debris_risk_matrix,
)

__all__ = [
    "MissionAnalyticsEngine",
    "MissionHealthSnapshot",
    "SubsystemTrendReport",
    "analyze_mission_health",
    "DebrisTrackerEngine",
    "TrackedObject",
    "DebrisFleetRiskMatrix",
    "build_debris_risk_matrix",
]
