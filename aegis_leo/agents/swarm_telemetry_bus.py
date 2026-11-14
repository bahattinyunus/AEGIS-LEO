"""
AEGIS-LEO 9-Agent Swarm High-Speed Telemetry Streaming Bus, Replay Buffer,
and Structured Tamper-Evident Event Logger.

Designed for onboard autonomous satellite operations, providing:
1. High-throughput in-memory publish/subscribe telemetry streaming bus
   with topic routing, wildcard patterns, and backpressure handling.
2. Circular ring replay buffer with multi-index temporal and priority querying,
   snapshot export/import, and variable-speed replay generators.
3. Cryptographically chained structured event logger providing an append-only,
   tamper-evident audit trail for consensus directives and flight anomalies.
4. Seamless integration with the 9-agent AEGIS-LEO swarm architecture
   (A1 Solar, A2 Thermosphere, A3 Risk, A4 Astrodynamics, A5 Edge AI,
    A6 Constellation, A7 Compliance, A8 Disposal, A9 Commander).

Uses exclusively the Python standard library with zero external dependencies.
"""

from typing import Dict, Any, List, Optional, Callable, Generator, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
import collections
import fnmatch
import hashlib
import io
import json
import queue
import threading
import time
import uuid


# ============================================================================
# Enums and Constants
# ============================================================================

class TelemetryPriority(IntEnum):
    """Priority levels for telemetry packets in the flight bus."""
    DEBUG = 0
    LOW = 1
    NOMINAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class EventSeverity(IntEnum):
    """Severity ratings for structured audit events."""
    DEBUG = 10
    INFO = 20
    NOTICE = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60


class BackpressureStrategy(str, Enum):
    """Queue backpressure policies when subscriber queues fill up."""
    DROP_OLDEST = "DROP_OLDEST"
    DROP_NEWEST = "DROP_NEWEST"
    BLOCK = "BLOCK"


# 9-Agent Swarm Canonical Identifiers & Default Topics
SWARM_AGENTS = {
    "A1_SOLAR": {
        "role": "SOLAR_PHYSICS",
        "name": "SolarStormAgent-01",
        "default_topic": "telemetry.space_weather",
    },
    "A2_THERMOSPHERE": {
        "role": "THERMOSPHERE_DYNAMICS",
        "name": "ThermosphereAgent-02",
        "default_topic": "telemetry.atmosphere",
    },
    "A3_RISK": {
        "role": "RISK_ASSESSMENT",
        "name": "RiskAssessmentAgent-03",
        "default_topic": "telemetry.conjunction_risk",
    },
    "A4_ASTRODYNAMICS": {
        "role": "ASTRODYNAMICS",
        "name": "AstrodynamicsAgent-04",
        "default_topic": "telemetry.trajectory_maneuver",
    },
    "A5_EDGE_INFERENCE": {
        "role": "EDGE_AI_INFERENCE",
        "name": "EdgeInferenceAgent-05",
        "default_topic": "telemetry.edge_hardware",
    },
    "A6_CONSTELLATION": {
        "role": "CONSTELLATION_CONSENSUS",
        "name": "ConstellationConsensusAgent-06",
        "default_topic": "telemetry.constellation_slot",
    },
    "A7_COMPLIANCE": {
        "role": "COMPLIANCE_WATCHDOG",
        "name": "ComplianceWatchdogAgent-07",
        "default_topic": "telemetry.compliance_audit",
    },
    "A8_DISPOSAL": {
        "role": "REENTRY_DISPOSAL",
        "name": "ReentryDisposalAgent-08",
        "default_topic": "telemetry.reentry_disposal",
    },
    "A9_COMMANDER": {
        "role": "MISSION_COMMANDER",
        "name": "MissionCommanderAgent-09",
        "default_topic": "telemetry.commander_consensus",
    },
}


def _canonical_json(data: Any) -> str:
    """Serialize dictionary/structure to canonical JSON for deterministic hashing."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)


def match_topic(pattern: str, topic: str) -> bool:
    """
    Match topic against hierarchical pattern.
    Supports:
      - '*' matches single level between dots (e.g., 'telemetry.*' matches 'telemetry.solar'),
        or universally matches everything if pattern == '*'.
      - '**' or '#' matches multi-level (e.g., 'telemetry.**' matches 'telemetry.a.b.c').
      - Glob pattern matching within levels (e.g. 'telemetry.space_*').
    """
    if pattern == topic or pattern == "*" or pattern == "#" or pattern == "**":
        return True

    pat_parts = pattern.split('.')
    top_parts = topic.split('.')

    # Handle multi-level wildcards
    if "**" in pat_parts or "#" in pat_parts:
        wildcard = "**" if "**" in pat_parts else "#"
        wildcard_idx = pat_parts.index(wildcard)
        prefix = pat_parts[:wildcard_idx]
        suffix = pat_parts[wildcard_idx + 1:]

        if len(top_parts) < len(prefix) + len(suffix):
            return False

        # Match prefix
        for p, t in zip(prefix, top_parts[:len(prefix)]):
            if p != "*" and not fnmatch.fnmatch(t, p):
                return False

        # Match suffix
        if suffix:
            for s, t in zip(suffix, top_parts[-len(suffix):]):
                if s != "*" and not fnmatch.fnmatch(t, s):
                    return False

        return True

    # Exact level count comparison
    if len(pat_parts) != len(top_parts):
        return False

    for p, t in zip(pat_parts, top_parts):
        if p == "*":
            continue
        if not fnmatch.fnmatch(t, p):
            return False

    return True

    for p, t in zip(pat_parts, top_parts):
        if p == "*":
            continue
        if not fnmatch.fnmatch(t, p):
            return False

    return True


# ============================================================================
# Telemetry Frame Data Model
# ============================================================================

@dataclass
class TelemetryFrame:
    """
    High-speed telemetry packet emitted by an agent, sensor, or subsystem.
    Contains cryptographic checksum and monotonic sequencing.
    """
    topic: str
    source_agent: str
    data: Dict[str, Any]
    priority: TelemetryPriority = TelemetryPriority.NOMINAL
    seq: int = 0
    timestamp: float = field(default_factory=time.time)
    frame_id: str = field(default_factory=lambda: f"TF-{uuid.uuid4().hex[:12].upper()}")
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum over data payload for tamper detection."""
        payload_bytes = _canonical_json(self.data).encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify that payload data has not been corrupted or modified."""
        return self.checksum == self.compute_checksum()

    @property
    def iso_timestamp(self) -> str:
        """UTC ISO 8601 formatted timestamp string."""
        return datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary representation."""
        return {
            "frame_id": self.frame_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "iso_timestamp": self.iso_timestamp,
            "source_agent": self.source_agent,
            "topic": self.topic,
            "priority": int(self.priority),
            "priority_name": self.priority.name,
            "data": self.data,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    def to_json(self) -> str:
        """Convert frame to JSON string."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TelemetryFrame":
        """Reconstruct TelemetryFrame from dictionary."""
        p_val = d.get("priority", TelemetryPriority.NOMINAL)
        if isinstance(p_val, str):
            priority = TelemetryPriority[p_val]
        else:
            priority = TelemetryPriority(p_val)

        return cls(
            topic=d["topic"],
            source_agent=d["source_agent"],
            data=d.get("data", {}),
            priority=priority,
            seq=d.get("seq", 0),
            timestamp=d.get("timestamp", time.time()),
            frame_id=d.get("frame_id", f"TF-{uuid.uuid4().hex[:12].upper()}"),
            metadata=d.get("metadata", {}),
            checksum=d.get("checksum", ""),
        )


# ============================================================================
# Structured Event Logger (Tamper-Evident Audit Trail)
# ============================================================================

@dataclass
class StructuredEvent:
    """
    Cryptographically chained flight event record.
    Forms an append-only tamper-evident audit ledger.
    """
    event_id: str
    seq: int
    timestamp: float
    agent_id: str
    severity: EventSeverity
    category: str
    message: str
    details: Dict[str, Any]
    prev_hash: str
    event_hash: str

    @property
    def iso_timestamp(self) -> str:
        return datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "iso_timestamp": self.iso_timestamp,
            "agent_id": self.agent_id,
            "severity": int(self.severity),
            "severity_name": self.severity.name,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "prev_hash": self.prev_hash,
            "event_hash": self.event_hash,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def compute_hash(self) -> str:
        """Calculate canonical SHA256 hash of this event."""
        body = {
            "event_id": self.event_id,
            "seq": self.seq,
            "timestamp": round(self.timestamp, 6),
            "agent_id": self.agent_id,
            "severity": int(self.severity),
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verify internal event hash matches computed body hash."""
        return self.event_hash == self.compute_hash()


class StructuredEventLogger:
    """
    High-reliability, space-grade event logger with tamper-evident hash chaining.
    Maintains a ring buffer of structured events with cryptographic verification.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, capacity: int = 10000, min_severity: EventSeverity = EventSeverity.DEBUG):
        self.capacity = capacity
        self.min_severity = min_severity
        self._lock = threading.RLock()
        self._events: collections.deque[StructuredEvent] = collections.deque(maxlen=capacity)
        self._seq = 0
        self._last_hash = self.GENESIS_HASH
        self._sinks: List[Callable[[StructuredEvent], None]] = []

    def add_sink(self, sink: Callable[[StructuredEvent], None]):
        """Register an external event consumer or streamer sink."""
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[StructuredEvent], None]):
        """Remove a previously registered sink."""
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def log(
        self,
        agent_id: str,
        severity: EventSeverity,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[StructuredEvent]:
        """
        Record a structured event into the tamper-evident log.
        Returns the created StructuredEvent, or None if below min_severity.
        """
        if severity < self.min_severity:
            return None

        with self._lock:
            self._seq += 1
            now = time.time()
            evt_id = f"EVT-{self._seq:08d}-{uuid.uuid4().hex[:6].upper()}"
            details_clean = details if details is not None else {}

            # Construct temporary event to compute canonical hash
            event = StructuredEvent(
                event_id=evt_id,
                seq=self._seq,
                timestamp=now,
                agent_id=agent_id,
                severity=severity,
                category=category,
                message=message,
                details=details_clean,
                prev_hash=self._last_hash,
                event_hash="",
            )
            event.event_hash = event.compute_hash()
            self._last_hash = event.event_hash
            self._events.append(event)

            # Dispatch to sinks
            sinks_snapshot = list(self._sinks)

        for sink in sinks_snapshot:
            try:
                sink(event)
            except Exception:
                pass

        return event

    def _parse_log_args(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        a = agent_id or agent or "SYSTEM"
        c = category or cat or "GENERAL"
        m = message or msg or ""
        d = dict(details or {})
        d.update(kwargs)
        return a, c, m, d

    def debug(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.DEBUG, c, m, d)

    def info(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.INFO, c, m, d)

    def notice(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.NOTICE, c, m, d)

    def warning(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.WARNING, c, m, d)

    def error(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.ERROR, c, m, d)

    def critical(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.CRITICAL, c, m, d)

    def fatal(
        self,
        agent: Optional[str] = None,
        cat: Optional[str] = None,
        msg: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[StructuredEvent]:
        a, c, m, d = self._parse_log_args(agent, cat, msg, agent_id=agent_id, category=category, message=message, details=details, **kwargs)
        return self.log(a, EventSeverity.FATAL, c, m, d)

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the complete cryptographic integrity of the audit chain.
        Returns (True, None) if completely valid, or (False, reason) on tampering.
        """
        with self._lock:
            events_list = list(self._events)

        if not events_list:
            return True, None

        prev_hash = events_list[0].prev_hash
        for idx, event in enumerate(events_list):
            # Check linkage
            if event.prev_hash != prev_hash:
                return False, f"Broken chain at seq {event.seq}: prev_hash mismatch"

            # Check individual event hash
            if not event.verify_hash():
                return False, f"Corrupted event body at seq {event.seq}: hash signature invalid"

            prev_hash = event.event_hash

        return True, None

    def query(
        self,
        agent_id: Optional[str] = None,
        min_severity: Optional[EventSeverity] = None,
        category: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        search_text: Optional[str] = None,
        limit: Optional[int] = None,
        reverse: bool = False,
    ) -> List[StructuredEvent]:
        """
        Query audit events using multi-field filtering criteria.
        """
        with self._lock:
            events_snapshot = list(self._events)

        if reverse:
            events_snapshot.reverse()

        results: List[StructuredEvent] = []
        for evt in events_snapshot:
            if agent_id and evt.agent_id != agent_id:
                continue
            if min_severity is not None and evt.severity < min_severity:
                continue
            if category and evt.category != category:
                continue
            if start_time is not None and evt.timestamp < start_time:
                continue
            if end_time is not None and evt.timestamp > end_time:
                continue
            if search_text and search_text.lower() not in evt.message.lower():
                continue

            results.append(evt)
            if limit is not None and len(results) >= limit:
                break

        return results

    def export_jsonl(self) -> str:
        """Export all events in current ring buffer as JSON Lines format."""
        with self._lock:
            events = list(self._events)
        return "\n".join(e.to_json() for e in events)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self):
        with self._lock:
            self._events.clear()
            self._last_hash = self.GENESIS_HASH
            self._seq = 0


# ============================================================================
# Telemetry Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    High-capacity circular ring buffer for storing, querying, and replaying
    telemetry frames across all 9 autonomous agents.
    """

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self._lock = threading.RLock()
        self._frames: collections.deque[TelemetryFrame] = collections.deque(maxlen=capacity)
        self._total_appended = 0
        self._dropped_on_overflow = 0
        self._agent_counts: Dict[str, int] = collections.defaultdict(int)
        self._topic_counts: Dict[str, int] = collections.defaultdict(int)

    def append(self, frame: TelemetryFrame) -> None:
        """Append a frame to the buffer. Handles ring eviction tracking."""
        with self._lock:
            if len(self._frames) >= self.capacity:
                self._dropped_on_overflow += 1
            self._frames.append(frame)
            self._total_appended += 1
            self._agent_counts[frame.source_agent] += 1
            self._topic_counts[frame.topic] += 1

    def append_batch(self, frames: List[TelemetryFrame]) -> None:
        """Append multiple frames efficiently."""
        with self._lock:
            for f in frames:
                self.append(f)

    def query(
        self,
        agent: Optional[str] = None,
        topic: Optional[str] = None,
        topic_pattern: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        min_priority: Optional[TelemetryPriority] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        reverse: bool = False,
    ) -> List[TelemetryFrame]:
        """
        Query frames from buffer matching specific filter conditions.
        """
        with self._lock:
            frame_list = list(self._frames)

        if reverse:
            frame_list.reverse()

        results: List[TelemetryFrame] = []
        skipped = 0

        for frame in frame_list:
            if agent and frame.source_agent != agent:
                continue
            if topic and frame.topic != topic:
                continue
            if topic_pattern and not match_topic(topic_pattern, frame.topic):
                continue
            if start_time is not None and frame.timestamp < start_time:
                continue
            if end_time is not None and frame.timestamp > end_time:
                continue
            if min_priority is not None and frame.priority < min_priority:
                continue

            if skipped < offset:
                skipped += 1
                continue

            results.append(frame)
            if limit is not None and len(results) >= limit:
                break

        return results

    def replay(
        self,
        filter_fn: Optional[Callable[[TelemetryFrame], bool]] = None,
        speed_multiplier: float = 0.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Generator[TelemetryFrame, None, None]:
        """
        Generator yielding frames sequentially for simulation or testing.
        If speed_multiplier > 0.0, respects relative real-time deltas scaled by multiplier.
        If speed_multiplier == 0.0, yields instantly without sleep.
        """
        with self._lock:
            frames = list(self._frames)

        last_ts: Optional[float] = None

        for frame in frames:
            if start_time is not None and frame.timestamp < start_time:
                continue
            if end_time is not None and frame.timestamp > end_time:
                break
            if filter_fn is not None and not filter_fn(frame):
                continue

            # Sleep proportional to time delta if pacing requested
            if speed_multiplier > 0.0 and last_ts is not None:
                delta = frame.timestamp - last_ts
                if delta > 0:
                    sleep_sec = delta / speed_multiplier
                    # Bound max sleep to prevent freeze on large pauses
                    sleep_sec = min(sleep_sec, 2.0)
                    time.sleep(sleep_sec)

            last_ts = frame.timestamp
            yield frame

    def export_snapshot(self, path_or_file: Optional[Union[str, io.TextIOBase]] = None) -> str:
        """
        Export complete buffer contents as JSON.
        Can write to a filepath, a file-like stream, or return JSON string if None.
        """
        with self._lock:
            data = {
                "snapshot_time": datetime.now(timezone.utc).isoformat(),
                "capacity": self.capacity,
                "total_appended": self._total_appended,
                "dropped_on_overflow": self._dropped_on_overflow,
                "frame_count": len(self._frames),
                "frames": [f.to_dict() for f in self._frames],
            }

        json_str = json.dumps(data, indent=2)

        if isinstance(path_or_file, str):
            with open(path_or_file, "w", encoding="utf-8") as f:
                f.write(json_str)
        elif path_or_file is not None:
            path_or_file.write(json_str)

        return json_str

    def import_snapshot(self, snapshot_source: Union[str, io.TextIOBase]) -> int:
        """
        Import frames from JSON string or file. Returns count of frames loaded.
        """
        if hasattr(snapshot_source, "read"):
            content = snapshot_source.read()
        elif isinstance(snapshot_source, str):
            if snapshot_source.strip().startswith("{"):
                content = snapshot_source
            else:
                with open(snapshot_source, "r", encoding="utf-8") as f:
                    content = f.read()
        else:
            raise ValueError("Unsupported snapshot source")

        data = json.loads(content)
        frames_data = data.get("frames", [])

        loaded_count = 0
        with self._lock:
            for item in frames_data:
                frame = TelemetryFrame.from_dict(item)
                self.append(frame)
                loaded_count += 1

        return loaded_count

    def get_statistics(self) -> Dict[str, Any]:
        """Compute buffer occupancy, agent breakdown, and health metrics."""
        with self._lock:
            count = len(self._frames)
            oldest_ts = self._frames[0].timestamp if count > 0 else None
            newest_ts = self._frames[-1].timestamp if count > 0 else None
            oldest_iso = (
                datetime.fromtimestamp(oldest_ts, timezone.utc).isoformat()
                if oldest_ts else None
            )
            newest_iso = (
                datetime.fromtimestamp(newest_ts, timezone.utc).isoformat()
                if newest_ts else None
            )

            return {
                "capacity": self.capacity,
                "current_size": count,
                "occupancy_pct": round((count / self.capacity) * 100.0, 2) if self.capacity > 0 else 0.0,
                "total_appended": self._total_appended,
                "dropped_on_overflow": self._dropped_on_overflow,
                "oldest_timestamp": oldest_iso,
                "newest_timestamp": newest_iso,
                "duration_seconds": (newest_ts - oldest_ts) if (newest_ts and oldest_ts) else 0.0,
                "agent_distribution": dict(self._agent_counts),
                "topic_distribution": dict(self._topic_counts),
            }

    def clear(self):
        """Purge all buffered telemetry frames."""
        with self._lock:
            self._frames.clear()
            self._total_appended = 0
            self._dropped_on_overflow = 0
            self._agent_counts.clear()
            self._topic_counts.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


# ============================================================================
# Subscriber Subscription Model
# ============================================================================

class TelemetrySubscription:
    """Represents an active subscription to one or more bus topics."""

    def __init__(
        self,
        sub_id: str,
        topic_pattern: str,
        callback: Optional[Callable[[TelemetryFrame], None]] = None,
        queue_size: int = 1000,
        min_priority: TelemetryPriority = TelemetryPriority.DEBUG,
        agent_filter: Optional[str] = None,
        backpressure: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
    ):
        self.sub_id = sub_id
        self.topic_pattern = topic_pattern
        self.callback = callback
        self.min_priority = min_priority
        self.agent_filter = agent_filter
        self.backpressure = backpressure
        self.queue: Optional[queue.Queue] = queue.Queue(maxsize=queue_size) if queue_size > 0 else None
        self.delivered_count = 0
        self.dropped_count = 0
        self.created_at = time.time()

    def matches(self, frame: TelemetryFrame) -> bool:
        """Evaluate if frame satisfies topic pattern, priority, and agent filters."""
        if frame.priority < self.min_priority:
            return False
        if self.agent_filter and frame.source_agent != self.agent_filter:
            return False
        return match_topic(self.topic_pattern, frame.topic)

    def dispatch(self, frame: TelemetryFrame):
        """Dispatch matching frame to callback and/or internal queue."""
        # 1. Execute direct callback if registered
        if self.callback is not None:
            try:
                self.callback(frame)
            except Exception:
                pass

        # 2. Push to queue if enabled
        if self.queue is not None:
            try:
                self.queue.put_nowait(frame)
                self.delivered_count += 1
            except queue.Full:
                self.dropped_count += 1
                if self.backpressure == BackpressureStrategy.DROP_OLDEST:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(frame)
                        self.delivered_count += 1
                    except (queue.Empty, queue.Full):
                        pass
                elif self.backpressure == BackpressureStrategy.DROP_NEWEST:
                    pass  # Frame dropped
                elif self.backpressure == BackpressureStrategy.BLOCK:
                    try:
                        self.queue.put(frame, timeout=0.1)
                        self.delivered_count += 1
                    except queue.Full:
                        pass
        else:
            self.delivered_count += 1

    def get(self, block: bool = True, timeout: Optional[float] = None) -> TelemetryFrame:
        """Pull frame from subscriber queue."""
        if self.queue is None:
            raise ValueError("Queue disabled for this subscription (callback-only).")
        return self.queue.get(block=block, timeout=timeout)

    def poll(self) -> Optional[TelemetryFrame]:
        """Non-blocking poll for available frame."""
        if self.queue is None:
            return None
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


# ============================================================================
# Swarm Telemetry Streaming Bus
# ============================================================================

class SwarmTelemetryBus:
    """
    High-speed, asynchronous telemetry streaming bus for the 9-agent autonomous swarm.
    Features:
      - Pub/Sub architecture with wildcard pattern matching ('*', '**', '#').
      - Built-in circular ReplayBuffer for flight playback and forensic reconstruction.
      - Built-in StructuredEventLogger with tamper-evident audit chaining.
      - Channels and role bindings for all 9 specialist agents.
      - Lock-guarded thread safety for multi-threaded flight software.
      - Interoperability bridge with AEGIS-LEO SwarmBlackboard.
    """

    def __init__(
        self,
        replay_capacity: int = 50000,
        event_capacity: int = 10000,
        bus_name: str = "AEGIS-LEO-BUS-01",
    ):
        self.bus_name = bus_name
        self._lock = threading.RLock()
        self._seq = 0
        self._subscriptions: Dict[str, TelemetrySubscription] = {}
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.event_logger = StructuredEventLogger(capacity=event_capacity)

        # Performance metrics
        self._published_count = 0
        self._dispatched_count = 0
        self._bytes_streamed = 0
        self._start_time = time.time()

        # Log bus initialization
        self.event_logger.info(
            agent_id="SYSTEM",
            cat="BUS_LIFECYCLE",
            msg=f"SwarmTelemetryBus '{self.bus_name}' initialized with replay_capacity={replay_capacity}",
            bus_name=self.bus_name,
        )

    # ------------------------------------------------------------------------
    # Publishing API
    # ------------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        source_agent: str,
        data: Dict[str, Any],
        priority: TelemetryPriority = TelemetryPriority.NOMINAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryFrame:
        """
        Publish a telemetry packet onto the bus.
        Automatically sequences, timestamps, computes checksum, stores in replay buffer,
        and dispatches to all matching subscribers.
        """
        with self._lock:
            self._seq += 1
            current_seq = self._seq

        frame = TelemetryFrame(
            topic=topic,
            source_agent=source_agent,
            data=data,
            priority=priority,
            seq=current_seq,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # 1. Record in replay buffer
        self.replay_buffer.append(frame)

        # 2. Estimate payload bytes for metrics
        try:
            payload_len = len(_canonical_json(data))
        except Exception:
            payload_len = 64

        with self._lock:
            self._published_count += 1
            self._bytes_streamed += payload_len
            subs_snapshot = list(self._subscriptions.values())

        # 3. Dispatch to matching subscribers
        dispatched = 0
        for sub in subs_snapshot:
            if sub.matches(frame):
                sub.dispatch(frame)
                dispatched += 1

        with self._lock:
            self._dispatched_count += dispatched

        return frame

    def publish_frame(self, frame: TelemetryFrame) -> TelemetryFrame:
        """Publish a pre-constructed TelemetryFrame."""
        with self._lock:
            if frame.seq <= 0:
                self._seq += 1
                frame.seq = self._seq
            self._published_count += 1
            subs_snapshot = list(self._subscriptions.values())

        self.replay_buffer.append(frame)

        dispatched = 0
        for sub in subs_snapshot:
            if sub.matches(frame):
                sub.dispatch(frame)
                dispatched += 1

        with self._lock:
            self._dispatched_count += dispatched

        return frame

    # ------------------------------------------------------------------------
    # Subscription API
    # ------------------------------------------------------------------------

    def subscribe(
        self,
        topic_pattern: str,
        callback: Optional[Callable[[TelemetryFrame], None]] = None,
        queue_size: int = 1000,
        min_priority: TelemetryPriority = TelemetryPriority.DEBUG,
        agent_filter: Optional[str] = None,
        backpressure: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
        sub_id: Optional[str] = None,
    ) -> TelemetrySubscription:
        """
        Subscribe to a topic pattern (supports '*', '**', '#', globs).
        Returns a TelemetrySubscription object.
        """
        if sub_id is None:
            sub_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"

        sub = TelemetrySubscription(
            sub_id=sub_id,
            topic_pattern=topic_pattern,
            callback=callback,
            queue_size=queue_size,
            min_priority=min_priority,
            agent_filter=agent_filter,
            backpressure=backpressure,
        )

        with self._lock:
            self._subscriptions[sub_id] = sub

        return sub

    def unsubscribe(self, sub_id_or_sub: Union[str, TelemetrySubscription]) -> bool:
        """Unsubscribe and release subscriber resources."""
        sub_id = (
            sub_id_or_sub.sub_id
            if isinstance(sub_id_or_sub, TelemetrySubscription)
            else sub_id_or_sub
        )
        with self._lock:
            return self._subscriptions.pop(sub_id, None) is not None

    # ------------------------------------------------------------------------
    # 9-Agent Swarm Helper Shortcuts
    # ------------------------------------------------------------------------

    def publish_agent_telemetry(
        self,
        agent_code: str,
        data: Dict[str, Any],
        priority: TelemetryPriority = TelemetryPriority.NOMINAL,
        subtopic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryFrame:
        """
        Convenience publisher for any of the 9 canonical agents (A1_SOLAR to A9_COMMANDER).
        Auto-resolves agent name and default topic.
        """
        agent_meta = SWARM_AGENTS.get(agent_code.upper())
        if agent_meta:
            source_agent = agent_meta["name"]
            topic = agent_meta["default_topic"]
        else:
            source_agent = agent_code
            topic = f"telemetry.{agent_code.lower()}"

        if subtopic:
            topic = f"{topic}.{subtopic}"

        meta = metadata or {}
        meta["agent_code"] = agent_code
        if agent_meta:
            meta["agent_role"] = agent_meta["role"]

        return self.publish(
            topic=topic,
            source_agent=source_agent,
            data=data,
            priority=priority,
            metadata=meta,
        )

    def subscribe_agent(
        self,
        agent_code: str,
        callback: Optional[Callable[[TelemetryFrame], None]] = None,
        queue_size: int = 1000,
    ) -> TelemetrySubscription:
        """Subscribe to all telemetry emitted by a specific swarm agent."""
        agent_meta = SWARM_AGENTS.get(agent_code.upper())
        agent_name = agent_meta["name"] if agent_meta else agent_code
        return self.subscribe(
            topic_pattern="**",
            callback=callback,
            queue_size=queue_size,
            agent_filter=agent_name,
        )

    # ------------------------------------------------------------------------
    # Interoperability Bridge with SwarmBlackboard
    # ------------------------------------------------------------------------

    def bridge_blackboard(self, blackboard: Any) -> Callable[[], None]:
        """
        Bridges this telemetry bus with an AEGIS-LEO SwarmBlackboard instance.
        Monitors SwarmMessage broadcasts and state postings, streaming them to the bus.
        Returns a detach/cleanup callable.
        """
        original_post = getattr(blackboard, "post", None)
        original_broadcast = getattr(blackboard, "broadcast", None)

        if original_post is None or original_broadcast is None:
            raise TypeError("Target object does not conform to SwarmBlackboard interface")

        bus_ref = self

        def hooked_post(key: str, value: Any, agent_name: str):
            original_post(key, value, agent_name)
            try:
                topic = f"blackboard.state.{key}"
                bus_ref.publish(
                    topic=topic,
                    source_agent=agent_name,
                    data={"key": key, "value": value},
                    priority=TelemetryPriority.NOMINAL,
                    metadata={"source": "blackboard_post"},
                )
            except Exception:
                pass

        def hooked_broadcast(message: Any):
            original_broadcast(message)
            try:
                topic = f"blackboard.message.{getattr(message, 'message_type', 'generic')}"
                source = getattr(message, "sender", "UNKNOWN")
                payload = getattr(message, "payload", {})
                priority_int = getattr(message, "priority", 1)
                prio_enum = TelemetryPriority(min(max(priority_int, 0), 5))

                bus_ref.publish(
                    topic=topic,
                    source_agent=source,
                    data=payload,
                    priority=prio_enum,
                    metadata={
                        "message_id": getattr(message, "message_id", ""),
                        "recipient": getattr(message, "recipient", ""),
                        "source": "blackboard_broadcast",
                    },
                )
            except Exception:
                pass

        blackboard.post = hooked_post
        blackboard.broadcast = hooked_broadcast

        def detach():
            blackboard.post = original_post
            blackboard.broadcast = original_broadcast

        return detach

    # ------------------------------------------------------------------------
    # Diagnostics, Telemetry Metrics, & Benchmark
    # ------------------------------------------------------------------------

    def get_bus_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive real-time telemetry throughput and status metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            rate = self._published_count / uptime if uptime > 0 else 0.0
            byte_rate_kb = (self._bytes_streamed / 1024.0) / uptime if uptime > 0 else 0.0

            sub_stats = {}
            for sid, s in self._subscriptions.items():
                sub_stats[sid] = {
                    "pattern": s.topic_pattern,
                    "agent_filter": s.agent_filter,
                    "delivered": s.delivered_count,
                    "dropped": s.dropped_count,
                    "queue_size": s.queue.qsize() if s.queue else None,
                }

            return {
                "bus_name": self.bus_name,
                "uptime_seconds": round(uptime, 3),
                "total_published": self._published_count,
                "total_dispatched": self._dispatched_count,
                "bytes_streamed": self._bytes_streamed,
                "rate_frames_per_sec": round(rate, 2),
                "throughput_kb_per_sec": round(byte_rate_kb, 2),
                "active_subscriptions": len(self._subscriptions),
                "subscriptions": sub_stats,
                "replay_buffer": self.replay_buffer.get_statistics(),
                "event_logger_count": self.event_logger.count(),
            }

    def benchmark_throughput(self, num_frames: int = 10000) -> Dict[str, Any]:
        """
        Runs an in-memory stress benchmark demonstrating streaming throughput.
        """
        test_sub = self.subscribe("benchmark.*", queue_size=num_frames + 100)

        t0 = time.perf_counter()
        for i in range(num_frames):
            self.publish(
                topic="benchmark.sensor",
                source_agent="BenchmarkRunner",
                data={"index": i, "val": i * 1.5, "status": "NOMINAL"},
                priority=TelemetryPriority.NOMINAL,
            )
        elapsed = time.perf_counter() - t0

        self.unsubscribe(test_sub)

        fps = num_frames / elapsed if elapsed > 0 else 0.0
        return {
            "num_frames": num_frames,
            "elapsed_seconds": round(elapsed, 4),
            "frames_per_second": round(fps, 1),
            "latency_microseconds_per_frame": round((elapsed / num_frames) * 1e6, 2) if num_frames > 0 else 0,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_swarm_bus(
    replay_capacity: int = 50000,
    event_capacity: int = 10000,
    bus_name: str = "AEGIS-LEO-BUS",
) -> SwarmTelemetryBus:
    """Instantiate and configure an AEGIS-LEO swarm telemetry bus."""
    return SwarmTelemetryBus(
        replay_capacity=replay_capacity,
        event_capacity=event_capacity,
        bus_name=bus_name,
    )
