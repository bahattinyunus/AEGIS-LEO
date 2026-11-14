"""
Unit tests for the AEGIS-LEO High-Speed Swarm Telemetry Bus, Replay Buffer,
and Structured Tamper-Evident Event Logger.
"""

import time
import io
import json
import pytest
from aegis_leo.agents import (
    TelemetryPriority,
    EventSeverity,
    BackpressureStrategy,
    TelemetryFrame,
    StructuredEvent,
    StructuredEventLogger,
    ReplayBuffer,
    TelemetrySubscription,
    SwarmTelemetryBus,
    create_swarm_bus,
    SWARM_AGENTS,
    match_topic,
    SwarmBlackboard,
    SwarmMessage,
)


# ============================================================================
# TelemetryFrame Tests
# ============================================================================

def test_telemetry_frame_creation_and_checksum():
    data = {"altitude_km": 420.5, "velocity_km_s": 7.66, "battery_pct": 98.4}
    frame = TelemetryFrame(
        topic="telemetry.orbit",
        source_agent="AstrodynamicsAgent-04",
        data=data,
        priority=TelemetryPriority.HIGH,
    )

    assert frame.topic == "telemetry.orbit"
    assert frame.source_agent == "AstrodynamicsAgent-04"
    assert frame.priority == TelemetryPriority.HIGH
    assert frame.checksum != ""
    assert frame.verify_checksum() is True

    # Mutate data payload and verify tamper detection
    frame.data["altitude_km"] = 999.9
    assert frame.verify_checksum() is False


def test_telemetry_frame_serialization_roundtrip():
    data = {"kp": 6.8, "solar_flux": 210.0}
    frame = TelemetryFrame(
        topic="telemetry.space_weather",
        source_agent="SolarStormAgent-01",
        data=data,
        priority=TelemetryPriority.CRITICAL,
        metadata={"cycle_id": 42},
    )

    frame_dict = frame.to_dict()
    assert frame_dict["topic"] == "telemetry.space_weather"
    assert frame_dict["priority"] == int(TelemetryPriority.CRITICAL)
    assert frame_dict["priority_name"] == "CRITICAL"

    json_str = frame.to_json()
    assert "telemetry.space_weather" in json_str

    reconstructed = TelemetryFrame.from_dict(frame_dict)
    assert reconstructed.frame_id == frame.frame_id
    assert reconstructed.topic == frame.topic
    assert reconstructed.source_agent == frame.source_agent
    assert reconstructed.priority == frame.priority
    assert reconstructed.data == frame.data
    assert reconstructed.metadata == frame.metadata
    assert reconstructed.verify_checksum() is True


# ============================================================================
# Topic Matching Tests
# ============================================================================

def test_match_topic_patterns():
    # Exact match
    assert match_topic("telemetry.solar", "telemetry.solar") is True
    assert match_topic("telemetry.solar", "telemetry.risk") is False

    # Universal wildcard
    assert match_topic("*", "telemetry.solar") is True
    assert match_topic("#", "telemetry.solar.kp") is True
    assert match_topic("**", "a.b.c.d.e") is True

    # Single-level wildcard
    assert match_topic("telemetry.*", "telemetry.solar") is True
    assert match_topic("telemetry.*", "telemetry.solar.sub") is False

    # Multi-level wildcard
    assert match_topic("telemetry.**", "telemetry.solar.sub") is True
    assert match_topic("telemetry.#", "telemetry.solar.sub.details") is True
    assert match_topic("*.solar.*", "space.solar.flare") is True
    assert match_topic("*.solar.*", "space.other.flare") is False


# ============================================================================
# Structured Event Logger Tests (Tamper-Evident Audit Trail)
# ============================================================================

def test_structured_event_logger_integrity():
    logger = StructuredEventLogger(capacity=100)

    # Log sequential flight events
    evt1 = logger.info("SolarStormAgent-01", "SPACE_WEATHER", "M-class solar flare detected", kp=6.2)
    evt2 = logger.warning("RiskAssessmentAgent-03", "CONJUNCTION", "Conjunction miss distance below threshold", miss_km=0.08)
    evt3 = logger.critical("MissionCommanderAgent-09", "CONSENSUS", "Autonomous avoidance burn approved", delta_v_m_s=3.4)

    assert evt1 is not None and evt2 is not None and evt3 is not None
    assert evt2.prev_hash == evt1.event_hash
    assert evt3.prev_hash == evt2.event_hash

    # Verify chain integrity
    is_valid, err = logger.verify_integrity()
    assert is_valid is True
    assert err is None


def test_structured_event_tamper_detection():
    logger = StructuredEventLogger(capacity=100)
    logger.info("AgentA", "CAT", "Message 1")
    logger.warning("AgentB", "CAT", "Message 2")
    logger.error("AgentC", "CAT", "Message 3")

    # Manually tamper with an event's message in the ledger
    with logger._lock:
        logger._events[1].message = "Forged altered message"

    is_valid, err = logger.verify_integrity()
    assert is_valid is False
    assert "Corrupted event body" in err or "Broken chain" in err


def test_structured_event_queries_and_filters():
    logger = StructuredEventLogger(capacity=100)
    logger.debug("A1", "PHYSICS", "Debug message")
    logger.info("A1", "PHYSICS", "Physics step complete")
    logger.warning("A2", "THERMO", "Atmospheric drag spike")
    logger.error("A3", "RISK", "Collision risk threshold exceeded")
    logger.critical("A9", "COMMAND", "Burn command executed")

    # Query by minimum severity (WARNING and above)
    high_sev = logger.query(min_severity=EventSeverity.WARNING)
    assert len(high_sev) == 3

    # Query by agent
    a1_events = logger.query(agent_id="A1")
    assert len(a1_events) == 2

    # Query by category
    risk_events = logger.query(category="RISK")
    assert len(risk_events) == 1
    assert risk_events[0].message == "Collision risk threshold exceeded"

    # Query text search
    drag_events = logger.query(search_text="spike")
    assert len(drag_events) == 1
    assert drag_events[0].agent_id == "A2"

    # Export to JSONL
    jsonl = logger.export_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 5


# ============================================================================
# Replay Buffer Tests
# ============================================================================

def test_replay_buffer_capacity_and_overflow():
    buffer = ReplayBuffer(capacity=5)

    for i in range(10):
        frame = TelemetryFrame(
            topic="test.metric",
            source_agent=f"Agent-{i%3}",
            data={"val": i},
            seq=i + 1,
        )
        buffer.append(frame)

    assert len(buffer) == 5
    stats = buffer.get_statistics()
    assert stats["current_size"] == 5
    assert stats["capacity"] == 5
    assert stats["total_appended"] == 10
    assert stats["dropped_on_overflow"] == 5


def test_replay_buffer_query_and_filtering():
    buffer = ReplayBuffer(capacity=50)

    now = time.time()
    for i in range(20):
        agent = "SolarStormAgent-01" if i % 2 == 0 else "AstrodynamicsAgent-04"
        topic = "telemetry.solar" if i % 2 == 0 else "telemetry.orbit"
        prio = TelemetryPriority.CRITICAL if i > 15 else TelemetryPriority.NOMINAL

        frame = TelemetryFrame(
            topic=topic,
            source_agent=agent,
            data={"counter": i},
            priority=prio,
            seq=i + 1,
            timestamp=now + (i * 10),
        )
        buffer.append(frame)

    # Filter by agent
    solar_frames = buffer.query(agent="SolarStormAgent-01")
    assert len(solar_frames) == 10
    assert all(f.source_agent == "SolarStormAgent-01" for f in solar_frames)

    # Filter by topic pattern
    orbit_frames = buffer.query(topic_pattern="telemetry.orbit")
    assert len(orbit_frames) == 10

    # Filter by priority
    crit_frames = buffer.query(min_priority=TelemetryPriority.CRITICAL)
    assert len(crit_frames) == 4

    # Filter by time window
    window = buffer.query(start_time=now + 50, end_time=now + 120)
    assert len(window) == 8


def test_replay_buffer_generator_and_snapshot_roundtrip():
    buffer = ReplayBuffer(capacity=20)
    for i in range(10):
        buffer.append(TelemetryFrame(
            topic=f"sensor.{i}",
            source_agent="TestAgent",
            data={"reading": i * 10.0},
            seq=i + 1,
        ))

    # Test generator
    replayed = list(buffer.replay(speed_multiplier=0.0))
    assert len(replayed) == 10
    assert [f.seq for f in replayed] == list(range(1, 11))

    # Test snapshot export & import
    snapshot_json = buffer.export_snapshot()
    assert "frames" in snapshot_json

    new_buffer = ReplayBuffer(capacity=20)
    count = new_buffer.import_snapshot(snapshot_json)
    assert count == 10
    assert len(new_buffer) == 10
    assert new_buffer.query()[0].data == {"reading": 0.0}


# ============================================================================
# Swarm Telemetry Bus Pub/Sub & Backpressure Tests
# ============================================================================

def test_bus_publish_and_subscribe_wildcards():
    bus = create_swarm_bus(replay_capacity=1000)

    received_all = []
    received_solar = []
    received_commands = []

    bus.subscribe("*", callback=lambda f: received_all.append(f))
    bus.subscribe("telemetry.space_*", callback=lambda f: received_solar.append(f))
    bus.subscribe("command.#", callback=lambda f: received_commands.append(f))

    bus.publish("telemetry.space_weather", "SolarStormAgent-01", {"kp": 7.0})
    bus.publish("telemetry.orbit", "AstrodynamicsAgent-04", {"alt": 450.0})
    bus.publish("command.burn.execute", "MissionCommanderAgent-09", {"dv": 2.5})

    assert len(received_all) == 3
    assert len(received_solar) == 1
    assert received_solar[0].data["kp"] == 7.0
    assert len(received_commands) == 1
    assert received_commands[0].data["dv"] == 2.5


def test_bus_queue_subscription_and_backpressure():
    bus = SwarmTelemetryBus(replay_capacity=100)

    # Queue size 2 with DROP_OLDEST policy
    sub = bus.subscribe(
        "telemetry.queue_test",
        queue_size=2,
        backpressure=BackpressureStrategy.DROP_OLDEST,
    )

    bus.publish("telemetry.queue_test", "AgentA", {"msg": "first"})
    bus.publish("telemetry.queue_test", "AgentA", {"msg": "second"})
    bus.publish("telemetry.queue_test", "AgentA", {"msg": "third"})

    # Queue capacity was 2, so "first" should have been dropped, leaving "second" and "third"
    f1 = sub.get(timeout=0.5)
    f2 = sub.get(timeout=0.5)
    assert f1.data["msg"] == "second"
    assert f2.data["msg"] == "third"
    assert sub.poll() is None
    assert sub.dropped_count >= 1

    bus.unsubscribe(sub)


def test_bus_9_agent_swarm_conveniences():
    bus = create_swarm_bus(replay_capacity=500)

    # Subscribe to commander
    commander_packets = []
    bus.subscribe_agent("A9_COMMANDER", callback=lambda f: commander_packets.append(f))

    # Publish telemetry using 9-agent convenience API
    f_solar = bus.publish_agent_telemetry(
        agent_code="A1_SOLAR",
        data={"storm_severity": "G3_STRONG", "f107": 220.0},
        priority=TelemetryPriority.HIGH,
    )
    assert f_solar.source_agent == SWARM_AGENTS["A1_SOLAR"]["name"]
    assert f_solar.topic == SWARM_AGENTS["A1_SOLAR"]["default_topic"]

    f_cmd = bus.publish_agent_telemetry(
        agent_code="A9_COMMANDER",
        data={"directive": "EXECUTE_AUTONOMOUS_MANEUVER", "consensus_ratio": 0.88},
        priority=TelemetryPriority.CRITICAL,
    )
    assert f_cmd.source_agent == SWARM_AGENTS["A9_COMMANDER"]["name"]
    assert len(commander_packets) == 1
    assert commander_packets[0].data["directive"] == "EXECUTE_AUTONOMOUS_MANEUVER"


def test_bus_blackboard_bridge():
    blackboard = SwarmBlackboard()
    bus = create_swarm_bus(replay_capacity=500)

    received_bridge_frames = []
    bus.subscribe("blackboard.**", callback=lambda f: received_bridge_frames.append(f))

    # Connect bridge
    detach = bus.bridge_blackboard(blackboard)

    # Post state to blackboard
    blackboard.post("atmospheric_density", 2.45e-12, "ThermosphereAgent-02")

    # Broadcast message on blackboard
    msg = SwarmMessage(
        sender="RiskAssessmentAgent-03",
        recipient="AstrodynamicsAgent-04",
        message_type="CONJUNCTION_ALERT",
        payload={"tca_hours": 2.1, "miss_distance_km": 0.045},
        priority=3,
    )
    blackboard.broadcast(msg)

    assert len(received_bridge_frames) == 2
    post_frame = [f for f in received_bridge_frames if "state" in f.topic][0]
    broadcast_frame = [f for f in received_bridge_frames if "message" in f.topic][0]

    assert post_frame.data["value"] == 2.45e-12
    assert broadcast_frame.data["miss_distance_km"] == 0.045

    # Detach bridge and ensure no more frames
    detach()
    blackboard.post("another_key", 100, "TestAgent")
    assert len(received_bridge_frames) == 2


def test_bus_benchmark_high_throughput():
    bus = create_swarm_bus(replay_capacity=20000)
    results = bus.benchmark_throughput(num_frames=5000)

    assert results["num_frames"] == 5000
    assert results["elapsed_seconds"] > 0
    assert results["frames_per_second"] > 5000  # High-speed expectation
    metrics = bus.get_bus_metrics()
    assert metrics["total_published"] >= 5000
