import json
import os
import tempfile
import pytest

from aegis_leo.compliance.odar_generator import (
    ODARGenerator,
    ComplianceStatus,
    OrbitRegime,
    DisposalStrategy,
    MissionOverview,
    SpacecraftSpecs,
    OrbitSpecs,
    ComplianceAssessmentInput,
    PassivationAction,
    DebrisFragment,
    RequirementResult,
)
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor


def test_standard_library_compliance_only():
    """Verify that odar_generator relies strictly on Python standard library."""
    import aegis_leo.compliance.odar_generator as odar_mod
    # Verify no numpy or scipy or external modules were imported into odar_mod globals
    assert "numpy" not in odar_mod.__dict__
    assert "np" not in odar_mod.__dict__
    assert "scipy" not in odar_mod.__dict__


def test_create_leo_mission_compliant():
    """Test standard compliant LEO mission generation."""
    generator = ODARGenerator.create_leo_mission(
        mission_name="Starlink-1007",
        norad_id=44713,
        altitude_km=540.0,
        inclination_deg=53.0,
        wet_mass_kg=260.0,
        dry_mass_kg=240.0,
        cross_section_area_m2=2.2,
        operational_pc=1.5e-5,
        has_passivation_valves=True,
    )

    assert generator.is_overall_compliant() is True

    eval_results = generator.evaluate_compliance()
    assert len(eval_results) >= 12

    # Check key Section 4 requirements
    req_map = {r.requirement_id: r for r in eval_results}
    assert "Requirement 4.3-1" in req_map
    assert req_map["Requirement 4.3-1"].status == ComplianceStatus.COMPLIANT

    assert "Requirement 4.4-1" in req_map
    assert req_map["Requirement 4.4-1"].status == ComplianceStatus.COMPLIANT

    assert "Requirement 4.4-2" in req_map
    assert req_map["Requirement 4.4-2"].status == ComplianceStatus.COMPLIANT

    assert "Requirement 4.5-1" in req_map
    assert req_map["Requirement 4.5-1"].status == ComplianceStatus.COMPLIANT

    assert "Requirement 4.6-1" in req_map
    assert req_map["Requirement 4.6-1"].status == ComplianceStatus.COMPLIANT

    assert "Requirement 4.7-1" in req_map
    assert req_map["Requirement 4.7-1"].status == ComplianceStatus.COMPLIANT

    assert "Requirement 4.8-1" in req_map
    assert req_map["Requirement 4.8-1"].status == ComplianceStatus.NOT_APPLICABLE


def test_non_compliant_cases():
    """Verify non-compliant cases trigger NON_COMPLIANT status for specific rules."""
    # Case 1: Excessive collision probability
    gen_high_pc = ODARGenerator.create_leo_mission(
        mission_name="Risky-Sat",
        altitude_km=540.0,
        operational_pc=2.5e-3,  # Exceeds 1e-4
    )
    assert gen_high_pc.is_overall_compliant() is False
    reqs = {r.requirement_id: r for r in gen_high_pc.evaluate_compliance()}
    assert reqs["Requirement 4.5-1"].status == ComplianceStatus.NON_COMPLIANT

    # Case 2: Missing passivation valves
    gen_no_passivation = ODARGenerator.create_leo_mission(
        mission_name="Unpassivated-Sat",
        altitude_km=540.0,
        has_passivation_valves=False,
    )
    assert gen_no_passivation.is_overall_compliant() is False
    reqs_p = {r.requirement_id: r for r in gen_no_passivation.evaluate_compliance()}
    assert reqs_p["Requirement 4.4-2"].status == ComplianceStatus.NON_COMPLIANT

    # Case 3: Excessive orbital lifetime (> 25 years)
    mission = MissionOverview(mission_name="High-Orbit-Sat")
    spacecraft = SpacecraftSpecs(wet_mass_kg=500.0, dry_mass_kg=480.0)
    orbit = OrbitSpecs(perigee_altitude_km=900.0, apogee_altitude_km=900.0)
    assessment = ComplianceAssessmentInput(
        orbital_lifetime_years=85.0,  # Exceeds 25.0 years
        is_controlled_reentry=False,
        disposal_strategy=DisposalStrategy.UNCONTROLLED_NATURAL_DECAY,
    )
    gen_long_life = ODARGenerator(mission, spacecraft, orbit, assessment)
    assert gen_long_life.is_overall_compliant() is False
    reqs_l = {r.requirement_id: r for r in gen_long_life.evaluate_compliance()}
    assert reqs_l["Requirement 4.6-1"].status == ComplianceStatus.NON_COMPLIANT


def test_controlled_reentry_case():
    """Verify controlled ocean disposal (Point Nemo) complies with zero casualty risk."""
    generator = ODARGenerator.create_leo_mission(
        mission_name="Heavy-Spacecraft",
        altitude_km=400.0,
        wet_mass_kg=4500.0,
        dry_mass_kg=4000.0,
        is_controlled_reentry=True,
    )
    assert generator.is_overall_compliant() is True
    reqs = {r.requirement_id: r for r in generator.evaluate_compliance()}
    assert reqs["Requirement 4.6-1"].status == ComplianceStatus.COMPLIANT
    assert reqs["Requirement 4.7-1"].status == ComplianceStatus.COMPLIANT
    assert "Point Nemo" in reqs["Requirement 4.7-1"].assessment_notes


def test_tether_system_evaluation():
    """Verify tether system requirement evaluation when tether is present."""
    mission = MissionOverview(mission_name="Tether-Sat")
    spacecraft = SpacecraftSpecs(
        wet_mass_kg=50.0,
        dry_mass_kg=45.0,
        has_tether=True,
        tether_length_m=5000.0,
    )
    orbit = OrbitSpecs()
    assessment = ComplianceAssessmentInput(tether_collision_risk=2.0e-5)
    gen_tether = ODARGenerator(mission, spacecraft, orbit, assessment)

    reqs = {r.requirement_id: r for r in gen_tether.evaluate_compliance()}
    assert reqs["Requirement 4.8-1"].status == ComplianceStatus.COMPLIANT


def test_json_and_markdown_generation():
    """Test full JSON serialization and Markdown generation."""
    generator = ODARGenerator.create_leo_mission(
        mission_name="AEGIS-DEMO-1",
        norad_id=55123,
        altitude_km=550.0,
    )

    # 1. JSON generation
    json_str = generator.generate_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)

    assert data["document_metadata"]["title"] == "Orbital Debris Assessment Report (ODAR)"
    assert data["document_metadata"]["standard"] == "NASA-STD-8719.14B Section 4"
    assert data["document_metadata"]["overall_status"] == "COMPLIANT"
    assert data["section_1_mission_overview"]["mission_name"] == "AEGIS-DEMO-1"
    assert len(data["requirements_compliance_matrix"]) >= 12
    assert data["compliance_summary"]["is_overall_compliant"] is True

    # 2. Markdown generation
    md_str = generator.generate_markdown()
    assert isinstance(md_str, str)
    assert "# NASA-STD-8719.14B Orbital Debris Assessment Report (ODAR)" in md_str
    assert "AEGIS-DEMO-1" in md_str
    assert "## 9. Requirement Compliance Matrix" in md_str
    assert "Requirement 4.3-1" in md_str
    assert "Requirement 4.4-1" in md_str
    assert "Requirement 4.5-1" in md_str
    assert "Requirement 4.6-1" in md_str
    assert "Requirement 4.7-1" in md_str
    assert "Team AYAZ" in md_str


def test_export_to_files():
    """Test exporting report to markdown and json files."""
    generator = ODARGenerator.create_leo_mission(mission_name="File-Export-Test")
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "report.md")
        json_path = os.path.join(tmpdir, "report.json")

        res = generator.export_to_files(md_path, json_path)
        assert res["status"] == "SUCCESS"
        assert os.path.exists(md_path)
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            j_data = json.load(f)
            assert j_data["section_1_mission_overview"]["mission_name"] == "File-Export-Test"


def test_from_audit_result_integration():
    """Verify interoperability with NASAComplianceAuditor."""
    auditor = NASAComplianceAuditor()
    audit_res = auditor.audit_mission(
        mission_name="Starlink-Integration-Test",
        norad_id=44713,
        altitude_km=548.0,
        inclination_deg=53.0,
        mass_kg=260.0,
        area_m2=2.2,
        operational_collision_prob=1.5e-5,
        has_passivation_valves=True,
    )

    generator = ODARGenerator.from_audit_result(audit_res)
    assert generator.mission.mission_name == "Starlink-Integration-Test"
    assert generator.is_overall_compliant() is True

    json_doc = generator.generate_json_dict()
    assert json_doc["document_metadata"]["overall_status"] == "COMPLIANT"


def test_from_dict_instantiation():
    """Verify dictionary-driven constructor."""
    cfg = {
        "mission": {
            "mission_name": "Dict-Config-Sat",
            "launch_vehicle": "Electron",
        },
        "spacecraft": {
            "wet_mass_kg": 150.0,
            "dry_mass_kg": 140.0,
        },
        "orbit": {
            "perigee_altitude_km": 500.0,
            "apogee_altitude_km": 500.0,
        },
        "assessment": {
            "collision_prob_large_objects": 2.0e-5,
            "orbital_lifetime_years": 4.5,
        },
    }

    generator = ODARGenerator.from_dict(cfg)
    assert generator.mission.mission_name == "Dict-Config-Sat"
    assert generator.mission.launch_vehicle == "Electron"
    assert generator.is_overall_compliant() is True


def test_analytical_utilities():
    """Verify analytical formulas with Python standard library."""
    ke = ODARGenerator.calculate_kinetic_energy(mass_kg=2.0, velocity_m_s=30.0)
    assert ke == 0.5 * 2.0 * (30.0 ** 2)
    assert ke == 900.0

    v_term = ODARGenerator.calculate_terminal_velocity(mass_kg=5.0, cross_section_area_m2=0.1)
    assert v_term > 0.0

    lifetime = ODARGenerator.estimate_natural_decay_lifetime_years(
        altitude_km=550.0,
        mass_kg=260.0,
        cross_section_area_m2=2.2,
    )
    assert 2.0 <= lifetime <= 15.0

    frag = DebrisFragment(
        component_name="Test Bolt",
        material="Titanium",
        dry_mass_kg=0.5,
        surviving_mass_kg=0.5,
        cross_section_area_m2=0.01,
    )
    assert frag.hazardous is True
    assert frag.impact_kinetic_energy_joules > 15.0
