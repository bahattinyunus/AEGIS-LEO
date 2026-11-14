import pytest
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor


def test_nasa_compliance_audit_compliant_case():
    auditor = NASAComplianceAuditor()
    res = auditor.audit_mission(
        mission_name="Starlink-1007",
        norad_id=44713,
        altitude_km=548.0,
        inclination_deg=53.0,
        mass_kg=260.0,
        area_m2=2.2,
        operational_collision_prob=1.5e-5,  # below 1e-4 threshold
        has_passivation_valves=True,
    )

    assert res["overall_status"] == "COMPLIANT"
    report = auditor.generate_markdown_report(res)
    assert "NASA-STD-8719.14" in report
    assert "Starlink-1007" in report


def test_nasa_compliance_audit_non_compliant_pc():
    auditor = NASAComplianceAuditor()
    res = auditor.audit_mission(
        mission_name="Risky-Sat",
        norad_id=99999,
        altitude_km=550.0,
        inclination_deg=53.0,
        mass_kg=260.0,
        area_m2=2.2,
        operational_collision_prob=2.5e-3,  # EXCEEDS 1e-4
        has_passivation_valves=True,
    )

    assert res["overall_status"] == "NON_COMPLIANT"
