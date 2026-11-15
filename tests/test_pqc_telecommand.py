"""
Unit tests for Post-Quantum Cryptographic (ML-KEM Lattice) & CCSDS 355.0-B-2 SDLS
Telecommand Authentication Engine.
"""

import numpy as np

from aegis_leo.security.pqc_telecommand import (
    CCSDSSDLSGuard,
    LATTICE_Q,
    mlkem_decapsulate,
    mlkem_encapsulate,
    mlkem_keygen,
    poly_mul_negacyclic,
    run_pqc_telecommand_audit,
)


def test_negacyclic_polynomial_ring_multiplication():
    # In Z_q[X]/(X^4 + 1), X^3 * X = X^4 = -1 = q - 1
    a = np.array([0, 0, 0, 1], dtype=np.int64)
    b = np.array([0, 1, 0, 0], dtype=np.int64)
    prod = poly_mul_negacyclic(a, b, q=LATTICE_Q)
    assert prod[0] == LATTICE_Q - 1
    assert np.all(prod[1:] == 0)


def test_mlkem_key_encapsulation_exact_recovery():
    kp = mlkem_keygen(seed=1337)
    ct, msg_bits, key_enc = mlkem_encapsulate(kp, seed=9001)
    rec_bits, key_dec = mlkem_decapsulate(kp, ct)
    assert np.array_equal(msg_bits, rec_bits)
    assert key_enc == key_dec
    assert len(key_enc) == 32


def test_ccsds_sdls_guard_merkle_and_anti_replay():
    guard = CCSDSSDLSGuard(session_key=b"0123456789abcdef0123456789abcdef", valid_spi=101)
    f1 = guard.sign_telecommand("SAT-01", "BURN_1", {"dv": 0.25})
    ok1, reason1 = guard.verify_and_commit(f1)
    assert ok1 is True
    assert reason1 == "AUTHENTICATED_PQC_SDLS"

    # Replay must fail
    ok_rep, reason_rep = guard.verify_and_commit(f1)
    assert ok_rep is False
    assert reason_rep == "ANTI_REPLAY_SEQUENCE_REJECTED"

    # Next valid command succeeds and updates Merkle head
    f2 = guard.sign_telecommand("SAT-01", "BURN_2", {"dv": 0.40})
    ok2, _ = guard.verify_and_commit(f2)
    assert ok2 is True
    assert guard.highest_seq == 2


def test_run_pqc_telecommand_audit_end_to_end():
    rep = run_pqc_telecommand_audit(spacecraft_id="STARLINK_LEO", delta_v_m_s=0.35)
    assert rep.kem_bit_agreement_pct == 100.0
    assert rep.authentic_command_verified is True
    assert rep.replay_attack_blocked is True
    assert rep.tamper_attack_blocked is True
    assert rep.rogue_spi_blocked is True
    assert len(rep.intrusion_test_ledger) == 4
