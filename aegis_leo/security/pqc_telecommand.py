"""
Post-Quantum Cryptographic (PQC) Space Data Link Security (CCSDS 355.0-B-2 SDLS)
================================================================================

Protects autonomous orbital maneuver directives and inter-satellite links (ISL)
against quantum-era eavesdropping, command injection, and replay attacks:

  1. Lattice Module-LWE Key Encapsulation Mechanism (ML-KEM / Kyber-inspired)
     over the cyclotomic polynomial ring R_q = Z_q[X] / (X^n + 1), q = 3329:
       - KeyGen:  t = A * s + e  (mod q)
       - Encaps:  u = A^T * r + e_1,  v = t^T * r + e_2 + round(q/2) * m
       - Decaps:  m' = round( (2/q) * (v - s^T * u) ) mod 2
  2. CCSDS 355.0-B-2 Space Data Link Security (SDLS) Telecommand Frame:
       - Security Parameter Index (SPI)
       - Monotonic Anti-Replay Sliding Window Sequence Counter
       - SHA3-256 / HMAC-SHA3-256 Frame Authentication Tag
       - Tamper-Evident Merkle Command Ledger
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LATTICE_Q = 3329
LATTICE_N = 64  # Degree of negacyclic polynomial ring Z_q[X]/(X^64 + 1) for fast onboard execution
LATTICE_K = 2   # Module rank


def poly_mul_negacyclic(a: np.ndarray, b: np.ndarray, q: int = LATTICE_Q) -> np.ndarray:
    """
    Multiply two polynomials in R_q = Z_q[X] / (X^n + 1).
    Coefficients of X^k with k >= n wrap around with a minus sign (X^n = -1).
    """
    n = len(a)
    full = np.convolve(a.astype(np.int64), b.astype(np.int64))
    res = full[:n].copy()
    if len(full) > n:
        res[: len(full) - n] -= full[n:]
    return np.mod(res, q)


def sample_cbd_poly(rng: np.random.Generator, n: int = LATTICE_N, eta: int = 2) -> np.ndarray:
    """Sample small-norm polynomial from Centered Binomial Distribution CBD_eta."""
    a = rng.integers(0, 2, size=(n, eta))
    b = rng.integers(0, 2, size=(n, eta))
    return (np.sum(a, axis=1) - np.sum(b, axis=1)).astype(np.int64)


@dataclass
class MLKEMKeyPair:
    matrix_a: np.ndarray   # Shape (k, k, n)
    public_t: np.ndarray   # Shape (k, n)
    secret_s: np.ndarray   # Shape (k, n)


@dataclass
class MLKEMCiphertext:
    u_vec: np.ndarray      # Shape (k, n)
    v_poly: np.ndarray     # Shape (n,)


def mlkem_keygen(seed: int = 2026) -> MLKEMKeyPair:
    """Generate a Module-LWE public/private keypair over R_q^k."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, LATTICE_Q, size=(LATTICE_K, LATTICE_K, LATTICE_N), dtype=np.int64)
    s = np.array([sample_cbd_poly(rng) for _ in range(LATTICE_K)], dtype=np.int64)
    e = np.array([sample_cbd_poly(rng) for _ in range(LATTICE_K)], dtype=np.int64)

    t = np.zeros((LATTICE_K, LATTICE_N), dtype=np.int64)
    for i in range(LATTICE_K):
        acc = np.zeros(LATTICE_N, dtype=np.int64)
        for j in range(LATTICE_K):
            acc = np.mod(acc + poly_mul_negacyclic(A[i, j], s[j]), LATTICE_Q)
        t[i] = np.mod(acc + e[i], LATTICE_Q)

    return MLKEMKeyPair(matrix_a=A, public_t=t, secret_s=s)


def mlkem_encapsulate(
    pk: MLKEMKeyPair,
    seed: int = 4096,
) -> Tuple[MLKEMCiphertext, np.ndarray, bytes]:
    """
    Encapsulate a 64-bit random lattice message polynomial and derive a
    256-bit SHA3-256 shared session key.
    """
    rng = np.random.default_rng(seed)
    msg_bits = rng.integers(0, 2, size=LATTICE_N, dtype=np.int64)
    r = np.array([sample_cbd_poly(rng) for _ in range(LATTICE_K)], dtype=np.int64)
    e1 = np.array([sample_cbd_poly(rng) for _ in range(LATTICE_K)], dtype=np.int64)
    e2 = sample_cbd_poly(rng)

    u = np.zeros((LATTICE_K, LATTICE_N), dtype=np.int64)
    for i in range(LATTICE_K):
        acc = np.zeros(LATTICE_N, dtype=np.int64)
        for j in range(LATTICE_K):
            # Transpose of A: A[j, i]
            acc = np.mod(acc + poly_mul_negacyclic(pk.matrix_a[j, i], r[j]), LATTICE_Q)
        u[i] = np.mod(acc + e1[i], LATTICE_Q)

    v = np.zeros(LATTICE_N, dtype=np.int64)
    for j in range(LATTICE_K):
        v = np.mod(v + poly_mul_negacyclic(pk.public_t[j], r[j]), LATTICE_Q)

    half_q = (LATTICE_Q + 1) // 2
    v = np.mod(v + e2 + half_q * msg_bits, LATTICE_Q)

    shared_key = hashlib.sha3_256(msg_bits.tobytes()).digest()
    return MLKEMCiphertext(u_vec=u, v_poly=v), msg_bits, shared_key


def mlkem_decapsulate(sk: MLKEMKeyPair, ct: MLKEMCiphertext) -> Tuple[np.ndarray, bytes]:
    """
    Decapsulate Module-LWE ciphertext using secret polynomial vector s
    to recover the exact message bits and 256-bit SHA3-256 session key.
    """
    s_dot_u = np.zeros(LATTICE_N, dtype=np.int64)
    for j in range(LATTICE_K):
        s_dot_u = np.mod(s_dot_u + poly_mul_negacyclic(sk.secret_s[j], ct.u_vec[j]), LATTICE_Q)

    diff = np.mod(ct.v_poly - s_dot_u, LATTICE_Q)
    # Closest to q/2 -> 1, closest to 0 or q -> 0
    lower = LATTICE_Q // 4
    upper = (3 * LATTICE_Q) // 4
    recovered_bits = ((diff >= lower) & (diff <= upper)).astype(np.int64)
    shared_key = hashlib.sha3_256(recovered_bits.tobytes()).digest()
    return recovered_bits, shared_key


@dataclass
class SDLSCommandFrame:
    """CCSDS 355.0-B-2 Space Data Link Security (SDLS) Protected Telecommand Frame."""
    spacecraft_id: str
    spi: int
    sequence_number: int
    command_op: str
    payload: Dict[str, Any]
    prev_merkle_hash: str
    auth_tag_hex: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CCSDSSDLSGuard:
    """
    Zero-Trust CCSDS 355.0-B-2 SDLS Telecommand Authenticator with anti-replay
    sequence enforcement and Merkle command chain verification.
    """

    def __init__(self, session_key: bytes, valid_spi: int = 101) -> None:
        self._session_key = session_key
        self.valid_spi = valid_spi
        self.highest_seq: int = 0
        self.merkle_head: str = hashlib.sha3_256(b"AEGIS-LEO-GENESIS-BLOCK").hexdigest()
        self.audit_log: List[Dict[str, Any]] = []

    def _compute_mac(
        self,
        spacecraft_id: str,
        spi: int,
        seq: int,
        command_op: str,
        payload: Dict[str, Any],
        prev_hash: str,
    ) -> str:
        canonical = json.dumps(
            {
                "sc_id": spacecraft_id,
                "spi": spi,
                "seq": seq,
                "op": command_op,
                "payload": payload,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hmac.new(self._session_key, canonical, hashlib.sha3_256).hexdigest()

    def sign_telecommand(
        self,
        spacecraft_id: str,
        command_op: str,
        payload: Dict[str, Any],
        sequence_number: Optional[int] = None,
    ) -> SDLSCommandFrame:
        seq = (self.highest_seq + 1) if sequence_number is None else sequence_number
        mac_hex = self._compute_mac(
            spacecraft_id=spacecraft_id,
            spi=self.valid_spi,
            seq=seq,
            command_op=command_op,
            payload=payload,
            prev_hash=self.merkle_head,
        )
        return SDLSCommandFrame(
            spacecraft_id=spacecraft_id,
            spi=self.valid_spi,
            sequence_number=seq,
            command_op=command_op,
            payload=payload,
            prev_merkle_hash=self.merkle_head,
            auth_tag_hex=mac_hex,
        )

    def verify_and_commit(self, frame: SDLSCommandFrame) -> Tuple[bool, str]:
        if frame.spi != self.valid_spi:
            self.audit_log.append({"seq": frame.sequence_number, "status": "REJECTED", "reason": "INVALID_SDLS_SPI"})
            return False, "INVALID_SDLS_SPI"

        if frame.sequence_number <= self.highest_seq:
            self.audit_log.append(
                {"seq": frame.sequence_number, "status": "REJECTED", "reason": "ANTI_REPLAY_SEQUENCE_REJECTED"}
            )
            return False, "ANTI_REPLAY_SEQUENCE_REJECTED"

        if frame.prev_merkle_hash != self.merkle_head:
            self.audit_log.append(
                {"seq": frame.sequence_number, "status": "REJECTED", "reason": "MERKLE_CHAIN_BROKEN"}
            )
            return False, "MERKLE_CHAIN_BROKEN"

        expected_mac = self._compute_mac(
            spacecraft_id=frame.spacecraft_id,
            spi=frame.spi,
            seq=frame.sequence_number,
            command_op=frame.command_op,
            payload=frame.payload,
            prev_hash=frame.prev_merkle_hash,
        )
        if not hmac.compare_digest(expected_mac, frame.auth_tag_hex):
            self.audit_log.append(
                {"seq": frame.sequence_number, "status": "REJECTED", "reason": "PAYLOAD_MAC_MISMATCH"}
            )
            return False, "PAYLOAD_MAC_MISMATCH"

        self.highest_seq = frame.sequence_number
        self.merkle_head = hashlib.sha3_256((self.merkle_head + frame.auth_tag_hex).encode("utf-8")).hexdigest()
        self.audit_log.append({"seq": frame.sequence_number, "status": "ACCEPTED", "reason": "AUTHENTICATED_PQC_SDLS"})
        return True, "AUTHENTICATED_PQC_SDLS"


@dataclass
class PQCSecurityReport:
    algorithm: str
    lattice_modulus_q: int
    polynomial_degree_n: int
    module_rank_k: int
    kem_bit_agreement_pct: float
    shared_key_fingerprint_sha3: str
    crypto_latency_us: float
    authentic_command_verified: bool
    replay_attack_blocked: bool
    tamper_attack_blocked: bool
    rogue_spi_blocked: bool
    merkle_root_hash: str
    signed_frame: Dict[str, Any]
    intrusion_test_ledger: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_pqc_telecommand_audit(
    spacecraft_id: str = "STARLINK_LEO",
    command_op: str = "EXECUTE_PROACTIVE_BURN",
    delta_v_m_s: float = 0.325,
    seed: int = 2026,
) -> PQCSecurityReport:
    """
    Execute an end-to-end Post-Quantum ML-KEM key exchange and CCSDS 355.0-B-2
    SDLS telecommand authentication & adversarial intrusion test suite.
    """
    t0 = time.perf_counter()
    kp = mlkem_keygen(seed=seed)
    ct, orig_bits, key_enc = mlkem_encapsulate(kp, seed=seed + 7)
    rec_bits, key_dec = mlkem_decapsulate(kp, ct)

    bit_match_pct = float(np.mean(orig_bits == rec_bits)) * 100.0
    guard = CCSDSSDLSGuard(session_key=key_dec, valid_spi=101)

    cmd_payload = {
        "delta_v_m_s": round(delta_v_m_s, 4),
        "burn_frame": "RIC_IN_TRACK",
        "swarm_quorum_approved": True,
    }
    valid_frame = guard.sign_telecommand(
        spacecraft_id=spacecraft_id,
        command_op=command_op,
        payload=cmd_payload,
    )
    ok_valid, _ = guard.verify_and_commit(valid_frame)

    # Adversarial Test 1: Replay Attack (re-sending sequence 1)
    ok_replay, reason_replay = guard.verify_and_commit(valid_frame)
    replay_blocked = (not ok_replay) and (reason_replay == "ANTI_REPLAY_SEQUENCE_REJECTED")

    # Adversarial Test 2: Man-in-the-Middle Payload Bit Tampering
    tampered_frame = SDLSCommandFrame(
        spacecraft_id=spacecraft_id,
        spi=101,
        sequence_number=2,
        command_op=command_op,
        payload={"delta_v_m_s": 99.9, "burn_frame": "DEORBIT_SABOTAGE", "swarm_quorum_approved": True},
        prev_merkle_hash=guard.merkle_head,
        auth_tag_hex=valid_frame.auth_tag_hex,
    )
    ok_tamper, reason_tamper = guard.verify_and_commit(tampered_frame)
    tamper_blocked = (not ok_tamper) and (reason_tamper == "PAYLOAD_MAC_MISMATCH")

    # Adversarial Test 3: Rogue Ground Station Invalid SPI
    rogue_frame = SDLSCommandFrame(
        spacecraft_id=spacecraft_id,
        spi=999,
        sequence_number=2,
        command_op=command_op,
        payload=cmd_payload,
        prev_merkle_hash=guard.merkle_head,
        auth_tag_hex=valid_frame.auth_tag_hex,
    )
    ok_rogue, reason_rogue = guard.verify_and_commit(rogue_frame)
    rogue_blocked = (not ok_rogue) and (reason_rogue == "INVALID_SDLS_SPI")

    latency_us = (time.perf_counter() - t0) * 1e6

    return PQCSecurityReport(
        algorithm="ML-KEM-Lattice + CCSDS-355.0-B-2-HMAC-SHA3-256",
        lattice_modulus_q=LATTICE_Q,
        polynomial_degree_n=LATTICE_N,
        module_rank_k=LATTICE_K,
        kem_bit_agreement_pct=round(bit_match_pct, 2),
        shared_key_fingerprint_sha3=key_enc.hex()[:32],
        crypto_latency_us=round(latency_us, 1),
        authentic_command_verified=bool(ok_valid and key_enc == key_dec),
        replay_attack_blocked=bool(replay_blocked),
        tamper_attack_blocked=bool(tamper_blocked),
        rogue_spi_blocked=bool(rogue_blocked),
        merkle_root_hash=guard.merkle_head[:32],
        signed_frame=valid_frame.to_dict(),
        intrusion_test_ledger=guard.audit_log,
    )
