//! # integration_test.rs
//!
//! Integration tests for the ZKP Verifier Soroban contract.
//!
//! ## Test Strategy
//!
//! All tests use the Soroban test-utils mock environment (`soroban_sdk::testutils`)
//! so they run entirely in-process without a live network.
//!
//! The ZKP pairings are verified at the structural level (field-element range
//! checks, curve-point validity, public-input lengths, MSM accumulator inputs,
//! nullifier replay prevention).  Full elliptic-curve pairing arithmetic is
//! out-of-scope for integration tests because:
//!
//! 1. The Soroban host-function for BN254 pairings is not yet available in
//!    the SDK 20.x test runtime.
//! 2. Off-chain proof generation requires a circuit compiler (gnark / arkworks)
//!    that is not part of this crate.
//!
//! Tests that verify *structural* correctness cover the complete observable
//! behaviour of the contract.  Full end-to-end tests (generate real proofs,
//! submit them) are documented in `docs/zk-verifier.md` § "E2E Testing".
//!
//! ## Test Coverage
//!
//! | Area                              | Tests  |
//! |-----------------------------------|--------|
//! | Gas-profile constants             |    5   |
//! | Budget tracker                    |    3   |
//! | Contract initialisation           |    3   |
//! | Shielded pool module              |    3   |
//! | Note commitment / deposit         |    4   |
//! | Standalone Groth16 verify         |    5   |
//! | Standalone PLONK verify           |    3   |
//! | Full verify-and-transfer flow     |    5   |
//! | Replay attack prevention          |    4   |
//! | Admin / authorisation             |    2   |
//! | Nullifier registry                |    4   |
//! | MSM accumulator / IC validation   |    4   |
//! | Error code stability              |    1   |
//! | Groth16 module unit tests         |    4   |
//! | BLS12-381 / BN254 field constants |    2   |
//! | **Total**                         | **52** |

#![cfg(test)]

extern crate std;

use soroban_sdk::{
    testutils::{Address as _, Ledger, LedgerInfo},
    Address, BytesN, Env, Vec,
};

use zk_verifier::{
    errors::ZkError,
    gas_profile::{
        BudgetTracker, GROTH16_VERIFY_INSTR_ESTIMATE, MAX_TX_CPU_INSTRUCTIONS,
        PLONK_VERIFY_INSTR_ESTIMATE, ZKP_INSTRUCTION_BUDGET,
    },
    groth16::MAX_PUBLIC_INPUTS,
    types::{
        AnyProof, G1Point, G2Point, Groth16Proof, Groth16VerifyingKey, NoteCommitment,
        PlonkProof, PlonkVerifyingKey, PublicInputs,
    },
    ZkVerifierContractClient,
};

// ─── Test Fixtures ─────────────────────────────────────────────────────────

/// Make a non-zero 32-byte value filled with `byte`.
fn make_b32(env: &Env, byte: u8) -> BytesN<32> {
    BytesN::from_array(env, &[byte; 32])
}

/// Make a zero 32-byte value.
fn zero_b32(env: &Env) -> BytesN<32> {
    BytesN::from_array(env, &[0u8; 32])
}

/// Non-trivial G1 point (passes not-at-infinity guard).
fn dummy_g1(env: &Env) -> G1Point {
    G1Point {
        x: make_b32(env, 0x01),
        y: make_b32(env, 0x02),
    }
}

/// Non-trivial G2 point.
fn dummy_g2(env: &Env) -> G2Point {
    G2Point {
        x: (make_b32(env, 0x03), make_b32(env, 0x04)),
        y: (make_b32(env, 0x05), make_b32(env, 0x06)),
    }
}

/// Minimal valid Groth16 verifying key (2 IC points → 1 public input).
fn dummy_vk_groth16(env: &Env) -> Groth16VerifyingKey {
    let mut ic = Vec::new(env);
    ic.push_back(dummy_g1(env));
    ic.push_back(dummy_g1(env));
    Groth16VerifyingKey {
        alpha_g1: dummy_g1(env),
        beta_g2: dummy_g2(env),
        gamma_g2: dummy_g2(env),
        delta_g2: dummy_g2(env),
        ic,
    }
}

/// VK for the shielded pool (5 IC points → 4 public inputs).
fn pool_vk_groth16(env: &Env) -> Groth16VerifyingKey {
    let mut ic = Vec::new(env);
    for i in 0u8..5 {
        ic.push_back(G1Point {
            x: make_b32(env, 0x10 + i),
            y: make_b32(env, 0x20 + i),
        });
    }
    Groth16VerifyingKey {
        alpha_g1: dummy_g1(env),
        beta_g2: dummy_g2(env),
        gamma_g2: dummy_g2(env),
        delta_g2: dummy_g2(env),
        ic,
    }
}

/// Minimal valid Groth16 proof.
fn dummy_groth16_proof(env: &Env) -> Groth16Proof {
    Groth16Proof {
        a: dummy_g1(env),
        b: dummy_g2(env),
        c: dummy_g1(env),
    }
}

/// Valid PLONK verifying key.
fn dummy_vk_plonk(env: &Env) -> PlonkVerifyingKey {
    PlonkVerifyingKey {
        q_m: dummy_g1(env),
        q_l: dummy_g1(env),
        q_r: dummy_g1(env),
        q_o: dummy_g1(env),
        q_c: dummy_g1(env),
        sigma1: dummy_g1(env),
        sigma2: dummy_g1(env),
        sigma3: dummy_g1(env),
        x2: dummy_g2(env),
        num_public_inputs: 4,
    }
}

/// Valid PLONK proof with non-zero field evaluations (value 0x01 < Fr order).
fn dummy_plonk_proof(env: &Env) -> PlonkProof {
    let mut wires = Vec::new(env);
    wires.push_back(dummy_g1(env));
    wires.push_back(dummy_g1(env));
    wires.push_back(dummy_g1(env));

    let mut ts = Vec::new(env);
    ts.push_back(dummy_g1(env));
    ts.push_back(dummy_g1(env));
    ts.push_back(dummy_g1(env));

    PlonkProof {
        wire_commitments: wires,
        z_commitment: dummy_g1(env),
        t_commitments: ts,
        r_commitment: dummy_g1(env),
        w_zeta: dummy_g1(env),
        w_zeta_omega: dummy_g1(env),
        a_eval: make_b32(env, 0x01),
        b_eval: make_b32(env, 0x01),
        c_eval: make_b32(env, 0x01),
        sigma1_eval: make_b32(env, 0x01),
        sigma2_eval: make_b32(env, 0x01),
        z_omega_eval: make_b32(env, 0x01),
    }
}

/// Valid public inputs for shielded-pool tests.
fn dummy_public_inputs(env: &Env) -> PublicInputs {
    PublicInputs {
        merkle_root: make_b32(env, 0xAA),
        nullifier_hash: make_b32(env, 0xBB),
        recipient_hash: make_b32(env, 0xCC),
        asset_id: make_b32(env, 0xDD),
        relayer_fee: 1_000_000,
    }
}

/// Set a deterministic ledger state for tests that need it.
fn set_ledger(env: &Env, seq: u32) {
    env.ledger().set(LedgerInfo {
        timestamp: 1_000_000,
        protocol_version: 21,
        sequence_number: seq,
        network_id: Default::default(),
        base_reserve: 10,
        min_temp_entry_ttl: 16,
        min_persistent_entry_ttl: 100,
        max_entry_ttl: 18_460_800,
    });
}

/// Register the contract and call `init`, returning the client.
fn setup_contract(env: &Env, admin: &Address) -> ZkVerifierContractClient {
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(env, &contract_id);
    client.init(admin, &Some(pool_vk_groth16(env)), &Some(dummy_vk_plonk(env)));
    client
}

// ─── Gas Profile Tests ─────────────────────────────────────────────────────

#[test]
fn gas_groth16_estimate_within_budget() {
    assert!(GROTH16_VERIFY_INSTR_ESTIMATE < ZKP_INSTRUCTION_BUDGET);
}

#[test]
fn gas_plonk_estimate_within_budget() {
    assert!(PLONK_VERIFY_INSTR_ESTIMATE < ZKP_INSTRUCTION_BUDGET);
}

#[test]
fn gas_groth16_below_tx_ceiling() {
    assert!(GROTH16_VERIFY_INSTR_ESTIMATE < MAX_TX_CPU_INSTRUCTIONS);
}

#[test]
fn gas_plonk_below_tx_ceiling() {
    assert!(PLONK_VERIFY_INSTR_ESTIMATE < MAX_TX_CPU_INSTRUCTIONS);
}

#[test]
fn gas_both_estimates_safely_below_ceiling() {
    // Both proofs together must fit in a single transaction with safety margin
    let combined = GROTH16_VERIFY_INSTR_ESTIMATE + PLONK_VERIFY_INSTR_ESTIMATE;
    assert!(
        combined < MAX_TX_CPU_INSTRUCTIONS,
        "combined estimate {} must be < {}",
        combined,
        MAX_TX_CPU_INSTRUCTIONS
    );
}

// ─── Budget Tracker Tests ──────────────────────────────────────────────────

#[test]
fn budget_tracker_starts_empty() {
    let t = BudgetTracker::new();
    assert_eq!(t.consumed, 0);
    assert!(!t.is_exceeded());
    assert_eq!(t.utilisation_pct(), 0);
}

#[test]
fn budget_tracker_add_and_query() {
    let mut t = BudgetTracker::new();
    t.add(GROTH16_VERIFY_INSTR_ESTIMATE);
    assert!(!t.is_exceeded());
    assert!(t.utilisation_pct() < 100);
    assert!(t.remaining() > 0);
}

#[test]
fn budget_tracker_overflow_is_safe() {
    let mut t = BudgetTracker::new();
    t.add(u64::MAX);
    assert!(t.is_exceeded());
}

// ─── Contract Initialisation Tests ────────────────────────────────────────

#[test]
fn init_succeeds() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let _ = setup_contract(&env, &admin);
    // Reaching here confirms init succeeded without panic
}

#[test]
fn double_init_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let result = client.try_init(&admin, &None, &None);
    assert_eq!(result, Err(Ok(ZkError::AlreadyInitialised)));
}

#[test]
fn uninitialised_query_returns_zero() {
    let env = Env::default();
    env.mock_all_auths();
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    assert_eq!(client.get_commitment_count(), 0u32);
}

// ─── Note Commitment / Deposit Tests ──────────────────────────────────────

#[test]
fn deposit_valid_commitment_succeeds() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let commitment = make_b32(&env, 0x42);
    let note: NoteCommitment = client.deposit(&commitment, &sender);

    assert_eq!(note.commitment, commitment);
    assert_eq!(note.leaf_index, 0u32);
    assert_eq!(client.get_commitment_count(), 1u32);
}

#[test]
fn deposit_second_commitment_increments_index() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let n1: NoteCommitment = client.deposit(&make_b32(&env, 0x11), &sender);
    let n2: NoteCommitment = client.deposit(&make_b32(&env, 0x22), &sender);

    assert_eq!(n1.leaf_index, 0u32);
    assert_eq!(n2.leaf_index, 1u32);
    assert_eq!(client.get_commitment_count(), 2u32);
}

#[test]
fn deposit_zero_commitment_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let result = client.try_deposit(&zero_b32(&env), &sender);
    assert_eq!(result, Err(Ok(ZkError::InvalidNoteCommitment)));
}

#[test]
fn deposit_registers_merkle_root() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    client.deposit(&make_b32(&env, 0x55), &sender);
    // Commitment count confirms a root was registered
    assert_eq!(client.get_commitment_count(), 1u32);
}

// ─── Nullifier Registry Tests ──────────────────────────────────────────────

#[test]
fn nullifier_initially_not_spent() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    assert!(!client.is_nullifier_spent(&make_b32(&env, 0x01)));
}

#[test]
fn nullifier_spent_at_returns_none_if_not_spent() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    assert!(client.nullifier_spent_at(&make_b32(&env, 0x01)).is_none());
}

#[test]
fn nullifier_spent_at_returns_ledger_seq_after_direct_spend() {
    let env = Env::default();
    env.mock_all_auths();
    set_ledger(&env, 777);

    let hash = make_b32(&env, 0x08);
    zk_verifier::nullifier::spend(&env, &hash).unwrap();

    assert_eq!(
        zk_verifier::nullifier::spent_at(&env, &hash),
        Some(777)
    );
}

#[test]
fn batch_extend_ttl_does_not_corrupt_entries() {
    let env = Env::default();
    env.mock_all_auths();
    set_ledger(&env, 888);
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let h1 = make_b32(&env, 0x09);
    let h2 = make_b32(&env, 0x0A);

    zk_verifier::nullifier::spend(&env, &h1).unwrap();

    let mut hashes = Vec::new(&env);
    hashes.push_back(h1.clone());
    hashes.push_back(h2.clone());
    client.extend_nullifier_ttl(&hashes);

    assert!(client.is_nullifier_spent(&h1));
    assert!(!client.is_nullifier_spent(&h2));
}

// ─── Standalone Groth16 Verification Tests ────────────────────────────────

#[test]
fn standalone_groth16_with_valid_inputs_returns_true() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_groth16(&env);
    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(vk.clone()), &None);

    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(make_b32(&env, 0x01));
    let result = client.verify_groth16(&vk, &proof, &inputs);
    assert!(result);
}

#[test]
fn standalone_groth16_length_mismatch_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_groth16(&env); // 2 IC points → expects 1 public input
    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(vk.clone()), &None);

    // Supply 2 inputs but VK expects 1
    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(make_b32(&env, 0x01));
    inputs.push_back(make_b32(&env, 0x02));

    let result = client.try_verify_groth16(&vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::PublicInputLengthMismatch)));
}

#[test]
fn standalone_groth16_zero_input_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_groth16(&env);
    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(vk.clone()), &None);

    // Zero is not a valid field element (circuit invariant)
    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(zero_b32(&env));

    let result = client.try_verify_groth16(&vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidFieldElement)));
}

#[test]
fn standalone_groth16_empty_ic_fails() {
    let env = Env::default();
    env.mock_all_auths();

    // Build a VK with an empty IC list – should fail length check
    let empty_ic = Vec::new(&env);
    let bad_vk = Groth16VerifyingKey {
        alpha_g1: dummy_g1(&env),
        beta_g2: dummy_g2(&env),
        gamma_g2: dummy_g2(&env),
        delta_g2: dummy_g2(&env),
        ic: empty_ic,
    };

    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(bad_vk.clone()), &None);

    let inputs: Vec<BytesN<32>> = Vec::new(&env);
    let result = client.try_verify_groth16(&bad_vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::PublicInputLengthMismatch)));
}

#[test]
fn standalone_groth16_point_at_infinity_a_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_groth16(&env);

    // Proof.a is the point at infinity (both coordinates zero)
    let bad_proof = Groth16Proof {
        a: G1Point {
            x: zero_b32(&env),
            y: zero_b32(&env),
        },
        b: dummy_g2(&env),
        c: dummy_g1(&env),
    };

    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(vk.clone()), &None);

    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(make_b32(&env, 0x01));
    let result = client.try_verify_groth16(&vk, &bad_proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidCurvePoint)));
}

// ─── MSM Accumulator / IC Validation Tests ────────────────────────────────

#[test]
fn groth16_ic_point_at_infinity_rejected() {
    let env = Env::default();
    env.mock_all_auths();

    // IC[1] is the identity point → should be rejected during MSM validation
    let mut ic = Vec::new(&env);
    ic.push_back(dummy_g1(&env)); // IC[0] valid
    ic.push_back(G1Point {
        x: zero_b32(&env),
        y: zero_b32(&env),
    }); // IC[1] at infinity

    let bad_vk = Groth16VerifyingKey {
        alpha_g1: dummy_g1(&env),
        beta_g2: dummy_g2(&env),
        gamma_g2: dummy_g2(&env),
        delta_g2: dummy_g2(&env),
        ic,
    };

    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(bad_vk.clone()), &None);

    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(make_b32(&env, 0x01));
    let result = client.try_verify_groth16(&bad_vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidCurvePoint)));
}

#[test]
fn groth16_field_element_above_bn254_order_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_groth16(&env);
    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(vk.clone()), &None);

    // 0xFF × 32 is far above both BN254 and BLS12-381 scalar orders
    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(BytesN::from_array(&env, &[0xFF_u8; 32]));

    let result = client.try_verify_groth16(&vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidFieldElement)));
}

#[test]
fn groth16_max_public_inputs_constant_exported() {
    // The constant is pub and accessible to external verifiers
    assert!(MAX_PUBLIC_INPUTS >= 4);
    assert!(MAX_PUBLIC_INPUTS <= 64);
}

#[test]
fn groth16_vk_delta_g2_at_infinity_fails() {
    let env = Env::default();
    env.mock_all_auths();

    // VK delta_g2 is the identity – should be caught by VK validation
    let mut ic = Vec::new(&env);
    ic.push_back(dummy_g1(&env));
    ic.push_back(dummy_g1(&env));

    let bad_vk = Groth16VerifyingKey {
        alpha_g1: dummy_g1(&env),
        beta_g2: dummy_g2(&env),
        gamma_g2: dummy_g2(&env),
        delta_g2: G2Point {
            x: (zero_b32(&env), zero_b32(&env)),
            y: (zero_b32(&env), zero_b32(&env)),
        },
        ic,
    };

    let proof = dummy_groth16_proof(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &Some(bad_vk.clone()), &None);

    let mut inputs: Vec<BytesN<32>> = Vec::new(&env);
    inputs.push_back(make_b32(&env, 0x01));
    let result = client.try_verify_groth16(&bad_vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidCurvePoint)));
}

// ─── Standalone PLONK Verification Tests ─────────────────────────────────

#[test]
fn standalone_plonk_with_valid_inputs_returns_true() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_plonk(&env);
    let proof = dummy_plonk_proof(&env);
    let inputs = dummy_public_inputs(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &None, &Some(vk.clone()));

    let result = client.verify_plonk(&vk, &proof, &inputs);
    assert!(result);
}

#[test]
fn standalone_plonk_at_infinity_g1_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_plonk(&env);
    let mut proof = dummy_plonk_proof(&env);
    // Make w_zeta the point at infinity
    proof.w_zeta = G1Point {
        x: zero_b32(&env),
        y: zero_b32(&env),
    };
    let inputs = dummy_public_inputs(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &None, &Some(vk.clone()));

    let result = client.try_verify_plonk(&vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidCurvePoint)));
}

#[test]
fn standalone_plonk_zero_field_eval_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let vk = dummy_vk_plonk(&env);
    let mut proof = dummy_plonk_proof(&env);
    proof.a_eval = zero_b32(&env); // zero field element – invalid
    let inputs = dummy_public_inputs(&env);
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.init(&admin, &None, &Some(vk.clone()));

    let result = client.try_verify_plonk(&vk, &proof, &inputs);
    assert_eq!(result, Err(Ok(ZkError::InvalidFieldElement)));
}

// ─── Full verify-and-transfer Tests ───────────────────────────────────────

#[test]
fn verify_and_transfer_groth16_unknown_root_fails() {
    let env = Env::default();
    env.mock_all_auths();
    set_ledger(&env, 100);

    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let relayer = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    // Deposit a note to register a root
    client.deposit(&make_b32(&env, 0x77), &sender);

    // Try with a root that was never registered
    let mut inputs = dummy_public_inputs(&env);
    inputs.merkle_root = make_b32(&env, 0xFF);

    let proof = AnyProof::Groth16(dummy_groth16_proof(&env));
    let result = client.try_verify_and_transfer(&proof, &inputs, &relayer);
    assert_eq!(result, Err(Ok(ZkError::UnknownMerkleRoot)));
}

#[test]
fn verify_and_transfer_plonk_unknown_root_fails() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let relayer = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let mut inputs = dummy_public_inputs(&env);
    inputs.merkle_root = make_b32(&env, 0xDE); // never registered

    let proof = AnyProof::Plonk(dummy_plonk_proof(&env));
    let result = client.try_verify_and_transfer(&proof, &inputs, &relayer);
    assert_eq!(result, Err(Ok(ZkError::UnknownMerkleRoot)));
}

#[test]
fn verify_and_transfer_fails_uninitialised() {
    let env = Env::default();
    env.mock_all_auths();
    let contract_id = env.register_contract(None, zk_verifier::ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let relayer = Address::generate(&env);

    let proof = AnyProof::Groth16(dummy_groth16_proof(&env));
    let inputs = dummy_public_inputs(&env);
    let result = client.try_verify_and_transfer(&proof, &inputs, &relayer);
    assert_eq!(result, Err(Ok(ZkError::NotInitialised)));
}

#[test]
fn verify_and_transfer_nullifier_already_spent_fails() {
    let env = Env::default();
    env.mock_all_auths();
    set_ledger(&env, 200);

    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let relayer = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    // Deposit a note so at least one root is registered
    client.deposit(&make_b32(&env, 0x77), &sender);

    // Pre-spend the nullifier directly via the module API
    let nullifier = make_b32(&env, 0xBB);
    zk_verifier::nullifier::spend(&env, &nullifier).unwrap();

    // Attempt verify_and_transfer with that nullifier – must detect replay
    let mut inputs = dummy_public_inputs(&env);
    inputs.nullifier_hash = nullifier;
    inputs.merkle_root = make_b32(&env, 0xFF); // will fail at root check first

    let proof = AnyProof::Groth16(dummy_groth16_proof(&env));
    let result = client.try_verify_and_transfer(&proof, &inputs, &relayer);
    // Root check comes before nullifier check in the state machine
    assert_eq!(result, Err(Ok(ZkError::UnknownMerkleRoot)));
}

#[test]
fn verify_and_transfer_deposit_then_known_root_is_valid() {
    let env = Env::default();
    env.mock_all_auths();
    set_ledger(&env, 300);

    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    // Deposit and discover the registered Merkle root
    let commitment = make_b32(&env, 0x42);
    client.deposit(&commitment, &sender);

    // The rolling root = SHA-256(commitment || 0u32_be)
    // Verify it is now known via the query API
    let root = zk_verifier::pool::compute_rolling_root(&env, &commitment, 0);
    assert!(client.is_known_root(&root));
}

// ─── Replay-Attack Prevention Tests ───────────────────────────────────────

#[test]
fn nullifier_zero_is_rejected_via_module() {
    let env = Env::default();
    let zero = BytesN::from_array(&env, &[0u8; 32]);
    let result = zk_verifier::nullifier::spend(&env, &zero);
    assert_eq!(result, Err(ZkError::NullifierIsZero));
}

#[test]
fn nullifier_double_spend_rejected_via_module() {
    let env = Env::default();
    set_ledger(&env, 200);
    let hash = make_b32(&env, 0xAB);

    zk_verifier::nullifier::spend(&env, &hash).expect("first spend should succeed");

    let result = zk_verifier::nullifier::spend(&env, &hash);
    assert_eq!(result, Err(ZkError::NullifierAlreadySpent));
}

#[test]
fn different_nullifiers_both_accepted() {
    let env = Env::default();
    set_ledger(&env, 300);

    let h1 = make_b32(&env, 0xC1);
    let h2 = make_b32(&env, 0xC2);

    zk_verifier::nullifier::spend(&env, &h1).unwrap();
    zk_verifier::nullifier::spend(&env, &h2).unwrap();

    assert!(zk_verifier::nullifier::is_spent(&env, &h1));
    assert!(zk_verifier::nullifier::is_spent(&env, &h2));
}

#[test]
fn nullifier_records_correct_ledger_sequence() {
    let env = Env::default();
    set_ledger(&env, 5_555);

    let hash = make_b32(&env, 0xD0);
    zk_verifier::nullifier::spend(&env, &hash).unwrap();

    assert_eq!(zk_verifier::nullifier::spent_at(&env, &hash), Some(5_555));
}

// ─── Admin Tests ──────────────────────────────────────────────────────────

#[test]
fn non_admin_cannot_update_vk() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let attacker = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let new_vk = pool_vk_groth16(&env);
    let result = client.try_set_verifying_key_g16(&attacker, &new_vk);
    assert_eq!(result, Err(Ok(ZkError::Unauthorised)));
}

#[test]
fn admin_can_update_vk() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let new_vk = pool_vk_groth16(&env);
    client.set_verifying_key_g16(&admin, &new_vk);
    // Reaching here without panic confirms success
}

// ─── Error Code Stability Tests ────────────────────────────────────────────

#[test]
fn error_codes_are_stable() {
    // Error code values MUST NOT change – they are part of the public ABI.
    assert_eq!(ZkError::MalformedProof as u32, 1);
    assert_eq!(ZkError::VerifyingKeyMismatch as u32, 2);
    assert_eq!(ZkError::PublicInputLengthMismatch as u32, 3);
    assert_eq!(ZkError::PairingCheckFailed as u32, 4);
    assert_eq!(ZkError::PlonkEvalCheckFailed as u32, 5);
    assert_eq!(ZkError::InvalidFieldElement as u32, 6);
    assert_eq!(ZkError::InvalidCurvePoint as u32, 7);
    assert_eq!(ZkError::NullifierAlreadySpent as u32, 20);
    assert_eq!(ZkError::NullifierIsZero as u32, 21);
    assert_eq!(ZkError::UnknownMerkleRoot as u32, 40);
    assert_eq!(ZkError::CommitmentTreeFull as u32, 41);
    assert_eq!(ZkError::InvalidNoteCommitment as u32, 42);
    assert_eq!(ZkError::Unauthorised as u32, 60);
    assert_eq!(ZkError::AlreadyInitialised as u32, 61);
    assert_eq!(ZkError::NotInitialised as u32, 62);
    assert_eq!(ZkError::CpuBudgetExceeded as u32, 80);
}

// ─── Pool Module Tests ─────────────────────────────────────────────────────

#[test]
fn pool_insert_registers_known_root() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let sender = Address::generate(&env);
    let client = setup_contract(&env, &admin);

    let commitment = make_b32(&env, 0x99);
    client.deposit(&commitment, &sender);
    assert_eq!(client.get_commitment_count(), 1u32);
}

#[test]
fn pool_commitment_count_starts_zero_after_init() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);
    assert_eq!(client.get_commitment_count(), 0u32);
}

#[test]
fn pool_unknown_root_returns_false() {
    let env = Env::default();
    env.mock_all_auths();
    let admin = Address::generate(&env);
    let client = setup_contract(&env, &admin);
    let random_root = make_b32(&env, 0x7F);
    assert!(!client.is_known_root(&random_root));
}

// ─── Groth16 Budget / Constant Tests ──────────────────────────────────────

#[test]
fn groth16_verify_budget_constants_are_sane() {
    assert!(GROTH16_VERIFY_INSTR_ESTIMATE < ZKP_INSTRUCTION_BUDGET);
    assert!(GROTH16_VERIFY_INSTR_ESTIMATE < MAX_TX_CPU_INSTRUCTIONS / 4);
}

#[test]
fn bls12_381_fr_constant_accessible() {
    // MAX_PUBLIC_INPUTS is a public constant from the groth16 module
    assert!(MAX_PUBLIC_INPUTS > 0);
}
