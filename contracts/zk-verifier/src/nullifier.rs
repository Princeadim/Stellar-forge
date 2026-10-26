//! # nullifier.rs
//!
//! Nullifier registry backed by Soroban **Persistent Storage**.
//!
//! ## Purpose
//!
//! A nullifier is a cryptographic one-time token derived inside the ZK circuit
//! from the note's secret key and its Merkle-tree leaf index:
//!
//! ```text
//! nullifier = PRF(note_secret_key, leaf_index)
//! ```
//!
//! Recording it on-chain prevents a spent note from being re-used (replay
//! attacks).  Because nullifiers must survive contract upgrades and arbitrary
//! ledger gaps, they are stored in **Persistent storage** with a long TTL.
//!
//! ## Storage Model
//!
//! Each nullifier is stored as a `Persistent` entry:
//!
//! ```text
//! key:   (Symbol("nf"), BytesN<32>)   // compact ledger key
//! value: u32                           // ledger sequence at which it was spent
//! ```
//!
//! Persistent storage survives ledger expiry, meaning nullifiers survive
//! indefinitely as long as the TTL is extended via the permissionless
//! TTL-bumper entry point (`extend_nullifier_ttl` in `lib.rs`).
//!
//! ## Replay-Attack Prevention Invariants
//!
//! The `spend` function is the **only** way to mark a nullifier as used.  It:
//!
//! 1. Rejects the zero nullifier (`NullifierIsZero`).
//! 2. Rejects a nullifier already present in storage (`NullifierAlreadySpent`).
//! 3. Writes the current ledger sequence as spend-time metadata.
//! 4. Immediately extends the entry's TTL to ~3 years.
//!
//! This ordering is enforced by the contract state machine in `lib.rs`:
//! `spend` is called *after* a successful proof verification and
//! *before* any asset transfer is recorded.
//!
//! ## TTL Maintenance
//!
//! Soroban Persistent entries expire if their TTL is not extended.  Expiry
//! of a nullifier entry would be catastrophic — an expired entry can be
//! re-spent by anyone holding the original note.  The contract mitigates this:
//!
//! * **Automatic extension on spend**: `spend` always extends TTL immediately.
//! * **Permissionless batch bumper**: `batch_extend_ttl` can be called by
//!   anyone (relayer, off-chain service) to refresh a list of entries.
//! * **Extension target**: `TTL_EXTEND_TO ≈ 18 460 800` ledgers ≈ 3 years
//!   at 5 seconds per ledger.
//!
//! ## Gas Costs
//!
//! | Operation               | Estimated Instructions |
//! |-------------------------|------------------------|
//! | `is_spent` (has)        |               ~4 000   |
//! | `spend` (write + TTL)   |               ~8 000   |
//! | `spent_at` (get)        |               ~4 000   |
//! | `extend_nullifier_ttl`  |               ~4 000   |
//!
//! ## References
//!
//! * Tornado Cash nullifier design: <https://tornado.cash/audits>
//! * Zcash Sapling Protocol Spec §4.19: Nullifier Sets
//! * Soroban TTL documentation: <https://developers.stellar.org/docs/learn/smart-contract-internals/persisting-data>

use soroban_sdk::{symbol_short, BytesN, Env, Symbol};

use crate::errors::ZkError;

// ─── Storage Constants ─────────────────────────────────────────────────────

/// Namespace prefix for all nullifier entries.  Using a short symbol keeps
/// ledger keys compact, reducing storage fees.
///
/// All nullifier entries are stored under `(NULLIFIER_NS, hash)` to namespace
/// them away from all other contract storage keys (e.g. admin, vk, root).
const NULLIFIER_NS: Symbol = symbol_short!("nf");

/// TTL extension threshold: extend unconditionally regardless of remaining TTL.
const TTL_EXTEND_THRESHOLD: u32 = 0;

/// Target TTL after extension: ~3 years at 5 seconds / ledger.
///
/// Calculation: 3 × 365.25 × 24 × 3600 / 5 ≈ 18 921 960.
/// We use 18 460 800 which is safely below the Protocol 21 max TTL ceiling.
const TTL_EXTEND_TO: u32 = 18_460_800;

// ─── Internal Helpers ──────────────────────────────────────────────────────

/// Build the composite ledger key for a nullifier.
///
/// The tuple `(NULLIFIER_NS, hash)` ensures no collision with other storage
/// keys used in the contract (pool_cnt, pool_root, admin, vk_g16, vk_plonk).
#[inline]
fn nullifier_key(hash: &BytesN<32>) -> (Symbol, BytesN<32>) {
    (NULLIFIER_NS, hash.clone())
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Check whether `hash` has already been spent.
///
/// Returns `true` if the nullifier is present in Persistent storage.
/// This is the **cheap pre-check** performed *before* the expensive pairing
/// verification in `verify_and_transfer` to fail fast on replay attempts.
///
/// # Gas Cost
///
/// One Persistent storage `has()` call: ~4 000 instructions.
pub fn is_spent(env: &Env, hash: &BytesN<32>) -> bool {
    env.storage()
        .persistent()
        .has(&nullifier_key(hash))
}

/// Mark `hash` as spent by writing the current ledger sequence number.
///
/// This function atomically:
/// 1. Validates `hash` is non-zero.
/// 2. Checks `hash` is not already present (replay guard).
/// 3. Writes `(NULLIFIER_NS, hash) → ledger_sequence` to Persistent storage.
/// 4. Extends the TTL of the new entry to [`TTL_EXTEND_TO`] ledgers.
///
/// # Errors
///
/// * [`ZkError::NullifierIsZero`] – `hash` is all-zero bytes.  The ZK circuit
///   should never produce a zero nullifier; this is a safe sentinel rejection.
/// * [`ZkError::NullifierAlreadySpent`] – `hash` was already recorded.
///   This is the primary replay-attack defence and must never be bypassed.
///
/// # Ordering Guarantee
///
/// The caller (`lib.rs::verify_and_transfer`) MUST call this function only
/// *after* a successful proof verification.  Calling it before verification
/// would allow a front-runner to pre-spend valid nullifiers without a proof.
///
/// # Panics
///
/// Does not panic.  All failure paths return `Err`.
pub fn spend(env: &Env, hash: &BytesN<32>) -> Result<(), ZkError> {
    // ── Guard: zero nullifier is never valid ───────────────────────────
    let zero = BytesN::from_array(env, &[0u8; 32]);
    if hash == &zero {
        return Err(ZkError::NullifierIsZero);
    }

    let key = nullifier_key(hash);

    // ── Guard: replay attack prevention ───────────────────────────────
    if env.storage().persistent().has(&key) {
        return Err(ZkError::NullifierAlreadySpent);
    }

    // ── Write: record the ledger sequence as spend-time metadata ──────
    let ledger_seq: u32 = env.ledger().sequence();
    env.storage().persistent().set(&key, &ledger_seq);

    // ── TTL extension: ensure the entry survives indefinitely ─────────
    // We extend immediately after writing so the entry never expires before
    // a bumper service has a chance to refresh it.
    env.storage()
        .persistent()
        .extend_ttl(&key, TTL_EXTEND_THRESHOLD, TTL_EXTEND_TO);

    Ok(())
}

/// Return the ledger sequence at which `hash` was spent, or `None` if not spent.
///
/// Useful for audit purposes: callers can determine exactly when a note was
/// consumed relative to the current ledger height.
///
/// # Gas Cost
///
/// One Persistent storage `get()` call: ~4 000 instructions.
pub fn spent_at(env: &Env, hash: &BytesN<32>) -> Option<u32> {
    env.storage()
        .persistent()
        .get::<(Symbol, BytesN<32>), u32>(&nullifier_key(hash))
}

/// Extend the TTL of an existing nullifier entry.
///
/// Can be called by anyone (permissionless) to prevent nullifier entries from
/// expiring.  Soroban Persistent entries have a maximum TTL of ~18 400 000
/// ledgers (~3 years); without periodic extension they will expire.
///
/// For a security-critical registry, nullifier expiry is catastrophic: it
/// would allow re-spending of already-consumed notes.  Off-chain relayers
/// and maintenance services SHOULD periodically call the contract's
/// `extend_nullifier_ttl` entry point for all known nullifiers.
///
/// # Idempotency
///
/// If `hash` is not present in storage (not yet spent), this function is a
/// no-op.  It only extends the TTL of existing entries.
pub fn extend_nullifier_ttl(env: &Env, hash: &BytesN<32>) {
    let key = nullifier_key(hash);
    if env.storage().persistent().has(&key) {
        env.storage()
            .persistent()
            .extend_ttl(&key, TTL_EXTEND_THRESHOLD, TTL_EXTEND_TO);
    }
}

/// Batch-extend TTLs for a list of nullifiers.
///
/// Useful for a maintenance transaction that refreshes many nullifiers at once
/// rather than one per call.  The gas cost scales linearly with the number of
/// entries: approximately `4 000 × n` instructions for `n` hashes.
///
/// # Note
///
/// There is no upper bound enforced by this function.  Callers should ensure
/// the batch does not exceed the per-transaction instruction budget.  A safe
/// batch size is approximately 1 000 nullifiers per transaction.
pub fn batch_extend_ttl(env: &Env, hashes: &soroban_sdk::Vec<BytesN<32>>) {
    for hash in hashes.iter() {
        extend_nullifier_ttl(env, &hash);
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use soroban_sdk::{testutils::LedgerInfo, Env};

    fn make_b32(env: &Env, byte: u8) -> BytesN<32> {
        BytesN::from_array(env, &[byte; 32])
    }

    fn set_ledger_seq(env: &Env, seq: u32) {
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

    // ─── is_spent ─────────────────────────────────────────────────────────

    #[test]
    fn is_spent_returns_false_for_unseen_hash() {
        let env = Env::default();
        let hash = make_b32(&env, 0x01);
        assert!(!is_spent(&env, &hash));
    }

    #[test]
    fn is_spent_returns_true_after_spend() {
        let env = Env::default();
        set_ledger_seq(&env, 500);
        let hash = make_b32(&env, 0x02);
        spend(&env, &hash).unwrap();
        assert!(is_spent(&env, &hash));
    }

    // ─── spend ────────────────────────────────────────────────────────────

    #[test]
    fn spend_zero_hash_fails() {
        let env = Env::default();
        let zero = BytesN::from_array(&env, &[0u8; 32]);
        let result = spend(&env, &zero);
        assert_eq!(result, Err(ZkError::NullifierIsZero));
    }

    #[test]
    fn spend_valid_hash_succeeds() {
        let env = Env::default();
        set_ledger_seq(&env, 100);
        let hash = make_b32(&env, 0x0A);
        assert!(spend(&env, &hash).is_ok());
    }

    #[test]
    fn spend_double_spend_fails() {
        let env = Env::default();
        set_ledger_seq(&env, 200);
        let hash = make_b32(&env, 0xAB);

        // First spend succeeds
        spend(&env, &hash).expect("first spend should succeed");

        // Second spend on same hash must fail
        let result = spend(&env, &hash);
        assert_eq!(result, Err(ZkError::NullifierAlreadySpent));
    }

    #[test]
    fn spend_different_hashes_both_succeed() {
        let env = Env::default();
        set_ledger_seq(&env, 300);

        let h1 = make_b32(&env, 0x11);
        let h2 = make_b32(&env, 0x22);

        spend(&env, &h1).unwrap();
        spend(&env, &h2).unwrap();

        assert!(is_spent(&env, &h1));
        assert!(is_spent(&env, &h2));
    }

    // ─── spent_at ─────────────────────────────────────────────────────────

    #[test]
    fn spent_at_returns_none_if_not_spent() {
        let env = Env::default();
        let hash = make_b32(&env, 0x05);
        assert!(spent_at(&env, &hash).is_none());
    }

    #[test]
    fn spent_at_returns_ledger_sequence() {
        let env = Env::default();
        set_ledger_seq(&env, 1_234);
        let hash = make_b32(&env, 0x06);
        spend(&env, &hash).unwrap();
        assert_eq!(spent_at(&env, &hash), Some(1_234));
    }

    #[test]
    fn spent_at_records_exact_sequence() {
        let env = Env::default();
        set_ledger_seq(&env, 99_999);
        let hash = make_b32(&env, 0x07);
        spend(&env, &hash).unwrap();
        assert_eq!(spent_at(&env, &hash), Some(99_999));
    }

    // ─── extend_nullifier_ttl ─────────────────────────────────────────────

    #[test]
    fn extend_ttl_on_unspent_nullifier_is_noop() {
        let env = Env::default();
        let hash = make_b32(&env, 0x30);
        // Should not panic on an unspent nullifier
        extend_nullifier_ttl(&env, &hash);
        assert!(!is_spent(&env, &hash));
    }

    #[test]
    fn extend_ttl_on_spent_nullifier_preserves_entry() {
        let env = Env::default();
        set_ledger_seq(&env, 400);
        let hash = make_b32(&env, 0x31);
        spend(&env, &hash).unwrap();

        // Extending TTL must not remove or corrupt the entry
        extend_nullifier_ttl(&env, &hash);
        assert!(is_spent(&env, &hash));
        assert_eq!(spent_at(&env, &hash), Some(400));
    }

    // ─── batch_extend_ttl ─────────────────────────────────────────────────

    #[test]
    fn batch_extend_mixed_spent_unspent() {
        let env = Env::default();
        set_ledger_seq(&env, 500);

        let h_spent = make_b32(&env, 0x40);
        let h_unspent = make_b32(&env, 0x41);
        spend(&env, &h_spent).unwrap();

        let mut hashes = soroban_sdk::Vec::new(&env);
        hashes.push_back(h_spent.clone());
        hashes.push_back(h_unspent.clone());

        // Should not panic on a mix of spent and unspent
        batch_extend_ttl(&env, &hashes);

        // State unchanged
        assert!(is_spent(&env, &h_spent));
        assert!(!is_spent(&env, &h_unspent));
    }

    // ─── Invariant Tests ──────────────────────────────────────────────────

    #[test]
    fn nullifier_key_is_namespaced() {
        // Spending one hash must not affect a different hash
        let env = Env::default();
        set_ledger_seq(&env, 1);
        let h1 = make_b32(&env, 0x50);
        let h2 = make_b32(&env, 0x51);

        spend(&env, &h1).unwrap();
        assert!(!is_spent(&env, &h2));
    }

    #[test]
    fn ttl_constants_are_sane() {
        // Should cover at least 1 year (3 153 600 ledgers at 5 s/ledger)
        assert!(TTL_EXTEND_TO >= 3_153_600);
        // Should be below the Protocol 21 max TTL (avoid rejection)
        assert!(TTL_EXTEND_TO <= 18_921_960);
        // Threshold must be <= extend target
        assert!(TTL_EXTEND_THRESHOLD <= TTL_EXTEND_TO);
    }
}
