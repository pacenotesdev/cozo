/*
 * Flatten: materialize the net effect of a windowed view into a stack's top layer (spec §5).
 *
 * One primitive covers merge-down, changeset extraction and cherry-pick; they differ only in
 * the windows of the source view.
 */

use miette::{bail, miette, IntoDiagnostic, Result, WrapErr};

use crate::data::memcmp::{restamp_tail_validity, tail_validity, validity_version_range};
use crate::storage::layered::iter::{LayeredTxn, StackMerge};
use crate::storage::layered::catalog::{catalog_relations, RelInfo};
use crate::storage::layered::tx::{bind_layers, key_state, BoundLayer};
use crate::storage::layered::{LayeredStorage, Seq, Stack, DEFAULT_LAYER};
use crate::Db;

/// What a flatten did. A consumer recording a merge needs to know which sequences the copied
/// rows occupy, hence `seq_range`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FlattenStats {
    /// Rows written into the destination's top layer.
    pub rows_copied: u64,
    /// Bytes of key and value written.
    pub bytes_copied: u64,
    /// Identical re-introductions of a record the destination already holds live.
    pub rows_deduped: u64,
    /// Retractions that bit nothing in the destination and were therefore dropped.
    pub tombstones_dropped: u64,
    /// The sequences the copied rows occupy. `None` exactly when `restamp` was false.
    pub seq_range: Option<(Seq, Seq)>,
}

/// The net effect of the view on one key.
struct NetRow {
    /// The key as the view holds it, validity included.
    key: Vec<u8>,
    val: Vec<u8>,
    assertive: bool,
}

/// The rows a view resolves to for one relation: for each key, its newest visible version,
/// retraction or not.
///
/// Shadowing and retraction resolve *within the view first* — a record created and retracted
/// inside the window contributes its tombstone, and only that.
fn net_effect(
    txn: &LayeredTxn<'_>,
    layers: &[BoundLayer<'_>],
    rel: &RelInfo,
    into: &mut Vec<NetRow>,
) -> Result<()> {
    let lower = rel.id.raw_encode().to_vec();
    let upper = rel.id.next().raw_encode().to_vec();
    let iters = layers
        .iter()
        .map(|l| (txn.raw_iterator_cf(&l.cf), l.window))
        .collect();
    let mut merge = StackMerge::new(iters, Some(upper));
    merge.seek(&lower)?;

    let mut last_identity: Option<Vec<u8>> = None;
    while let Some(row) = merge.next_kv() {
        let (key, val) = row?;
        let Some(vld) = tail_validity(&key) else {
            bail!(
                "relation '{}' has no validity but holds rows in a view being flattened; \
                 a relation without one cannot express a cross-layer retraction",
                rel.name
            );
        };
        let (identity, _, _) = validity_version_range(&key);
        if last_identity.as_deref() == Some(identity) {
            // An older version of a key we have already resolved.
            continue;
        }
        last_identity = Some(identity.to_vec());
        into.push(NetRow {
            key,
            val,
            assertive: vld.is_assert.0,
        });
    }
    Ok(())
}

impl Db<LayeredStorage> {
    /// Materialize the net effect of `src` into `dst`'s top layer.
    ///
    /// `src` is any windowed sub-stack: an unwindowed single layer is a merge-down, a windowed
    /// one a cherry-pick. `dst`'s lower layers are read for the liveness and identity checks
    /// below and never written.
    ///
    /// This specifies no atomicity against concurrent writers on either stack — the caller must
    /// quiesce. The intended pattern, flattening sealed layers into a single-writer head, makes
    /// that free.
    pub fn flatten(&self, src: &Stack, dst: &Stack, restamp: bool) -> Result<FlattenStats> {
        let src_spec = self.db.resolve(src)?;
        let dst_spec = self.db.resolve(dst)?;
        let inner = &*self.db.inner;
        let txn = inner.db.transaction();

        let src_layers = bind_layers(inner, &src_spec)?;
        let dst_layers = bind_layers(inner, &dst_spec)?;
        let catalog = inner
            .db
            .cf_handle(DEFAULT_LAYER)
            .ok_or_else(|| miette!("the default layer is missing"))?;
        let top = dst_layers[0].cf.clone();

        let relations = catalog_relations(&txn, &catalog)?;

        // 1. What the view resolves to.
        let mut net = vec![];
        for rel in relations.values() {
            if rel.is_index {
                // Copying index rows would splice two independently evolved graphs into
                // something structurally invalid, whose only symptom is silent recall loss.
                // The destination's indexes are dropped and rebuilt instead (spec §5).
                continue;
            }
            if !rel.stackable {
                // A relation with no validity has no retraction mechanism and no stamp, so
                // there is no net effect to compute. Leaving its rows behind silently would be
                // the worst outcome, so check that there are none rather than assume it.
                if holds_rows(&txn, &src_layers[0], rel)? {
                    bail!(
                        "cannot flatten: relation '{}' has no validity column, and the source \
                         layer holds rows for it",
                        rel.name
                    );
                }
                continue;
            }
            net_effect(&txn, &src_layers, rel, &mut net)?;
        }

        // 2. Decide each row against the destination, before writing anything: a flatten that
        //    fails must leave `dst` byte-identical (spec §10.8 F6).
        let mut stats = FlattenStats::default();
        let mut to_write: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        for row in net.iter() {
            let state = key_state(&txn, &dst_layers, &row.key)?;
            if row.assertive {
                match state.asserted {
                    Some(ref existing) if *existing != row.val => {
                        // Under value immutability the only reachable collision is an
                        // identical re-introduction; anything else is an invariant breach.
                        bail!(
                            "flatten aborted: the destination already holds a different value \
                             under a key the view asserts"
                        );
                    }
                    Some(_) if state.live => {
                        // Already there, identical: the same record cherry-picked into a
                        // lineage that already carries it.
                        stats.rows_deduped += 1;
                        continue;
                    }
                    _ => to_write.push((row.key.clone(), row.val.clone())),
                }
            } else if state.live {
                to_write.push((row.key.clone(), row.val.clone()));
            } else {
                // A tombstone for a record this lineage never saw. Dropping it is what keeps
                // every flatten simpler than the union of its inputs.
                stats.tombstones_dropped += 1;
            }
        }

        // 3. Write, under the lock that orders commits (spec §4).
        let _ordered = inner
            .commit_lock
            .lock()
            .map_err(|_| miette!("commit lock poisoned"))?;
        let seq = inner.next_stamp();
        for (key, val) in to_write {
            let mut key = key;
            if restamp {
                // Without this, a later time-travel read of the destination would report the
                // rows as present at sequences they were not (spec §5).
                restamp_tail_validity(&mut key, seq);
            }
            stats.bytes_copied += (key.len() + val.len()) as u64;
            stats.rows_copied += 1;
            txn.put_cf(&top, &key, &val)
                .into_diagnostic()
                .wrap_err("failed to write a flattened row")?;
        }
        if restamp {
            // A flatten is one transaction, so it is one point in commit order.
            stats.seq_range = Some((seq, seq));
        }
        txn.commit()
            .into_diagnostic()
            .wrap_err("failed to commit the flatten")?;
        Ok(stats)
    }
}

/// Whether one layer holds any row of a relation, window included.
fn holds_rows(txn: &LayeredTxn<'_>, layer: &BoundLayer<'_>, rel: &RelInfo) -> Result<bool> {
    let lower = rel.id.raw_encode().to_vec();
    let upper = rel.id.next().raw_encode().to_vec();
    let mut it = txn.raw_iterator_cf(&layer.cf);
    it.seek(&lower);
    while let Some(k) = it.key() {
        if k >= upper.as_slice() {
            break;
        }
        if layer.window.admits(k) {
            return Ok(true);
        }
        it.next();
    }
    it.status()
        .into_diagnostic()
        .wrap_err("failed to scan a layer")?;
    Ok(false)
}
