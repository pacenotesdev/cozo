/*
 * Flatten: materialize the net effect of a windowed view into a stack's top layer.
 *
 * One primitive covers merge-down, changeset extraction and cherry-pick; they differ only in
 * the windows of the source view.
 */

use miette::{bail, miette, IntoDiagnostic, Result, WrapErr};

use std::collections::BTreeMap;
use std::ops::ControlFlow;

use crate::data::memcmp::{restamp_tail_validity, tail_validity, validity_version_range};
use crate::data::tuple::decode_tuple_from_key;
use crate::data::value::DataValue;
use crate::runtime::relation::extend_tuple_from_v;
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

/// Which two sides of a flatten disagree about a key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConflictKind {
    /// The source view asserts a value the destination disagrees with.
    Destination,
    /// Two layers of the source view assert different values for one key. Reachable when the
    /// source spans branches that forked from a common base and never saw one another's
    /// writes, so each write was legal where it was made.
    ///
    /// This detects disagreement about a *value*, and only that. It is not a three-way merge:
    /// one branch retracting a key while another asserts it is not reported here, because a
    /// retraction followed by an assertion is also what ordinary undeleting looks like within
    /// a single lineage, and the two are indistinguishable without per-row layer provenance.
    /// Differing values need no provenance — under value immutability a write that could see
    /// the other value would have been refused, so the difference is itself the evidence.
    ///
    /// Merging branches that have diverged means scanning each against their common base and
    /// adjudicating the two results, where which side did what is known by construction.
    Source,
}

/// One key given two different values by two lineages that could not see each other.
///
/// Under value immutability a key's value never changes, so this never arises within one
/// lineage. It arises between them: each branch's window predates the other's write, so both
/// writes were legal where they were made, and neither the destination's version nor the
/// larger stamp is entitled to win. No merge can choose without a policy the storage layer
/// does not have, so it reports and refuses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlattenConflict {
    /// Which two sides disagree.
    pub kind: ConflictKind,
    /// The relation the key belongs to.
    pub relation: String,
    /// The key, decoded. Its last element is the validity the view holds the row at.
    pub key: Vec<DataValue>,
    /// For [`ConflictKind::Destination`], the value the destination already holds. For
    /// [`ConflictKind::Source`], the value carried by the older of the two disagreeing
    /// assertions.
    pub existing: Vec<DataValue>,
    /// The value the view would write: its newest assertion under this key.
    pub incoming: Vec<DataValue>,
}

/// What a flatten *would* do, computed without writing anything.
///
/// This is the same computation [`Db::flatten`] performs before it writes — resolving the
/// view's net effect and deciding each row against the destination's liveness — so a caller
/// that needs to preview a merge, or to report its conflicts, does not have to reproduce it.
///
/// A plan is not a lock. It is subject to the same caveat as the flatten itself: nothing here
/// is atomic against concurrent writers on either stack, so a plan can be made stale by a write
/// that lands between planning and acting.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FlattenPlan {
    /// What the flatten would report, had it run. `seq_range` is always `None`: a plan occupies
    /// no sequence because it commits nothing.
    pub stats: FlattenStats,
    /// Every collision, not merely the first. Empty exactly when the flatten would succeed.
    pub conflicts: Vec<FlattenConflict>,
}

impl FlattenPlan {
    /// Whether the flatten this plans would succeed.
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}


/// A row of a paginated preview, decoded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FlattenItem {
    /// A row the flatten would write into the destination's top layer.
    Copy {
        /// The relation the row belongs to.
        relation: String,
        /// The key, decoded. Its last element is the validity the view holds the row at.
        key: Vec<DataValue>,
        /// The value columns.
        value: Vec<DataValue>,
    },
    /// A record the destination already holds live, identically.
    Dedupe {
        /// The relation the row belongs to.
        relation: String,
        /// The key, decoded.
        key: Vec<DataValue>,
    },
    /// A retraction that bites nothing in the destination.
    TombstoneDropped {
        /// The relation the row belongs to.
        relation: String,
        /// The key, decoded.
        key: Vec<DataValue>,
    },
    /// A key the two lineages gave different values.
    Conflict(FlattenConflict),
}

/// Where a paginated scan left off. Opaque, and meaningful only to the same `(src, dst)` pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlattenCursor(Vec<u8>);

impl FlattenCursor {
    /// The token's bytes, for a caller that needs to carry it across a process boundary.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    /// Rebuild a cursor from [`FlattenCursor::as_bytes`].
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        FlattenCursor(bytes)
    }
}

/// One page of a preview.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlattenPage {
    /// The items in this page, in key order.
    pub items: Vec<FlattenItem>,
    /// Where to resume, or `None` at the end of the scan.
    pub next: Option<FlattenCursor>,
}

/// What one resolved source row means against the destination, borrowed from the scan.
///
/// Named for what the row *is* rather than for what a flatten does about it, because the scan
/// is not flatten's alone: a diff reads the same stream and classifies it differently — an
/// effective assertion is an addition, an effective retraction a removal.
///
/// Nothing here is owned. A caller that only counts rows allocates nothing; a caller that
/// renders them decides for itself what to build. That asymmetry is the point: conflicts are
/// bounded by construction and worth materializing, rows are not.
enum RowEffect<'r> {
    /// The destination does not already agree: applying this row would change it.
    Effective {
        key: &'r [u8],
        val: &'r [u8],
        /// Whether the row asserts. A diff reads this as added versus removed.
        #[allow(dead_code)]
        asserts: bool,
    },
    /// The destination already holds this record, identically and live.
    Redundant { key: &'r [u8] },
    /// A retraction of a record the destination never held live.
    Inert { key: &'r [u8] },
    /// One key, two values.
    Divergent {
        key: &'r [u8],
        existing: &'r [u8],
        incoming: &'r [u8],
        kind: ConflictKind,
    },
}

/// The O(1) state one identity accumulates while its versions stream past.
struct Resolved {
    identity: Vec<u8>,
    /// The newest visible version: what the view resolves this identity to.
    key: Vec<u8>,
    val: Vec<u8>,
    asserts: bool,
    /// The newest assertive value, which under value immutability every assertion under this
    /// key should carry.
    asserted: Option<Vec<u8>>,
    /// An assertion that carries a different one. Its existence is the invariant breach.
    divergent: Option<Vec<u8>>,
}

/// Decide one resolved identity against the destination and hand it to the visitor.
fn emit<F>(
    txn: &LayeredTxn<'_>,
    dst_layers: &[BoundLayer<'_>],
    rel: &RelInfo,
    done: Resolved,
    visit: &mut F,
) -> Result<ControlFlow<Vec<u8>>>
where
    F: FnMut(&RelInfo, RowEffect<'_>) -> Result<ControlFlow<()>>,
{
    let state = key_state(txn, dst_layers, &done.key)?;
    let effect = if let (Some(other), Some(newest)) = (&done.divergent, &done.asserted) {
        // Two source layers disagree. Reported before the destination is consulted: the view
        // is not self-consistent, so what the destination holds cannot settle it.
        RowEffect::Divergent {
            key: &done.key,
            existing: other,
            incoming: newest,
            kind: ConflictKind::Source,
        }
    } else if done.asserts {
        match state.asserted {
            Some(ref existing) if *existing != done.val => RowEffect::Divergent {
                key: &done.key,
                existing,
                incoming: &done.val,
                kind: ConflictKind::Destination,
            },
            // Already there, identical: the same record cherry-picked into a lineage that
            // already carries it.
            Some(_) if state.live => RowEffect::Redundant { key: &done.key },
            _ => RowEffect::Effective {
                key: &done.key,
                val: &done.val,
                asserts: true,
            },
        }
    } else if state.live {
        RowEffect::Effective {
            key: &done.key,
            val: &done.val,
            asserts: false,
        }
    } else {
        // A tombstone for a record this lineage never saw. Dropping it is what keeps every
        // flatten simpler than the union of its inputs.
        RowEffect::Inert { key: &done.key }
    };
    Ok(match visit(rel, effect)? {
        ControlFlow::Break(()) => ControlFlow::Break(done.key),
        ControlFlow::Continue(()) => ControlFlow::Continue(()),
    })
}

/// Resolve the source view and decide each row against the destination, one row at a time.
///
/// This is the whole of a flatten except the writing. `visit` sees every row in key order and
/// may stop early; the returned key is where a later call should resume from, and is `None`
/// when the scan ran to the end.
///
/// Resolution happens *within the view first*: a record created and retracted inside the
/// window contributes its tombstone and only that, which is why the merge is drained per
/// identity rather than per row.
fn scan<F>(
    txn: &LayeredTxn<'_>,
    src_layers: &[BoundLayer<'_>],
    dst_layers: &[BoundLayer<'_>],
    relations: &BTreeMap<u64, RelInfo>,
    after: Option<&[u8]>,
    mut visit: F,
) -> Result<Option<Vec<u8>>>
where
    F: FnMut(&RelInfo, RowEffect<'_>) -> Result<ControlFlow<()>>,
{
    for rel in relations.values() {
        if rel.is_index {
            // Copying index rows would splice two independently evolved graphs into something
            // structurally invalid, whose only symptom is silent recall loss. The destination's
            // indexes are dropped and rebuilt instead.
            continue;
        }
        let lower = rel.id.raw_encode().to_vec();
        let upper = rel.id.next().raw_encode().to_vec();
        if let Some(after) = after {
            if after >= upper.as_slice() {
                // A relation an earlier page already finished.
                continue;
            }
        }
        if !rel.stackable {
            // A relation with no validity has no retraction mechanism and no stamp, so there is
            // no net effect to compute. Leaving its rows behind silently would be the worst
            // outcome, so check that there are none rather than assume it.
            if holds_rows(txn, &src_layers[0], rel)? {
                bail!(
                    "cannot flatten: relation '{}' has no validity column, and the source \
                     layer holds rows for it",
                    rel.name
                );
            }
            continue;
        }

        // Resume past every version of the last key reported, not merely past that key: older
        // versions of one identity sort *after* the newest, so seeking to the key itself would
        // re-resolve the identity to a stale version.
        let start = match after {
            Some(after) if after > lower.as_slice() => validity_version_range(after).2,
            _ => lower.clone(),
        };

        let iters = src_layers
            .iter()
            .map(|l| (txn.raw_iterator_cf(&l.cf), l.window))
            .collect();
        let mut merge = StackMerge::new(iters, Some(upper.clone()));
        merge.seek(&start)?;

        // One identity at a time, and only O(1) of it: the newest version, the newest
        // assertive value, and one assertion that disagrees with it. Never the chain — a
        // record asserted and retracted many times has a long one.
        let mut cur: Option<Resolved> = None;
        while let Some(row) = merge.next_kv() {
            let (key, val) = row?;
            let Some(vld) = tail_validity(&key) else {
                bail!(
                    "relation '{}' has no validity but holds rows in a view being flattened; \
                     a relation without one cannot express a cross-layer retraction",
                    rel.name
                );
            };
            let asserts = vld.is_assert.0;
            let (identity, _, _) = validity_version_range(&key);

            match cur.as_mut() {
                // An older version of the identity being resolved. It does not change what the
                // view resolves to, but if it asserts a *different* value than the newest
                // assertion then two layers of the source disagree, and stamp order is not
                // entitled to pick between them.
                Some(c) if c.identity == identity => {
                    if asserts {
                        match &c.asserted {
                            None => c.asserted = Some(val),
                            Some(newest) if *newest != val && c.divergent.is_none() => {
                                c.divergent = Some(val)
                            }
                            Some(_) => {}
                        }
                    }
                }
                _ => {
                    if let Some(done) = cur.take() {
                        if let ControlFlow::Break(at) =
                            emit(txn, dst_layers, rel, done, &mut visit)?
                        {
                            return Ok(Some(at));
                        }
                    }
                    cur = Some(Resolved {
                        identity: identity.to_vec(),
                        asserted: if asserts { Some(val.clone()) } else { None },
                        key,
                        val,
                        asserts,
                        divergent: None,
                    });
                }
            }
        }
        if let Some(done) = cur.take() {
            if let ControlFlow::Break(at) = emit(txn, dst_layers, rel, done, &mut visit)? {
                return Ok(Some(at));
            }
        }
    }
    Ok(None)
}

/// Describe one collision in decoded terms, so a caller never has to parse a stored key.
fn conflict_at(
    rel: &RelInfo,
    key: &[u8],
    existing: &[u8],
    incoming: &[u8],
    kind: ConflictKind,
) -> FlattenConflict {
    FlattenConflict {
        kind,
        relation: rel.name.clone(),
        key: decode_tuple_from_key(key, rel.n_keys),
        existing: decode_values(existing),
        incoming: decode_values(incoming),
    }
}

fn decode_values(val: &[u8]) -> Vec<DataValue> {
    let mut ret = vec![];
    extend_tuple_from_v(&mut ret, val);
    ret
}

/// Everything a scan needs, bound for the life of one call.
///
/// Every field borrows from the open database, never from a sibling, so this is an ordinary
/// struct rather than a self-referential one. That is also why the context is returned instead
/// of being lent to a callback: `Transaction::commit` consumes the transaction, so `flatten`
/// has to be able to move it out.
struct ScanContext<'a> {
    txn: LayeredTxn<'a>,
    src_layers: Vec<BoundLayer<'a>>,
    dst_layers: Vec<BoundLayer<'a>>,
    relations: BTreeMap<u64, RelInfo>,
}

impl Db<LayeredStorage> {
    fn scan_context<'a>(&'a self, src: &Stack, dst: &Stack) -> Result<ScanContext<'a>> {
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
        let relations = catalog_relations(&txn, &catalog)?;
        Ok(ScanContext {
            txn,
            src_layers,
            dst_layers,
            relations,
        })
    }

    /// What [`Db::flatten`] would do, without doing it.
    ///
    /// Every check a flatten makes runs here — the view's net effect, dedupe, tombstone
    /// liveness against the whole destination stack — so a merge can be previewed, and its
    /// conflicts reported in full, without the caller reimplementing any of it.
    ///
    /// Memory is proportional to the number of conflicts, not to the size of the view: rows are
    /// counted as they stream past. Use [`Db::flatten_page`] to see the rows themselves.
    ///
    /// A clean plan is not a promise: see [`FlattenPlan`] on staleness.
    pub fn flatten_plan(&self, src: &Stack, dst: &Stack) -> Result<FlattenPlan> {
        let ctx = self.scan_context(src, dst)?;
        let mut plan = FlattenPlan::default();
        scan(
            &ctx.txn,
            &ctx.src_layers,
            &ctx.dst_layers,
            &ctx.relations,
            None,
            |rel, effect| {
                match effect {
                    RowEffect::Effective { key, val, .. } => {
                        plan.stats.rows_copied += 1;
                        plan.stats.bytes_copied += (key.len() + val.len()) as u64;
                    }
                    RowEffect::Redundant { .. } => plan.stats.rows_deduped += 1,
                    RowEffect::Inert { .. } => plan.stats.tombstones_dropped += 1,
                    RowEffect::Divergent {
                        key,
                        existing,
                        incoming,
                        kind,
                    } => plan
                        .conflicts
                        .push(conflict_at(rel, key, existing, incoming, kind)),
                }
                Ok(ControlFlow::Continue(()))
            },
        )?;
        Ok(plan)
    }

    /// One page of what [`Db::flatten`] would do, decoded.
    ///
    /// Pass `after: None` for the first page and the previous page's `next` thereafter. Each
    /// call is self-contained — it opens a transaction, reads its page and closes — so no
    /// snapshot is held while a caller decides what to do with the rows.
    ///
    /// Consecutive pages are therefore *not* one snapshot: a write landing between them is
    /// visible to the later page. Bounding the source stack's layers makes the view immutable
    /// and the paging repeatable.
    pub fn flatten_page(
        &self,
        src: &Stack,
        dst: &Stack,
        after: Option<&FlattenCursor>,
        limit: usize,
    ) -> Result<FlattenPage> {
        let ctx = self.scan_context(src, dst)?;
        let mut items = Vec::with_capacity(limit.min(1024));
        let next = scan(
            &ctx.txn,
            &ctx.src_layers,
            &ctx.dst_layers,
            &ctx.relations,
            after.map(|c| c.0.as_slice()),
            |rel, effect| {
                if limit == 0 {
                    return Ok(ControlFlow::Break(()));
                }
                let relation = rel.name.clone();
                items.push(match effect {
                    RowEffect::Effective { key, val, .. } => FlattenItem::Copy {
                        relation,
                        key: decode_tuple_from_key(key, rel.n_keys),
                        value: decode_values(val),
                    },
                    RowEffect::Redundant { key } => FlattenItem::Dedupe {
                        relation,
                        key: decode_tuple_from_key(key, rel.n_keys),
                    },
                    RowEffect::Inert { key } => FlattenItem::TombstoneDropped {
                        relation,
                        key: decode_tuple_from_key(key, rel.n_keys),
                    },
                    RowEffect::Divergent {
                        key,
                        existing,
                        incoming,
                        kind,
                    } => FlattenItem::Conflict(conflict_at(
                        rel, key, existing, incoming, kind,
                    )),
                });
                Ok(if items.len() >= limit {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                })
            },
        )?;
        Ok(FlattenPage {
            items,
            next: next.map(FlattenCursor),
        })
    }

    /// Materialize the net effect of `src` into `dst`'s top layer.
    ///
    /// `src` is any windowed sub-stack: an unwindowed single layer is a merge-down, a windowed
    /// one a cherry-pick. `dst`'s lower layers are read for the liveness and identity checks
    /// and never written.
    ///
    /// Rows are decided and written as they stream, so memory is proportional to the number of
    /// conflicts rather than to the size of the view. A collision does not stop the scan: every
    /// one is collected, and the transaction is then dropped without committing, which is what
    /// leaves `dst` byte-identical. [`Db::flatten_plan`] reports the same conflicts without
    /// attempting the write at all.
    ///
    /// `dst`'s top layer may not appear in `src`: the scan reads the source while the write
    /// lands in that layer, and a layer that is both would be read as it is written.
    ///
    /// This specifies no atomicity against concurrent writers on either stack — the caller must
    /// quiesce. The intended pattern, flattening sealed layers into a single-writer head, makes
    /// that free. The commit lock is held for the whole call, not merely the writes, because
    /// the stamp is allocated before the first row is decided.
    pub fn flatten(&self, src: &Stack, dst: &Stack, restamp: bool) -> Result<FlattenStats> {
        let ctx = self.scan_context(src, dst)?;
        let top = ctx.dst_layers[0].cf.clone();
        if let Some(clash) = ctx.src_layers.iter().find(|l| l.name == ctx.dst_layers[0].name) {
            bail!(
                "cannot flatten: layer '{}' is both the source of this flatten and the \
                 destination's top layer",
                clash.name
            );
        }

        // The stamp is commit order, so it has to be read where commits are serialized,
        // and the batch has to land before the next commit proceeds.
        let _ordered = self
            .db
            .inner
            .commit_lock
            .lock()
            .map_err(|_| miette!("commit lock poisoned"))?;
        let seq = self.db.inner.next_stamp();

        let mut stats = FlattenStats::default();
        let mut conflicts = vec![];
        let mut failed = None;
        scan(
            &ctx.txn,
            &ctx.src_layers,
            &ctx.dst_layers,
            &ctx.relations,
            None,
            |rel, effect| {
                match effect {
                    RowEffect::Effective { key, val, .. } => {
                        let mut key = key.to_vec();
                        if restamp {
                            // Without this, a later time-travel read of the destination
                            // would report the rows as present at sequences they were not.
                            restamp_tail_validity(&mut key, seq);
                        }
                        stats.rows_copied += 1;
                        stats.bytes_copied += (key.len() + val.len()) as u64;
                        // Writing as we go is safe because each identity is visited once:
                        // no later liveness check can read a row this loop just wrote.
                        if let Err(err) = ctx.txn.put_cf(&top, &key, val) {
                            failed = Some(err);
                            return Ok(ControlFlow::Break(()));
                        }
                    }
                    RowEffect::Redundant { .. } => stats.rows_deduped += 1,
                    RowEffect::Inert { .. } => stats.tombstones_dropped += 1,
                    RowEffect::Divergent {
                        key,
                        existing,
                        incoming,
                        kind,
                    } => conflicts.push(conflict_at(rel, key, existing, incoming, kind)),
                }
                Ok(ControlFlow::Continue(()))
            },
        )?;
        if let Some(err) = failed {
            return Err(err).into_diagnostic().wrap_err("failed to write a flattened row");
        }
        if !conflicts.is_empty() {
            // Dropping the transaction unwritten is what makes a failed flatten a no-op.
            bail!("{}", describe_conflicts(&conflicts));
        }
        if restamp {
            // A flatten is one transaction, so it is one point in commit order.
            stats.seq_range = Some((seq, seq));
        }
        ctx.txn.commit()
            .into_diagnostic()
            .wrap_err("failed to commit the flatten")?;
        Ok(stats)
    }
}

/// Name every collision in an error, capped so that a merge of a large divergent branch does
/// not produce an unreadable message. The full list is what [`Db::flatten_plan`] is for.
fn describe_conflicts(conflicts: &[FlattenConflict]) -> String {
    const SHOWN: usize = 5;
    let mut msg = format!(
        "flatten aborted: {} key(s) collide — the destination already holds a different value \
         under a key the view asserts",
        conflicts.len()
    );
    for c in conflicts.iter().take(SHOWN) {
        msg.push_str(&format!(
            "\n  {} {:?}: destination {:?}, view {:?}",
            c.relation, c.key, c.existing, c.incoming
        ));
    }
    if conflicts.len() > SHOWN {
        msg.push_str(&format!(
            "\n  ... and {} more; `flatten_plan` reports every one",
            conflicts.len() - SHOWN
        ));
    }
    msg
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
