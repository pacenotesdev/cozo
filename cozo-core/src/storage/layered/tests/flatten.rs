/*
 * Flatten (spec §10.8): net effect through a view, then dedupe, collision and tombstone
 * liveness against the destination.
 */

use miette::Result;

use super::{base, frontier, Fixture};
use crate::storage::layered::{LayerRef, Stack};

/// A branch over the base at the current sequence, plus the whole-layer view of it.
fn branch(f: &Fixture, name: &str) -> Result<(Stack, Stack)> {
    f.db.create_layer(name)?;
    let fork = f.seq();
    let stack = vec![LayerRef::new(name), LayerRef::bounded("default", fork)];
    let view = vec![LayerRef::new(name)];
    Ok((stack, view))
}

/// F1. A record created and retracted inside the window contributes nothing: its net effect is
/// a tombstone, and a tombstone for a record the destination never saw is dropped.
#[test]
fn a_record_born_and_died_inside_the_view_contributes_nothing() -> Result<()> {
    let f = Fixture::new()?;
    let (work, view) = branch(&f, "work")?;
    f.assert_rec(&work, "ephemeral", "v1")?;
    f.retract_rec(&work, "ephemeral")?;

    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 0);
    assert_eq!(stats.tombstones_dropped, 1);
    assert_eq!(f.live(&base(), None)?, frontier(&[]));
    Ok(())
}

/// F2. A retraction stamped above the view's ceiling is not in the view, so the live row is
/// what gets written: a view cannot see its own future.
#[test]
fn a_retraction_above_the_ceiling_is_not_in_the_view() -> Result<()> {
    let f = Fixture::new()?;
    let (work, _) = branch(&f, "work")?;
    f.assert_rec(&work, "k", "v1")?;
    let head = f.seq();
    f.retract_rec(&work, "k")?;

    let view: Stack = vec![LayerRef::bounded("work", head)];
    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 1);
    assert_eq!(f.live(&base(), None)?, frontier(&[("k", "v1")]));
    Ok(())
}

/// F3. A record created below the floor and retracted inside the window nets to a tombstone,
/// which is copied exactly when the destination holds the record live.
#[test]
fn a_net_tombstone_is_copied_only_where_it_bites() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "k", "v1")?;
    let (work, view) = branch(&f, "work")?;
    f.retract_rec(&work, "k")?;

    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 1);
    assert_eq!(stats.tombstones_dropped, 0);
    assert_eq!(f.live(&base(), None)?, frontier(&[]));
    Ok(())
}

/// F5. A record the destination already holds live, cherry-picked again: skipped, counted, and
/// the destination is unchanged.
#[test]
fn an_identical_record_dedupes() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "k", "v1")?;
    let (work, view) = branch(&f, "work")?;
    f.assert_rec(&work, "k", "v1")?;

    let before = f.versions(&base())?;
    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 0);
    assert_eq!(stats.rows_deduped, 1);
    assert_eq!(f.versions(&base())?, before);
    Ok(())
}

/// F7. The view's net assertion over a key the destination has tombstoned: the record returns.
#[test]
fn a_re_introduction_revives_a_retracted_record() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "k", "v1")?;
    let (work, view) = branch(&f, "work")?;
    // The base retracts it after the fork; the branch, which cannot see that, still holds it.
    f.retract_rec(&base(), "k")?;
    f.assert_rec(&work, "k", "v1")?;
    assert_eq!(f.live(&base(), None)?, frontier(&[]));

    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 1);
    assert_eq!(f.live(&base(), None)?, frontier(&[("k", "v1")]));
    Ok(())
}

/// F8. The liveness check reads the destination *stack*, not its top layer: a tombstone whose
/// key is live further down the destination's lineage is still copied.
#[test]
fn liveness_is_checked_through_the_whole_destination_stack() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "k", "v1")?;
    let fork = f.seq();

    // `dst` is a branch whose own layer is empty: `k` lives one layer down.
    f.db.create_layer("dst")?;
    let dst: Stack = vec![LayerRef::new("dst"), LayerRef::bounded("default", fork)];

    f.db.create_layer("src")?;
    let src_stack: Stack = vec![LayerRef::new("src"), LayerRef::bounded("default", fork)];
    f.retract_rec(&src_stack, "k")?;

    let stats = f.db.flatten(&vec![LayerRef::new("src")], &dst, true)?;
    assert_eq!(stats.rows_copied, 1);
    assert_eq!(stats.tombstones_dropped, 0);
    assert_eq!(f.live(&dst, None)?, frontier(&[]));
    // The base still holds it: the tombstone landed in `dst`'s own layer.
    assert_eq!(f.live(&base(), None)?, frontier(&[("k", "v1")]));
    Ok(())
}

/// F9. Both sides deleted the same record: the second tombstone bites nothing and is dropped.
#[test]
fn a_second_tombstone_is_dropped() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "k", "v1")?;
    let (work, view) = branch(&f, "work")?;
    f.retract_rec(&work, "k")?;
    f.retract_rec(&base(), "k")?;

    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 0);
    assert_eq!(stats.tombstones_dropped, 1);
    assert_eq!(f.live(&base(), None)?, frontier(&[]));
    Ok(())
}

/// F10. With restamping, the destination's history stays true: a read at a sequence captured
/// before the flatten shows nothing the flatten brought in.
#[test]
fn restamping_keeps_the_destination_history_true() -> Result<()> {
    let f = Fixture::new()?;
    let (work, view) = branch(&f, "work")?;
    let authored = f.assert_rec(&work, "k", "v1")?;
    let before_flatten = f.seq();

    let stats = f.db.flatten(&view, &base(), true)?;
    let (lo, hi) = stats.seq_range.expect("restamping assigns a sequence range");
    assert!(lo > before_flatten && hi >= lo);

    assert_eq!(f.live(&base(), Some(before_flatten))?, frontier(&[]));
    assert_eq!(f.live(&base(), None)?, frontier(&[("k", "v1")]));

    // The row is in the base at the flatten's sequence, not the one it was authored at.
    let stamps: Vec<_> = f.versions(&base())?.into_iter().map(|v| v.seq).collect();
    assert_eq!(stamps, vec![lo]);
    assert!(lo != authored);
    Ok(())
}

/// F11. Without restamping, rows keep their authoring stamps and interleave into the
/// destination's history where they were written.
#[test]
fn without_restamping_the_authoring_stamps_survive() -> Result<()> {
    let f = Fixture::new()?;
    let (work, view) = branch(&f, "work")?;
    let authored = f.assert_rec(&work, "k", "v1")?;

    let stats = f.db.flatten(&view, &base(), false)?;
    assert_eq!(stats.seq_range, None);

    let stamps: Vec<_> = f.versions(&base())?.into_iter().map(|v| v.seq).collect();
    assert_eq!(stamps, vec![authored]);
    // A read of the base as of the authoring sequence now shows the row — which is exactly the
    // historical inaccuracy restamping exists to avoid.
    assert_eq!(f.live(&base(), Some(authored))?, frontier(&[("k", "v1")]));
    Ok(())
}

/// F12. The identical flatten run twice copies nothing the second time. This is also the
/// crash-recovery story: recovery is rerunning it.
#[test]
fn flatten_is_idempotent() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "kept", "v0")?;
    let (work, view) = branch(&f, "work")?;
    f.assert_rec(&work, "added", "v1")?;
    f.retract_rec(&work, "kept")?;

    let first = f.db.flatten(&view, &base(), true)?;
    assert_eq!(first.rows_copied, 2);
    let after_first = f.live(&base(), None)?;

    let second = f.db.flatten(&view, &base(), true)?;
    assert_eq!(second.rows_copied, 0);
    assert_eq!(second.rows_deduped, 1);
    assert_eq!(second.tombstones_dropped, 1);
    assert_eq!(f.live(&base(), None)?, after_first);
    Ok(())
}

/// F13. An empty view writes nothing and leaves the destination alone.
#[test]
fn an_empty_view_is_a_no_op() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "k", "v1")?;
    let (_work, view) = branch(&f, "work")?;

    let before = f.versions(&base())?;
    let stats = f.db.flatten(&view, &base(), true)?;
    assert_eq!(stats.rows_copied, 0);
    assert_eq!(stats.rows_deduped, 0);
    assert_eq!(stats.tombstones_dropped, 0);
    assert_eq!(f.versions(&base())?, before);
    Ok(())
}

/// F16. Flattening into a destination whose top layer already holds unmerged work — a
/// cherry-pick into an active branch — leaves that work alone, and checks against the whole
/// destination stack.
#[test]
fn flattening_into_a_dirty_head_leaves_its_work_alone() -> Result<()> {
    let f = Fixture::new()?;
    f.assert_rec(&base(), "root", "v0")?;
    let fork = f.seq();

    f.db.create_layer("target")?;
    let target: Stack = vec![LayerRef::new("target"), LayerRef::bounded("default", fork)];
    f.assert_rec(&target, "target-work", "v1")?;

    f.db.create_layer("source")?;
    let source: Stack = vec![LayerRef::new("source"), LayerRef::bounded("default", fork)];
    f.assert_rec(&source, "picked", "v2")?;

    let stats = f.db.flatten(&vec![LayerRef::new("source")], &target, true)?;
    assert_eq!(stats.rows_copied, 1);
    assert_eq!(
        f.live(&target, None)?,
        frontier(&[("root", "v0"), ("target-work", "v1"), ("picked", "v2")])
    );
    Ok(())
}
