/*
 * Layered storage: windowed iterators and the k-way stack merge.
 */

use miette::{miette, Result};
use rocksdb::{DBRawIteratorWithThreadMode, MultiThreaded, OptimisticTransactionDB, Transaction};

use crate::data::memcmp::tail_validity;
use crate::data::tuple::{check_key_for_validity, key_ends_in_validity, Tuple};
use crate::data::value::ValidityTs;
use crate::runtime::relation::{decode_tuple_from_kv, extend_tuple_from_v};
use crate::storage::layered::Seq;

pub(crate) type LayeredDb = OptimisticTransactionDB<MultiThreaded>;
pub(crate) type LayeredTxn<'a> = Transaction<'a, LayeredDb>;
type RawIter<'a> = DBRawIteratorWithThreadMode<'a, LayeredTxn<'a>>;

/// The visibility window of one layer within one stack.
///
/// `since` is an exclusive floor and `bound` an inclusive ceiling, so the window `(fork, head]`
/// is exactly "everything since the fork" with the fork-point row belonging to the parent.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Window {
    /// Rows stamped at or below this sequence are invisible. `None` is an unbounded floor.
    pub since: Option<Seq>,
    /// Rows stamped above this sequence are invisible. `None` is an unbounded ceiling.
    pub bound: Option<Seq>,
}

impl Window {
    pub(crate) const OPEN: Window = Window {
        since: None,
        bound: None,
    };

    /// Whether a row's key falls inside this window.
    ///
    /// A row with no validity carries no sequence, so no window excludes it: a window selects
    /// by *when*, and such a row has no when. The relations this applies to are index
    /// relations (everything else without a validity is refused by a multi-layer stack
    /// altogether), and it is what makes an index compose through a stack at all:
    /// a bounded base layer must keep contributing its edges, or traversal from a branch sees
    /// only the branch's own nodes and silently loses the rest.
    pub(crate) fn admits(&self, key: &[u8]) -> bool {
        let vld = tail_validity(key);
        debug_assert_eq!(
            vld.is_some(),
            key_ends_in_validity(key),
            "the cheap validity probe disagreed with a full decode"
        );
        match vld {
            None => true,
            Some(vld) => {
                let seq = vld.timestamp.0 .0;
                if let Some(since) = self.since {
                    if seq <= since {
                        return false;
                    }
                }
                if let Some(bound) = self.bound {
                    if seq > bound {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// One layer's contribution to a scan: a raw RocksDB iterator that only ever rests on a row
/// inside the layer's window and below the scan's upper bound.
struct LayerIter<'a> {
    inner: RawIter<'a>,
    window: Window,
    exhausted: bool,
}

impl<'a> LayerIter<'a> {
    fn new(inner: RawIter<'a>, window: Window) -> Self {
        Self {
            inner,
            window,
            exhausted: false,
        }
    }

    fn seek(&mut self, from: &[u8], upper: Option<&[u8]>) -> Result<()> {
        self.exhausted = false;
        self.inner.seek(from);
        self.settle(upper)
    }

    /// Advance past rows that this stack cannot see: those beyond the scan's upper bound, and
    /// those outside the layer's window.
    fn settle(&mut self, upper: Option<&[u8]>) -> Result<()> {
        loop {
            if self.exhausted {
                return Ok(());
            }
            match self.inner.key() {
                None => {
                    self.exhausted = true;
                    return self
                        .inner
                        .status()
                        .map_err(|err| miette!("layer iteration failed: {}", err));
                }
                Some(key) => {
                    if let Some(upper) = upper {
                        if key >= upper {
                            self.exhausted = true;
                            return Ok(());
                        }
                    }
                    if self.window.admits(key) {
                        return Ok(());
                    }
                    self.inner.next();
                }
            }
        }
    }

    fn key(&self) -> Option<&[u8]> {
        if self.exhausted {
            None
        } else {
            self.inner.key()
        }
    }

    fn value(&self) -> Option<&[u8]> {
        if self.exhausted {
            None
        } else {
            self.inner.value()
        }
    }

    fn advance(&mut self, upper: Option<&[u8]>) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }
        self.inner.next();
        self.settle(upper)
    }
}

/// The k-way merge across a stack.
///
/// Entries are emitted in *full-key* order. For a stackable relation the key embeds the
/// validity, which sorts descending, so every version of a relation key arrives newest-first
/// across the whole stack, irrespective of which layer holds it.
///
/// Shadowing between layers is therefore not positional. [`StackSkipIter`] applies the
/// ordinary single-store validity rule to that newest-first stream, so the winner is the
/// newest version, whichever layer holds it. Stack position decides only which copy of a
/// byte-identical key is emitted.
///
/// Layers are compared linearly rather than through a heap: a stack is a short-lived divergence
/// merged down promptly, so depth stays in the single digits and a scan of the array
/// beats maintaining a heap.
pub(crate) struct StackMerge<'a> {
    layers: Vec<LayerIter<'a>>,
    upper: Option<Vec<u8>>,
}

impl<'a> StackMerge<'a> {
    pub(crate) fn new(iters: Vec<(RawIter<'a>, Window)>, upper: Option<Vec<u8>>) -> Self {
        Self {
            layers: iters
                .into_iter()
                .map(|(it, win)| LayerIter::new(it, win))
                .collect(),
            upper,
        }
    }

    pub(crate) fn seek(&mut self, from: &[u8]) -> Result<()> {
        let upper = self.upper.as_deref();
        for layer in self.layers.iter_mut() {
            layer.seek(from, upper)?;
        }
        Ok(())
    }

    /// The index of the layer holding the smallest key, preferring the topmost on a tie.
    fn front(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            let Some(key) = layer.key() else { continue };
            match best {
                None => best = Some(idx),
                // Strictly less, so a tie leaves the slot with the earlier (higher) layer.
                // A tie is byte-identical keys (same relation key *and* same stamp), so this
                // chooses which copy of one row to emit. It does not decide which version of a
                // key wins; the stamp does, in `StackSkipIter`.
                Some(b) => {
                    if key < self.layers[b].key().unwrap() {
                        best = Some(idx)
                    }
                }
            }
        }
        best
    }

    pub(crate) fn next_kv(&mut self) -> Option<Result<(Vec<u8>, Vec<u8>)>> {
        let front = self.front()?;
        let key = self.layers[front].key().unwrap().to_vec();
        let val = self.layers[front].value().unwrap_or_default().to_vec();
        let upper = self.upper.clone();
        // Advance every layer sitting on this exact key, not just the front one: that is what
        // collapses a row present identically in several layers into a single emission.
        for layer in self.layers.iter_mut() {
            if layer.key() == Some(key.as_slice()) {
                if let Err(err) = layer.advance(upper.as_deref()) {
                    return Some(Err(err));
                }
            }
        }
        Some(Ok((key, val)))
    }
}

/// Raw `(key, value)` scan across a stack.
pub(crate) struct StackRawIter<'a> {
    pub(crate) merge: StackMerge<'a>,
    pub(crate) started: bool,
    pub(crate) lower: Vec<u8>,
}

impl<'a> Iterator for StackRawIter<'a> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;
            if let Err(err) = self.merge.seek(&self.lower) {
                return Some(Err(err));
            }
        }
        self.merge.next_kv()
    }
}

/// Decoded-tuple scan across a stack.
pub(crate) struct StackTupleIter<'a> {
    pub(crate) inner: StackRawIter<'a>,
}

impl<'a> Iterator for StackTupleIter<'a> {
    type Item = Result<Tuple>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((k, v))) => Some(Ok(decode_tuple_from_kv(&k, &v, None))),
            Some(Err(err)) => Some(Err(err)),
            None => None,
        }
    }
}

/// Frontier scan across a stack: for each key, the newest version at or before `valid_at`,
/// skipped entirely when that version is a retraction.
///
/// The merge below it already presents each key's versions newest-first across all layers, so
/// cross-layer shadowing and cross-layer retraction both fall out of running the
/// single-store validity rule over the merged stream.
pub(crate) struct StackSkipIter<'a> {
    pub(crate) merge: StackMerge<'a>,
    pub(crate) valid_at: ValidityTs,
    pub(crate) next_bound: Vec<u8>,
}

impl<'a> Iterator for StackSkipIter<'a> {
    type Item = Result<Tuple>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Err(err) = self.merge.seek(&self.next_bound) {
                return Some(Err(err));
            }
            match self.merge.next_kv() {
                None => return None,
                Some(Err(err)) => return Some(Err(err)),
                Some(Ok((k, v))) => {
                    let (ret, nxt_bound) = check_key_for_validity(&k, self.valid_at, None);
                    self.next_bound = nxt_bound;
                    if let Some(mut tup) = ret {
                        extend_tuple_from_v(&mut tup, &v);
                        return Some(Ok(tup));
                    }
                }
            }
        }
    }
}
