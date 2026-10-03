/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Latency and throughput of queries over the Pokec social graph. The dataset is loaded from the
//! environment, see `support/pokec.rs`. Load tests that do not fit a per-iteration model are in
//! `pokec_scenarios.rs`.

use std::collections::BTreeMap;
use std::time::{Duratgition, Instant};

use criterion::{criterion_group, criterion_main, BatchSize, Bencher, Criterion, Throughput};
use lazy_static::initialize;
use rand::Rng;
use rayon::prelude::*;

use cozo::{DataValue, ScriptMutability};

#[path = "support/pokec.rs"]
mod pokec;
use pokec::*;

/// Queries per sample in the `throughput` group, from `COZO_BENCH_ITERATIONS`.
fn throughput_batch() -> usize {
    *ITERATIONS
}

/// Queries per sample in the `qps` group, large enough to saturate the pool on point queries.
const QPS_BATCH: usize = 1_000_000;

fn id_param(name: &str, id: usize) -> (String, DataValue) {
    (name.to_string(), DataValue::from(id as i64))
}

/// A user row as it stood before a benchmark wrote to it. Dropping it puts the row back, or
/// removes it if there was none, so every iteration sees the same dataset. Benchmarks drop it
/// outside the timed region.
struct RestoreUser {
    uid: usize,
    old: Option<Vec<DataValue>>,
}

impl RestoreUser {
    fn capture(uid: usize) -> Self {
        let rows = TEST_DB
            .run_script(
                "?[cmpl_pct, gender, age] := *user{uid: $id, cmpl_pct, gender, age}",
                BTreeMap::from([id_param("id", uid)]),
                ScriptMutability::Immutable,
            )
            .unwrap()
            .rows;
        Self {
            uid,
            old: rows.into_iter().next(),
        }
    }
}

impl Drop for RestoreUser {
    fn drop(&mut self) {
        let mut params = BTreeMap::from([id_param("id", self.uid)]);
        let script = match self.old.take() {
            Some(row) => {
                let mut row = row.into_iter();
                for name in ["c", "g", "a"] {
                    params.insert(name.to_string(), row.next().unwrap());
                }
                "?[uid, cmpl_pct, gender, age] <- [[$id, $c, $g, $a]] \
                 :put user {uid => cmpl_pct, gender, age}"
            }
            None => "?[uid] <- [[$id]] :rm user {uid}",
        };
        TEST_DB
            .run_script(script, params, ScriptMutability::Mutable)
            .unwrap();
    }
}

/// Whether an edge existed before a benchmark wrote it. Dropping it removes the edge from both
/// directions if it did not.
struct RestoreEdge {
    fr: usize,
    to: usize,
    existed: bool,
}

impl RestoreEdge {
    fn capture((fr, to): (usize, usize)) -> Self {
        let existed = !TEST_DB
            .run_script(
                "?[fr] := *friends{fr: $i, to: $j}",
                BTreeMap::from([id_param("i", fr), id_param("j", to)]),
                ScriptMutability::Immutable,
            )
            .unwrap()
            .rows
            .is_empty();
        Self { fr, to, existed }
    }
}

impl Drop for RestoreEdge {
    fn drop(&mut self) {
        if self.existed {
            return;
        }
        TEST_DB
            .run_script(
                r#"
            {?[fr, to] <- [[$i, $j]] :rm friends {fr, to}}
            {?[fr, to] <- [[$i, $j]] :rm friends.rev {fr, to}}
            "#,
                BTreeMap::from([id_param("i", self.fr), id_param("j", self.to)]),
                ScriptMutability::Mutable,
            )
            .unwrap();
    }
}

fn prepare_vertex_write() -> RestoreUser {
    RestoreUser::capture(rand::thread_rng().gen_range(1..SIZES.0 * 10))
}

fn prepare_vertex_update() -> RestoreUser {
    RestoreUser::capture(rand::thread_rng().gen_range(1..SIZES.0))
}

fn prepare_edge_write() -> RestoreEdge {
    RestoreEdge::capture(random_edge())
}

fn apply_vertex_write(op: &RestoreUser) {
    single_vertex_write_at(op.uid)
}

fn apply_vertex_update(op: &RestoreUser) {
    single_vertex_update_at(op.uid)
}

fn apply_edge_write(op: &RestoreEdge) {
    single_edge_write_at(op.fr, op.to)
}

fn latency(c: &mut Criterion, group: &str, cases: &[(&str, QueryFn)]) {
    let mut g = c.benchmark_group(group);
    for (name, f) in cases {
        g.bench_function(*name, |b| {
            initialize(&TEST_DB);
            b.iter(f)
        });
    }
    g.finish();
}

/// A write measured one at a time. The key is chosen and the prior state captured before the
/// timer starts, and the write is undone after it stops.
fn write_latency<P>(b: &mut Bencher, prepare: fn() -> P, apply: fn(&P)) {
    initialize(&TEST_DB);
    b.iter_batched(
        prepare,
        |op| {
            apply(&op);
            op
        },
        BatchSize::SmallInput,
    )
}

/// `batch` executions of `f` spread across the rayon pool, timed as a whole. Reported as
/// queries per second through the group's `Throughput`.
fn parallel_reads(b: &mut Bencher, batch: usize, f: QueryFn) {
    initialize(&TEST_DB);
    b.iter_custom(|iters| {
        let start = Instant::now();
        for _ in 0..iters {
            (0..batch).into_par_iter().for_each(|_| f());
        }
        start.elapsed()
    })
}

/// As `parallel_reads`, for writes: only the parallel writes are timed. Choosing keys and
/// capturing prior state happen before, undoing the writes after.
fn parallel_writes<P: Sync>(b: &mut Bencher, batch: usize, prepare: fn() -> P, apply: fn(&P)) {
    initialize(&TEST_DB);
    b.iter_custom(|iters| {
        let mut taken = Duration::ZERO;
        for _ in 0..iters {
            let ops: Vec<P> = (0..batch).map(|_| prepare()).collect();
            let start = Instant::now();
            ops.par_iter().for_each(apply);
            taken += start.elapsed();
            drop(ops);
        }
        taken
    })
}

fn reads(c: &mut Criterion) {
    latency(c, "read", &[("single_vertex", single_vertex_read)]);
}

fn aggregations(c: &mut Criterion) {
    latency(
        c,
        "aggregation",
        &[
            ("group", aggregation_group),
            ("distinct", aggregation_count),
            ("filter", aggregation_filter),
            ("min_max", aggregation_min_max),
        ],
    );
}

fn expansions(c: &mut Criterion) {
    latency(
        c,
        "expansion",
        &[
            ("1", expansion_1_plain),
            ("1_filter", expansion_1_filter),
            ("2", expansion_2_plain),
            ("2_filter", expansion_2_filter),
            ("3", expansion_3_plain),
            ("3_filter", expansion_3_filter),
            ("4", expansion_4_plain),
            ("4_filter", expansion_4_filter),
        ],
    );
}

fn neighbours(c: &mut Criterion) {
    latency(
        c,
        "neighbours_2",
        &[
            ("plain", neighbours_2_plain),
            ("filter", neighbours_2_filter_only),
            ("data", neighbours_2_data_only),
            ("filter_data", neighbours_2_filter_data),
        ],
    );
}

fn patterns(c: &mut Criterion) {
    latency(
        c,
        "pattern",
        &[
            ("cycle", pattern_cycle),
            ("long", pattern_long),
            ("short", pattern_short),
        ],
    );
}

fn writes(c: &mut Criterion) {
    let mut g = c.benchmark_group("write");
    g.bench_function("single_vertex", |b| {
        write_latency(b, prepare_vertex_write, apply_vertex_write)
    });
    g.bench_function("single_edge", |b| {
        write_latency(b, prepare_edge_write, apply_edge_write)
    });
    g.bench_function("single_vertex_update", |b| {
        write_latency(b, prepare_vertex_update, apply_vertex_update)
    });
    g.finish();
}

fn throughput(c: &mut Criterion) {
    let batch = throughput_batch();
    let mut g = c.benchmark_group("throughput");
    // Each sample is a whole parallel batch, so the default sample count would run for a long
    // time without tightening the estimate much.
    g.sample_size(10);
    g.throughput(Throughput::Elements(batch as u64));
    let read_cases: [(&str, QueryFn); 20] = [
        ("expansion_1_plain", expansion_1_plain),
        ("expansion_1_filter", expansion_1_filter),
        ("expansion_2_plain", expansion_2_plain),
        ("expansion_2_filter", expansion_2_filter),
        ("expansion_3_plain", expansion_3_plain),
        ("expansion_3_filter", expansion_3_filter),
        ("expansion_4_plain", expansion_4_plain),
        ("expansion_4_filter", expansion_4_filter),
        ("neighbours_2_plain", neighbours_2_plain),
        ("neighbours_2_filter_only", neighbours_2_filter_only),
        ("neighbours_2_data_only", neighbours_2_data_only),
        ("neighbours_2_filter_data", neighbours_2_filter_data),
        ("pattern_cycle", pattern_cycle),
        ("pattern_long", pattern_long),
        ("pattern_short", pattern_short),
        ("aggregation_group", aggregation_group),
        ("aggregation_count", aggregation_count),
        ("aggregation_filter", aggregation_filter),
        ("aggregation_min_max", aggregation_min_max),
        ("single_vertex_read", single_vertex_read),
    ];
    for (name, f) in read_cases {
        g.bench_function(name, |b| parallel_reads(b, batch, f));
    }
    g.bench_function("single_vertex_write", |b| {
        parallel_writes(b, batch, prepare_vertex_write, apply_vertex_write)
    });
    g.bench_function("single_edge_write", |b| {
        parallel_writes(b, batch, prepare_edge_write, apply_edge_write)
    });
    g.bench_function("single_vertex_update", |b| {
        parallel_writes(b, batch, prepare_vertex_update, apply_vertex_update)
    });
    g.bench_function("pagerank", |b| parallel_reads(b, batch, pagerank));
    g.finish();
}

fn qps(c: &mut Criterion) {
    let mut g = c.benchmark_group("qps");
    g.sample_size(10);
    g.throughput(Throughput::Elements(QPS_BATCH as u64));
    g.bench_function("single_vertex_read", |b| {
        parallel_reads(b, QPS_BATCH, single_vertex_read)
    });
    g.bench_function("single_vertex_write", |b| {
        parallel_writes(b, QPS_BATCH, prepare_vertex_write, apply_vertex_write)
    });
    g.finish();
}

criterion_group!(
    benches,
    reads,
    aggregations,
    expansions,
    neighbours,
    patterns,
    writes,
    throughput,
    qps
);
criterion_main!(benches);
