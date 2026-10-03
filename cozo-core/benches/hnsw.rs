/*
 *  Copyright 2026, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */

//! Vector-index benchmarks: build, incremental insert, and search latency.
//!
//! Latency alone does not describe an approximate index: a search can always be made faster by
//! looking at less of the graph. Every timing here therefore has a recall figure beside it in
//! `hnsw_report.rs`, which measures recall, filtered recall, build scaling and concurrent
//! throughput, and prints them as tables.
//!
//! Dataset size, shape and engine come from the environment; see `support/hnsw.rs`.

use cozo::DbInstance;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::cell::OnceCell;

#[path = "support/hnsw.rs"]
mod hnsw;
use hnsw::*;

/// Index construction from scratch. Each iteration loads the rows and builds the index over
/// them; the store is dropped outside the measurement.
fn build(c: &mut Criterion) {
    let mut g = c.benchmark_group("build");
    // An iteration is a whole build, seconds each on an on-disk engine.
    g.sample_size(10);
    let cases: [(&str, &[Vec<f32>], &str); 3] = [
        ("l2", &UNIFORM[..build_n()], "L2"),
        ("cosine", &UNIFORM[..build_n()], "Cosine"),
        ("clustered", &CLUSTERED[..build_n()], "L2"),
    ];
    for (name, data, distance) in cases {
        g.bench_function(name, |b| {
            b.iter_with_large_drop(|| {
                let db = load(data);
                create_index(&db, distance);
                db
            })
        });
    }
    g.finish();
}

/// A row inserted by an iteration, removed again when dropped so that the index is the same
/// size for every iteration. Dropped outside the measurement.
struct RemoveOnDrop<'a> {
    db: &'a DbInstance,
    id: i64,
}

impl Drop for RemoveOnDrop<'_> {
    fn drop(&mut self) {
        run(self.db, &format!("?[id] <- [[{}]] :rm pts {{id}}", self.id));
    }
}

/// One row inserted into an existing index: the incremental maintenance path, which is what a
/// live system pays per write rather than the bulk build above.
fn insert(c: &mut Criterion) {
    // Built on first use, so a run filtered to other benchmarks does not pay for it.
    let fixture = OnceCell::new();
    let base = n_vectors() as i64;
    let mut next = 0usize;
    c.bench_function("insert_one_into_an_existing_index", |b| {
        let (db, extra) = fixture.get_or_init(|| {
            let db = load(&UNIFORM);
            create_index(&db, "L2");
            (db, make_vectors(2000, dim(), Shape::Uniform, 0xF00D))
        });
        b.iter_batched(
            || {
                next += 1;
                let id = base + next as i64;
                let script = format!(
                    "?[id, v] <- [[{id}, {}]] :put pts {{id => v}}",
                    vec_literal(&extra[next % extra.len()])
                );
                (id, script)
            },
            |(id, script)| {
                run(db, &script);
                RemoveOnDrop { db, id }
            },
            BatchSize::SmallInput,
        )
    });
}

/// One query per iteration, cycling through the shared query set.
fn search(c: &mut Criterion) {
    let mut g = c.benchmark_group("search");
    // Stores are named by function rather than by reference, so that only the ones a selected
    // benchmark uses get built.
    let cases: [(&str, fn() -> &'static DbInstance, usize, usize, &str); 14] = [
        // `ef` is the dial that trades latency for recall, so it is swept at fixed `k`. Read
        // these against the recall-against-ef report, which gives the recall each one buys.
        ("k10_ef16", l2_db, 10, 16, ""),
        ("k10_ef32", l2_db, 10, 32, ""),
        ("k10_ef64", l2_db, 10, 64, ""),
        ("k10_ef128", l2_db, 10, 128, ""),
        ("k10_ef256", l2_db, 10, 256, ""),
        // `k` at fixed `ef`: how much the result set itself costs.
        ("k1_ef64", l2_db, 1, 64, ""),
        ("k50_ef64", l2_db, 50, 64, ""),
        ("k100_ef128", l2_db, 100, 128, ""),
        // The metrics differ in arithmetic per comparison, not in graph shape.
        ("cosine_k10_ef64", cosine_db, 10, 64, ""),
        ("inner_product_k10_ef64", ip_db, 10, 64, ""),
        // Clustered data: shorter hops, denser neighbourhoods.
        ("clustered_k10_ef64", clustered_db, 10, 64, ""),
        // A predicate is evaluated during the walk, and a node it rejects still routes. The
        // selective case is the one that makes the walk widen, so it costs more than the
        // unfiltered search by design; the filtered-recall report shows what that buys.
        ("filter_permissive", l2_db, 10, 64, ", filter: id % 2 == 0"),
        ("filter_selective", l2_db, 10, 64, ", filter: id % 64 == 0"),
        // Squared L2 between random unit vectors centres near 2.0, so this radius keeps
        // roughly the nearer half: tight enough to exercise the cut, loose enough that results
        // come back.
        ("radius", l2_db, 10, 64, ", radius: 1.6"),
    ];
    for (name, db, k, ef, extra) in cases {
        let mut i = 0usize;
        g.bench_function(name, |b| {
            let db = db();
            b.iter(|| {
                let q = &QUERIES[i % QUERIES.len()];
                i += 1;
                knn(db, q, k, ef, extra)
            })
        });
    }
    g.finish();
}

fn l2_db() -> &'static DbInstance {
    &L2_DB
}

fn cosine_db() -> &'static DbInstance {
    &COSINE_DB
}

fn ip_db() -> &'static DbInstance {
    &IP_DB
}

fn clustered_db() -> &'static DbInstance {
    &CLUSTERED_DB
}

criterion_group!(benches, build, insert, search);
criterion_main!(benches);
