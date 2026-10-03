/*
 *  Copyright 2026, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */

//! Vector-index measurements that are not per-iteration timings: recall against `ef`, recall
//! spread between builds, filtered recall, build scaling, and concurrent query throughput. Each
//! prints a table. Timings are in `hnsw.rs`.
//!
//! Dataset size, shape and engine come from the environment; see `support/hnsw.rs`.

use cozo::DbInstance;
use std::collections::HashSet;
use std::time::Instant;

#[path = "support/hnsw.rs"]
mod hnsw;
#[path = "support/scenario.rs"]
mod scenario;
use hnsw::*;

fn main() {
    scenario::run(&[
        ("recall_vs_ef", report_recall_vs_ef),
        ("recall_vs_ef_clustered", report_recall_vs_ef_clustered),
        ("recall_spread", report_recall_spread),
        ("filtered_recall", report_filtered_recall),
        ("build_scaling", report_build_scaling),
        ("query_throughput", report_query_throughput),
    ]);
}

/// What each `ef` actually buys. Run alongside the `search_k10_ef*` timings: a change that
/// makes the search faster at unchanged recall is an improvement, and one that makes it faster
/// by looking at less of the graph is not.
fn report_recall_vs_ef() {
    let db: &DbInstance = &L2_DB;
    println!(
        "\nrecall@10 vs ef  (n={}, dim={}, m={}, ef_construction={}, {} queries)",
        n_vectors(),
        dim(),
        m(),
        ef_construction(),
        QUERIES.len()
    );
    for ef in [16usize, 32, 64, 128, 256] {
        let started = Instant::now();
        let recall = recall_at(db, &UNIFORM, &QUERIES, 10, ef);
        let per_query = started.elapsed() / QUERIES.len() as u32;
        println!("  ef={ef:<4} recall={recall:.4}  {per_query:?}/query (incl. ground truth)");
    }
}

/// The same curve on clustered data. Real corpora cluster, and a graph built over clusters
/// behaves differently from one built over noise.
fn report_recall_vs_ef_clustered() {
    let db: &DbInstance = &CLUSTERED_DB;
    println!("\nrecall@10 vs ef, clustered data");
    for ef in [16usize, 32, 64, 128] {
        let recall = recall_at(db, &CLUSTERED, &QUERIES, 10, ef);
        println!("  ef={ef:<4} recall={recall:.4}");
    }
}

/// How much recall moves between builds of identical data, because each node's level is drawn
/// at random. Anything smaller than this spread is noise, not a result: without it a change of
/// a few points looks like a finding.
fn report_recall_spread() {
    println!("\nrecall@10 at ef=64 across repeated builds of identical data");
    let mut seen: Vec<f64> = vec![];
    let data = &UNIFORM[..build_n()];
    for round in 0..5 {
        let db = load(data);
        create_index(&db, "L2");
        let recall = recall_at(&db, data, &QUERIES, 10, 64);
        seen.push(recall);
        println!("  build {round}: recall={recall:.4}");
    }
    let lo = seen.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = seen.iter().sum::<f64>() / seen.len() as f64;
    println!("  spread: {lo:.4}..{hi:.4}  mean={mean:.4}  width={:.4}", hi - lo);
}

/// Whether a selective predicate still returns `k`. The search evaluates it during the walk,
/// so a filter should cost time rather than results; a shortfall here means the walk is
/// stopping while the result set is still short.
fn report_filtered_recall() {
    let db: &DbInstance = &L2_DB;
    println!("\nfilter: returned rows against the k asked for (k=10)");
    for (label, modulus) in [("1 in 2", 2i64), ("1 in 8", 8), ("1 in 64", 64), ("1 in 256", 256)] {
        let eligible = (n_vectors() as i64 + modulus - 1) / modulus;
        for ef in [16usize, 64] {
            let mut returned = 0usize;
            let mut exact = 0usize;
            for q in QUERIES.iter() {
                let got = knn(db, q, 10, ef, &format!(", filter: id % {modulus} == 0"));
                returned += got.len();
                // Ground truth restricted to the eligible rows, which is what the filter means.
                let mut scored: Vec<(usize, f32)> = UNIFORM
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i as i64 % modulus == 0)
                    .map(|(i, v)| {
                        (i, v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>())
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                let truth: HashSet<i64> =
                    scored[..10.min(scored.len())].iter().map(|(i, _)| *i as i64).collect();
                exact += got.into_iter().filter(|id| truth.contains(id)).count();
            }
            let wanted = QUERIES.len() * 10;
            println!(
                "  {label:<9} ({eligible} eligible) ef={ef:<4} returned={:.3} of k   recall={:.4}",
                returned as f64 / wanted as f64,
                exact as f64 / wanted as f64
            );
        }
    }
}

/// Build cost against index size, so the shape of the curve is visible rather than a single
/// point. Bounded by `COZO_HNSW_N`; the sweep stops there.
fn report_build_scaling() {
    println!("\nbuild time against index size (dim={}, m={})", dim(), m());
    let mut size = 250usize.min(build_n());
    while size <= build_n() {
        let subset: Vec<Vec<f32>> = UNIFORM[..size].to_vec();
        let db = load(&subset);
        let started = Instant::now();
        create_index(&db, "L2");
        let elapsed = started.elapsed();
        println!(
            "  n={size:<8} {elapsed:?}  ({:?}/vector)",
            elapsed / size as u32
        );
        size *= 2;
    }
}

/// Concurrent query throughput, which is the number that matters for a served index: searches
/// are independent and share an immutable graph, so this is the surface under load.
///
/// Reported against thread count so the scaling is visible rather than implied. Perfect
/// scaling would be `threads x` the single-threaded rate; the gap is contention, and on this
/// engine the candidates are the storage transaction taken per query and the per-neighbour
/// point lookups into the base relation.
///
/// Latency is reported alongside, because the two move in opposite directions under load: a
/// saturated pool can raise throughput while every individual query gets slower.
fn report_query_throughput() {
    use rayon::prelude::*;

    let db: &DbInstance = &L2_DB;
    let total = env_usize("COZO_HNSW_THROUGHPUT_QUERIES", 4000);
    println!(
        "\nquery throughput, k=10 ef=64 (n={}, dim={}, {total} queries per point)",
        n_vectors(),
        dim()
    );
    let mut single: Option<f64> = None;
    for threads in [1usize, 2, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        // Warm the pool so thread spawn is not inside the measurement.
        pool.install(|| (0..threads).into_par_iter().for_each(|_| {}));
        let started = Instant::now();
        pool.install(|| {
            (0..total).into_par_iter().for_each(|i| {
                let q = &QUERIES[i % QUERIES.len()];
                let got = knn(db, q, 10, 64, "");
                assert!(!got.is_empty(), "a query returned nothing");
            });
        });
        let elapsed = started.elapsed();
        let qps = total as f64 / elapsed.as_secs_f64();
        let base = *single.get_or_insert(qps);
        println!(
            "  threads={threads:<3} {qps:>9.0} q/s   {:>9?}/query   scaling={:.2}x of ideal",
            elapsed / total as u32,
            qps / (base * threads as f64)
        );
    }
}
