/*
 *  Copyright 2026, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */

//! Vector-index benchmarks.
//!
//! Latency alone does not describe an approximate index: a search can always be made faster by
//! looking at less of the graph. Every timing here therefore has a recall figure beside it,
//! reported by the `report_*` functions, which measure rather than time and print their results
//! the way `pokec.rs`'s `qps_*` functions do.
//!
//! Recall varies between builds of the same data, because each node's level is drawn at
//! random. `report_recall_spread` shows how much, which is what says whether a change to the
//! engine moved recall or the dice did.
//!
//! The dataset is generated, not loaded, so these run unattended. Size and shape come from the
//! environment:
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `COZO_HNSW_N` | vectors in the index | 5000 |
//! | `COZO_HNSW_DIM` | dimensions | 64 |
//! | `COZO_HNSW_M` | graph degree | 16 |
//! | `COZO_HNSW_EF_C` | build-time candidate width | 50 |
//! | `COZO_HNSW_QUERIES` | probes per measurement | 100 |
//! | `COZO_HNSW_ENGINE` | `mem` or `rocksdb` | mem |
//!
//! `rocksdb` is the one that exercises the point-lookup path per visited neighbour, so the
//! gap between the two engines is the cost of a disk-resident graph.

#![feature(test)]

extern crate test;

use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};
use lazy_static::lazy_static;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use test::Bencher;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn n_vectors() -> usize {
    env_usize("COZO_HNSW_N", 5000)
}
fn dim() -> usize {
    env_usize("COZO_HNSW_DIM", 64)
}
fn m() -> usize {
    env_usize("COZO_HNSW_M", 16)
}
fn ef_construction() -> usize {
    env_usize("COZO_HNSW_EF_C", 50)
}
fn n_queries() -> usize {
    env_usize("COZO_HNSW_QUERIES", 100)
}

/// Index construction is single-threaded on this engine, and the build benchmarks rebuild from
/// scratch on every iteration, so they get their own size. Raising `COZO_HNSW_N` to make the
/// query benchmarks realistic would otherwise make the build ones take hours.
fn build_n() -> usize {
    env_usize("COZO_HNSW_BUILD_N", 2000).min(n_vectors())
}

/// A deterministic stream, so a run is reproducible and two runs of different engine code see
/// the identical dataset. Seeding matters more than quality here.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32) - 0.5
    }
}

/// How the points are laid out. Real corpora are clustered; uniform is the harder case for a
/// graph index and the easier one to reason about.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    Uniform,
    Clustered,
}

/// `n` unit vectors of `dim` dimensions. Normalised, so cosine and inner product are
/// comparable to L2 and no distance can come back unmeasurable.
fn make_vectors(n: usize, dim: usize, shape: Shape, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    let centroids: Vec<Vec<f32>> = if shape == Shape::Clustered {
        (0..(n / 200).max(4))
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect()
    } else {
        vec![]
    };
    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = match shape {
                Shape::Uniform => (0..dim).map(|_| rng.next_f32()).collect(),
                Shape::Clustered => {
                    let c = &centroids[i % centroids.len()];
                    c.iter().map(|x| x + rng.next_f32() * 0.15).collect()
                }
            };
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::EPSILON);
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// A database together with the temporary directory backing it, for the on-disk engines. The
/// directory has to outlive the `DbInstance`, so the guard is kept here rather than in the
/// constructor's stack frame; dropping the store removes it. Derefs to the database, so callers
/// that only want to run scripts need not know which engine they got.
struct Store {
    db: DbInstance,
    _dir: Option<tempfile::TempDir>,
}

impl std::ops::Deref for Store {
    type Target = DbInstance;

    fn deref(&self) -> &DbInstance {
        &self.db
    }
}

/// Every on-disk store this benchmark opens goes under one directory, which is emptied when the
/// first store is opened. Dropping a `Store` already removes its own directory, but the fixtures
/// below are `lazy_static`, and Rust does not drop statics at exit: without a sweep their stores
/// would accumulate one set per run. Two copies of this benchmark running at once would clear
/// each other's stores, which `cargo bench` does not do.
fn store_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = std::env::temp_dir().join("cozo-hnsw-bench");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    })
}

fn new_db() -> Store {
    match std::env::var("COZO_HNSW_ENGINE").unwrap_or_else(|_| "mem".into()).as_str() {
        "mem" => Store {
            db: DbInstance::new("mem", "", "").unwrap(),
            _dir: None,
        },
        engine => {
            let dir = tempfile::Builder::new().tempdir_in(store_root()).unwrap();
            let path = dir.path().join("db");
            let db = DbInstance::new(engine, path.to_str().unwrap(), "").unwrap();
            Store {
                db,
                _dir: Some(dir),
            }
        }
    }
}

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n--- script ---\n{script}"))
}

/// A store holding `vectors` under `pts`, with no index yet. Bulk-loaded, because building the
/// rows through the parser would dominate what is being measured.
fn load(vectors: &[Vec<f32>]) -> Store {
    let db = new_db();
    run(&db, &format!(":create pts {{id: Int => v: <F32; {}>}}", dim()));
    let rows = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            vec![
                DataValue::from(i as i64),
                DataValue::Vec(Vector::F32(ndarray::arr1(v))),
            ]
        })
        .collect();
    let mut to_import = BTreeMap::new();
    to_import.insert(
        "pts".to_string(),
        NamedRows {
            headers: vec!["id".to_string(), "v".to_string()],
            rows,
            next: None,
        },
    );
    db.import_relations(to_import).unwrap();
    db
}

fn create_index(db: &DbInstance, distance: &str) {
    run(
        db,
        &format!(
            "::hnsw create pts:i {{dim: {}, m: {}, dtype: F32, fields: [v], distance: {distance}, \
             ef_construction: {}}}",
            dim(),
            m(),
            ef_construction()
        ),
    );
}

fn vec_literal(v: &[f32]) -> String {
    let body = v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",");
    format!("vec([{body}])")
}

/// The query vector is bound as a parameter rather than formatted into the script. Inlining it
/// means re-parsing `dim` floats of text per query, which on a 64-dimension vector costs more
/// than the search does: the benchmark would be measuring the parser.
fn knn(db: &DbInstance, q: &[f32], k: usize, ef: usize, extra: &str) -> Vec<i64> {
    let mut params = BTreeMap::new();
    params.insert(
        "q".to_string(),
        DataValue::Vec(Vector::F32(ndarray::arr1(q))),
    );
    db.run_script(
        &format!(
            "?[id, dist] := ~pts:i{{id | query: $q, k: {k}, ef: {ef}, bind_distance: dist{extra}}} \
             :order dist"
        ),
        params,
        ScriptMutability::Immutable,
    )
    .unwrap_or_else(|e| panic!("query failed: {e:?}"))
    .rows
    .iter()
    .map(|r| r[0].get_int().unwrap())
    .collect()
}

/// Exact nearest neighbours by brute force, for recall to be measured against.
fn ground_truth(vectors: &[Vec<f32>], q: &[f32], k: usize) -> HashSet<i64> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored[..k].iter().map(|(i, _)| *i as i64).collect()
}

/// The queries every measurement uses: held-out points, so a probe is never its own answer.
fn query_set(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    make_vectors(n_queries(), dim(), Shape::Uniform, 0xBEEF)
        .into_iter()
        .take(n_queries().min(vectors.len()))
        .collect()
}

fn recall_at(db: &DbInstance, vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let truth = ground_truth(vectors, q, k);
        hits += knn(db, q, k, ef, "")
            .into_iter()
            .filter(|id| truth.contains(id))
            .count();
    }
    hits as f64 / (queries.len() * k) as f64
}

lazy_static! {
    static ref UNIFORM: Vec<Vec<f32>> = make_vectors(n_vectors(), dim(), Shape::Uniform, 0x5EED);
    static ref CLUSTERED: Vec<Vec<f32>> =
        make_vectors(n_vectors(), dim(), Shape::Clustered, 0x5EED);
    static ref QUERIES: Vec<Vec<f32>> = query_set(&UNIFORM);
    /// One indexed store per distance, reused by every query benchmark.
    static ref L2_DB: Store = {
        let db = load(&UNIFORM);
        create_index(&db, "L2");
        db
    };
    static ref COSINE_DB: Store = {
        let db = load(&UNIFORM);
        create_index(&db, "Cosine");
        db
    };
    static ref IP_DB: Store = {
        let db = load(&UNIFORM);
        create_index(&db, "IP");
        db
    };
    static ref CLUSTERED_DB: Store = {
        let db = load(&CLUSTERED);
        create_index(&db, "L2");
        db
    };
}

// ---------------------------------------------------------------- build

#[bench]
fn build_l2(b: &mut Bencher) {
    let data = &UNIFORM[..build_n()];
    b.iter(|| {
        let db = load(data);
        create_index(&db, "L2");
        db
    });
}

#[bench]
fn build_cosine(b: &mut Bencher) {
    let data = &UNIFORM[..build_n()];
    b.iter(|| {
        let db = load(data);
        create_index(&db, "Cosine");
        db
    });
}

#[bench]
fn build_clustered(b: &mut Bencher) {
    let data = &CLUSTERED[..build_n()];
    b.iter(|| {
        let db = load(data);
        create_index(&db, "L2");
        db
    });
}

// ------------------------------------------------------- steady-state write

/// One row inserted into an existing index: the incremental maintenance path, which is what a
/// live system pays per write rather than the bulk build above.
///
/// The index grows as this runs, so the figure depends on how many iterations the harness
/// chose and is comparable only against itself at the same iteration count. Read it as an
/// order of magnitude, not a number.
#[bench]
fn insert_one_into_an_existing_index(b: &mut Bencher) {
    let db = load(&UNIFORM);
    create_index(&db, "L2");
    let extra = make_vectors(2000, dim(), Shape::Uniform, 0xF00D);
    let mut next = 0usize;
    let base = n_vectors() as i64;
    b.iter(|| {
        let id = base + next as i64;
        next += 1;
        run(
            &db,
            &format!(
                "?[id, v] <- [[{id}, {}]] :put pts {{id => v}}",
                vec_literal(&extra[next % extra.len()])
            ),
        );
    });
}

// ---------------------------------------------------------------- search

macro_rules! query_bench {
    ($name:ident, $db:ident, $k:expr, $ef:expr, $extra:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let db: &DbInstance = &$db;
            let mut i = 0usize;
            b.iter(|| {
                let q = &QUERIES[i % QUERIES.len()];
                i += 1;
                knn(db, q, $k, $ef, $extra)
            });
        }
    };
}

// `ef` is the dial that trades latency for recall, so it is swept at fixed `k`. Read these
// against `report_recall_vs_ef`, which gives the recall each one buys.
query_bench!(search_k10_ef16, L2_DB, 10, 16, "");
query_bench!(search_k10_ef32, L2_DB, 10, 32, "");
query_bench!(search_k10_ef64, L2_DB, 10, 64, "");
query_bench!(search_k10_ef128, L2_DB, 10, 128, "");
query_bench!(search_k10_ef256, L2_DB, 10, 256, "");

// `k` at fixed `ef`: how much the result set itself costs.
query_bench!(search_k1_ef64, L2_DB, 1, 64, "");
query_bench!(search_k50_ef64, L2_DB, 50, 64, "");
query_bench!(search_k100_ef128, L2_DB, 100, 128, "");

// The metrics differ in arithmetic per comparison, not in graph shape.
query_bench!(search_cosine_k10_ef64, COSINE_DB, 10, 64, "");
query_bench!(search_inner_product_k10_ef64, IP_DB, 10, 64, "");

// Clustered data: shorter hops, denser neighbourhoods.
query_bench!(search_clustered_k10_ef64, CLUSTERED_DB, 10, 64, "");

// A predicate is evaluated during the walk, and a node it rejects still routes. The selective
// case is the one that makes the walk widen, so it costs more than the unfiltered search
// above by design; `report_filtered_recall` shows what that buys.
query_bench!(search_filter_permissive, L2_DB, 10, 64, ", filter: id % 2 == 0");
query_bench!(search_filter_selective, L2_DB, 10, 64, ", filter: id % 64 == 0");
// Squared L2 between random unit vectors centres near 2.0, so this radius keeps roughly the
// nearer half: tight enough to exercise the cut, loose enough that results come back.
query_bench!(search_radius, L2_DB, 10, 64, ", radius: 1.6");

// ---------------------------------------------------------------- reports
//
// These measure rather than time, and print. They take a `Bencher` only because that is how
// the harness finds them; `pokec.rs`'s `qps_*` functions do the same.

/// What each `ef` actually buys. Run alongside the `search_k10_ef*` timings: a change that
/// makes the search faster at unchanged recall is an improvement, and one that makes it faster
/// by looking at less of the graph is not.
#[bench]
fn report_recall_vs_ef(_b: &mut Bencher) {
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
#[bench]
fn report_recall_vs_ef_clustered(_b: &mut Bencher) {
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
#[bench]
fn report_recall_spread(_b: &mut Bencher) {
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
#[bench]
fn report_filtered_recall(_b: &mut Bencher) {
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
#[bench]
fn report_build_scaling(_b: &mut Bencher) {
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
#[bench]
fn report_query_throughput(_b: &mut Bencher) {
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
