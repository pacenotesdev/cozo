/*
 *  Copyright 2026, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */

//! Datasets, stores and queries shared by the `hnsw` benchmarks and the `hnsw_report` tables.
//!
//! The dataset is generated, not loaded, so both run unattended. Size and shape come from the
//! environment:
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `COZO_HNSW_N` | vectors in the searched index | 5000 |
//! | `COZO_HNSW_BUILD_N` | vectors in each build measurement | 2000 |
//! | `COZO_HNSW_DIM` | dimensions | 64 |
//! | `COZO_HNSW_M` | graph degree | 16 |
//! | `COZO_HNSW_EF_C` | build-time candidate width | 50 |
//! | `COZO_HNSW_QUERIES` | probes per recall measurement | 100 |
//! | `COZO_HNSW_THROUGHPUT_QUERIES` | queries per throughput point | 4000 |
//! | `COZO_HNSW_ENGINE` | storage engine, e.g. `mem` or `newrocksdb` | mem |
//!
//! An on-disk engine needs its storage feature, e.g. `--features storage-new-rocksdb`. It is
//! the one that exercises the point lookup per visited neighbour, so the gap between it and
//! `mem` is the cost of a disk-resident graph.

#![allow(dead_code)]

use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};
use lazy_static::lazy_static;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn n_vectors() -> usize {
    env_usize("COZO_HNSW_N", 5000)
}
pub fn dim() -> usize {
    env_usize("COZO_HNSW_DIM", 64)
}
pub fn m() -> usize {
    env_usize("COZO_HNSW_M", 16)
}
pub fn ef_construction() -> usize {
    env_usize("COZO_HNSW_EF_C", 50)
}
pub fn n_queries() -> usize {
    env_usize("COZO_HNSW_QUERIES", 100)
}

/// Index construction is single-threaded on this engine, and the build benchmarks rebuild from
/// scratch on every iteration, so they get their own size. Raising `COZO_HNSW_N` to make the
/// query benchmarks realistic would otherwise make the build ones take hours.
pub fn build_n() -> usize {
    env_usize("COZO_HNSW_BUILD_N", 2000).min(n_vectors())
}

/// A deterministic stream, so a run is reproducible and two runs of different engine code see
/// the identical dataset. Seeding matters more than quality here.
pub struct Lcg(u64);

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
pub enum Shape {
    Uniform,
    Clustered,
}

/// `n` unit vectors of `dim` dimensions. Normalised, so cosine and inner product are
/// comparable to L2 and no distance can come back unmeasurable.
pub fn make_vectors(n: usize, dim: usize, shape: Shape, seed: u64) -> Vec<Vec<f32>> {
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
pub struct Store {
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
pub fn store_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = std::env::temp_dir().join("cozo-hnsw-bench");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    })
}

pub fn new_db() -> Store {
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

pub fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n--- script ---\n{script}"))
}

/// A store holding `vectors` under `pts`, with no index yet. Bulk-loaded, because building the
/// rows through the parser would dominate what is being measured.
pub fn load(vectors: &[Vec<f32>]) -> Store {
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

pub fn create_index(db: &DbInstance, distance: &str) {
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

pub fn vec_literal(v: &[f32]) -> String {
    let body = v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",");
    format!("vec([{body}])")
}

/// The query vector is bound as a parameter rather than formatted into the script. Inlining it
/// means re-parsing `dim` floats of text per query, which on a 64-dimension vector costs more
/// than the search does: the benchmark would be measuring the parser.
pub fn knn(db: &DbInstance, q: &[f32], k: usize, ef: usize, extra: &str) -> Vec<i64> {
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
pub fn ground_truth(vectors: &[Vec<f32>], q: &[f32], k: usize) -> HashSet<i64> {
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
pub fn query_set(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    make_vectors(n_queries(), dim(), Shape::Uniform, 0xBEEF)
        .into_iter()
        .take(n_queries().min(vectors.len()))
        .collect()
}

pub fn recall_at(db: &DbInstance, vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
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
    pub static ref UNIFORM: Vec<Vec<f32>> = make_vectors(n_vectors(), dim(), Shape::Uniform, 0x5EED);
    pub static ref CLUSTERED: Vec<Vec<f32>> =
        make_vectors(n_vectors(), dim(), Shape::Clustered, 0x5EED);
    pub static ref QUERIES: Vec<Vec<f32>> = query_set(&UNIFORM);
    /// One indexed store per distance, reused by every query benchmark.
    pub static ref L2_DB: Store = {
        let db = load(&UNIFORM);
        create_index(&db, "L2");
        db
    };
    pub static ref COSINE_DB: Store = {
        let db = load(&UNIFORM);
        create_index(&db, "Cosine");
        db
    };
    pub static ref IP_DB: Store = {
        let db = load(&UNIFORM);
        create_index(&db, "IP");
        db
    };
    pub static ref CLUSTERED_DB: Store = {
        let db = load(&CLUSTERED);
        create_index(&db, "L2");
        db
    };
}

