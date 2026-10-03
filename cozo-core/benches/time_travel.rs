/*
 *  Copyright 2022, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */
use cozo::{DataValue, DbInstance, NamedRows, Validity, ScriptMutability};
use criterion::{criterion_group, criterion_main, BenchmarkId, Bencher, Criterion, Throughput};
use itertools::Itertools;
use lazy_static::{initialize, lazy_static};
use rand::Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

fn insert_data(db: &DbInstance) {
    let insert_plain_time = Instant::now();
    let mut to_import = BTreeMap::new();
    to_import.insert(
        "plain".to_string(),
        NamedRows {
            headers: vec!["k".to_string(), "v".to_string()],
            rows: (0..10000).map(|i| vec![DataValue::from(i as i64), DataValue::from(i as i64)]).collect_vec(),
            next: None,
        },
    );
    db.import_relations(to_import).unwrap();
    dbg!(insert_plain_time.elapsed());

    let insert_tt1_time = Instant::now();
    let mut to_import = BTreeMap::new();
    to_import.insert(
        "tt1".to_string(),
        NamedRows {
            headers: vec!["k".to_string(), "vld".to_string(), "v".to_string()],
            rows: (0..10000)
                .map(|i| vec![
                    DataValue::from(i as i64),
                    DataValue::Validity(Validity::from((0, true))),
                    DataValue::from(i as i64),
                ])
                .collect_vec(),
            next: None,
        },
    );
    db.import_relations(to_import).unwrap();
    dbg!(insert_tt1_time.elapsed());

    let insert_tt10_time = Instant::now();
    let mut to_import = BTreeMap::new();
    to_import.insert(
        "tt10".to_string(),
        NamedRows {
            headers: vec!["k".to_string(), "vld".to_string(), "v".to_string()],
            rows: (0..10000)
                .flat_map(|i| (0..10).map(move |vld| vec![
                    DataValue::from(i as i64),
                    DataValue::Validity(Validity::from((vld, true))),
                    DataValue::from(i as i64),
                ]))
                .collect_vec(),
            next: None,
        },
    );
    db.import_relations(to_import).unwrap();
    dbg!(insert_tt10_time.elapsed());

    let insert_tt100_time = Instant::now();
    let mut to_import = BTreeMap::new();
    to_import.insert(
        "tt100".to_string(),
        NamedRows {
            headers: vec!["k".to_string(), "vld".to_string(), "v".to_string()],
            rows: (0..10000)
                .flat_map(|i| (0..100).map(move |vld| vec![
                    DataValue::from(i as i64),
                    DataValue::Validity(Validity::from((vld, true))),
                    DataValue::from(i as i64),
                ]))
                .collect_vec(),
            next: None,
        },
    );
    db.import_relations(to_import).unwrap();
    dbg!(insert_tt100_time.elapsed());

    let insert_tt1000_time = Instant::now();
    let mut to_import = BTreeMap::new();
    to_import.insert(
        "tt1000".to_string(),
        NamedRows {
            headers: vec!["k".to_string(), "vld".to_string(), "v".to_string()],
            rows: (0..10000)
                .flat_map(|i| {
                    (0..1000).map(move |vld| vec![
                        DataValue::from(i as i64),
                        DataValue::Validity((vld, true).into()),
                        DataValue::from(i as i64),
                    ])
                })
                .collect_vec(),
            next: None,
        },
    );
    db.import_relations(to_import).unwrap();
    dbg!(insert_tt1000_time.elapsed());
}

lazy_static! {
    static ref TEST_DB: DbInstance = {
        // Needs `--features storage-new-rocksdb` to run.
        let db_path = "_time_travel_rocks.db";
        let db = DbInstance::new("newrocksdb", db_path, "").unwrap();

        let create_res = db.run_script(
            r#"
        {:create plain {k: Int => v}}
        {:create tt1 {k: Int, vld: Validity => v}}
        {:create tt10 {k: Int, vld: Validity => v}}
        {:create tt100 {k: Int, vld: Validity => v}}
        {:create tt1000 {k: Int, vld: Validity => v}}
        "#,
            Default::default(),
            ScriptMutability::Mutable,
        );

        if create_res.is_ok() {
            insert_data(&db);
        } else {
            println!("database already exists, skip import");
        }

        db
    };
}

fn single_plain_read() {
    let i = rand::thread_rng().gen_range(0..10000);
    TEST_DB
        .run_script(
            "?[v] := *plain{k: $id, v}",
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

fn plain_aggr() {
    TEST_DB
        .run_script(
            r#"
    ?[sum(v)] := *plain{v}
    "#,
            BTreeMap::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

fn tt_stupid_aggr(k: usize) {
    TEST_DB
        .run_script(
            &format!(
                r#"
    r[k, smallest_by(pack)] := *tt{}{{k, vld, v}}, pack = [v, vld]
    ?[sum(v)] := r[k, v]
    "#,
                k
            ),
            BTreeMap::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

fn tt_travel_aggr(k: usize) {
    TEST_DB
        .run_script(
            &format!(
                r#"
    ?[sum(v)] := *tt{}{{v @ "NOW"}}
    "#,
                k
            ),
            BTreeMap::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

fn single_tt_read(k: usize) {
    let i = rand::thread_rng().gen_range(0..10000);
    TEST_DB
        .run_script(
            &format!(
                r#"
            ?[smallest_by(pack)] := *tt{}{{k: $id, vld, v}}, pack = [v, vld]
            "#,
                k
            ),
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

fn single_tt_travel_read(k: usize) {
    let i = rand::thread_rng().gen_range(0..10000);
    TEST_DB
        .run_script(
            &format!(
                r#"
            ?[v] := *tt{}{{k: $id, v @ "NOW"}}
            "#,
                k
            ),
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

/// History depths: each `ttN` relation holds `N` versions of every key.
const DEPTHS: [usize; 4] = [1, 10, 100, 1000];

/// Point reads per sample in `point_read`, spread across the rayon pool.
const READ_BATCH: usize = 100_000;

fn parallel_reads(b: &mut Bencher, f: impl Fn() + Sync) {
    initialize(&TEST_DB);
    b.iter_custom(|iters| {
        let start = Instant::now();
        for _ in 0..iters {
            (0..READ_BATCH).into_par_iter().for_each(|_| f());
        }
        start.elapsed()
    })
}

/// Point-read throughput against history depth: the plain relation, the latest version found
/// by aggregation, and the latest version found by time travel.
fn point_reads(c: &mut Criterion) {
    let mut g = c.benchmark_group("point_read");
    // Each sample is a whole parallel batch.
    g.sample_size(10);
    g.throughput(Throughput::Elements(READ_BATCH as u64));
    g.bench_function("plain", |b| parallel_reads(b, single_plain_read));
    for k in DEPTHS {
        g.bench_with_input(BenchmarkId::new("tt", k), &k, |b, &k| {
            parallel_reads(b, || single_tt_read(k))
        });
    }
    for k in DEPTHS {
        g.bench_with_input(BenchmarkId::new("tt_travel", k), &k, |b, &k| {
            parallel_reads(b, || single_tt_travel_read(k))
        });
    }
    g.finish();
}

/// Whole-relation aggregation latency against history depth, by the same two routes.
fn aggregations(c: &mut Criterion) {
    let mut g = c.benchmark_group("aggregation");
    // The deep relations scan millions of versions per iteration.
    g.sample_size(10);
    g.bench_function("plain", |b| {
        initialize(&TEST_DB);
        b.iter(plain_aggr)
    });
    for k in DEPTHS {
        g.bench_with_input(BenchmarkId::new("tt_stupid", k), &k, |b, &k| {
            initialize(&TEST_DB);
            b.iter(|| tt_stupid_aggr(k))
        });
        g.bench_with_input(BenchmarkId::new("tt_travel", k), &k, |b, &k| {
            initialize(&TEST_DB);
            b.iter(|| tt_travel_aggr(k))
        });
    }
    g.finish();
}

criterion_group!(benches, point_reads, aggregations);
criterion_main!(benches);
