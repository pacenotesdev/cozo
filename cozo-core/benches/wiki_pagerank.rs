/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;
use std::{env, io};

use criterion::{criterion_group, criterion_main, Criterion};

use lazy_static::{initialize, lazy_static};

use cozo::{DbInstance, NamedRows, DataValue, ScriptMutability};

lazy_static! {
    static ref TEST_DB: DbInstance = {
        let data_dir = PathBuf::from(env::var("COZO_BENCH_WIKI_DIR").unwrap());

        let db = DbInstance::new("mem", "", "").unwrap();
        let mut file_path = data_dir.clone();
        file_path.push("wikipedia-articles.el");

        // dbg!(&db_kind);
        // dbg!(&data_dir);
        // dbg!(&file_path);
        // dbg!(&data_size);
        // dbg!(&n_threads);

        db.run_script(":create article {fr: Int, to: Int}",
            Default::default(),
            ScriptMutability::Mutable,
        ).unwrap();

        let file = File::open(&file_path).unwrap();
        let mut articles = vec![];

        let import_time = Instant::now();
        for line in io::BufReader::new(file).lines() {
            let line = line.unwrap();
            if line.len() < 2 {
                continue
            }
            let mut splits = line.split_whitespace();
            let fr = splits.next().unwrap();
            let to = splits.next().unwrap();
            articles.push(vec![DataValue::from(fr.parse::<i64>().unwrap()), DataValue::from(to.parse::<i64>().unwrap())])
        }
        db.import_relations(BTreeMap::from([("article".to_string(), NamedRows {
            headers: vec![
                "fr".to_string(),
                "to".to_string(),
            ],
            rows: articles,
            next: None,
        })])).unwrap();
        dbg!(import_time.elapsed());
        db
    };
}

fn wikipedia(c: &mut Criterion) {
    let mut g = c.benchmark_group("wikipedia");
    // Each iteration is a whole-graph algorithm over the link dump; the default 100 samples
    // would take a very long time.
    g.sample_size(10);
    g.bench_function("pagerank", |b| {
        initialize(&TEST_DB);
        b.iter(|| {
            TEST_DB
                .run_script(
                    "?[id, rank] <~ PageRank(*article[])",
                    Default::default(),
                    ScriptMutability::Immutable,
                )
                .unwrap()
        })
    });
    g.bench_function("louvain", |b| {
        initialize(&TEST_DB);
        b.iter(|| {
            TEST_DB
                .run_script(
                    "?[grp, idx] <~ CommunityDetectionLouvain(*article[])",
                    Default::default(),
                    ScriptMutability::Immutable,
                )
                .unwrap()
        })
    });
    g.finish();
}

criterion_group!(benches, wikipedia);
criterion_main!(benches);
