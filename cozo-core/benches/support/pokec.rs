/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The Pokec dataset and the queries run against it, shared by the `pokec` benchmarks and the
//! `pokec_scenarios` load tests. Each target uses a different subset.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{env, io, mem};

use lazy_static::lazy_static;
use rand::Rng;
use regex::Regex;

use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability};

lazy_static! {
    pub static ref ITERATIONS: usize = {
        let size = env::var("COZO_BENCH_ITERATIONS").unwrap_or("100".to_string());
        size.parse::<usize>().unwrap()
    };
    pub static ref SIZES: (usize, usize) = {
        let size = env::var("COZO_BENCH_POKEC_SIZE").unwrap_or("medium".to_string());
        match &size as &str {
            "small" => (10000, 121716),
            "medium" => (100000, 1768515),
            "large" => (1632803, 30622564),
            _ => panic!()
        }
    };

    pub static ref TEST_DB: DbInstance = {
        let data_dir = PathBuf::from(env::var("COZO_BENCH_POKEC_DIR").unwrap());
        let db_kind = env::var("COZO_TEST_DB_ENGINE").unwrap_or("mem".to_string());
        let mut db_path = data_dir.clone();
        let data_size = env::var("COZO_BENCH_POKEC_SIZE").unwrap_or("medium".to_string());
        let batch_size = env::var("COZO_BENCH_POKEC_BATCH")
            .unwrap()
            .parse::<usize>()
            .unwrap();
        db_path.push(format!("{}-{}.db", db_kind, data_size));
        // let _ = std::fs::remove_file(&db_path);
        // let _ = std::fs::remove_dir_all(&db_path);
        let path_exists = Path::exists(&db_path);
        let db = DbInstance::new(&db_kind, db_path.to_str().unwrap(), "").unwrap();
        if path_exists {
            db.run_script("::compact", Default::default(), ScriptMutability::Mutable).unwrap();
            return db
        }

        let mut backup_path = data_dir.clone();
        backup_path.push(format!("backup-{}.db", data_size));
        if Path::exists(&backup_path) {
            println!("restore from backup");
            let import_time = Instant::now();
            db.restore_backup(backup_path.to_str().unwrap()).unwrap();
            dbg!(import_time.elapsed());
            dbg!(((SIZES.0 + 2 * SIZES.1) as f64) / import_time.elapsed().as_secs_f64());
        } else {
            println!("parse data from text file");
            let mut file_path = data_dir.clone();
            file_path.push(format!("pokec_{}_import.cypher", data_size));

            // dbg!(&db_kind);
            // dbg!(&data_dir);
            // dbg!(&file_path);
            // dbg!(&data_size);
            // dbg!(&n_threads);

            if db.run_script(
                r#"
            {:create user {uid: Int => cmpl_pct: Int, gender: String?, age: Int?}}
            {:create friends {fr: Int, to: Int}}
            {:create friends.rev {to: Int, fr: Int}}
            "#,
                Default::default(),
                ScriptMutability::Mutable,
            ).is_err() {
                return db
            }

            let node_re = Regex::new(r#"CREATE \(:User \{id: (\d+), completion_percentage: (\d+), gender: "(\w+)", age: (\d+)}\);"#).unwrap();
            let node_partial_re =
                Regex::new(r#"CREATE \(:User \{id: (\d+), completion_percentage: (\d+)}\);"#).unwrap();
            let edge_re = Regex::new(r#"MATCH \(n:User \{id: (\d+)}\), \(m:User \{id: (\d+)}\) CREATE \(n\)-\[e: Friend]->\(m\);"#).unwrap();

            let file = File::open(&file_path).unwrap();
            let mut friends = Vec::with_capacity(batch_size);
            let mut users = Vec::with_capacity(batch_size);
            let mut push_to_users = |row: Option<Vec<DataValue>>, force: bool| {
                if let Some(row) = row {
                    users.push(row);
                }
                if users.len() >= batch_size || (force && !users.is_empty()) {
                    let mut new_rows = Vec::with_capacity(batch_size);
                    mem::swap(&mut new_rows, &mut users);
                    db.import_relations(BTreeMap::from([(
                        "user".to_string(),
                        NamedRows {
                            headers: vec![
                                "uid".to_string(),
                                "cmpl_pct".to_string(),
                                "gender".to_string(),
                                "age".to_string(),
                            ],
                            rows: new_rows,
                            next: None
                        },
                    )]))
                    .unwrap();
                }
            };

            let mut push_to_friends = |row: Option<Vec<DataValue>>, force: bool| {
                if let Some(row) = row {
                    friends.push(row);
                }
                if friends.len() >= batch_size || (force && !friends.is_empty()) {
                    let mut new_rows = Vec::with_capacity(batch_size);
                    mem::swap(&mut new_rows, &mut friends);
                    db.import_relations(BTreeMap::from([
                        (
                            "friends".to_string(),
                            NamedRows {
                                headers: vec!["fr".to_string(), "to".to_string()],
                                rows: new_rows.clone(),
                                next: None,
                            },
                        ),
                        (
                            "friends.rev".to_string(),
                            NamedRows {
                                headers: vec!["fr".to_string(), "to".to_string()],
                                rows: new_rows,
                                next: None,
                            },
                        ),
                    ]))
                    .unwrap();
                }
            };

            let import_time = Instant::now();
            let mut n_rows = 0usize;
            for line in io::BufReader::new(file).lines() {
                let line = line.unwrap();
                if let Some(data) = edge_re.captures(&line) {
                    n_rows += 2;
                    let fr = data.get(1).unwrap().as_str().parse::<i64>().unwrap();
                    let to = data.get(2).unwrap().as_str().parse::<i64>().unwrap();
                    push_to_friends(Some(vec![DataValue::from(fr), DataValue::from(to)]), false);
                    continue;
                }
                if let Some(data) = node_re.captures(&line) {
                    n_rows += 1;
                    let uid = data.get(1).unwrap().as_str().parse::<i64>().unwrap();
                    let cmpl_pct = data.get(2).unwrap().as_str().parse::<i64>().unwrap();
                    let gender = data.get(3).unwrap().as_str();
                    let age = data.get(4).unwrap().as_str().parse::<i64>().unwrap();
                    push_to_users(
                        Some(vec![DataValue::from(uid), DataValue::from(cmpl_pct), DataValue::from(gender), DataValue::from(age)]),
                        false,
                    );
                    continue;
                }
                if let Some(data) = node_partial_re.captures(&line) {
                    n_rows += 1;
                    let uid = data.get(1).unwrap().as_str().parse::<i64>().unwrap();
                    let cmpl_pct = data.get(2).unwrap().as_str().parse::<i64>().unwrap();
                    push_to_users(
                        Some(vec![
                            DataValue::from(uid),
                            DataValue::from(cmpl_pct),
                            DataValue::Null,
                            DataValue::Null,
                        ]),
                        false,
                    );
                    continue;
                }
                if line.len() < 3 {
                    continue;
                }
                panic!("Err: {}", line)
            }
            push_to_users(None, true);
            push_to_friends(None, true);
            dbg!(import_time.elapsed());
            dbg!((n_rows as f64) / import_time.elapsed().as_secs_f64());
        }
        db
    };
}

pub type QueryFn = fn() -> ();

pub const READ_QUERIES: [QueryFn; 1] = [single_vertex_read];
pub const WRITE_QUERIES: [QueryFn; 2] = [single_edge_write, single_vertex_write];
pub const UPDATE_QUERIES: [QueryFn; 1] = [single_vertex_update];
#[allow(dead_code)]
pub const AGGREGATE_QUERIES: [QueryFn; 4] = [
    aggregation_group,
    aggregation_filter,
    aggregation_count,
    aggregation_min_max,
];
pub const ANALYTICAL_QUERIES: [QueryFn; 15] = [
    expansion_1_plain,
    expansion_2_plain,
    expansion_3_plain,
    expansion_4_plain,
    expansion_1_filter,
    expansion_2_filter,
    expansion_3_filter,
    expansion_4_filter,
    neighbours_2_plain,
    neighbours_2_filter_only,
    neighbours_2_data_only,
    neighbours_2_filter_data,
    pattern_cycle,
    pattern_long,
    pattern_short,
];

pub fn single_vertex_read() {
    let i = rand::thread_rng().gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            "?[cmpl_pct, gender, age] := *user{uid: $id, cmpl_pct, gender, age}",
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn single_vertex_write() {
    single_vertex_write_at(rand::thread_rng().gen_range(1..SIZES.0 * 10));
}

pub fn single_vertex_write_at(i: usize) {
    for _ in 0..10 {
        if TEST_DB
            .run_script(
                "?[uid, cmpl_pct, gender, age] <- [[$id, 0, null, null]] :put user {uid => cmpl_pct, gender, age}",
                BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
                ScriptMutability::Mutable,
            )
            .is_ok() {
            return;
        }
    }
    panic!()
}

pub fn single_edge_write() {
    let (i, j) = random_edge();
    single_edge_write_at(i, j);
}

/// Two distinct user ids.
pub fn random_edge() -> (usize, usize) {
    let i = rand::thread_rng().gen_range(1..SIZES.0);
    let mut j = rand::thread_rng().gen_range(1..SIZES.0);
    while j == i {
        j = rand::thread_rng().gen_range(1..SIZES.0);
    }
    (i, j)
}

pub fn single_edge_write_at(i: usize, j: usize) {
    for _ in 0..10 {
        if TEST_DB
            .run_script(
                r#"
            {?[fr, to] <- [[$i, $j]] :put friends {fr, to}}
            {?[fr, to] <- [[$i, $j]] :put friends.rev {fr, to}}
            "#,
                BTreeMap::from([("i".to_string(), DataValue::from(i as i64)), ("j".to_string(), DataValue::from(j as i64))]),
                ScriptMutability::Mutable,
            )
            .is_ok()
        {
            return;
        }
    }
    panic!()
}

pub fn pagerank() {
    TEST_DB
        .run_script(
            r#"
            ?[] <~ PageRank(*friends[])
            "#,
            Default::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn single_vertex_update() {
    single_vertex_update_at(rand::thread_rng().gen_range(1..SIZES.0));
}

pub fn single_vertex_update_at(i: usize) {
    for _ in 0..10 {
        if TEST_DB
            .run_script(
                r#"
            ?[uid, cmpl_pct, age, gender] := uid = $id, *user{uid, age, gender}, cmpl_pct = -1
            :put user {uid => cmpl_pct, age, gender}
            "#,
                BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
                ScriptMutability::Mutable,
            )
            .is_ok()
        {
            return;
        }
    }
    panic!()
}

pub fn aggregation_group() {
    TEST_DB
        .run_script("?[age, count(uid)] := *user{uid, age}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
}

pub fn aggregation_count() {
    TEST_DB
        .run_script(
            "?[count(uid), count(age)] := *user{uid, age}",
            Default::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn aggregation_filter() {
    TEST_DB
        .run_script(
            "?[age, count(age)] := *user{age}, age ~ 0 >= 18",
            Default::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn aggregation_min_max() {
    TEST_DB
        .run_script(
            "?[min(uid), max(uid), mean(uid)] := *user{uid, age}",
            Default::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_1_plain() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            "?[to] := *friends{fr: $id, to}",
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_1_filter() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            "?[to] := *friends{fr: $id, to}, *user{uid: to, age}, age ~ 0 >= 18",
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_2_plain() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            "?[to] := *friends{fr: $id, to: a}, *friends{fr: a, to}",
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_2_filter() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            "?[to] := *friends{fr: $id, to: a}, *friends{fr: a, to}, *user{uid: to, age}, age ~ 0 >= 18",
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_3_plain() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            l1[to] := *friends{fr: $id, to}
            l2[to] := l1[fr], *friends{fr, to}
            ?[to] := l2[fr], *friends{fr, to}
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_3_filter() {
    let i = rand::thread_rng().gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
                        l1[to] := *friends{fr: $id, to}
                        l2[to] := l1[fr], *friends{fr, to}
                        ?[to] := l2[fr], *friends{fr, to}, *user{uid: to, age}, age ~ 0 >= 18
                        "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_4_plain() {
    let i = rand::thread_rng().gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
                        l1[to] := *friends{fr: $id, to}
                        l2[to] := l1[fr], *friends{fr, to}
                        l3[to] := l2[fr], *friends{fr, to}
                        ?[to] := l3[fr], *friends{fr, to}
                        "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn expansion_4_filter() {
    let i = rand::thread_rng().gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
                        l1[to] := *friends{fr: $id, to}
                        l2[to] := l1[fr], *friends{fr, to}
                        l3[to] := l2[fr], *friends{fr, to}
                        ?[to] := l3[fr], *friends{fr, to}
                        "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn neighbours_2_plain() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            l1[to] := *friends{fr: $id, to}
            ?[to] := l1[to]
            ?[to] := l1[fr], *friends{fr, to}
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn neighbours_2_filter_only() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            l1[to] := *friends{fr: $id, to}
            ?[to] := l1[to], *user{uid: to, age}, age ~ 0 >= 18
            ?[to] := l1[fr], *friends{fr, to}, *user{uid: to, age}, age ~ 0 >= 18
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn neighbours_2_data_only() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            l1[to] := *friends{fr: $id, to}
            ?[to, age, cmpl_pct, gender] := l1[to], *user{uid: to, age, cmpl_pct, gender}
            ?[to, age, cmpl_pct, gender] := l1[fr], *friends{fr, to}, *user{uid: to, age, cmpl_pct, gender}
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn neighbours_2_filter_data() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            l1[to] := *friends{fr: $id, to}
            ?[to] := l1[to], *user{uid: to, age, cmpl_pct, gender}, age ~ 0 >= 18
            ?[to] := l1[fr], *friends{fr, to}, *user{uid: to, age, cmpl_pct, gender}, age ~ 0 >= 18
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn pattern_cycle() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            ?[n, m] := n = $id, *friends{fr: n, to: m}, *friends.rev{fr: m, to: n}
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn pattern_long() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
                ?[n] := *friends{fr: $id, to: n2},
                        *friends{fr: n2, to: n3},
                        *friends{fr: n3, to: n4},
                        *friends.rev{to: n4, fr: n}

                :limit 1
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}

pub fn pattern_short() {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(1..SIZES.0);
    TEST_DB
        .run_script(
            r#"
            ?[to] := *friends{fr: $id, to}

            :limit 1
            "#,
            BTreeMap::from([("id".to_string(), DataValue::from(i as i64))]),
            ScriptMutability::Immutable,
        )
        .unwrap();
}
