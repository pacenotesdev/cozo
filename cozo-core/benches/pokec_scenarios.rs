/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Load tests over the Pokec dataset that time themselves and print their results: dataset
//! import, backup, and mixed workloads. Per-query latency and throughput are in `pokec.rs`.

use std::env;
use std::time::Instant;

use lazy_static::initialize;

#[path = "support/pokec.rs"]
mod pokec;
#[path = "support/scenario.rs"]
mod scenario;
use pokec::*;

fn main() {
    scenario::run(&[
        ("load", load),
        ("backup_db", backup_db),
        ("realistic", realistic),
        ("mixed", mixed),
    ]);
}

/// Builds or restores the dataset; the import timings are printed as it loads.
fn load() {
    initialize(&TEST_DB);
}

fn backup_db() {
    initialize(&TEST_DB);
    let data_size = env::var("COZO_BENCH_POKEC_SIZE").unwrap_or("medium".to_string());
    let backup_taken = Instant::now();
    TEST_DB
        .backup_db(format!("backup-{}.db", data_size))
        .unwrap();
    dbg!(backup_taken.elapsed());
    dbg!(((SIZES.0 + 2 * SIZES.1) as f64) / backup_taken.elapsed().as_secs_f64());
}

fn wrap(mixed_pct: f64, f: QueryFn) {
    use rand::prelude::*;

    let mut gen = rand::thread_rng();
    if gen.gen_bool(mixed_pct) {
        let wtr = WRITE_QUERIES.choose(&mut gen).unwrap();
        wtr();
    } else {
        f();
    }
}

fn realistic() {
    use rand::prelude::*;
    use rayon::prelude::*;

    println!("realistic benchmarks");
    dbg!(rayon::current_num_threads());
    let init_time = Instant::now();
    initialize(&TEST_DB);
    dbg!(init_time.elapsed());

    let percentages = [
        [0.0, 0.9, 0.05, 0.05],
        [0.0, 0.7, 0.15, 0.15],
        [0.0, 0.5, 0.25, 0.25],
        [0.0, 0.3, 0.35, 0.35],
    ];

    for [analytical, read, update, write] in percentages {
        dbg!((analytical, read, update, write));
        let taken = Instant::now();
        (0..*ITERATIONS).into_par_iter().for_each(|_| {
            let mut gen = thread_rng();
            let p = gen.gen::<f64>();
            let f = if p < analytical {
                ANALYTICAL_QUERIES.choose(&mut gen)
            } else if p < analytical + read {
                READ_QUERIES.choose(&mut gen)
            } else if p < analytical + read + update {
                UPDATE_QUERIES.choose(&mut gen)
            } else {
                WRITE_QUERIES.choose(&mut gen)
            };
            f.unwrap()()
        });
        dbg!((*ITERATIONS as f64) / taken.elapsed().as_secs_f64());
    }
}

fn mixed() {
    use rayon::prelude::*;

    println!("mixed benchmarks");
    dbg!(rayon::current_num_threads());
    let init_time = Instant::now();
    initialize(&TEST_DB);
    dbg!(init_time.elapsed());

    let mixed_pct = env::var("COZO_BENCH_POKEC_MIX_PCT").unwrap_or("0.3".to_string());
    let mixed_pct = mixed_pct.parse::<f64>().unwrap();
    dbg!(mixed_pct);
    assert!(mixed_pct >= 0.);
    assert!(mixed_pct <= 1.);

    let expansion_1_time = Instant::now();
    let count = 100;
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_1_plain);
    });
    dbg!((count as f64) / expansion_1_time.elapsed().as_secs_f64());

    let expansion_1_filter_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_1_filter);
    });
    dbg!((count as f64) / expansion_1_filter_time.elapsed().as_secs_f64());

    let expansion_2_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_2_plain);
    });
    dbg!((count as f64) / expansion_2_time.elapsed().as_secs_f64());

    let expansion_2_filter_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_2_filter);
    });
    dbg!((count as f64) / expansion_2_filter_time.elapsed().as_secs_f64());

    let expansion_3_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_3_plain);
    });
    dbg!((count as f64) / expansion_3_time.elapsed().as_secs_f64());

    let expansion_3_filter_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_3_filter);
    });
    dbg!((count as f64) / expansion_3_filter_time.elapsed().as_secs_f64());

    let expansion_4_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_4_plain);
    });
    dbg!((count as f64) / expansion_4_time.elapsed().as_secs_f64());

    let expansion_4_filter_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, expansion_4_filter);
    });
    dbg!((count as f64) / expansion_4_filter_time.elapsed().as_secs_f64());

    let neighbours_2_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, neighbours_2_plain);
    });
    dbg!((count as f64) / neighbours_2_time.elapsed().as_secs_f64());

    let neighbours_2_filter_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, neighbours_2_filter_only);
    });
    dbg!((count as f64) / neighbours_2_filter_time.elapsed().as_secs_f64());

    let neighbours_2_data_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, neighbours_2_data_only);
    });
    dbg!((count as f64) / neighbours_2_data_time.elapsed().as_secs_f64());

    let neighbours_2_filter_data_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, neighbours_2_filter_data);
    });
    dbg!((count as f64) / neighbours_2_filter_data_time.elapsed().as_secs_f64());

    let pattern_cycle_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, pattern_cycle);
    });
    dbg!((count as f64) / pattern_cycle_time.elapsed().as_secs_f64());

    let pattern_long_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, pattern_long);
    });
    dbg!((count as f64) / pattern_long_time.elapsed().as_secs_f64());

    let pattern_short_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, pattern_short);
    });
    dbg!((count as f64) / pattern_short_time.elapsed().as_secs_f64());

    let aggregation_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, aggregation_group);
    });
    dbg!((count as f64) / aggregation_time.elapsed().as_secs_f64());

    let aggregation_distinct_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, aggregation_count);
    });
    dbg!((count as f64) / aggregation_distinct_time.elapsed().as_secs_f64());

    let aggregation_filter_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, aggregation_filter);
    });
    dbg!((count as f64) / aggregation_filter_time.elapsed().as_secs_f64());

    let min_max_time = Instant::now();

    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, aggregation_min_max);
    });
    dbg!((count as f64) / min_max_time.elapsed().as_secs_f64());

    let single_vertex_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, single_vertex_read);
    });
    dbg!((count as f64) / single_vertex_time.elapsed().as_secs_f64());

    let single_vertex_update_time = Instant::now();
    (0..count).into_par_iter().for_each(|_| {
        wrap(mixed_pct, single_vertex_update);
    });
    dbg!((count as f64) / single_vertex_update_time.elapsed().as_secs_f64());
}
