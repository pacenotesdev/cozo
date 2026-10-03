/*
 *  Copyright 2026, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */

//! Entry point for a scenario binary: named measurements that time themselves and print their
//! own results, for the cases a per-iteration benchmark model does not fit.
//!
//! Scenarios are selected by substring, as benchmarks are:
//! `cargo bench --bench pokec_scenarios -- mixed`. With no filter, all of them run.

/// Runs the scenarios whose names match the command line filters.
///
/// Cargo passes `--bench` when it runs a bench target to measure it, and omits it under
/// `cargo test`, which runs bench targets as a smoke test. Scenarios need staged data and take
/// minutes, so without `--bench` this only reports what it would have run.
pub fn run(scenarios: &[(&str, fn())]) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let filters: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with('-'))
        .map(String::as_str)
        .collect();
    let selected = scenarios
        .iter()
        .filter(|(name, _)| filters.is_empty() || filters.iter().any(|f| name.contains(f)));
    if !args.iter().any(|a| a == "--bench") {
        for (name, _) in selected {
            println!("scenario {name}: skipped, runs under `cargo bench` only");
        }
        return;
    }
    for (name, f) in selected {
        println!("\n=== {name}");
        f();
    }
}
