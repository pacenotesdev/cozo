# HNSW benchmark baseline

Measurements from `cozo-core/benches/hnsw.rs`, recorded so that a later change to the vector
index has something to be compared against.

## Machine

| | |
|---|---|
| CPU | 11th Gen Intel Core i9-11900H, 8 cores / 16 threads, single socket |
| Core clocks | 4 cores at 4.90 GHz max, 12 at 4.80 GHz; 800 MHz minimum |
| Cache | L2 10 MiB across 8 instances, L3 24 MiB |
| Memory | 62 GiB |
| Storage | Samsung PM9A1 NVMe, 1 TB, non-rotational, write-back caching, `none` I/O scheduler |

The PM9A1's DRAM buffer size is not reported through sysfs, so it is omitted.

At these sizes the whole dataset is resident in page cache, on both engines.

## Method

Every figure is the median of three runs of the whole suite, six runs in total. Latencies come
from the libtest harness (`ns/iter`); the `report_*` functions measure rather than time and
print their own tables. The `+/-` column is the harness's own spread within a run, not the
spread between runs.

### Parameters

| Knob | Value | Meaning |
|---|---|---|
| `COZO_HNSW_N` | 10000 | vectors in the searched index |
| `COZO_HNSW_BUILD_N` | 2000 | vectors in the build benchmarks |
| `COZO_HNSW_DIM` | 64 | dimensions, `F32`, unit-normalised |
| `COZO_HNSW_M` | 16 | graph degree |
| `COZO_HNSW_EF_C` | 50 | `ef_construction` |
| `COZO_HNSW_QUERIES` | 100 | query vectors per latency or recall figure |
| `COZO_HNSW_THROUGHPUT_QUERIES` | 4000 | queries per throughput measurement |

Vectors come from a seeded LCG, so both engines see byte-identical data. The build benchmarks
rebuild the index from scratch on every harness iteration, so they take their size from
`COZO_HNSW_BUILD_N` rather than `COZO_HNSW_N`.

Distance is L2 unless the benchmark name says otherwise. The uniform dataset is random unit
vectors; the clustered dataset is 16 centroids with 0.15 jitter.

## Search latency

Median ns/iter, converted to milliseconds. One iteration is `COZO_HNSW_QUERIES` = 100 queries,
so divide by 100 for per-query cost.

| Benchmark | mem | rocksdb | ratio |
|---|---:|---:|---:|
| `search_k10_ef16` | 1.049 ms +/- 0.05 | 1.780 ms +/- 0.16 | 1.70x |
| `search_k10_ef32` | 1.672 ms +/- 0.08 | 2.859 ms +/- 0.13 | 1.71x |
| `search_k10_ef64` | 2.736 ms +/- 0.07 | 4.771 ms +/- 0.16 | 1.74x |
| `search_k10_ef128` | 4.731 ms +/- 0.09 | 8.537 ms +/- 0.21 | 1.80x |
| `search_k10_ef256` | 8.464 ms +/- 0.20 | 14.946 ms +/- 0.43 | 1.77x |
| `search_k1_ef64` | 2.723 ms +/- 0.06 | 4.762 ms +/- 0.24 | 1.75x |
| `search_k50_ef64` | 2.764 ms +/- 0.07 | 4.857 ms +/- 0.16 | 1.76x |
| `search_k100_ef128` | 4.809 ms +/- 0.09 | 8.723 ms +/- 0.18 | 1.81x |
| `search_cosine_k10_ef64` | 2.799 ms +/- 0.07 | 4.757 ms +/- 0.16 | 1.70x |
| `search_inner_product_k10_ef64` | 2.782 ms +/- 0.09 | 4.778 ms +/- 0.19 | 1.72x |
| `search_clustered_k10_ef64` | 0.925 ms +/- 0.08 | 1.267 ms +/- 0.09 | 1.37x |
| `search_radius` | 2.740 ms +/- 0.07 | 4.795 ms +/- 0.23 | 1.75x |
| `search_filter_permissive` | 5.208 ms +/- 0.23 | 9.194 ms +/- 0.45 | 1.77x |
| `search_filter_selective` | 91.227 ms +/- 13.90 | 119.540 ms +/- 17.03 | 1.31x |

`search_filter_permissive` is `id % 2 == 0`, `search_filter_selective` is `id % 64 == 0`. Both
are k=10, ef=64. `search_radius` is k=10, ef=64 with `radius: 1.6`.

## Build and insert

| Benchmark | mem | rocksdb | ratio |
|---|---:|---:|---:|
| `build_l2` (2000 vectors) | 1834.492 ms +/- 39.09 | 5111.341 ms +/- 89.03 | 2.79x |
| `build_cosine` (2000 vectors) | 1831.362 ms +/- 41.15 | 5124.331 ms +/- 80.71 | 2.80x |
| `build_clustered` (2000 vectors) | 709.678 ms +/- 13.97 | 1936.216 ms +/- 49.92 | 2.73x |
| `insert_one_into_an_existing_index` | 4.175 ms +/- 0.41 | 6.439 ms +/- 0.72 | 1.54x |

`insert_one_into_an_existing_index` adds a single vector to an index already holding
`COZO_HNSW_BUILD_N` = 2000.

### Build scaling

Microseconds per vector for a from-scratch build.

| n | mem | rocksdb |
|---:|---:|---:|
| 250 | 295.8 us/vec | 856.5 us/vec |
| 500 | 447.4 us/vec | 1253.1 us/vec |
| 1000 | 647.9 us/vec | 1834.1 us/vec |
| 2000 | 917.9 us/vec | 2549.9 us/vec |

## Recall

### Recall@10 against `ef`, uniform data, n=10000

| ef | mem | rocksdb |
|---:|---:|---:|
| 16 | 0.506 | 0.528 |
| 32 | 0.708 | 0.693 |
| 64 | 0.865 | 0.861 |
| 128 | 0.957 | 0.957 |
| 256 | 0.991 | 0.992 |

### Recall@10 against `ef`, clustered data, n=10000

| ef | mem | rocksdb |
|---:|---:|---:|
| 16 | 0.455 | 0.456 |
| 32 | 0.525 | 0.526 |
| 64 | 0.605 | 0.608 |
| 128 | 0.676 | 0.681 |

### Build-to-build spread

Recall@10 at `ef=64` over five builds of identical data, n=2000. Node levels are drawn at
random, so two builds of the same data are not the same graph.

| | mem | rocksdb |
|---|---:|---:|
| range | 0.971 .. 0.975 | 0.971 .. 0.975 |
| mean | 0.9718 | 0.9730 |
| width | 0.004 | 0.004 |

This is measured at n=2000, unlike the n=10000 of the tables above.

### Filtered recall

k=10, n=10000. `returned` is the fraction of the requested k delivered; recall is against
ground truth restricted to the rows the filter admits.

| Filter | eligible | ef | returned (mem) | recall (mem) | returned (rocksdb) | recall (rocksdb) |
|---|---:|---:|---:|---:|---:|---:|
| `1 in 2` | 5000 | 16 | 1.000 | 0.670 | 1.000 | 0.645 |
| `1 in 2` | 5000 | 64 | 1.000 | 0.949 | 1.000 | 0.943 |
| `1 in 8` | 1250 | 16 | 1.000 | 0.903 | 1.000 | 0.900 |
| `1 in 8` | 1250 | 64 | 1.000 | 0.999 | 1.000 | 0.998 |
| `1 in 64` | 157 | 16 | 1.000 | 0.998 | 1.000 | 0.999 |
| `1 in 64` | 157 | 64 | 1.000 | 1.000 | 1.000 | 1.000 |
| `1 in 256` | 40 | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| `1 in 256` | 40 | 64 | 1.000 | 1.000 | 1.000 | 1.000 |

## Query throughput

Concurrent k=10, ef=64 queries, 4000 per measurement. `scaling` is the measured rate against a
perfect multiple of the single-thread rate.

| threads | mem q/s | scaling | rocksdb q/s | scaling |
|---:|---:|---:|---:|---:|
| 1 | 406 | 1.00x | 211 | 1.00x |
| 2 | 816 | 1.00x | 406 | 0.96x |
| 4 | 1531 | 0.94x | 725 | 0.86x |
| 8 | 2811 | 0.87x | 1317 | 0.77x |
| 16 | 3254 | 0.50x | 1390 | 0.41x |

## Reproducing

```
cd cozo-core
export COZO_HNSW_N=10000 COZO_HNSW_BUILD_N=2000 COZO_HNSW_DIM=64 \
       COZO_HNSW_M=16 COZO_HNSW_EF_C=50 \
       COZO_HNSW_QUERIES=100 COZO_HNSW_THROUGHPUT_QUERIES=4000

COZO_HNSW_ENGINE=mem     cargo bench --bench hnsw -- --nocapture
COZO_HNSW_ENGINE=rocksdb cargo bench --features storage-rocksdb --bench hnsw -- --nocapture
```

A full suite is about 27 minutes on `mem` and about 69 minutes on `rocksdb`. The benchmarks
require a nightly toolchain for the `test` harness. On-disk engines put their stores under
`$TMPDIR/cozo-hnsw-bench`, which is emptied at the start of each run.
