# jgrapht-tornadovm-proto

A small proof of concept for [jgrapht/jgrapht#733](https://github.com/jgrapht/jgrapht/issues/733): JGraphT's PageRank run on a GPU with [TornadoVM](https://github.com/beehive-lab/TornadoVM), compared with JGraphT itself.

## What it does

`PageRankKernels` implements the same update as `org.jgrapht.alg.scoring.PageRank` for unweighted graphs, including the teleport term and the handling of vertices with no outgoing edges. Each iteration is three kernels, written in plain Java and compiled by TornadoVM for CUDA, OpenCL or Metal at run time:

- **`prepare`** computes each vertex's contribution, `d · score / outdegree`, plus per-work-group partial sums of the teleport term.
- **`reduce`** adds up those partial sums.
- **`gather`** pulls the contributions over each vertex's incoming edges, stored as CSR (compressed sparse row) arrays.

The graph and the scores stay on the GPU for all iterations. Only the CSR arrays are uploaded, and only the final scores are read back.

`PageRankBench` loads a SNAP edge list and runs PageRank three ways, for a fixed number of iterations:

1. JGraphT's `PageRank` on a `SparseIntDirectedGraph`. This is single-threaded.
2. The same update on the CSR arrays, using Java parallel streams across all cores.
3. The TornadoVM kernels. Timing includes building the plan, uploading the graph and reading the scores back.

## Requirements

- JDK 22 or newer (tested with 25).
- One GPU backend and its driver:
  - **CUDA:** the NVIDIA driver.
  - **OpenCL:** an OpenCL 1.2+ runtime.
  - **Metal:** selectable, but not tested.
- Maven 3.8 or newer.

JGraphT 1.5.3 and TornadoVM 8.1.0 come from Maven Central; there's no TornadoVM installation step.

```bash
curl -LO https://snap.stanford.edu/data/web-Google.txt.gz
curl -LO https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
./run.sh web-Google.txt.gz
JAVA_OPTS="-Xmx56g -Dreps=3" ./run.sh soc-LiveJournal1.txt.gz 100
BACKEND=opencl ./run.sh web-Google.txt.gz
```

## Results

Setup: RTX 4090 with an i9-13900K (32 threads), JDK 25, JGraphT 1.5.3, TornadoVM 8.1.0.
- 100 iterations with damping 0.85.
- Each figure is the median of 3–5 runs, after warm-up.
- In every run, the scores from both the Java-threads version and TornadoVM match JGraphT to 1e-14 absolute, at most 3e-11 relative to the largest score.

| Graph | JGraphT `PageRank`, 1 thread | CSR, 32 Java threads | TornadoVM CUDA | TornadoVM OpenCL |
|---|---|---|---|---|
| web-Google: 0.88M vertices, 5.1M edges | 2.73 s | 0.216 s | **0.061 s** | 0.065 s |
| soc-LiveJournal1: 4.8M vertices, 69M edges | 37.1 s | 1.67 s | **0.30 s** | 0.31 s |

- **Against JGraphT's `PageRank`**, the GPU is 45× faster on web-Google and 125× faster on LiveJournal.
- **Most of that gain comes from the flat CSR layout and threads, not from the GPU.** The Java-threads version is already 13–22× faster than JGraphT.
- **Against that multithreaded CSR version,** the GPU adds 3.6× on web-Google and 5.6× on LiveJournal.
- **Why the GPU helps:** PageRank is memory-bound. The graph stays in GPU memory for all 100 iterations, and GPU memory bandwidth is much higher than the CPU's.

The full output is in `results/bench-rtx4090.txt`.

## Limitations

- Unweighted PageRank only; JGraphT's weighted variant is not covered.
- One thread per vertex in `gather`, with no special handling for very high-degree vertices.
- There is no convergence check on the GPU: it runs the fixed number of iterations, which must be even.

## License

Apache License 2.0.
