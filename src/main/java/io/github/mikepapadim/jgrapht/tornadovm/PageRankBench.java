package io.github.mikepapadim.jgrapht.tornadovm;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;
import org.jgrapht.Graph;
import org.jgrapht.alg.scoring.PageRank;
import org.jgrapht.alg.util.Pair;
import org.jgrapht.opt.graph.sparse.IncomingEdgesSupport;
import org.jgrapht.opt.graph.sparse.SparseIntDirectedGraph;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Runs PageRank on a SNAP edge list three ways: JGraphT's {@code PageRank} on a {@code
 * SparseIntDirectedGraph}, the same update on CSR arrays with Java threads, and the same update on a
 * GPU with TornadoVM. All run a fixed number of iterations (JGraphT with the smallest positive tolerance).
 *
 * <p>Usage: {@code PageRankBench <edges.txt[.gz]> [iterations]}.
 */
public final class PageRankBench {

  private PageRankBench() {}

  static final double DAMPING = 0.85;

  /** Edge list with vertex ids renumbered densely in order of first appearance. */
  record Edges(int n, int[] src, int[] dst) {}

  static Edges load(Path file) throws Exception {
    int[] src = new int[1 << 20];
    int[] dst = new int[1 << 20];
    int m = 0;
    int[] remap = new int[1 << 20];
    Arrays.fill(remap, -1);
    int n = 0;
    var in = Files.newInputStream(file);
    var stream = file.toString().endsWith(".gz") ? new GZIPInputStream(in, 1 << 16) : in;
    try (BufferedReader r = new BufferedReader(new InputStreamReader(stream), 1 << 20)) {
      String line;
      while ((line = r.readLine()) != null) {
        if (line.isEmpty() || line.charAt(0) == '#') {
          continue;
        }
        int tab = line.indexOf('\t');
        if (tab < 0) {
          tab = line.indexOf(' ');
        }
        int a = Integer.parseInt(line.substring(0, tab).trim());
        int b = Integer.parseInt(line.substring(tab + 1).trim());
        int max = Math.max(a, b);
        if (max >= remap.length) {
          int old = remap.length;
          remap = Arrays.copyOf(remap, Math.max(max + 1, old * 2));
          Arrays.fill(remap, old, remap.length, -1);
        }
        if (remap[a] < 0) {
          remap[a] = n++;
        }
        if (remap[b] < 0) {
          remap[b] = n++;
        }
        if (m == src.length) {
          src = Arrays.copyOf(src, m * 2);
          dst = Arrays.copyOf(dst, m * 2);
        }
        src[m] = remap[a];
        dst[m] = remap[b];
        m++;
      }
    }
    return new Edges(n, Arrays.copyOf(src, m), Arrays.copyOf(dst, m));
  }

  /** Incoming-edge CSR plus out-degrees. */
  record Csr(int n, int[] rowStart, int[] sources, int[] outDegree) {}

  static Csr csr(Edges e) {
    int n = e.n();
    int[] rowStart = new int[n + 1];
    int[] outDegree = new int[n];
    for (int i = 0; i < e.dst().length; i++) {
      rowStart[e.dst()[i] + 1]++;
      outDegree[e.src()[i]]++;
    }
    for (int v = 0; v < n; v++) {
      rowStart[v + 1] += rowStart[v];
    }
    int[] fill = Arrays.copyOf(rowStart, n);
    int[] sources = new int[e.dst().length];
    for (int i = 0; i < e.dst().length; i++) {
      sources[fill[e.dst()[i]]++] = e.src()[i];
    }
    return new Csr(n, rowStart, sources, outDegree);
  }

  /** The same update as the kernels, with Java threads. */
  static double[] cpu(Csr g, int iterations) {
    int n = g.n();
    double[] cur = new double[n];
    double[] next = new double[n];
    double[] contrib = new double[n];
    Arrays.fill(cur, 1.0 / n);
    for (int it = 0; it < iterations; it++) {
      final double[] c = cur;
      double tele =
          IntStream.range(0, n)
              .parallel()
              .mapToDouble(
                  v -> {
                    int deg = g.outDegree()[v];
                    contrib[v] = deg > 0 ? DAMPING * c[v] / deg : 0.0;
                    return deg > 0 ? (1.0 - DAMPING) * c[v] : c[v];
                  })
              .sum();
      double r = tele / n;
      final double[] nx = next;
      IntStream.range(0, n)
          .parallel()
          .forEach(
              v -> {
                double sum = 0.0;
                for (int e = g.rowStart()[v]; e < g.rowStart()[v + 1]; e++) {
                  sum += contrib[g.sources()[e]];
                }
                nx[v] = r + sum;
              });
      next = cur;
      cur = nx;
    }
    return cur;
  }

  static TornadoDevice device() {
    String[] parts = System.getProperty("tornado.backend.device", "0:0").split(":");
    return TornadoRuntimeProvider.getTornadoRuntime()
        .getBackend(Integer.parseInt(parts[0]))
        .getDevice(Integer.parseInt(parts[1]));
  }

  /** PageRank on the GPU; returns the scores and the time in ms (upload, iterations, download). */
  static double[] gpu(Csr g, int iterations, double[] timeMs) {
    int n = g.n();
    int groups = (n + PageRankKernels.GROUP - 1) / PageRankKernels.GROUP;
    long t0 = System.nanoTime();
    IntArray rowStart = IntArray.fromArray(g.rowStart());
    IntArray sources = IntArray.fromArray(g.sources());
    IntArray outDegree = IntArray.fromArray(g.outDegree());
    DoubleArray a = new DoubleArray(n);
    a.init(1.0 / n);
    DoubleArray b = new DoubleArray(n);
    DoubleArray contrib = new DoubleArray(n);
    DoubleArray partials = new DoubleArray(groups);
    DoubleArray teleport = new DoubleArray(1);
    DoubleArray damping = DoubleArray.fromElements(DAMPING);
    IntArray dims = IntArray.fromElements(n, groups);
    KernelContext ctx = new KernelContext();
    // Two iterations per execution: a -> b, then b -> a.
    TaskGraph graph =
        new TaskGraph("pr")
            .transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                rowStart, sources, outDegree, a, b, contrib, partials, teleport, damping, dims)
            .task("p1", PageRankKernels::prepare, ctx, a, outDegree, damping, contrib, partials, dims)
            .task("r1", PageRankKernels::reduce, ctx, partials, teleport, dims)
            .task("g1", PageRankKernels::gather, ctx, rowStart, sources, contrib, teleport, b, dims)
            .task("p2", PageRankKernels::prepare, ctx, b, outDegree, damping, contrib, partials, dims)
            .task("r2", PageRankKernels::reduce, ctx, partials, teleport, dims)
            .task("g2", PageRankKernels::gather, ctx, rowStart, sources, contrib, teleport, a, dims)
            .transferToHost(DataTransferMode.UNDER_DEMAND, a);
    WorkerGrid1D vertices = new WorkerGrid1D(groups * PageRankKernels.GROUP);
    vertices.setLocalWork(PageRankKernels.GROUP, 1, 1);
    WorkerGrid1D single = new WorkerGrid1D(PageRankKernels.GROUP);
    single.setLocalWork(PageRankKernels.GROUP, 1, 1);
    GridScheduler grid = new GridScheduler();
    for (String t : new String[] {"p1", "g1", "p2", "g2"}) {
      grid.addWorkerGrid("pr." + t, vertices);
    }
    grid.addWorkerGrid("pr.r1", single);
    grid.addWorkerGrid("pr.r2", single);
    try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
      plan.withGridScheduler(grid).withDevice(device());
      var result = plan.execute();
      for (int it = 2; it < iterations; it += 2) {
        result = plan.execute();
      }
      result.transferToHost(a);
      timeMs[0] = (System.nanoTime() - t0) / 1e6;
    } catch (Exception ex) {
      throw new IllegalStateException(ex);
    }
    return a.toHeapArray();
  }

  public static void main(String[] args) throws Exception {
    Path file = Path.of(args[0]);
    int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 100;
    int reps = Integer.getInteger("reps", 5);
    long t0 = System.nanoTime();
    Edges e = load(file);
    Csr g = csr(e);
    System.out.printf(
        "%s: %d vertices, %d edges, loaded in %.1f s; %d iterations, damping %.2f%n",
        file.getFileName(), e.n(), e.src().length, (System.nanoTime() - t0) / 1e9, iterations,
        DAMPING);

    // JGraphT.
    int[] src = e.src();
    int[] dst = e.dst();
    Graph<Integer, Integer> graph =
        new SparseIntDirectedGraph(
            e.n(), src.length,
            () -> IntStream.range(0, src.length).mapToObj(i -> Pair.of(src[i], dst[i])),
            IncomingEdgesSupport.FULL_INCOMING_EDGES);
    double[] jgt = new double[e.n()];
    double[] tJ = new double[Math.min(reps, 3)];
    for (int r = 0; r < tJ.length; r++) {
      long a = System.nanoTime();
      Map<Integer, Double> scores = new PageRank<>(graph, DAMPING, iterations, Double.MIN_VALUE).getScores();
      tJ[r] = (System.nanoTime() - a) / 1e6;
      for (int v = 0; v < e.n(); v++) {
        jgt[v] = scores.get(v);
      }
    }

    double[] tC = new double[reps];
    double[] cpu = null;
    for (int r = 0; r < reps + 1; r++) {
      long a = System.nanoTime();
      cpu = cpu(g, iterations);
      if (r > 0) {
        tC[r - 1] = (System.nanoTime() - a) / 1e6;
      }
    }

    double[] tG = new double[reps];
    double[] gpu = null;
    double[] t = new double[1];
    gpu(g, iterations, t); // compile and warm up
    for (int r = 0; r < reps; r++) {
      gpu = gpu(g, iterations, t);
      tG[r] = t[0];
    }

    double maxCpu = 0;
    double maxGpu = 0;
    double ref = 0;
    for (int v = 0; v < e.n(); v++) {
      maxCpu = Math.max(maxCpu, Math.abs(cpu[v] - jgt[v]));
      maxGpu = Math.max(maxGpu, Math.abs(gpu[v] - jgt[v]));
      ref = Math.max(ref, jgt[v]);
    }
    Arrays.sort(tJ);
    Arrays.sort(tC);
    Arrays.sort(tG);
    System.out.printf(
        "JGraphT PageRank (1 thread) %.0f ms | CSR Java %d threads %.0f ms | TornadoVM %s %.0f ms"
            + " (incl. upload/download) | %.1fx vs JGraphT, %.1fx vs Java threads%n",
        tJ[tJ.length / 2], Runtime.getRuntime().availableProcessors(), tC[reps / 2],
        device().getDeviceName(), tG[reps / 2], tJ[tJ.length / 2] / tG[reps / 2],
        tC[reps / 2] / tG[reps / 2]);
    System.out.printf(
        "max |score - JGraphT|: Java threads %.2e, TornadoVM %.2e (max score %.2e)%n",
        maxCpu, maxGpu, ref);
  }
}
