package io.github.mikepapadim.jgrapht.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * PageRank with the same update as JGraphT's {@code PageRank} (unweighted): every iteration
 *
 * <pre>
 *   r       = ( sum over v with outdeg &gt; 0 of (1 - d) * cur[v]  +  sum over dangling v of cur[v] ) / n
 *   next[v] = r + sum over edges u -&gt; v of d * cur[u] / outdeg[u]
 * </pre>
 *
 * <p>One iteration is three kernels: {@link #prepare} writes {@code d * cur[u] / outdeg[u]} per vertex
 * and one partial sum of the teleport term per work group, {@link #reduce} adds the partials in a
 * single work group, and {@link #gather} pulls the contributions over the incoming edges in CSR form
 * ({@code rowStart}, {@code sources}). {@code dims} holds {@code n, number of partials}; {@code
 * damping} holds d.
 */
public final class PageRankKernels {

  /** Work-group size of {@link #prepare} and {@link #reduce}. */
  public static final int GROUP = 256;

  private PageRankKernels() {}

  public static void prepare(
      KernelContext context,
      DoubleArray cur,
      IntArray outDegree,
      DoubleArray damping,
      DoubleArray contrib,
      DoubleArray partials,
      IntArray dims) {
    double[] local = context.allocateDoubleLocalArray(GROUP);
    int v = context.globalIdx;
    int lid = context.localIdx;
    int n = dims.get(0);
    double d = damping.get(0);
    double tele = 0.0;
    if (v < n) {
      double score = cur.get(v);
      int deg = outDegree.get(v);
      if (deg > 0) {
        tele = (1.0 - d) * score;
        contrib.set(v, d * score / deg);
      } else {
        tele = score;
        contrib.set(v, 0.0);
      }
    }
    local[lid] = tele;
    for (int stride = GROUP / 2; stride > 0; stride /= 2) {
      context.localBarrier();
      if (lid < stride) {
        local[lid] += local[lid + stride];
      }
    }
    if (lid == 0) {
      partials.set(context.groupIdx, local[0]);
    }
  }

  public static void reduce(
      KernelContext context, DoubleArray partials, DoubleArray teleport, IntArray dims) {
    double[] local = context.allocateDoubleLocalArray(GROUP);
    int lid = context.localIdx;
    int count = dims.get(1);
    double sum = 0.0;
    for (int i = lid; i < count; i += GROUP) {
      sum += partials.get(i);
    }
    local[lid] = sum;
    for (int stride = GROUP / 2; stride > 0; stride /= 2) {
      context.localBarrier();
      if (lid < stride) {
        local[lid] += local[lid + stride];
      }
    }
    if (lid == 0) {
      teleport.set(0, local[0] / dims.get(0));
    }
  }

  public static void gather(
      KernelContext context,
      IntArray rowStart,
      IntArray sources,
      DoubleArray contrib,
      DoubleArray teleport,
      DoubleArray next,
      IntArray dims) {
    int v = context.globalIdx;
    if (v < dims.get(0)) {
      double sum = 0.0;
      int end = rowStart.get(v + 1);
      for (int e = rowStart.get(v); e < end; e++) {
        sum += contrib.get(sources.get(e));
      }
      next.set(v, teleport.get(0) + sum);
    }
  }
}
