#!/usr/bin/env bash
# Builds the prototype and runs PageRank on a SNAP edge list.
#
#   ./run.sh web-Google.txt.gz            # https://snap.stanford.edu/data/web-Google.html
#   ./run.sh soc-LiveJournal1.txt.gz 100  # https://snap.stanford.edu/data/soc-LiveJournal1.html
#
# BACKEND=cuda|opencl|metal picks the TornadoVM backend (default cuda), DEVICE its device index.
# Extra JVM options go in JAVA_OPTS, e.g. JAVA_OPTS="-Dreps=3 -Xmx56g".
set -euo pipefail
BACKEND=${BACKEND:-cuda}
DEVICE=${DEVICE:-0}
GRAPH=$(realpath "$1"); shift
cd "$(dirname "$0")"
mvn -q -Dtornadovm.backend="$BACKEND" compile dependency:build-classpath -Dmdep.outputFile=cp.txt
exec java -Xmx24g --enable-native-access=ALL-UNNAMED \
  -Dtornado.enabled.backends="$BACKEND" -Dtornado.backend.device="0:$DEVICE" \
  ${JAVA_OPTS:-} -cp "target/classes:$(cat cp.txt)" \
  io.github.mikepapadim.jgrapht.tornadovm.PageRankBench "$GRAPH" "$@"
