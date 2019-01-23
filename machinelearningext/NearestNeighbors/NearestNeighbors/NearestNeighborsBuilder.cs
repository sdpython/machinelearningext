// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using Scikit.ML.PipelineHelper;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.NearestNeighbors
{
    public static class NearestNeighborsBuilder
    {
        public static NearestNeighborsTrees NearestNeighborsBuild<TLabel>(IChannel ch, IDataView data,
                    int featureIndex, int labelIndex, int idIndex, int weightIndex,
                    out Dictionary<long, Tuple<TLabel, float>> outLabelsWeights,
                    NearestNeighborsArguments args)
                    where TLabel : IComparable<TLabel>
        {
            var indexes = new HashSet<int>() { featureIndex, labelIndex, weightIndex, idIndex };

            if (idIndex != -1)
            {
                var colType = data.Schema[idIndex].Type;
                if (idIndex != -1 && (colType.IsVector() || colType.RawKind() != DataKind.I8))
                    throw ch.Except("Column '{0}' must be of type '{1}' not '{2}'", args.colId, DataKind.I8, colType);
            }

            int nt = args.numThreads ?? 1;
            Random rand = RandomUtils.Create(args.seed);
            var cursors = (nt == 1)
                                ? new RowCursor[] { data.GetRowCursor(data.Schema.Where(c=>indexes.Contains(c.Index)), rand) }
                                : data.GetRowCursorSet(data.Schema.Where(c => indexes.Contains(c.Index)), nt, rand);
            KdTree[] kdtrees;
            Dictionary<long, Tuple<TLabel, float>>[] labelsWeights;
            if (nt == 1)
            {
                labelsWeights = new Dictionary<long, Tuple<TLabel, float>>[1];
                kdtrees = new KdTree[] { BuildKDTree<TLabel>(data,cursors[0],  featureIndex, labelIndex, idIndex, weightIndex,
                    out labelsWeights[0], args) };
            }
            else
            {
                // Multithreading. We assume the distributed set of cursor is well distributed.
                // No KdTree will be much smaller than the others.
                Action[] ops = new Action[cursors.Length];
                kdtrees = new KdTree[cursors.Length];
                labelsWeights = new Dictionary<long, Tuple<TLabel, float>>[cursors.Length];
                for (int i = 0; i < ops.Length; ++i)
                {
                    int chunkId = i;
                    kdtrees[i] = null;
                    ops[i] = new Action(() =>
                    {
                        kdtrees[chunkId] = BuildKDTree<TLabel>(data, cursors[chunkId],
                            featureIndex, labelIndex, idIndex, weightIndex,
                            out labelsWeights[chunkId], args);
                    });
                }

                Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = cursors.Length }, ops);
            }

            kdtrees = kdtrees.Where(c => c.Any()).ToArray();
            labelsWeights = labelsWeights.Where(c => c.Any()).ToArray();
            var merged = labelsWeights[0];
            long start = merged.Count;
            long newKey;
            for (int i = 1; i < labelsWeights.Length; ++i)
            {
                kdtrees[i].MoveId(start);
                foreach (var pair in labelsWeights[i])
                {
                    newKey = pair.Key + start;
                    if (merged.ContainsKey(newKey))
                        throw ch.Except("The same key appeared twice in two differents threads: {0}", newKey);
                    else
                        merged.Add(newKey, pair.Value);
                }
                start += labelsWeights[i].Count;
            }

            // Id checking.
            var labelId = merged.Select(c => c.Key).ToList();
            var treeId = new List<long>();
            for (int i = 0; i < kdtrees.Length; ++i)
                treeId.AddRange(kdtrees[i].EnumeratePoints().Select(c => c.id));
            var h1 = new HashSet<long>(labelId);
            var h2 = new HashSet<long>(treeId);
            if (h1.Count != labelId.Count)
                throw ch.Except("Duplicated label ids.");
            if (h2.Count != treeId.Count)
                throw ch.Except("Duplicated label ids.");
            if (h1.Count != h2.Count)
                throw ch.Except("Mismatch (1) in ids.");
            var inter = h1.Intersect(h2);
            if (inter.Count() != h1.Count)
                throw ch.Except("Mismatch (2) in ids.");

            // End.
            outLabelsWeights = merged;
            return new NearestNeighborsTrees(ch, kdtrees);
        }

        private static KdTree BuildKDTree<TLabel>(IDataView data, RowCursor cursor,
                        int featureIndex, int labelIndex, int idIndex, int weightIndex,
                        out Dictionary<long, Tuple<TLabel, float>> labelsWeights, NearestNeighborsArguments args)
            where TLabel : IComparable<TLabel>
        {
            using (cursor)
            {
                var featureGetter = cursor.GetGetter<VBuffer<float>>(featureIndex);
                var labelGetter = labelIndex >= 0 ? cursor.GetGetter<TLabel>(labelIndex) : null;
                var weightGetter = weightIndex >= 0 ? cursor.GetGetter<float>(weightIndex) : null;
                var idGetter = idIndex >= 0 ? cursor.GetGetter<long>(idIndex) : null;
                var kdtree = new KdTree(distance: args.distance, seed: args.seed);
                labelsWeights = new Dictionary<long, Tuple<TLabel, float>>();
                VBuffer<float> features = new VBuffer<float>();
                TLabel label = default(TLabel);
                float weight = 1;
                long lid = default(long);
                while (cursor.MoveNext())
                {
                    featureGetter(ref features);
                    if (labelGetter != null)
                        labelGetter(ref label);
                    if (weightGetter != null)
                        weightGetter(ref weight);
                    if (idGetter != null)
                        idGetter(ref lid);
                    else
                        lid = labelsWeights.Count;
                    labelsWeights[lid] = new Tuple<TLabel, float>(label, weight);
                    var point = new PointIdFloat(lid, features, true);
                    kdtree.Add(point);
                }
                return kdtree;
            }
        }
    }
}
