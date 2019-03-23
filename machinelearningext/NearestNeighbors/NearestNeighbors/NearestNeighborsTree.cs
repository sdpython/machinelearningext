// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Threading.Tasks;
using Microsoft.ML;


namespace Scikit.ML.NearestNeighbors
{
    public enum NearestNeighborsWeights
    {
        uniform = 1,
        distance = 2
    }

    public enum NearestNeighborsAlgorithm
    {
        kdtree = 1
    }

    public enum NearestNeighborsDistance
    {
        cosine = 2,
        L1 = 3,
        L2 = 4
    }

    public class NearestNeighborsTrees
    {
        readonly IExceptionContext _host;
        readonly KdTree[] _kdtrees;

        readonly DataViewType _inputType;
        public DataViewType InputType { get { return _inputType; } }
        public KdTree[] Trees { get { return _kdtrees; } }

        public long Count() { return _kdtrees.Select(c => c.Count()).Sum(); }

        public NearestNeighborsTrees(IExceptionContext host, KdTree[] kdtrees)
        {
            Contracts.CheckValue(host, "host");
            _host = host;
            _host.Check(!kdtrees.Where(c => c == null).Any(), "kdtree");
            _kdtrees = kdtrees;
            _inputType = new VectorDataViewType(NumberDataViewType.Single, _kdtrees[0].dimension);
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.Writer.Write(_kdtrees.Length);
            for (int i = 0; i < _kdtrees.Length; ++i)
            {
                _host.CheckValue(_kdtrees[i], "kdtree");
                _kdtrees[i].Save(ctx);
            }
        }

        public NearestNeighborsTrees(IHost env, ModelLoadContext ctx)
        {
            _host = env;
            int nb = ctx.Reader.ReadInt32();
            _kdtrees = new KdTree[nb];
            for (int i = 0; i < nb; ++i)
            {
                _kdtrees[i] = new KdTree(ctx);
                _host.CheckValue(_kdtrees[i], "kdtree");
            }
            _inputType = new VectorDataViewType(NumberDataViewType.Single, _kdtrees[0].dimension);
        }

        public KeyValuePair<float, long>[] NearestNNeighbors(VBuffer<float> target, int k)
        {
            var point = new PointIdFloat(-1, target, false);
            KeyValuePair<float, long>[] neighbors;
            if (_kdtrees.Length == 1)
                // kdtrees returns the opposite of the distance.
                neighbors = _kdtrees[0].NearestNNeighborsAndDistance(point, k).Select(c => new KeyValuePair<float, long>(-c.Key, c.Value.id)).ToArray();
            else
            {
                KeyValuePair<float, long>[][] stack = new KeyValuePair<float, long>[_kdtrees.Length][];
                var ops = new Action[_kdtrees.Length];
                for (int i = 0; i < ops.Length; ++i)
                {
                    int chunkId = i;
                    ops[i] = () =>
                    {
                        // kdtrees returns the opposite of the distance.
                        stack[chunkId] = _kdtrees[chunkId].NearestNNeighborsAndDistance(point, k).Select(c => new KeyValuePair<float, long>(-c.Key, c.Value.id)).ToArray();
                    };
                }
                Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = ops.Length }, ops);
                var merged = new List<KeyValuePair<float, long>>();
                for (int i = 0; i < ops.Length; ++i)
                    merged.AddRange(stack[i]);
                neighbors = merged.OrderBy(c => c.Key).Take(k).ToArray();
            }
            return neighbors;
        }
    }
}
