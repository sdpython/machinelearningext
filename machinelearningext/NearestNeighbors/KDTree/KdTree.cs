// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.NearestNeighbors
{
    /// <summary>
    /// Implements a kd tree.
    /// </summary>
    public class KdTree
    {
        /// <summary>
        /// Maximum depth for the kdtree.
        /// </summary>
        public const int MaxDepth = 1500;

        #region static

        private static readonly KeyValuePair<float, IPointIdFloat> InfinitePoint = new KeyValuePair<float, IPointIdFloat>(float.MinValue, null);

        internal static IKdTreeNode CreateTree(IList<IPointIdFloat> points, Random rnd, int depth = 0)
        {
            switch (points.Count())
            {
                case 0:
                    return new KdTreeLeaf(depth);
                case 1:
                    return new KdTreeNode(points.First(), new KdTreeLeaf(depth + 1), new KdTreeLeaf(depth + 1), depth);
                default:
                    IList<IPointIdFloat> leftPoints = null;
                    IList<IPointIdFloat> rightPoints = null;
                    IPointIdFloat median = MedianPoint(points, ref leftPoints, ref rightPoints, depth, rnd);
                    IKdTreeNode left = CreateTree(leftPoints, rnd, depth + 1);
                    IKdTreeNode right = CreateTree(rightPoints, rnd, depth + 1);
                    return new KdTreeNode(median, left, right, depth);
            }
        }

        public static IPointIdFloat MedianPoint(
            IList<IPointIdFloat> points,
            ref IList<IPointIdFloat> left,
            ref IList<IPointIdFloat> right,
            int depth,
            Random rnd)
        {
            int m = (points.Count() - 1) / 2;
            left = new List<IPointIdFloat>();
            right = new List<IPointIdFloat>();
            return RandomSelect(points, m, ref left, ref right, depth, rnd);
        }

        internal static IPointIdFloat RandomSelect(
            IList<IPointIdFloat> points,
            int which,
            ref IList<IPointIdFloat> left,
            ref IList<IPointIdFloat> right,
            int depth,
            Random rnd)
        {
            int n = points.Count();
            switch (n)
            {
                case 0:
                    throw new IndexOutOfRangeException();
                case 1:
                    return points.First();
                default:
                    IPointIdFloat pivot;
                    int i;

                    //lock (syncLock)
                    //{ // synchronize
                    i = rnd.Next(n);

                    List<IPointIdFloat> localMiddle = new List<IPointIdFloat>();
                    List<IPointIdFloat> localLeft = new List<IPointIdFloat>();
                    List<IPointIdFloat> localRight = new List<IPointIdFloat>();
                    float pivotKey = KdTreeNode.KeyByDepth(points[i], depth);

                    foreach (IPointIdFloat p in points)
                    {
                        float pKey = KdTreeNode.KeyByDepth(p, depth);
                        if (pKey < pivotKey)
                            localLeft.Add(p);
                        else if (pKey > pivotKey)
                            localRight.Add(p);
                        else
                            localMiddle.Add(p);
                    }
                    int sizeLeft = localLeft.Count();
                    int sizeMiddle = localMiddle.Count();
                    if (which < sizeLeft)
                    {
                        foreach (IPointIdFloat p in localMiddle.Concat(localRight))
                            right.Add(p);
                        pivot = RandomSelect(localLeft, which, ref left, ref right, depth, rnd);
                    }
                    else if (which < sizeLeft + sizeMiddle)
                    {
                        pivot = localMiddle.First();
                        //All points with the same key as this node's, must go left, to make search faster
                        foreach (IPointIdFloat p in localLeft.Concat(localMiddle.Skip(1)))
                            left.Add(p);
                        foreach (IPointIdFloat p in localRight)
                            right.Add(p);
                    }
                    else
                    {
                        foreach (IPointIdFloat p in localLeft.Concat(localMiddle))
                            left.Add(p);
                        pivot = RandomSelect(localRight, which - sizeLeft - sizeMiddle, ref left, ref right, depth, rnd);
                    }
                    return pivot;
            }
        }

        #endregion

        #region Properties

        public int dimension { get; private set; }
        private IKdTreeNode root;
        NearestNeighborsDistance _distance;
        Func<VBuffer<float>, VBuffer<float>, float> _distFunc;
        Random _rnd;
        int? _seed;

        #endregion

        #region constructor

        public KdTree(IEnumerable<IPointIdFloat> points, int dimension = -1, int? seed = null,
                      NearestNeighborsDistance distance = NearestNeighborsDistance.L2)
        {
            _seed = seed;
            _rnd = seed.HasValue ? new Random(seed.Value) : new Random();
            _distance = distance;
            this.dimension = dimension;
            if (points.Any())
            {
                ValidatePointsArray(points);
                root = CreateTree(points.ToList(), _rnd);
            }
            else
                root = null;
            SetDistanceFunction();
        }

        public KdTree(NearestNeighborsDistance distance, int? seed, params IPointIdFloat[] points) :
            this(points.ToList(), distance: distance, seed: seed)
        {
        }

        public KdTree(int? seed, params IPointIdFloat[] points) :
            this(points.ToList(), distance: NearestNeighborsDistance.L2, seed: seed)
        {
        }

        public KdTree(ModelLoadContext ctx)
        {
            bool hasSeed = ctx.Reader.ReadByte() == 1;
            if (hasSeed)
            {
                var seed = ctx.Reader.ReadInt32();
                _rnd = new Random(seed);
                _seed = seed;
            }
            else
            {
                _rnd = new Random();
                _seed = null;
            }
            dimension = ctx.Reader.ReadInt32();
            int i = ctx.Reader.ReadInt32();
            _distance = (NearestNeighborsDistance)i;
            root = ReadNode(ctx);
            byte b = ctx.Reader.ReadByte();
            if (b != 168)
                throw Contracts.Except("Detected inconsistency in deserializing.");
            SetDistanceFunction();
        }

        #endregion

        void SetDistanceFunction()
        {
            switch (_distance)
            {
                case NearestNeighborsDistance.cosine:
                    _distFunc = (VBuffer<float> v1, VBuffer<float> v2) => { return VectorDistanceHelper.Cosine(v1, v2); };
                    break;
                case NearestNeighborsDistance.L1:
                    _distFunc = (VBuffer<float> v1, VBuffer<float> v2) => { return VectorDistanceHelper.L1(v1, v2); };
                    break;
                case NearestNeighborsDistance.L2:
                    _distFunc = (VBuffer<float> v1, VBuffer<float> v2) => { return VectorDistanceHelper.L2(v1, v2); };
                    break;
                default:
                    throw Contracts.Except("No associated distance for {0}", _distance);
            }
        }

        public bool Any()
        {
            return root != null;
        }

        /// <summary>
        /// Add a constant to every id. Needs when elements are loaded from multiple threads.
        /// </summary>
        public void MoveId(long add)
        {
            foreach (var el in EnumeratePoints())
                el.ChangeId(el.id + add);
        }

        public IEnumerable<IPointIdFloat> EnumeratePoints()
        {
            if (root != null)
                foreach (var el in root.EnumerateNodes())
                    if (el.point != null)
                        yield return el.point;
        }

        public void Save(ModelSaveContext ctx)
        {
            if (_seed.HasValue)
            {
                ctx.Writer.Write((byte)1);
                ctx.Writer.Write(_seed.Value);
            }
            else
                ctx.Writer.Write((byte)0);
            ctx.Writer.Write(dimension);
            ctx.Writer.Write((int)_distance);
            SaveNode(ctx, root);
            ctx.Writer.Write((byte)168);
        }

        public static void SaveNode(ModelSaveContext ctx, IKdTreeNode node)
        {
            ctx.Writer.Write((byte)(node != null ? (node as KdTreeLeaf == null ? 1 : 2) : 0));
            if (node != null)
                node.Save(ctx);
            ctx.Writer.Write((byte)169);
        }

        public static IKdTreeNode ReadNode(ModelLoadContext ctx)
        {
            var typeNode = ctx.Reader.ReadByte();
            IKdTreeNode res;
            switch (typeNode)
            {
                case 0:
                    res = null;
                    break;
                case 1:
                    res = new KdTreeNode(ctx);
                    break;
                case 2:
                    res = new KdTreeLeaf(ctx);
                    break;
                default:
                    throw Contracts.Except("Bad value for type: {0}", typeNode);
            }
            byte b = ctx.Reader.ReadByte();
            if (b != 169)
                throw Contracts.Except("Detected inconsistency in deserializing.");
            return res;
        }

        #region API

        public void Add(IPointIdFloat p)
        {
            if (root == null)
            {
                var points = new IPointIdFloat[] { p };
                ValidatePointsArray(points);
                root = CreateTree(points.ToList(), _rnd);
            }
            else
            {
                ValidatePoint(p);
                root.Add(p);
            }
        }

        public bool Contains(IPointIdFloat p)
        {
            return root.Contains(p);
        }

        public IPointIdFloat NearestNeighbour(IPointIdFloat p)
        {
            ValidatePoint(p);
            float tmp = float.MaxValue;
            return root.NearestNeighbour(p, null, ref tmp, _distFunc);
        }

        /// <summary>
        /// Returns the N points in the tree which are closest to the target.
        /// </summary>
        public IList<IPointIdFloat> NearestNNeighbors(IPointIdFloat target, int N)
        {
            var nns = NearestNNeighborsAndDistance(target, N)
                                          .ToList()
                                          .OrderBy(kvp => -kvp.Key)
                                          .Select(kvp => kvp.Value)
                                          .ToList();

            Comparison<IPointIdFloat> comparePointsByDistanceToTarget = (a, b) =>
            {
                float dA = _distFunc(a.coordinates, target.coordinates);
                float dB = _distFunc(b.coordinates, target.coordinates);
                if (dA < dB)
                    return -1;
                else if (dA > dB)
                    return 1;
                else
                    return 0;
            };

            nns.Sort(comparePointsByDistanceToTarget);
            return nns;
        }

        public FixedSizePriorityQueue<float, IPointIdFloat> NearestNNeighborsAndDistance(IPointIdFloat target, int N)
        {
            ValidatePoint(target);
            ValidateSize(N, "N");

            if (N <= 0)
                throw new ArgumentException("N must be positive.");

            var results = new FixedSizePriorityQueue<float, IPointIdFloat>(N);
            var nns = root.NearestNNeighbors(target, results, _distFunc);
            return nns;
        }

        public IList<IPointIdFloat> PointsWithinDistance(IPointIdFloat center, float distance)
        {
            ValidatePoint(center);
            IList<IPointIdFloat> results = new List<IPointIdFloat>();
            root.PointsWithinDistance(center, distance, ref results, _distFunc);
            return results;
        }

        public long Count()
        {
            return root.size;
        }

        #endregion

        #region Internal

        internal void ValidateSize(int size, string argName = null)
        {
            if (size <= 0)
                throw new ArgumentException(string.Format("Argument '{0}': passed {1} while it must be positive", argName ?? "size", size));
        }

        internal void ValidatePoint(IPointIdFloat p)
        {
            if (p == null)
                throw new ArgumentNullException();
            else if (p.dimension != dimension)
                throw new ArgumentException(string.Format("Wrong Point dimension: expected {0}, got {1}", dimension, p.dimension));
        }

        internal void ValidatePointsArray(IEnumerable<IPointIdFloat> points)
        {
            int d = 0;

            if (points == null)
                throw new ArgumentNullException();
            else if (points.Count() > 0)
            {
                if (dimension > 0)
                    d = dimension;
                else
                    this.dimension = d = points.First().dimension;

                if (points.All(p => p.dimension == d))
                    return;
            }
            throw new ArgumentException(string.Format("Points array must be non-empty and all points need to have the same dimension{0}", d > 0 ? " " + d : ""));
        }

        #endregion

        #region Node

        public interface IKdTreeNode
        {
            IPointIdFloat point { get; }
            IKdTreeNode left { get; }
            IKdTreeNode right { get; }
            long size { get; }
            int depth { get; }

            float Key();

            IKdTreeNode Add(IPointIdFloat point);
            bool Delete(IPointIdFloat point);
            bool Contains(IPointIdFloat p);
            void PointsWithinDistance(IPointIdFloat center, float distance, ref IList<IPointIdFloat> results, Func<VBuffer<float>, VBuffer<float>, float> distFunc);
            IPointIdFloat NearestNeighbour(IPointIdFloat target, IPointIdFloat nn, ref float nnDist, Func<VBuffer<float>, VBuffer<float>, float> distFunc);
            FixedSizePriorityQueue<float, IPointIdFloat> NearestNNeighbors(IPointIdFloat target, FixedSizePriorityQueue<float, IPointIdFloat> nns, Func<VBuffer<float>, VBuffer<float>, float> distFunc);
            void Save(ModelSaveContext ctx);
            IEnumerable<IKdTreeNode> EnumerateNodes();
        }

        public class KdTreeNode : IKdTreeNode
        {
            public IPointIdFloat point { get; private set; }
            public IKdTreeNode left { get; private set; }
            public IKdTreeNode right { get; private set; }
            public long size { get; private set; }
            public int depth { get; private set; }

            public IEnumerable<IKdTreeNode> EnumerateNodes()
            {
                yield return this;
                if (left != null)
                {
                    foreach (var el in left.EnumerateNodes())
                        yield return el;
                }
                if (right != null)
                {
                    foreach (var el in right.EnumerateNodes())
                        yield return el;
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                ctx.Writer.Write((byte)(point != null ? 1 : 0));
                if (point != null)
                    point.Save(ctx);
                ctx.Writer.Write(size);
                ctx.Writer.Write(depth);
                SaveNode(ctx, left);
                SaveNode(ctx, right);
            }

            public KdTreeNode(ModelLoadContext ctx)
            {
                bool isnotnull = ctx.Reader.ReadByte() == 1;
                point = isnotnull ? new PointIdFloat(ctx) : null;
                size = ctx.Reader.ReadInt64();
                depth = ctx.Reader.ReadInt32();
                left = ReadNode(ctx);
                right = ReadNode(ctx);
            }

            public static float KeyByDepth(IPointIdFloat p, int depth)
            {
                return p.ElementAt(depth % p.dimension);
            }

            public KdTreeNode(IPointIdFloat point, IKdTreeNode left, IKdTreeNode right, int depth)
            {
                this.point = point;
                this.left = left;
                this.right = right;
                this.depth = depth;
                size = 1 + left.size + right.size;
                if (depth > MaxDepth)
                    throw Contracts.Except("The k-d tree depth is too high (> {0}). This probably means the datasets is too sparse to be effective. You should reduce the number of dimensions (PCA for example).", MaxDepth);
            }

            public float Key()
            {
                return KeyByDepth(point, depth);
            }

            public IKdTreeNode Add(IPointIdFloat p)
            {
                float key = this.Key();
                float pKey = KeyByDepth(p, depth);

                if (pKey <= key)
                {
                    left = left.Add(p);
                    size = 1 + left.size + right.size;
                }
                else
                {
                    right = right.Add(p);
                    size = 1 + left.size + right.size;
                }
                return this;
            }

            public bool Delete(IPointIdFloat point)
            {
                throw new NotImplementedException();
            }

            public bool Contains(IPointIdFloat p)
            {
                if (point == p)
                    return true;
                else if (KeyByDepth(p, depth) <= KeyByDepth(point, depth))
                    return left.Contains(p);
                else
                    return right.Contains(p);
            }

            public void PointsWithinDistance(IPointIdFloat center, float distance, ref IList<IPointIdFloat> results,
                                             Func<VBuffer<float>, VBuffer<float>, float> distFunc)
            {
                float key = KeyByDepth(point, depth);
                float centerKey = KeyByDepth(center, depth);

                float d = distFunc(point.coordinates, center.coordinates);
                IKdTreeNode closestBranch, fartherBranch;

                if (d <= distance)
                    results.Add(point);

                if (centerKey <= key)
                {
                    closestBranch = left;
                    fartherBranch = right;
                }
                else
                {
                    closestBranch = right;
                    fartherBranch = left;
                }

                closestBranch.PointsWithinDistance(center, distance, ref results, distFunc);

                if (Math.Abs(centerKey - key) <= distance)
                    fartherBranch.PointsWithinDistance(center, distance, ref results, distFunc);
            }

            public IPointIdFloat NearestNeighbour(IPointIdFloat target, IPointIdFloat nn, ref float nnDist,
                                                  Func<VBuffer<float>, VBuffer<float>, float> distFunc)
            {
                float key = KeyByDepth(point, depth);
                float targetKey = KeyByDepth(target, depth);

                float d = distFunc(point.coordinates, target.coordinates);
                IKdTreeNode closestBranch, fartherBranch;

                if (d < nnDist)
                {
                    nnDist = d;
                    nn = point;
                }

                if (targetKey <= key)
                {
                    closestBranch = left;
                    fartherBranch = right;
                }
                else
                {
                    closestBranch = right;
                    fartherBranch = left;
                }

                nn = closestBranch.NearestNeighbour(target, nn, ref nnDist, distFunc);

                if (Math.Abs(targetKey - key) <= nnDist)
                {
                    nn = fartherBranch.NearestNeighbour(target, nn, ref nnDist, distFunc);
                }
                return nn;
            }

            public FixedSizePriorityQueue<float, IPointIdFloat> NearestNNeighbors(IPointIdFloat target,
                                                        FixedSizePriorityQueue<float, IPointIdFloat> nns,
                                                        Func<VBuffer<float>, VBuffer<float>, float> distFunc)
            {
                float key = KeyByDepth(point, depth);
                float targetKey = KeyByDepth(target, depth);

                float d = distFunc(point.coordinates, target.coordinates);
                IKdTreeNode closestBranch, fartherBranch;
                float nthNnDist = -nns.Peek().GetValueOrDefault(InfinitePoint).Key;

                if (!nns.IsFull || d < nthNnDist)
                    nns.Enqueue(-d, point);

                if (targetKey <= key)
                {
                    closestBranch = left;
                    fartherBranch = right;
                }
                else
                {
                    closestBranch = right;
                    fartherBranch = left;
                }

                nns = closestBranch.NearestNNeighbors(target, nns, distFunc);

                nthNnDist = -nns.Peek().GetValueOrDefault(InfinitePoint).Key;
                if (Math.Abs(targetKey - key) <= nthNnDist)
                    nns = fartherBranch.NearestNNeighbors(target, nns, distFunc);
                return nns;
            }
        }

        public class KdTreeLeaf : IKdTreeNode
        {
            public IPointIdFloat point { get; private set; }
            public IKdTreeNode left { get; private set; }
            public IKdTreeNode right { get; private set; }
            public long size { get; private set; }
            public int depth { get; private set; }

            public KdTreeLeaf(int depth)
            {
                this.point = null;
                this.depth = depth;
                if (depth > MaxDepth)
                    throw Contracts.Except("The k-d tree depth is too high (> {0}). This probably means the datasets is too sparse to be effective. You should reduce the number of dimensions (PCA for example).", MaxDepth);
            }

            public IEnumerable<IKdTreeNode> EnumerateNodes()
            {
                yield return this;
                if (left != null)
                {
                    foreach (var el in left.EnumerateNodes())
                        yield return el;
                }
                if (right != null)
                {
                    foreach (var el in right.EnumerateNodes())
                        yield return el;
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                ctx.Writer.Write((byte)(point != null ? 1 : 0));
                if (point != null)
                    point.Save(ctx);
                ctx.Writer.Write(size);
                ctx.Writer.Write(depth);
                SaveNode(ctx, left);
                SaveNode(ctx, right);
            }

            public KdTreeLeaf(ModelLoadContext ctx)
            {
                bool isnotnull = ctx.Reader.ReadByte() == 1;
                point = isnotnull ? new PointIdFloat(ctx) : null;
                size = ctx.Reader.ReadInt64();
                depth = ctx.Reader.ReadInt32();
                left = ReadNode(ctx);
                right = ReadNode(ctx);
            }

            public IKdTreeNode Add(IPointIdFloat point)
            {
                left = new KdTreeLeaf(depth + 1);
                right = new KdTreeLeaf(depth + 1);
                return new KdTreeNode(point, left, right, depth);
            }

            public bool Delete(IPointIdFloat point)
            {
                throw new NotImplementedException();
            }

            public float Key()
            {
                throw new NotImplementedException();
            }

            public bool Contains(IPointIdFloat p)
            {
                return false;
            }

            public void PointsWithinDistance(IPointIdFloat center, float distance, ref IList<IPointIdFloat> results, Func<VBuffer<float>, VBuffer<float>, float> distFunc)
            {
                //Nothing to do
            }

            public IPointIdFloat NearestNeighbour(IPointIdFloat target, IPointIdFloat nn, ref float nnDist, Func<VBuffer<float>, VBuffer<float>, float> distFunc)
            {
                return nn;
            }

            public FixedSizePriorityQueue<float, IPointIdFloat> NearestNNeighbors(IPointIdFloat target, FixedSizePriorityQueue<float, IPointIdFloat> nns, Func<VBuffer<float>, VBuffer<float>, float> distFunc)
            {
                return nns;
            }
        }

        #endregion
    }
}
