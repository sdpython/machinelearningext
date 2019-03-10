// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Internal.Utilities;


namespace Scikit.ML.NearestNeighbors
{
    /// <summary>
    /// Common interface for observations clustered with DBScan and Optics algorithms.
    /// </summary>
    public interface IPointIdFloat
    {
        VBuffer<float> coordinates { get; }
        float ElementAt(int index);
        int dimension { get; }
        long id { get; }
        void ChangeId(long newId);
        void Save(ModelSaveContext ctx);
        float DistanceTo(IPointIdFloat other);
    }

    /// <summary>
    /// Implements the interface IPoint for DBScan and Optics algorithms.
    /// Sealed for performance (no virtual methods).
    /// </summary>
    public sealed class PointIdFloat : IPointIdFloat
    {
        #region static functions

        public static IList<IPointIdFloat> RandomShuffle(IReadOnlyCollection<IPointIdFloat> points, Random rnd)
        {
            int j;
            IPointIdFloat tmp;
            List<IPointIdFloat> pts = points.ToList();

            for (int i = 0, n = pts.Count(); i < n; i++)
            {
                j = rnd.Next(0, i);
                tmp = pts[i];
                pts[i] = pts[j];
                pts[j] = tmp;
            }
            return pts.AsReadOnly();
        }

        public static Comparison<IPointIdFloat> PointsComparison = (p1, p2) =>
        {
            if (p1 == null || p2 == null)
                throw new ArgumentNullException("Points must be non-null");
            if (p1.dimension != p2.dimension)
                throw new ArgumentException(string.Format("Incompatible points dimensions: {0} != {1}", p1.dimension, p2.dimension));

            if (p1.coordinates.IsDense && p2.coordinates.IsDense)
            {
                var cs1 = p1.coordinates.Values;
                var cs2 = p2.coordinates.Values;

                for (int i = 0; i < p1.coordinates.Count; i++)
                {
                    if (cs1[i] < cs2[i])
                        return -1;
                    else if (cs1[i] > cs2[i])
                        return 1;
                }
                return 0;
            }
            else
            {
                var v1 = p1.coordinates;
                var v2 = p2.coordinates;

                int i = 0;
                int j = 0;
                while (true)
                {
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++i;
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++j;
                    if (i < v1.Count)
                    {
                        if (j < v2.Count)
                        {
                            if (v1.Values[i] != v2.Values[j])
                                return v1.Values[i] < v2.Values[j] ? -1 : 1;
                            ++i;
                            ++j;
                        }
                        else
                        {
                            if (v1.Values[i] != 0)
                                return v1.Values[i] < 0 ? -1 : 1;
                            ++i;
                        }
                    }
                    else if (j < v2.Count)
                    {
                        if (v2.Values[j] != 0)
                            return v2.Values[j] < 0 ? -1 : 1;
                        ++j;
                    }
                    else
                        break;
                }
                return 0;
            }
        };

        #endregion

        #region Properties

        public long id { get; private set; }
        public int dimension => _coordinates.Length;
        private VBuffer<float> _coordinates;
        public VBuffer<float> coordinates => _coordinates;

        public void ChangeId(long newId)
        {
            id = newId;
        }

        public static void SetIds(IEnumerable<IPointIdFloat> points)
        {
            long id = 0;
            foreach (var p in points)
                p.ChangeId(id++);
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.Writer.Write(id);
            ctx.Writer.Write(_coordinates.Length);
            ctx.Writer.Write(_coordinates.Count);
            if (_coordinates.Length < _coordinates.Count)
                throw Contracts.Except("Detected inconsistency where serializing.");
            Utils.WriteSingleArray(ctx.Writer, _coordinates.Values);
            Utils.WriteIntArray(ctx.Writer, _coordinates.Indices);
            ctx.Writer.Write((byte)167);
        }

        public PointIdFloat(ModelLoadContext ctx)
        {
            id = ctx.Reader.ReadInt64();
            int length = ctx.Reader.ReadInt32();
            int count = ctx.Reader.ReadInt32();
            if (length < count)
                throw Contracts.Except("Detected inconsistency where serializing.");
            var floatArray = Utils.ReadFloatArray(ctx.Reader);
            var intArray = Utils.ReadIntArray(ctx.Reader);
            _coordinates = new VBuffer<float>(length, count, floatArray, intArray);
            byte b = ctx.Reader.ReadByte();
            if (b != 167)
                throw Contracts.Except("Serialization failed. Check code is wrong.");
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Package internal version for the constructor: allows to set an id directly.
        /// </summary>
        public PointIdFloat(long id, VBuffer<float> coordinates, bool copy)
        {
            this.id = id;
            if (copy)
            {
                _coordinates = new VBuffer<float>();
                coordinates.CopyTo(ref this._coordinates);
            }
            else
                this._coordinates = coordinates;
        }

        /// <summary>
        /// Package internal version for the constructor: allows to set an id directly.
        /// </summary>
        public PointIdFloat(IEnumerable<float> coordinates)
        {
            this.id = 0;
            var arr = coordinates.ToArray();
            _coordinates = new VBuffer<float>(arr.Length, arr);
        }

        public PointIdFloat(long id, IEnumerable<float> denseValues)
        {
            this.id = id;
            var array = denseValues.ToArray();
            _coordinates = new VBuffer<float>(array.Length, array);
        }

        public PointIdFloat(long id, params float[] denseValues)
        {
            this.id = id;
            var array = denseValues.ToArray();
            _coordinates = new VBuffer<float>(array.Length, array);
        }

        #endregion

        #region API

        public float ElementAt(int i)
        {
            if (_coordinates.IsDense)
                return _coordinates.Values[i];
            else
            {
                int j = 0;
                for (; j < _coordinates.Indices.Length && _coordinates.Indices[j] >= i; ++j) ;
                if (j < _coordinates.Indices.Length && _coordinates.Indices[j] == i)
                    return _coordinates.Values[j];
                return 0f;
            }
        }

        public float DistanceTo(IPointIdFloat other)
        {
            return VectorDistanceHelper.L2Norm(coordinates, other.coordinates);
        }

        #endregion

        #region Overrides

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            var p = obj as IPointIdFloat;
            if ((System.Object)p == null)
                return false;
            if (p.dimension != this.dimension)
                return false;
            return PointsComparison(this, obj as PointIdFloat) == 0;
        }

        public static bool operator ==(PointIdFloat a, PointIdFloat b)
        {
            return PointsComparison(a, b) == 0;
        }

        public static bool operator !=(PointIdFloat a, PointIdFloat b)
        {
            return !(a == b);
        }

        public override int GetHashCode()
        {
            int hash = 19;
            foreach (var c in coordinates.DenseValues())
                hash = hash * 31 + c.GetHashCode();
            return hash;
        }

        public override string ToString()
        {
            return string.Format("PointIdFloat{0}D({1})", dimension,
                string.Join(",", coordinates.Values.Select(x => string.Format("{0:N3}", x).Replace(",", "."))));
        }

        public int CompareTo(IPointIdFloat other)
        {
            return PointsComparison(this, other);
        }

        #endregion
    }
}
