// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements sorting functions for dataframe.
    /// </summary>
    public static class DataFrameSorting
    {
        /// <summary>
        /// The number of sorting columns which can be used is limited.
        /// </summary>
        public const int LimitNumberSortingColumns = 3;

        #region typed version

        public static void TSort<T>(IDataFrameView df, ref int[] order, T[] keys, bool ascending)
            where T : IComparable<T>
        {
            if (order == null)
            {
                order = new int[df.Length];
                for (int i = 0; i < order.Length; ++i)
                    order[i] = i;
            }
            if (ascending)
                Array.Sort(order, (x, y) => keys[x].CompareTo(keys[y]));
            else
                Array.Sort(order, (x, y) => -keys[x].CompareTo(keys[y]));
        }

        public static ImmutableTuple<T1>[] TSort<T1>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var keys = df.EnumerateItems<T1>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2>[] TSort<T1, T2>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var keys = df.EnumerateItems<T1, T2>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2, T3>[] TSort<T1, T2, T3>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var keys = df.EnumerateItems<T1, T2, T3>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1>[] TSort<T1>(IDataFrameView df, ref int[] order, IEnumerable<int> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var keys = df.EnumerateItems<T1>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2>[] TSort<T1, T2>(IDataFrameView df, ref int[] order, IEnumerable<int> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var keys = df.EnumerateItems<T1, T2>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2, T3>[] TSort<T1, T2, T3>(IDataFrameView df, ref int[] order, IEnumerable<int> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var keys = df.EnumerateItems<T1, T2, T3>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        #endregion

        #region untyped

        static void RecSort(IDataFrameView df, int[] icols, bool ascending)
        {
            var kind = df.Kinds[icols[0]];
            if (icols.Length == 1)
            {
                if (kind.IsVector())
                {
                    switch (kind.ItemType().RawKind())
                    {
                        case DataKind.Boolean: df.TSort<VBufferEqSort<bool>>(icols, ascending); break;
                        case DataKind.Int32: df.TSort<VBufferEqSort<int>>(icols, ascending); break;
                        case DataKind.UInt32: df.TSort<VBufferEqSort<uint>>(icols, ascending); break;
                        case DataKind.Int64: df.TSort<VBufferEqSort<long>>(icols, ascending); break;
                        case DataKind.Single: df.TSort<VBufferEqSort<float>>(icols, ascending); break;
                        case DataKind.Double: df.TSort<VBufferEqSort<double>>(icols, ascending); break;
                        case DataKind.String: df.TSort<VBufferEqSort<DvText>>(icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: df.TSort<bool>(icols, ascending); break;
                        case DataKind.Int32: df.TSort<int>(icols, ascending); break;
                        case DataKind.UInt32: df.TSort<uint>(icols, ascending); break;
                        case DataKind.Int64: df.TSort<long>(icols, ascending); break;
                        case DataKind.Single: df.TSort<float>(icols, ascending); break;
                        case DataKind.Double: df.TSort<double>(icols, ascending); break;
                        case DataKind.String: df.TSort<DvText>(icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
            }
            else
            {
                if (kind.IsVector())
                {
                    switch (kind.ItemType().RawKind())
                    {
                        case DataKind.Boolean: RecSort<VBufferEqSort<bool>>(df, icols, ascending); break;
                        case DataKind.Int32: RecSort<VBufferEqSort<int>>(df, icols, ascending); break;
                        case DataKind.UInt32: RecSort<VBufferEqSort<uint>>(df, icols, ascending); break;
                        case DataKind.Int64: RecSort<VBufferEqSort<long>>(df, icols, ascending); break;
                        case DataKind.Single: RecSort<VBufferEqSort<float>>(df, icols, ascending); break;
                        case DataKind.Double: RecSort<VBufferEqSort<double>>(df, icols, ascending); break;
                        case DataKind.String: RecSort<VBufferEqSort<DvText>>(df, icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: RecSort<bool>(df, icols, ascending); break;
                        case DataKind.Int32: RecSort<int>(df, icols, ascending); break;
                        case DataKind.UInt32: RecSort<uint>(df, icols, ascending); break;
                        case DataKind.Int64: RecSort<long>(df, icols, ascending); break;
                        case DataKind.Single: RecSort<float>(df, icols, ascending); break;
                        case DataKind.Double: RecSort<double>(df, icols, ascending); break;
                        case DataKind.String: RecSort<DvText>(df, icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
            }
        }

        static void RecSort<T1>(IDataFrameView df, int[] icols, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var kind = df.Kinds[icols[1]];
            if (icols.Length == 2)
            {
                if (kind.IsVector())
                {
                    switch (kind.ItemType().RawKind())
                    {
                        case DataKind.Boolean: df.TSort<T1, VBufferEqSort<bool>>(icols, ascending); break;
                        case DataKind.Int32: df.TSort<T1, VBufferEqSort<int>>(icols, ascending); break;
                        case DataKind.UInt32: df.TSort<T1, VBufferEqSort<uint>>(icols, ascending); break;
                        case DataKind.Int64: df.TSort<T1, VBufferEqSort<long>>(icols, ascending); break;
                        case DataKind.Single: df.TSort<T1, VBufferEqSort<float>>(icols, ascending); break;
                        case DataKind.Double: df.TSort<T1, VBufferEqSort<double>>(icols, ascending); break;
                        case DataKind.String: df.TSort<T1, VBufferEqSort<DvText>>(icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: df.TSort<T1, bool>(icols, ascending); break;
                        case DataKind.Int32: df.TSort<T1, int>(icols, ascending); break;
                        case DataKind.UInt32: df.TSort<T1, uint>(icols, ascending); break;
                        case DataKind.Int64: df.TSort<T1, long>(icols, ascending); break;
                        case DataKind.Single: df.TSort<T1, float>(icols, ascending); break;
                        case DataKind.Double: df.TSort<T1, double>(icols, ascending); break;
                        case DataKind.String: df.TSort<T1, DvText>(icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
            }
            else
            {
                if (kind.IsVector())
                {
                    switch (kind.ItemType().RawKind())
                    {
                        case DataKind.Boolean: RecSort<T1, VBufferEqSort<bool>>(df, icols, ascending); break;
                        case DataKind.Int32: RecSort<T1, VBufferEqSort<int>>(df, icols, ascending); break;
                        case DataKind.UInt32: RecSort<T1, VBufferEqSort<uint>>(df, icols, ascending); break;
                        case DataKind.Int64: RecSort<T1, VBufferEqSort<long>>(df, icols, ascending); break;
                        case DataKind.Single: RecSort<T1, VBufferEqSort<float>>(df, icols, ascending); break;
                        case DataKind.Double: RecSort<T1, VBufferEqSort<double>>(df, icols, ascending); break;
                        case DataKind.String: RecSort<T1, VBufferEqSort<DvText>>(df, icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: RecSort<T1, bool>(df, icols, ascending); break;
                        case DataKind.Int32: RecSort<T1, int>(df, icols, ascending); break;
                        case DataKind.UInt32: RecSort<T1, uint>(df, icols, ascending); break;
                        case DataKind.Int64: RecSort<T1, long>(df, icols, ascending); break;
                        case DataKind.Single: RecSort<T1, float>(df, icols, ascending); break;
                        case DataKind.Double: RecSort<T1, double>(df, icols, ascending); break;
                        case DataKind.String: RecSort<T1, DvText>(df, icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
            }
        }

        static void RecSort<T1, T2>(IDataFrameView df, int[] icols, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var kind = df.Kinds[icols[2]];
            if (icols.Length == 3)
            {
                if (kind.IsVector())
                {
                    switch (kind.ItemType().RawKind())
                    {
                        case DataKind.Boolean: df.TSort<T1, T2, VBufferEqSort<bool>>(icols, ascending); break;
                        case DataKind.Int32: df.TSort<T1, T2, VBufferEqSort<int>>(icols, ascending); break;
                        case DataKind.UInt32: df.TSort<T1, T2, VBufferEqSort<uint>>(icols, ascending); break;
                        case DataKind.Int64: df.TSort<T1, T2, VBufferEqSort<long>>(icols, ascending); break;
                        case DataKind.Single: df.TSort<T1, T2, VBufferEqSort<float>>(icols, ascending); break;
                        case DataKind.Double: df.TSort<T1, T2, VBufferEqSort<double>>(icols, ascending); break;
                        case DataKind.String: df.TSort<T1, T2, VBufferEqSort<DvText>>(icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: df.TSort<T1, T2, bool>(icols, ascending); break;
                        case DataKind.Int32: df.TSort<T1, T2, int>(icols, ascending); break;
                        case DataKind.UInt32: df.TSort<T1, T2, uint>(icols, ascending); break;
                        case DataKind.Int64: df.TSort<T1, T2, long>(icols, ascending); break;
                        case DataKind.Single: df.TSort<T1, T2, float>(icols, ascending); break;
                        case DataKind.Double: df.TSort<T1, T2, double>(icols, ascending); break;
                        case DataKind.String: df.TSort<T1, T2, DvText>(icols, ascending); break;
                        default:
                            throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                    }
                }
            }
            else
            {
                throw new NotImplementedException($"Sort is not implemented for {icols.Length} columns.");
            }
        }

        public static void Sort(IDataFrameView df, IEnumerable<int> columns, bool ascending = true)
        {
            int[] icols = columns.ToArray();
            RecSort(df, icols, ascending);
        }

        #endregion
    }
}
