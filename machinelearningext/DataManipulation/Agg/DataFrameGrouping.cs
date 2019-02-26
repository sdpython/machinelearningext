// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements grouping functions for dataframe.
    /// </summary>
    public static class DataFrameGrouping
    {
        #region type version

        public static IEnumerable<KeyValuePair<TKey, DataFrameViewGroup>> TGroupBy<TKey>(
                                IDataFrameView df, int[] order, TKey[] keys, int[] columns,
                                Func<TKey, DataFrameGroupKey[]> func)
            where TKey : IEquatable<TKey>
        {
            TKey last = keys.Any() ? keys[order[0]] : default(TKey);
            List<int> subrows = new List<int>();
            foreach (var pos in order)
            {
                var cur = keys[pos];
                if (cur.Equals(last))
                    subrows.Add(pos);
                else if (subrows.Any())
                {
                    yield return new KeyValuePair<TKey, DataFrameViewGroup>(last,
                                    new DataFrameViewGroup(func(last), df.Source ?? df, subrows.ToArray(), df.ColumnsSet));
                    subrows.Clear();
                    subrows.Add(pos);
                }
                last = cur;
            }
            if (subrows.Any())
                yield return new KeyValuePair<TKey, DataFrameViewGroup>(last,
                            new DataFrameViewGroup(func(last), df.Source ?? df, subrows.ToArray(), df.ColumnsSet));
        }

        public static DataFrameViewGroupResults<TImutKey> TGroupBy<TMutKey, TImutKey>(
                            IDataFrameView df, int[] rows, int[] columns, IEnumerable<int> cols, bool sort,
                            MultiGetterAt<TMutKey> getter,
                            Func<TMutKey, TImutKey> conv,
                            Func<TImutKey, DataFrameGroupKey[]> conv2)
            where TMutKey : ITUple, new()
            where TImutKey : IComparable<TImutKey>, IEquatable<TImutKey>
        {
            var icols = cols.ToArray();
            int[] order = rows == null ? rows.Select(c => c).ToArray() : Enumerable.Range(0, df.Length).ToArray();
            var keys = df.EnumerateItems(icols, true, rows, getter).Select(c => conv(c)).ToArray();
            if (sort)
                DataFrameSorting.TSort(df, ref order, keys, true);
            var iter = TGroupBy(df, order, keys, columns, conv2);
            return new DataFrameViewGroupResults<TImutKey>(iter);
        }

        #endregion

        #region agnostic groupby

        static IDataFrameViewGroupResults RecGroupBy(IDataFrameView df, int[] icols, bool sort)
        {
            var kind = df.Kinds[icols[0]];
            if (icols.Length == 1)
            {
                if (kind.IsVector())
                    throw new NotImplementedException();
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: return df.TGroupBy<bool>(icols, sort);
                        case DataKind.Int32: return df.TGroupBy<int>(icols, sort);
                        case DataKind.UInt32: return df.TGroupBy<uint>(icols, sort);
                        case DataKind.Int64: return df.TGroupBy<long>(icols, sort);
                        case DataKind.Single: return df.TGroupBy<float>(icols, sort);
                        case DataKind.Double: return df.TGroupBy<double>(icols, sort);
                        case DataKind.String: return df.TGroupBy<DvText>(icols, sort);
                        default:
                            throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                    }
                }
            }
            else
            {
                if (kind.IsVector())
                    throw new NotImplementedException();
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: return RecGroupBy<bool>(df, icols, sort);
                        case DataKind.Int32: return RecGroupBy<int>(df, icols, sort);
                        case DataKind.UInt32: return RecGroupBy<uint>(df, icols, sort);
                        case DataKind.Int64: return RecGroupBy<long>(df, icols, sort);
                        case DataKind.Single: return RecGroupBy<float>(df, icols, sort);
                        case DataKind.Double: return RecGroupBy<double>(df, icols, sort);
                        case DataKind.String: return RecGroupBy<DvText>(df, icols, sort);
                        default:
                            throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                    }
                }
            }
        }

        static IDataFrameViewGroupResults RecGroupBy<T1>(IDataFrameView df, int[] icols, bool sort)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var kind = df.Kinds[icols[1]];
            if (icols.Length == 2)
            {
                if (kind.IsVector())
                    throw new NotImplementedException();
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: return df.TGroupBy<T1, bool>(icols, sort);
                        case DataKind.Int32: return df.TGroupBy<T1, int>(icols, sort);
                        case DataKind.UInt32: return df.TGroupBy<T1, uint>(icols, sort);
                        case DataKind.Int64: return df.TGroupBy<T1, long>(icols, sort);
                        case DataKind.Single: return df.TGroupBy<T1, float>(icols, sort);
                        case DataKind.Double: return df.TGroupBy<T1, double>(icols, sort);
                        case DataKind.String: return df.TGroupBy<T1, DvText>(icols, sort);
                        default:
                            throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                    }
                }
            }
            else
            {
                if (kind.IsVector())
                    throw new NotImplementedException();
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: return RecGroupBy<T1, bool>(df, icols, sort);
                        case DataKind.Int32: return RecGroupBy<T1, int>(df, icols, sort);
                        case DataKind.UInt32: return RecGroupBy<T1, uint>(df, icols, sort);
                        case DataKind.Int64: return RecGroupBy<T1, long>(df, icols, sort);
                        case DataKind.Single: return RecGroupBy<T1, float>(df, icols, sort);
                        case DataKind.Double: return RecGroupBy<T1, double>(df, icols, sort);
                        case DataKind.String: return RecGroupBy<T1, DvText>(df, icols, sort);
                        default:
                            throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                    }
                }
            }
        }

        static IDataFrameViewGroupResults RecGroupBy<T1, T2>(IDataFrameView df, int[] icols, bool sort)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var kind = df.Kinds[icols[2]];
            if (icols.Length == 3)
            {
                if (kind.IsVector())
                    throw new NotImplementedException();
                else
                {
                    switch (kind.RawKind())
                    {
                        case DataKind.Boolean: return df.TGroupBy<T1, T2, bool>(icols, sort);
                        case DataKind.Int32: return df.TGroupBy<T1, T2, int>(icols, sort);
                        case DataKind.UInt32: return df.TGroupBy<T1, T2, uint>(icols, sort);
                        case DataKind.Int64: return df.TGroupBy<T1, T2, long>(icols, sort);
                        case DataKind.Single: return df.TGroupBy<T1, T2, float>(icols, sort);
                        case DataKind.Double: return df.TGroupBy<T1, T2, double>(icols, sort);
                        case DataKind.String: return df.TGroupBy<T1, T2, DvText>(icols, sort);
                        default:
                            throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                    }
                }
            }
            else
            {
                throw new NotImplementedException($"soGroupByrt is not implemented for {icols.Length} columns.");
            }
        }

        public static IDataFrameViewGroupResults GroupBy(IDataFrameView df, IEnumerable<int> columns, bool ascending = true)
        {
            int[] icols = columns.ToArray();
            return RecGroupBy(df, icols, ascending);
        }

        #endregion
    }
}
