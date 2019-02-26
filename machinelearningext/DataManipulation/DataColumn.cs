// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements a dense column container.
    /// </summary>
    public class DataColumn<DType> : IDataColumn, IEquatable<DataColumn<DType>>, IEnumerable<DType>
        where DType : IEquatable<DType>, IComparable<DType>
    {
        #region members and easy functions

        /// <summary>
        /// Data for the column.
        /// </summary>
        DType[] _data;

        /// <summary>
        /// Number of elements.
        /// </summary>
        int _length;

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public IDataColumn Copy()
        {
            var res = new DataColumn<DType>(Length);
            Array.Copy(_data, res._data, Length);
            return res;
        }

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        public IDataColumn Copy(IEnumerable<int> rows)
        {
            var arows = rows.ToArray();
            var res = new DataColumn<DType>(arows.Length);
            for (int i = 0; i < arows.Length; ++i)
                res._data[i] = _data[arows[i]];
            return res;
        }

        /// <summary>
        /// Creates a new column with the same type but a new length and a constant value.
        /// </summary>
        public IDataColumn Create(int n, bool NA = false)
        {
            var res = new DataColumn<DType>(n);
            if (NA)
            {
                if (Kind.IsVector())
                    res.Set(null);
                else
                {
                    switch (Kind.RawKind())
                    {
                        case DataKind.Boolean: res.Set(false); break;
                        case DataKind.Int32: res.Set(0); break;
                        case DataKind.UInt32: res.Set(0); break;
                        case DataKind.Int64: res.Set(0); break;
                        case DataKind.Single: res.Set(float.NaN); break;
                        case DataKind.Double: res.Set(double.NaN); break;
                        case DataKind.String: res.Set(DvText.NA); break;
                        default:
                            throw new NotImplementedException($"No missing value convention for type '{Kind}'.");
                    }
                }
            }
            return res;
        }

        /// <summary>
        /// Concatenates multiple columns for the same type.
        /// </summary>
        public IDataColumn Concat(IEnumerable<IDataColumn> cols)
        {
            var data = new List<DType>();
            foreach (var col in cols)
            {
                var cast = col as DataColumn<DType>;
                if (cast == null)
                    throw new DataTypeError($"Unable to cast {col.GetType()} in {GetType()}.");
                data.AddRange(cast._data);
            }
            return new DataColumn<DType>(data.ToArray());
        }

        /// <summary>
        /// Number of elements.
        /// </summary>
        public int Length => _length;

        /// <summary>
        /// Number of elements in memory.
        /// </summary>
        public int MemoryLength => (_data == null ? 0 : _data.Length);

        /// <summary>
        /// Get a pointer on the raw data.
        /// </summary>
        public DType[] Data
        {
            get
            {
                if (_data == null || _length == _data.Length)
                    return _data;
                else
                {
                    var res = new DType[Length];
                    Array.Copy(_data, res, Length);
                    return res;
                }
            }
        }

        public object Get(int row) { return _data[row]; }

        public void Set(int row, object value)
        {
            DType dt;
            ObjectConversion.Convert(ref value, out dt);
            Set(row, dt);
        }

        /// <summary>
        /// Returns type data kind.
        /// </summary>
        public DataViewType Kind => SchemaHelper.GetColumnType<DType>();

        public IEnumerator<DType> GetEnumerator() { foreach (var v in _data) yield return v; }
        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }

        #endregion

        #region constructor

        /// <summary>
        /// Builds the columns.
        /// </summary>
        /// <param name="nb">number of rows</param>
        public DataColumn(int nb)
        {
            _data = new DType[nb];
            _length = nb;
        }

        /// <summary>
        /// Builds the columns.
        /// </summary>
        /// <param name="data">column data</param>
        public DataColumn(DType[] data)
        {
            _data = data;
            _length = data.Length;
        }

        /// <summary>
        /// Resizes the columns.
        /// </summary>
        /// <param name="keepData">keeps existing data</param>
        /// <param name="length">new length</param>
        public void Resize(int length, bool keepData = false)
        {
            if (length > _data.Length || length < _data.Length / 2)
            {
                _data = new DType[length];
                _length = length;
            }
            else
                _length = length;
        }

        /// <summary>
        /// Changes the value at a specific row.
        /// </summary>
        public void Set(int row, DType value)
        {
            _data[row] = value;
        }

        /// <summary>
        /// Changes the value at a specific row.
        /// </summary>
        public void Set(int row, ValueGetter<DType> getter)
        {
            getter(ref _data[row]);
        }

        /// <summary>
        /// Changes all values.
        /// </summary>
        public void Set(DType value)
        {
            for (int i = 0; i < Length; ++i)
                _data[i] = value;
        }

        /// <summary>
        /// Changes all values.
        /// </summary>
        public void SetDefault()
        {
            Set(default(DType));
        }

        /// <summary>
        /// Changes all values.
        /// </summary>
        public void Set(object value)
        {
            var numCol = value as NumericColumn;
            if (numCol is null)
            {
                var enumerable = value as IEnumerable;
                if (enumerable == null || value is string || value is ReadOnlyMemory<char>)
                {
                    DType dt;
                    ObjectConversion.Convert(ref value, out dt);
                    for (var row = 0; row < Length; ++row)
                        _data[row] = dt;
                }
                else
                {
                    DType[] dt;
                    ObjectConversion.Convert(ref value, out dt);
                    for (var row = 0; row < Length; ++row)
                        _data[row] = dt[row];
                }
            }
            else
            {
                var arr = numCol.Column as DataColumn<DType>;
                if (arr != null)
                {
                    DType[] dt = arr.Data;
                    for (var row = 0; row < Length; ++row)
                        _data[row] = dt[row];
                }
                else
                {
                    var t = typeof(DataColumn<DType>);
                    throw new DataValueError($"Column oof kind {numCol.Column.Kind} cannot be converted into {t}");
                }
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<bool> rows, object value)
        {
            var irow = 0;
            foreach (var row in rows)
            {
                if (row)
                    Set(irow, value);
                ++irow;
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<int> rows, object value)
        {
            foreach (var row in rows)
                Set(row, value);
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<bool> rows, IEnumerable<object> values)
        {
            var iter = values.GetEnumerator();
            var irow = 0;
            foreach (var row in rows)
            {
                iter.MoveNext();
                if (row)
                    Set(irow, iter.Current);
                ++irow;
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<int> rows, IEnumerable<object> values)
        {
            var iter = values.GetEnumerator();
            foreach (var row in rows)
            {
                iter.MoveNext();
                Set(row, iter.Current);
            }
        }

        /// <summary>
        /// Splits a vector column into multiple ones.
        /// </summary>
        public IDataFrameView Flatten(string name, IEnumerable<int> rows = null)
        {
            var kind = Kind;
            if (kind.IsVector())
            {
                switch (kind.ItemType().RawKind())
                {
                    case DataKind.Boolean: return BuildDataFrame(name, Flatten<bool>(rows));
                    case DataKind.Int32: return BuildDataFrame(name, Flatten<int>(rows));
                    case DataKind.Int64: return BuildDataFrame(name, Flatten<long>(rows));
                    case DataKind.UInt32: return BuildDataFrame(name, Flatten<uint>(rows));
                    case DataKind.Single: return BuildDataFrame(name, Flatten<float>(rows));
                    case DataKind.Double: return BuildDataFrame(name, Flatten<double>(rows));
                    case DataKind.String: return BuildDataFrame(name, Flatten<DvText>(rows));
                    default:
                        throw new NotImplementedException($"Unable to flatten column of type {kind}.");
                }

            }
            else
            {
                var df = new DataFrame();
                df.AddColumn(name, new DataColumn<DType>(rows == null ? _data : rows.Select(c => _data[c]).ToArray()));
                return df;
            }
        }

        private DT[][] Flatten<DT>(IEnumerable<int> rows = null)
            where DT : IEquatable<DT>, IComparable<DT>
        {
            var srows = rows == null ? Enumerable.Range(0, Length).ToArray() : rows.ToArray();
            var data = _data as VBufferEqSort<DT>[];
            var max = data.Select(c => c.Count).Max();
            var res = new DT[max][];
            for (int i = 0; i < max; ++i)
            {
                res[i] = new DT[srows.Length];
                var col = res[i];
                for (int j = 0; j < srows.Length; ++j)
                    col[j] = data[j].GetItemOrDefault(i);
            }
            return res;
        }

        public IDataFrameView BuildDataFrame<DT>(string name, DT[][] values)
        {
            var kind = SchemaHelper.GetColumnType<DT>();
            var df = new DataFrame();
            for (int i = 0; i < values.Length; ++i)
                df.AddColumn($"{name}.{i}", values[i]);
            return df;
        }

        /// <summary>
        /// Raises an exception if two columns do not have the same
        /// shape or are two much different.
        /// </summary>
        /// <param name="col">columns</param>
        /// <param name="precision">precision</param>
        /// <param name="exc">raises an exception if too different</param>
        /// <returns>max difference</returns>
        public double AssertAlmostEqual(IDataColumn col, double precision = 1e-5, bool exc = true)
        {
            var colt = (col is NumericColumn ? (col as NumericColumn).Column : col) as DataColumn<DType>;
            if (colt is null)
                throw new DataValueError(string.Format("Column types are different {0} != {1}",
                                                       GetType(), col.GetType()));
            if (Length != colt.Length)
                throw new DataValueError(string.Format("Column have different length {0} != {1}",
                                                       Length, colt.Length));
            if (Kind.IsVector())
            {
                double oks = 0;
                for (int i = 0; i < Length; ++i)
                {
                    switch (Kind.ItemType().RawKind())
                    {
                        case DataKind.Boolean:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<bool>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<bool>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        case DataKind.Int32:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<int>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<int>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        case DataKind.UInt32:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<uint>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<uint>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        case DataKind.Int64:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<Int64>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<Int64>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        case DataKind.Single:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<float>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<float>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        case DataKind.Double:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<double>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<double>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        case DataKind.String:
                            oks += NumericHelper.AssertAlmostEqual((_data as VBufferEqSort<DvText>[])[i].DenseValues().ToArray(),
                                                                   (colt._data as VBufferEqSort<DvText>[])[i].DenseValues().ToArray(),
                                                                   precision, exc);
                            break;
                        default:
                            throw new DataTypeError($"Unable to handle kind '{Kind}'");
                    }
                }
                return oks;
            }
            else
            {
                switch (Kind.RawKind())
                {
                    case DataKind.Boolean:
                        return NumericHelper.AssertAlmostEqual(_data as bool[], colt._data as bool[], precision, exc, Length, colt.Length);
                    case DataKind.Int32:
                        return NumericHelper.AssertAlmostEqual(_data as int[], colt._data as int[], precision, exc, Length, colt.Length);
                    case DataKind.UInt32:
                        return NumericHelper.AssertAlmostEqual(_data as uint[], colt._data as uint[], precision, exc, Length, colt.Length);
                    case DataKind.Int64:
                        return NumericHelper.AssertAlmostEqual(_data as long[], colt._data as long[], precision, exc, Length, colt.Length);
                    case DataKind.Single:
                        return NumericHelper.AssertAlmostEqual(_data as float[], colt._data as float[], precision, exc, Length, colt.Length);
                    case DataKind.Double:
                        return NumericHelper.AssertAlmostEqual(_data as double[], colt._data as double[], precision, exc, Length, colt.Length);
                    case DataKind.String:
                        return NumericHelper.AssertAlmostEqual(_data as DvText[], colt._data as DvText[], precision, exc, Length, colt.Length);
                    default:
                        throw new DataTypeError($"Unable to handle kind '{Kind}'");
                }
            }
        }

        /// <summary>
        /// Converts a column into another type.
        /// </summary>
        /// <param name="colType"column type></param>
        /// <returns>new column</returns>
        public IDataColumn AsType(DataViewType colType)
        {
            if (Kind == colType)
                return this;
            if (Kind.IsVector() || colType.IsVector())
                throw new NotImplementedException();
            else
            {
                switch (Kind.RawKind())
                {
                    case DataKind.Int32:
                        switch (colType.RawKind())
                        {
                            case DataKind.Single:
                                return new DataColumn<float>(NumericHelper.Convert(_data as int[], float.NaN));
                            default:
                                throw new NotImplementedException($"No conversion from '{Kind}' to '{colType.RawKind()}'.");
                        }
                    default:
                        throw new NotImplementedException($"No conversion from '{Kind}' to '{colType.RawKind()}'.");
                }
            }
        }

        #endregion

        #region linq

        public IEnumerable<bool> Filter<DType2>(Func<DType2, bool> predicate)
        {
            return (_data as DType2[]).Select(c => predicate(c));
        }

        public int[] Sort(bool ascending = true, bool inplace = true)
        {
            if (inplace)
            {
                Array.Sort(_data);
                if (!ascending)
                    Array.Reverse(_data);
                return null;
            }
            else
            {
                int[] order = null;
                Sort(ref order, ascending);
                return order;
            }
        }

        public void Sort(ref int[] order, bool ascending = true)
        {
            if (order == null)
            {
                order = new int[Length];
                for (int i = 0; i < order.Length; ++i)
                    order[i] = i;
            }

            if (ascending)
                Array.Sort(order, (x, y) => _data[x].CompareTo(_data[y]));
            else
                Array.Sort(order, (x, y) => -_data[x].CompareTo(_data[y]));
        }

        public void Order(int[] order)
        {
            var data = new DType[Length];
            for (int i = 0; i < Length; ++i)
                data[i] = _data[order[i]];
            _data = data;
        }

        public GetterAt<DType2> GetGetterAt<DType2>()
            where DType2 : IEquatable<DType2>, IComparable<DType2>
        {
            var res = GetGetterAtCore() as GetterAt<DType2>;
            if (res == null)
                throw new DataTypeError(string.Format("Type mismatch bytween {0} (expected) and {1} (given).", typeof(DType), typeof(DType2)));
            return res;
        }

        public GetterAt<DType> GetGetterAtCore()
        {
            return (int i, ref DType value) => { value = _data[i]; };
        }

        #endregion

        #region getter and comparison

        /// <summary>
        /// Creates a getter on the column. The getter returns the element at
        /// cursor.Position.
        /// </summary>
        public ValueGetter<DType2> GetGetter<DType2>(DataViewRowCursor cursor)
        {
            if (typeof(DType2) == typeof(ReadOnlyMemory<char>))
            {
                var res = GetGetterReadOnlyMemory(cursor) as ValueGetter<DType2>;
                if (res != null)
                    return res;
                throw new DataTypeError($"Unable to get a getter for type {typeof(DType2)} from type {typeof(DType)}.");
            }
            else
            {
                var _data2 = _data as DType2[];
                if (_data2 == null)
                    throw new NotSupportedException(string.Format("Unable to convert into {0}", typeof(DType2)));
                var missing = DataFrameMissingValue.GetMissingOrDefaultValue(Kind);
                return (ref DType2 value) =>
                {
                    value = cursor.Position < (long)Length
                            ? _data2[cursor.Position]
                            : (DType2)missing;
                };
            }
        }

        private ValueGetter<ReadOnlyMemory<char>> GetGetterReadOnlyMemory(DataViewRowCursor cursor)
        {
            var _data2 = _data as DvText[];
            if (_data2 == null)
                throw new NotSupportedException(string.Format("Unable to convert into DvText"));
            var missing = DataFrameMissingValue.GetMissingOrDefaultValue(Kind);
            return (ref ReadOnlyMemory<char> value) =>
            {
                value = cursor.Position < (long)Length
                        ? _data2[cursor.Position].str
                        : null;
            };
        }

        private static ValueGetter<VBuffer<DType2>> CheckNotEmpty<DType2, DT0>(ValueGetter<VBuffer<DT0>> dele)
        {
            var dele2 = dele as ValueGetter<VBuffer<DType2>>;
            if (dele2 == null)
                throw new DataTypeError($"Unable to get a getter for type {typeof(DType2)} from type {typeof(DT0)}.");
            return dele2;
        }

        /// <summary>
        /// Creates a getter on the column. The getter returns the element at
        /// cursor.Position.
        /// </summary>
        public ValueGetter<VBuffer<DType2>> GetGetterVector<DType2>(DataViewRowCursor cursor)
        {
            var colType = SchemaHelper.GetColumnType<DType2>();
            if (colType.IsVector())
                throw new DataValueError($"Unable to handle vector of kind {colType.ItemType().RawKind()}.");
            else
            {
                switch (colType.ItemType().RawKind())
                {
                    case DataKind.Boolean: return CheckNotEmpty<DType2, bool>(GetGetterVectorEqSort<bool>(cursor));
                    case DataKind.Int32: return CheckNotEmpty<DType2, int>(GetGetterVectorEqSort<int>(cursor));
                    case DataKind.UInt32: return CheckNotEmpty<DType2, uint>(GetGetterVectorEqSort<uint>(cursor));
                    case DataKind.Int64: return CheckNotEmpty<DType2, long>(GetGetterVectorEqSort<long>(cursor));
                    case DataKind.Single: return CheckNotEmpty<DType2, float>(GetGetterVectorEqSort<float>(cursor));
                    case DataKind.Double: return CheckNotEmpty<DType2, double>(GetGetterVectorEqSort<double>(cursor));
                    case DataKind.String: return GetGetterVectorEqSortText<DType2>(cursor);
                    default:
                        throw new DataValueError($"Unable to handle kind {colType.RawKind()}.");
                }
            }
        }

        public ValueGetter<VBuffer<DType2>> GetGetterVectorEqSort<DType2>(DataViewRowCursor cursor)
            where DType2 : IEquatable<DType2>, IComparable<DType2>
        {
            var _data2 = _data as VBufferEqSort<DType2>[];
            var missing = new VBuffer<DType2>();
            return (ref VBuffer<DType2> value) =>
            {
                value = cursor.Position < Length
                        ? _data2[cursor.Position].data
                        : missing;
            };
        }

        public ValueGetter<VBuffer<DType2>> GetGetterVectorEqSortText<DType2>(DataViewRowCursor cursor)
        {
            if (typeof(DType2) == typeof(DvText))
                return GetGetterVectorEqSort<DvText>(cursor) as ValueGetter<VBuffer<DType2>>;
            if (typeof(DType2) != typeof(ReadOnlyMemory<char>))
                throw new DataValueError($"Unable to create a getter of type {typeof(DType2)} from type {typeof(DType)}.");
            var getter = GetGetterVectorEqSortReadOnlyMemoryChar(cursor) as ValueGetter<VBuffer<DType2>>;
            if (getter == null)
                throw new DataValueError($"Unable to create a getter of type {typeof(DType2)} from type {typeof(DType)}.");
            return getter;
        }

        public ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetGetterVectorEqSortReadOnlyMemoryChar(DataViewRowCursor cursor)
        {
            var _data2 = _data as VBufferEqSort<DvText>[];
            var missing = new VBuffer<ReadOnlyMemory<char>>();
            return (ref VBuffer<ReadOnlyMemory<char>> value) =>
            {
                if (cursor.Position < Length)
                {
                    var el = _data2[cursor.Position].data;
                    value = new VBuffer<ReadOnlyMemory<char>>(el.Length, el.Count, el.Values.Select(c => c.str).ToArray(), el.Indices);
                }
                else
                    value = missing;
            };
        }

        public bool Equals(IDataColumn c)
        {
            var obj = c as DataColumn<DType>;
            if (obj == null)
                return false;
            return Equals(obj);
        }

        public bool Equals(DataColumn<DType> c)
        {
            if (Length != c.Length)
                return false;
            for (int i = 0; i < Length; ++i)
                if (!_data[i].Equals(c._data[i]))
                    return false;
            return true;
        }

        #endregion

        #region dataframe functions

        /// <summary>
        /// Applies the same function on every value of the column.
        /// </summary>
        public NumericColumn Apply<TSrc, TDst>(ValueMapper<TSrc, TDst> mapper)
            where TDst : IEquatable<TDst>, IComparable<TDst>
        {
            var maptyped = mapper as ValueMapper<DType, TDst>;
            if (maptyped == null)
                throw new DataValueError("Unexpected input type for this column.");
            var res = new DataColumn<TDst>(Length);
            for (int i = 0; i < res.Length; ++i)
                maptyped(in Data[i], ref res.Data[i]);
            return new NumericColumn(res);
        }

        public TSource Aggregate<TSource>(Func<TSource, TSource, TSource> func, int[] rows = null)
        {
            var funcTyped = func as Func<DType, DType, DType>;
            if (func == null)
                throw new NotSupportedException($"Type '{typeof(TSource)}' is not compatible with '{typeof(DType)}'.");
            var mapper = GetGenericConverter() as ValueMapper<DType, TSource>;
            var res = AggregateTyped(funcTyped, rows);
            var converted = default(TSource);
            mapper(in res, ref converted);
            return converted;
        }

        public TSource Aggregate<TSource>(Func<TSource[], TSource> func, int[] rows = null)
        {
            var funcTyped = func as Func<DType[], DType>;
            if (funcTyped == null)
                throw new NotSupportedException($"Type '{typeof(TSource)}' is not compatible with '{typeof(DType)}'.");
            var mapper = GetGenericConverter() as ValueMapper<DType, TSource>;
            var res = AggregateTyped(funcTyped, rows);
            var converted = default(TSource);
            mapper(in res, ref converted);
            return converted;
        }

        static ValueMapper<DType, DType> GetGenericConverter()
        {
            return (in DType src, ref DType dst) => { dst = src; };
        }

        public DType AggregateTyped(Func<DType, DType, DType> func, int[] rows = null)
        {
            if (rows == null)
                return _data.Aggregate(func);
            else
                return rows.Select(c => _data[c]).Aggregate(func);
        }

        public DType AggregateTyped(Func<DType[], DType> func, int[] rows = null)
        {
            if (rows == null)
                return func(_data);
            else
                return func(rows.Select(c => _data[c]).ToArray());
        }

        public IDataColumn Aggregate(AggregatedFunction func, int[] rows = null)
        {
            if (typeof(DType) == typeof(bool))
                return new DataColumn<bool>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(bool)), rows) });
            if (typeof(DType) == typeof(int))
                return new DataColumn<int>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(int)), rows) });
            if (typeof(DType) == typeof(uint))
                return new DataColumn<uint>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(uint)), rows) });
            if (typeof(DType) == typeof(long))
                return new DataColumn<long>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(long)), rows) });
            if (typeof(DType) == typeof(float))
                return new DataColumn<float>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(float)), rows) });
            if (typeof(DType) == typeof(double))
                return new DataColumn<double>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(double)), rows) });
            if (typeof(DType) == typeof(ReadOnlyMemory<char>))
                return new DataColumn<DvText>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(DvText)), rows) });
            if (typeof(DType) == typeof(DvText))
                return new DataColumn<DvText>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(DvText)), rows) });
            throw new NotImplementedException($"Unkown type '{typeof(DType)}'.");
        }
    }

    #endregion
}
