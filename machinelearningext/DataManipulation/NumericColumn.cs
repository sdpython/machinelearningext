// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Wraps a column and adds numerical operators.
    /// </summary>
    public class NumericColumn : IDataColumn
    {
        #region members and simple functions

        protected IDataColumn _column;

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public IDataColumn Copy()
        {
            return new NumericColumn(_column.Copy());
        }

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        public IDataColumn Copy(IEnumerable<int> rows)
        {
            return new NumericColumn(_column.Copy(rows));
        }

        public NumericColumn(IDataColumn column)
        {
            _column = column;
        }

        public IDataColumn Column { get { return _column; } }
        public int Length => _column.Length;
        public int MemoryLength => _column.MemoryLength;
        public DataViewType Kind => _column.Kind;
        public object Get(int row) => _column.Get(row);
        public void Set(int row, object value) { _column.Set(row, value); }
        public void Set<T>(int row, T value) { _column.Set(row, value); }
        public void Set(object value) { _column.Set(value); }
        public void SetDefault() { _column.SetDefault(); }
        public IDataFrameView Flatten(string name, IEnumerable<int> rows = null) { return _column.Flatten(name, rows); }
        public void Resize(int length, bool keepData = false) { _column.Resize(length, keepData); }
        public ValueGetter<DType> GetGetter<DType>(DataViewRowCursor cursor) => _column.GetGetter<DType>(cursor);
        public ValueGetter<VBuffer<DType>> GetGetterVector<DType>(DataViewRowCursor cursor) => _column.GetGetterVector<DType>(cursor);
        public bool Equals(IDataColumn col) => _column.Equals(col);

        public NumericColumn Apply<TSrc, TDst>(ValueMapper<TSrc, TDst> mapper) where TDst : IEquatable<TDst>, IComparable<TDst>
        {
            return Column.Apply(mapper);
        }

        public void Sort(ref int[] order, bool ascending = true) { Column.Sort(ref order, ascending); }
        public int[] Sort(bool ascending = true, bool inplace = true) { return Column.Sort(ascending, inplace); }
        public void Order(int[] order) { Column.Order(order); }
        public GetterAt<DType> GetGetterAt<DType>() where DType : IEquatable<DType>, IComparable<DType> { return Column.GetGetterAt<DType>(); }
        public IDataColumn Create(int n, bool NA = false) { return Column.Create(n, NA); }

        public IDataColumn Concat(IEnumerable<IDataColumn> cols)
        {
            var filt = new List<IDataColumn>();
            foreach (var col in cols)
            {
                var cast = col as NumericColumn;
                filt.Add(cast is null ? col : cast.Column);
            }
            return new NumericColumn(filt[0].Concat(filt));
        }

        /// <summary>
        /// Raises an exception if two columns do not have the same
        /// shape or are two much different.
        /// </summary>
        /// <param name="df">columns</param>
        /// <param name="precision">precision</param>
        /// <param name="exc">raises an exception if too different</param>
        /// <returns>max difference</returns>
        public double AssertAlmostEqual(IDataColumn col, double precision = 1e-5, bool exc = true)
        {
            return _column.AssertAlmostEqual(col, precision, exc);
        }

        /// <summary>
        /// Converts a column into another type.
        /// </summary>
        /// <param name="colType">column type</param>
        /// <returns>new columns</returns>
        public IDataColumn AsType(DataViewType colType)
        {
            return _column.AsType(colType);
        }

        #endregion

        #region Set + Enumerator

        public void Set(IEnumerable<bool> rows, object value) { Column.Set(rows, value); }
        public void Set(IEnumerable<int> rows, object value) { Column.Set(rows, value); }
        public void Set(IEnumerable<bool> rows, IEnumerable<object> values) { Column.Set(rows, values); }
        public void Set(IEnumerable<int> rows, IEnumerable<object> values) { Column.Set(rows, values); }

        #endregion

        #region linq

        public IEnumerable<bool> Filter<TSource>(Func<TSource, bool> predicate) { return Column.Filter(predicate); }

        #endregion

        public virtual DType[] GetData<DType>()
        {
            throw new NotImplementedException("This function must be overwritten.");
        }

        #region +

        public static NumericColumn operator +(NumericColumn c1, NumericColumn c2) { return DataFrameOpAdditionHelper.Operation(c1, c2); }
        public static NumericColumn operator +(NumericColumn c1, int value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, long value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, float value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, double value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, DvText value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, string value) { return DataFrameOpAdditionHelper.Operation(c1, value); }

        #endregion

        #region *

        public static NumericColumn operator *(NumericColumn c1, NumericColumn c2) { return DataFrameOpMultiplicationHelper.Operation(c1, c2); }
        public static NumericColumn operator *(NumericColumn c1, int value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, long value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, float value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, double value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }

        #endregion

        #region /

        public static NumericColumn operator /(NumericColumn c1, NumericColumn c2) { return DataFrameOpDivisionHelper.Operation(c1, c2); }
        public static NumericColumn operator /(NumericColumn c1, int value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, long value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, float value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, double value) { return DataFrameOpDivisionHelper.Operation(c1, value); }

        #endregion

        #region -

        public static NumericColumn operator -(NumericColumn c1, NumericColumn c2) { return DataFrameOpSoustractionHelper.Operation(c1, c2); }
        public static NumericColumn operator -(NumericColumn c1, int value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, long value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, float value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, double value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }

        public static NumericColumn operator -(NumericColumn c1) { return DataFrameOpMinusHelper.Operation(c1); }

        #endregion

        #region ==

        public override bool Equals(object o) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static NumericColumn operator ==(NumericColumn c1, NumericColumn c2) { return DataFrameOpEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator ==(NumericColumn c1, int value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, long value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, float value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, double value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, bool value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, string value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, DvText value) { return DataFrameOpEqualHelper.Operation(c1, value); }

        #endregion

        #region !=, !

        public static NumericColumn operator !=(NumericColumn c1, NumericColumn c2) { return DataFrameOpNotEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator !=(NumericColumn c1, int value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, long value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, float value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, double value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, bool value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, string value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, DvText value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }

        public static NumericColumn operator !(NumericColumn c1) { return DataFrameOpNotHelper.Operation(c1); }

        #endregion

        #region >

        public static NumericColumn operator >(NumericColumn c1, NumericColumn c2) { return DataFrameOpSupHelper.Operation(c1, c2); }
        public static NumericColumn operator >(NumericColumn c1, int value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, long value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, float value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, double value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, string value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, DvText value) { return DataFrameOpSupHelper.Operation(c1, value); }

        #endregion

        #region >=

        public static NumericColumn operator >=(NumericColumn c1, NumericColumn c2) { return DataFrameOpSupEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator >=(NumericColumn c1, int value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, long value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, float value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, double value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, string value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, DvText value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }

        #endregion

        #region <

        public static NumericColumn operator <(NumericColumn c1, NumericColumn c2) { return DataFrameOpInfHelper.Operation(c1, c2); }
        public static NumericColumn operator <(NumericColumn c1, int value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, long value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, float value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, double value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, string value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, DvText value) { return DataFrameOpInfHelper.Operation(c1, value); }

        #endregion

        #region <=

        public static NumericColumn operator <=(NumericColumn c1, NumericColumn c2) { return DataFrameOpInfEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator <=(NumericColumn c1, int value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, long value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, float value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, double value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, ReadOnlyMemory<char> value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, string value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, DvText value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }

        #endregion

        #region &, |

        public static NumericColumn operator &(NumericColumn c1, NumericColumn c2) { return DataFrameOpAndHelper.Operation(c1, c2); }
        public static NumericColumn operator &(NumericColumn c1, bool value) { return DataFrameOpAndHelper.Operation(c1, value); }

        public static NumericColumn operator |(NumericColumn c1, NumericColumn c2) { return DataFrameOpOrHelper.Operation(c1, c2); }
        public static NumericColumn operator |(NumericColumn c1, bool value) { return DataFrameOpOrHelper.Operation(c1, value); }

        #endregion

        #region aggregation

        public TSource Aggregate<TSource>(Func<TSource, TSource, TSource> func, int[] rows = null) { return Column.Aggregate(func, null); }

        public TSource Aggregate<TSource>(Func<TSource[], TSource> func, int[] rows = null) { return Column.Aggregate(func, null); }

        public IDataColumn Aggregate(AggregatedFunction func, int[] rows = null) { return Column.Aggregate(func, rows); }

        #endregion
    }
}
