﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
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
        public DataKind Kind => _column.Kind;
        public object Get(int row) => _column.Get(row);
        public void Set(int row, object value) { _column.Set(row, value); }
        public void Set(object value) { _column.Set(value); }
        public ValueGetter<DType> GetGetter<DType>(IRowCursor cursor) => _column.GetGetter<DType>(cursor);
        public bool Equals(IDataColumn col) => _column.Equals(col);

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
        public static NumericColumn operator +(NumericColumn c1, DvInt4 value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, Int64 value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, DvInt8 value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, float value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, double value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, DvText value) { return DataFrameOpAdditionHelper.Operation(c1, value); }
        public static NumericColumn operator +(NumericColumn c1, string value) { return DataFrameOpAdditionHelper.Operation(c1, value); }

        #endregion

        #region *

        public static NumericColumn operator *(NumericColumn c1, NumericColumn c2) { return DataFrameOpMultiplicationHelper.Operation(c1, c2); }
        public static NumericColumn operator *(NumericColumn c1, int value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, DvInt4 value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, Int64 value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, DvInt8 value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, float value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }
        public static NumericColumn operator *(NumericColumn c1, double value) { return DataFrameOpMultiplicationHelper.Operation(c1, value); }

        #endregion

        #region /

        public static NumericColumn operator /(NumericColumn c1, NumericColumn c2) { return DataFrameOpDivisionHelper.Operation(c1, c2); }
        public static NumericColumn operator /(NumericColumn c1, int value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, DvInt4 value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, Int64 value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, DvInt8 value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, float value) { return DataFrameOpDivisionHelper.Operation(c1, value); }
        public static NumericColumn operator /(NumericColumn c1, double value) { return DataFrameOpDivisionHelper.Operation(c1, value); }

        #endregion

        #region -

        public static NumericColumn operator -(NumericColumn c1, NumericColumn c2) { return DataFrameOpSoustractionHelper.Operation(c1, c2); }
        public static NumericColumn operator -(NumericColumn c1, int value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, DvInt4 value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, Int64 value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, DvInt8 value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, float value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }
        public static NumericColumn operator -(NumericColumn c1, double value) { return DataFrameOpSoustractionHelper.Operation(c1, value); }

        public static NumericColumn operator -(NumericColumn c1) { return DataFrameOpMinusHelper.Operation(c1); }

        #endregion

        #region ==

        public override bool Equals(object o) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static NumericColumn operator ==(NumericColumn c1, NumericColumn c2) { return DataFrameOpEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator ==(NumericColumn c1, int value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, DvInt4 value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, Int64 value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, DvInt8 value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, float value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, double value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, bool value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, DvBool value) { return DataFrameOpEqualHelper.Operation(c1, value); }
        public static NumericColumn operator ==(NumericColumn c1, string value) { return DataFrameOpEqualHelper.Operation(c1, value); }

        #endregion

        #region !=

        public static NumericColumn operator !=(NumericColumn c1, NumericColumn c2) { return DataFrameOpNotEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator !=(NumericColumn c1, int value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, DvInt4 value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, Int64 value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, DvInt8 value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, float value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, double value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, bool value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, DvBool value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }
        public static NumericColumn operator !=(NumericColumn c1, string value) { return DataFrameOpNotEqualHelper.Operation(c1, value); }

        #endregion

        #region >

        public static NumericColumn operator >(NumericColumn c1, NumericColumn c2) { return DataFrameOpSupHelper.Operation(c1, c2); }
        public static NumericColumn operator >(NumericColumn c1, int value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, DvInt4 value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, Int64 value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, DvInt8 value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, float value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, double value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, DvText value) { return DataFrameOpSupHelper.Operation(c1, value); }
        public static NumericColumn operator >(NumericColumn c1, string value) { return DataFrameOpSupHelper.Operation(c1, value); }

        #endregion

        #region >=

        public static NumericColumn operator >=(NumericColumn c1, NumericColumn c2) { return DataFrameOpSupEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator >=(NumericColumn c1, int value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, DvInt4 value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, Int64 value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, DvInt8 value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, float value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, double value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, DvText value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }
        public static NumericColumn operator >=(NumericColumn c1, string value) { return DataFrameOpSupEqualHelper.Operation(c1, value); }

        #endregion

        #region <

        public static NumericColumn operator <(NumericColumn c1, NumericColumn c2) { return DataFrameOpInfHelper.Operation(c1, c2); }
        public static NumericColumn operator <(NumericColumn c1, int value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, DvInt4 value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, Int64 value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, DvInt8 value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, float value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, double value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, DvText value) { return DataFrameOpInfHelper.Operation(c1, value); }
        public static NumericColumn operator <(NumericColumn c1, string value) { return DataFrameOpInfHelper.Operation(c1, value); }

        #endregion

        #region <=

        public static NumericColumn operator <=(NumericColumn c1, NumericColumn c2) { return DataFrameOpInfEqualHelper.Operation(c1, c2); }
        public static NumericColumn operator <=(NumericColumn c1, int value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, DvInt4 value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, Int64 value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, DvInt8 value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, float value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, double value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, DvText value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }
        public static NumericColumn operator <=(NumericColumn c1, string value) { return DataFrameOpInfEqualHelper.Operation(c1, value); }

        #endregion
    }
}
