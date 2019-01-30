﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.DataManipulation;


namespace Scikit.ML.ProductionPrediction
{
    #region interface

    interface IInfiniteLoopViewCursorDataFrame
    {
        void Set(DataFrame value, int position = 0);
    }

    #endregion

    #region replace multiple columns

    /// <summary>
    /// Creates a view on one row and can loop on it after it was replaced.
    /// </summary>
    public class InfiniteLoopViewCursorDataFrame : IDataView, IInfiniteLoopViewCursorDataFrame
    {
        readonly int[] _columns;
        readonly Schema _schema;
        readonly RowCursor _otherValues;
        readonly Schema _columnsSchema;
        CursorType _ownCursor;

        public int[] ReplacedCol => _columns;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="columns">columns to be replaced</param>
        /// <param name="schema">schema of the view</param>
        /// <param name="otherValues">cursor which contains the others values</param>
        public InfiniteLoopViewCursorDataFrame(int[] columns = null, Schema schema = null, RowCursor otherValues = null)
        {
            if (columns == null)
                columns = Enumerable.Range(0, schema.Count).ToArray();
            _columns = columns;
            _columnsSchema = schema;
            if (columns.Length != _columnsSchema.Count)
                throw Contracts.Except($"Dimension mismatch expected columns is {columns.Length} not {_columnsSchema.Count}.");
            _otherValues = otherValues;
            _schema = otherValues == null ? schema : otherValues.Schema;
            _ownCursor = null;
            Contracts.AssertValue(_schema);
        }

        public bool CanShuffle { get { return false; } }
        public long? GetRowCount() { return null; }
        public Schema Schema { get { return _schema; } }

        public void Set(DataFrame value, int position = 0)
        {
            if (_ownCursor == null)
                throw Contracts.Except("GetRowCursor on this view was never called. No cursor is registered.");
            _ownCursor.Set(ref value, position);
        }

        public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            if (_ownCursor != null)
                throw Contracts.Except("GetRowCursor was called a second time which probably means this function was called from multiple threads. " +
                    "Be sure that an environment is called by parameter conc:1.");
            _ownCursor = new CursorType(this, columnsNeeded, _otherValues);
            return _ownCursor;
        }

        public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            var cur = GetRowCursor(columnsNeeded, rand);
            if (n >= 2)
            {
                /*
                var setColumns = new HashSet<int>(_columns);
                var res = new RowCursor[n];
                var empty = new EmptyCursor(this,
                                    col => setColumns.Contains(col) || needCol(col) || (_otherValues != null && _otherValues.IsColumnActive(col)));
                for (int i = 0; i < n; ++i)
                    res[i] = i == 0 ? cur : empty;
                return res.Take(1).ToArray();
                */
                return new RowCursor[] { cur };
            }
            else
                return new RowCursor[] { cur };
        }

        enum CursorState { Good, NotStarted, Done };

        class CursorType : RowCursor
        {
            readonly IEnumerable<Schema.Column> _columnsNeeded;
            readonly InfiniteLoopViewCursorDataFrame _view;
            readonly Schema _columnsSchema;
            CursorState _state;
            RowCursor _otherValues;
            DataFrame _container;
            int _positionDataFrame;
            Dictionary<int, int> _columns;
            bool _wait;
            long _position;
            long _batch;

            public CursorType(InfiniteLoopViewCursorDataFrame view, IEnumerable<Schema.Column> columnsNeeded, RowCursor otherValues)
            {
                _columnsNeeded = columnsNeeded;
                _view = view;
                _state = CursorState.NotStarted;
                _otherValues = otherValues;
                _wait = true;
                _position = 0;
                _batch = 1;
                _container = null;
                _positionDataFrame = -1;
                _columnsSchema = _view._columnsSchema;
                _columns = new Dictionary<int, int>();
                for (int i = 0; i < view.ReplacedCol.Length; ++i)
                    _columns[view.ReplacedCol[i]] = i;
            }

            public long? GetRowCount() { return 1; }
            public override long Batch { get { return _batch; } }
            public override long Position { get { return _position; } }
            public override Schema Schema { get { return _view.Schema; } }
            public override ValueGetter<RowId> GetIdGetter() { return (ref RowId uid) => { uid = new RowId(0, 1); }; }

            protected override void Dispose(bool disposing)
            {
                if (_otherValues != null)
                {
                    // Do not dispose the cursor. The current one does not call MoveNext,
                    // it does not own any cursor and should free any of them.
                    // _otherValues.Dispose();
                    _otherValues = null;
                }
                GC.SuppressFinalize(this);
            }

            public override bool MoveNext()
            {
                if (_state == CursorState.Done)
                    throw Contracts.Except("The state of the cursor should not be Done.");
                if (_wait)
                    throw Contracts.Except("The cursor has no value to show. This exception happens because a different thread is " +
                        "requested the next value or because a view is requesting for more than one value at a time.");
                _state = CursorState.Good;
                ++_positionDataFrame;
                ++_position;
                ++_batch;
                _wait = _positionDataFrame >= _container.Length;
                return true;
            }

            public void Set(ref DataFrame value, int position)
            {
                _container = value;
                _positionDataFrame = position;
                _wait = false;
            }

            public override bool IsColumnActive(int col)
            {
                return _columns.ContainsKey(col) ||
                    _columnsNeeded.Where(c => c.Index == col).Any() ||
                    (_otherValues != null && _otherValues.IsColumnActive(col));
            }

            /// <summary>
            /// Switches between the replaced column or the values coming
            /// from a view which has a column to be replaced,
            /// or the entire row (ReplacedCol == -1, _otherValues).
            /// </summary>
            /// <typeparam name="TValue">column type</typeparam>
            /// <param name="col">column index</param>
            /// <returns>ValueGetter</returns>
            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (_columns.ContainsKey(col))
                    return GetGetterPrivate<TValue>(col) as ValueGetter<TValue>;
                else if (_otherValues != null)
                    return _otherValues.GetGetter<TValue>(col);
                else
                    throw Contracts.Except("otherValues is null, unable to access other columns." +
                        "If you are using PrePostTransformPredictor, it means the preprossing transform cannot be " +
                        "converted into a IValueMapper because it relies on more than one columns.");
            }

            ValueGetter<TValue> GetGetterPrivate<TValue>(int col)
            {
                var coor = SchemaHelper.GetColumnType<TValue>();
                if (coor.IsVector())
                {
                    switch (coor.ItemType().RawKind())
                    {
                        case DataKind.BL:
                            return typeof(TValue) == typeof(VBufferEqSort<bool>)
                                         ? GetGetterPrivateI<VBufferEqSort<bool>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVector<bool>(col) as ValueGetter<TValue>;
                        case DataKind.I4:
                            return typeof(TValue) == typeof(VBufferEqSort<int>)
                                         ? GetGetterPrivateI<VBufferEqSort<int>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVector<int>(col) as ValueGetter<TValue>;
                        case DataKind.U4:
                            return typeof(TValue) == typeof(VBufferEqSort<uint>)
                                         ? GetGetterPrivateI<VBufferEqSort<uint>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVector<uint>(col) as ValueGetter<TValue>;
                        case DataKind.I8:
                            return typeof(TValue) == typeof(VBufferEqSort<Int64>)
                                         ? GetGetterPrivateI<VBufferEqSort<Int64>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVector<Int64>(col) as ValueGetter<TValue>;
                        case DataKind.R4:
                            return typeof(TValue) == typeof(VBufferEqSort<float>)
                                         ? GetGetterPrivateI<VBufferEqSort<float>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVector<float>(col) as ValueGetter<TValue>;
                        case DataKind.R8:
                            return typeof(TValue) == typeof(VBufferEqSort<double>)
                                         ? GetGetterPrivateI<VBufferEqSort<double>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVector<double>(col) as ValueGetter<TValue>;
                        case DataKind.TX:
                            return typeof(TValue) == typeof(VBufferEqSort<DvText>)
                                         ? GetGetterPrivateI<VBufferEqSort<DvText>>(col) as ValueGetter<TValue>
                                         : GetGetterPrivateIVectorText(col) as ValueGetter<TValue>;
                        default:
                            throw new DataTypeError(string.Format("Type '{0}' is not handled.", coor));
                    }
                }
                else
                {
                    switch (coor.RawKind())
                    {
                        case DataKind.BL: return GetGetterPrivateI<bool>(col) as ValueGetter<TValue>;
                        case DataKind.I4: return GetGetterPrivateI<int>(col) as ValueGetter<TValue>;
                        case DataKind.U4: return GetGetterPrivateI<uint>(col) as ValueGetter<TValue>;
                        case DataKind.I8: return GetGetterPrivateI<Int64>(col) as ValueGetter<TValue>;
                        case DataKind.R4: return GetGetterPrivateI<float>(col) as ValueGetter<TValue>;
                        case DataKind.R8: return GetGetterPrivateI<double>(col) as ValueGetter<TValue>;
                        case DataKind.TX:
                            {
                                if (typeof(TValue) == typeof(DvText))
                                    return GetGetterPrivateI<DvText>(col) as ValueGetter<TValue>;
                                else
                                    return GetGetterPrivateIText(col) as ValueGetter<TValue>;
                            }
                        default:
                            throw new DataTypeError(string.Format("Type {0} is not handled.", coor.RawKind()));
                    }
                }
            }

            ValueGetter<TValue> GetGetterPrivateI<TValue>(int col)
                where TValue : IEquatable<TValue>, IComparable<TValue>
            {
                var schema = Schema;
                Contracts.CheckValue(schema, nameof(schema));
                return (ref TValue value) =>
                {
                    DataColumn<TValue> column;
                    _container.GetTypedColumn(col, out column);
                    value = column.Data[_positionDataFrame - 1];
                };
            }

            ValueGetter<VBuffer<TValue>> GetGetterPrivateIVector<TValue>(int col)
                where TValue : IEquatable<TValue>, IComparable<TValue>
            {
                var schema = Schema;
                Contracts.CheckValue(schema, nameof(schema));
                return (ref VBuffer<TValue> value) =>
                {
                    DataColumn<VBufferEqSort<TValue>> column;
                    _container.GetTypedColumn(col, out column);
                    var t = column.Data[_positionDataFrame - 1];
                    value = new VBuffer<TValue>(t.Length, t.Count, t.Values, t.Indices);
                };
            }

            ValueGetter<ReadOnlyMemory<char>> GetGetterPrivateIText(int col)
            {
                var schema = Schema;
                Contracts.CheckValue(schema, nameof(schema));
                return (ref ReadOnlyMemory<char> value) =>
                {
                    DataColumn<DvText> column;
                    _container.GetTypedColumn(col, out column);
                    value = column.Data[_positionDataFrame - 1].str;
                };
            }

            ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetGetterPrivateIVectorText(int col)
            {
                var schema = Schema;
                Contracts.CheckValue(schema, nameof(schema));
                return (ref VBuffer<ReadOnlyMemory<char>> value) =>
                {
                    DataColumn<VBufferEqSort<DvText>> column;
                    _container.GetTypedColumn(col, out column);
                    var t = column.Data[_positionDataFrame - 1];
                    value = new VBuffer<ReadOnlyMemory<char>>(t.Length, t.Count, t.Values.Select(c => c.str).ToArray(), t.Indices);
                };
            }
        }
    }

    #endregion
}
