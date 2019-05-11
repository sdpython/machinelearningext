// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    #region interface

    interface IInfiniteLoopViewCursor<TRepValue>
    {
        void Set(in TRepValue value);
    }

    #endregion

    #region replace 1 column

    /// <summary>
    /// Creates a view on one row and can loop on it after it was replaced.
    /// </summary>
    public class InfiniteLoopViewCursorColumn<TRepValue> : IDataView, IInfiniteLoopViewCursor<TRepValue>
    {
        readonly int _column;
        readonly DataViewSchema _schema;
        readonly DataViewRowCursor _otherValues;
        readonly bool _ignoreOtherColumn;
        CursorType _ownCursor;

        public int ReplacedCol { get { return _column; } }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="column">column to be replaced</param>
        /// <param name="schema">schema of the view</param>
        /// <param name="otherValues">cursor which contains the others values</param>
        /// <param name="ignoreOtherColumn">ignore other column if they are being requested</param>
        public InfiniteLoopViewCursorColumn(int column, DataViewSchema schema = null, DataViewRowCursor otherValues = null, bool ignoreOtherColumn = false)
        {
            _column = column;
            _otherValues = otherValues;
            _schema = otherValues == null ? schema : otherValues.Schema;
            _ownCursor = null;
            _ignoreOtherColumn = ignoreOtherColumn;
            Contracts.AssertValue(_schema);
        }

        public bool CanShuffle { get { return false; } }
        public long? GetRowCount() { return 1; }
        public DataViewSchema Schema { get { return _schema; } }

        public void Set(in TRepValue value)
        {
            if (_ownCursor == null)
                throw Contracts.Except("GetRowCursor on this view was never called. No cursor is registered.");
            _ownCursor.Set(in value);
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            if (_ownCursor != null)
                throw Contracts.Except("GetRowCursor was called a second time which probably means this function was called from multiple threads.");
            _ownCursor = new CursorType(this, columnsNeeded, _otherValues);
            return _ownCursor;
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            var cur = GetRowCursor(columnsNeeded, rand);
            if (n >= 2)
            {
                // This trick avoids the cursor to be split into multiple later.
                /*
                var res = new DataViewRowCursor[n];
                var empty = new EmptyCursor(this,
                                    col => col == _column || needCol(col) || (_otherValues != null && _otherValues.IsColumnActive(col)));
                for (int i = 0; i < n; ++i)
                    res[i] = i == 0 ? cur : empty;
                return res;
                */
                return new DataViewRowCursor[] { cur };
            }
            else
                return new DataViewRowCursor[] { cur };
        }

        enum CursorState { Good, NotStarted, Done };

        class CursorType : DataViewRowCursor
        {
            IEnumerable<DataViewSchema.Column> _columnsNeeded;
            InfiniteLoopViewCursorColumn<TRepValue> _view;
            CursorState _state;
            DataViewRowCursor _otherValues;
            TRepValue[] _container;
            bool _wait;
            long _position;
            long _batch;
            bool _ignoreOtherColumn;

            public CursorType(InfiniteLoopViewCursorColumn<TRepValue> view, IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewRowCursor otherValues)
            {
                _columnsNeeded = columnsNeeded;
                _view = view;
                _state = CursorState.NotStarted;
                _container = new TRepValue[1];
                _otherValues = otherValues;
                _wait = true;
                _position = 0;
                _batch = 1;
                _ignoreOtherColumn = view._ignoreOtherColumn;
            }

            //public override int Count() { return 1; }
            public override long Batch { get { return _batch; } }
            public override long Position { get { return _position; } }
            public override DataViewSchema Schema { get { return _view.Schema; } }
            public override ValueGetter<DataViewRowId> GetIdGetter() { return (ref DataViewRowId uid) => { uid = new DataViewRowId(0, 1); }; }

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
                    throw Contracts.Except("The cursor has no value to show. This exception happens because a different " +
                                           "thread is requested the next value or because a view is requesting for " +
                                           "more than one value at a time.");
                _state = CursorState.Good;
                ++_position;
                ++_batch;
                _wait = false;
                return true;
            }

            public void Set(in TRepValue value)
            {
                _container[0] = value;
                _wait = false;
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                return col.Index == _view._column ||
                    _columnsNeeded.Where(c => c.Index == col.Index).Any() ||
                    (_otherValues != null && _otherValues.IsColumnActive(col));
            }

            /// <summary>
            /// Switches between the replaced column or the values coming
            /// from a view which has a column to be replaced,
            /// or the entire row (ReplacedCol == -1, _otherValues).
            /// </summary>
            /// <typeparam name="TValue">column type</typeparam>
            /// <param name="col">column number</param>
            /// <returns>ValueGetter</returns>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                if (col.Index == _view.ReplacedCol)
                    return GetGetterPrivate(col) as ValueGetter<TValue>;
                else if (_otherValues != null)
                    return _otherValues.GetGetter<TValue>(col);
                else if (_ignoreOtherColumn)
                    return (ref TValue value) =>
                    {
                        value = default(TValue);
                    };
                else
                    throw Contracts.Except("otherValues is null, unable to access other columns." +
                        "If you are using PrePostTransformPredictor, it means the preprossing transform cannot be " +
                        "converted into a IValueMapper because it relies on more than one columns.");
            }

            public ValueGetter<TRepValue> GetGetterPrivate(DataViewSchema.Column col)
            {
                if (col.Index == _view.ReplacedCol)
                {
                    var type = _view.Schema[col.Index].Type;
                    if (type.IsVector())
                    {
                        switch (type.AsVector().ItemType().RawKind())
                        {
                            case DataKind.Single:
                                return GetGetterPrivateVector<float>(col) as ValueGetter<TRepValue>;
                            default:
                                throw Contracts.ExceptNotSupp("Unable to get a getter for type {0}", type.ToString());
                        }
                    }
                    else
                    {
                        return (ref TRepValue value) =>
                        {
                            value = _container[0];
                        };
                    }
                }
                else
                    throw Contracts.ExceptNotSupp();
            }

            public ValueGetter<VBuffer<TRepValueItem>> GetGetterPrivateVector<TRepValueItem>(DataViewSchema.Column col)
            {
                if (col.Index == _view.ReplacedCol)
                {
                    return (ref VBuffer<TRepValueItem> value) =>
                    {
                        VBuffer<TRepValueItem> cast = (VBuffer<TRepValueItem>)(object)_container[0];
                        cast.CopyTo(ref value);
                    };
                }
                else
                    throw Contracts.ExceptNotSupp("Unable to create a vector getter.");
            }
        }
    }

    #endregion

    #region replace multiple columns

    /// <summary>
    /// Creates a view on one row and can loop on it after it was replaced.
    /// </summary>
    public class InfiniteLoopViewCursorRow<TRowValue> : IDataView, IInfiniteLoopViewCursor<TRowValue>
        where TRowValue : class
    {
        readonly int[] _columns;
        readonly DataViewSchema _schema;
        readonly DataViewRowCursor _otherValues;
        readonly SchemaDefinition _columnsSchema;
        readonly Dictionary<string, Delegate> _overwriteRowGetter;
        CursorType _ownCursor;

        public int[] ReplacedCol => _columns;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="columns">columns to be replaced</param>
        /// <param name="schema">schema of the view</param>
        /// <param name="otherValues">cursor which contains the others values</param>
        public InfiniteLoopViewCursorRow(int[] columns = null, DataViewSchema schema = null, DataViewRowCursor otherValues = null,
                                         Dictionary<string, Delegate> overwriteRowGetter = null)
        {
            var columnsSchema = SchemaDefinition.Create(typeof(TRowValue), SchemaDefinition.Direction.Read);
            if (columns == null)
                columns = columnsSchema.Select((c, i) => i).ToArray();
            if (columns.Length != columnsSchema.Count)
                throw Contracts.Except($"Dimension mismatch, expected columns is {columns.Length}, number of fields for {typeof(TRowValue)} is {columnsSchema.Count}.");
            _columns = columns;
            _otherValues = otherValues;
            _schema = otherValues == null ? schema : otherValues.Schema;
            _ownCursor = null;
            _columnsSchema = columnsSchema;
            _overwriteRowGetter = overwriteRowGetter;
            Contracts.AssertValue(_schema);
        }

        public bool CanShuffle { get { return false; } }
        public long? GetRowCount() { return null; }
        public DataViewSchema Schema { get { return _schema; } }

        public void Set(in TRowValue value)
        {
            if (_ownCursor == null)
                throw Contracts.Except("GetRowCursor on this view was never called. No cursor is registered.");
            _ownCursor.Set(in value);
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            if (_ownCursor != null)
                throw Contracts.Except("GetRowCursor was called a second time which probably means this function was called from multiple threads. " +
                    "Be sure that an environment is called by parameter conc:1.");
            _ownCursor = new CursorType(this, columnsNeeded, _otherValues);
            return _ownCursor;
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            var cur = GetRowCursor(columnsNeeded, rand);
            if (n >= 2)
            {
                /*
                // This trick avoids the cursor to be split into multiple later.
                var setColumns = new HashSet<int>(_columns);
                var res = new DataViewRowCursor[n];
                var empty = new EmptyCursor(this,
                                    col => setColumns.Contains(col) || needCol(col) || (_otherValues != null && _otherValues.IsColumnActive(col)));
                for (int i = 0; i < n; ++i)
                    res[i] = i == 0 ? cur : empty;
                return res;
                */
                return new DataViewRowCursor[] { cur };
            }
            else
                return new DataViewRowCursor[] { cur };
        }

        enum CursorState { Started, Done, NotStarted, Good };

        class CursorType : DataViewRowCursor
        {
            IEnumerable<DataViewSchema.Column> _columnsNeeded;
            InfiniteLoopViewCursorRow<TRowValue> _view;
            SchemaDefinition _columnsSchema;
            CursorState _state;
            DataViewRowCursor _otherValues;
            TRowValue[] _container;
            Dictionary<int, int> _columns;
            Dictionary<string, Delegate> _overwriteRowGetter;
            bool _wait;
            long _position;
            long _batch;

            public CursorType(InfiniteLoopViewCursorRow<TRowValue> view, IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewRowCursor otherValues)
            {
                _columnsNeeded = columnsNeeded;
                _view = view;
                _state = CursorState.NotStarted;
                _container = new TRowValue[1];
                _otherValues = otherValues;
                _wait = true;
                _position = 0;
                _batch = 1;
                _columnsSchema = _view._columnsSchema;
                _overwriteRowGetter = _view._overwriteRowGetter;
                _columns = new Dictionary<int, int>();
                for (int i = 0; i < view.ReplacedCol.Length; ++i)
                    _columns[view.ReplacedCol[i]] = i;
            }

            //public override int Count() { return 1; }
            public override long Batch { get { return _batch; } }
            public override long Position { get { return _position; } }
            public override DataViewSchema Schema { get { return _view.Schema; } }
            public override ValueGetter<DataViewRowId> GetIdGetter() { return (ref DataViewRowId uid) => { uid = new DataViewRowId(0, 1); }; }

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
                    throw Contracts.Except("The cursor has no value to show. This exception happens because a different " +
                                           "thread is requested the next value or because a view is requesting for more " +
                                           "than one value at a time.");
                _state = CursorState.Good;
                ++_position;
                ++_batch;
                _wait = false;
                return true;
            }

            public void Set(in TRowValue value)
            {
                _container[0] = value;
                _wait = false;
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                return _columns.ContainsKey(col.Index) ||
                        _columnsNeeded.Where(c => c.Index == col.Index).Any() ||
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
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                if (_columns.ContainsKey(col.Index))
                    return GetGetterPrivate<TValue>(col) as ValueGetter<TValue>;
                else if (_otherValues != null)
                    return _otherValues.GetGetter<TValue>(col);
                else
                    throw Contracts.Except("otherValues is null, unable to access other columns." +
                        "If you are using PrePostTransformPredictor, it means the preprossing transform cannot be " +
                        "converted into a IValueMapper because it relies on more than one columns.");
            }

            public ValueGetter<TValue> GetGetterPrivate<TValue>(DataViewSchema.Column col)
            {
                int rowColumns = _columns[col.Index];
                var name = _columnsSchema[rowColumns].ColumnName;
                if (_overwriteRowGetter.ContainsKey(name))
                {
                    var getter = _overwriteRowGetter[name] as ValueGetterInstance<TRowValue, TValue>;
                    if (getter == null)
                        throw Contracts.Except($"Irreconcilable types {_overwriteRowGetter[name].GetType()} != {typeof(ValueGetterInstance<TRowValue, TValue>)}.");
                    return (ref TValue value) =>
                    {
                        getter(ref _container[0], ref value);
                    };
                }
                else
                {
                    var prop = typeof(TRowValue).GetProperty(name);
                    var getMethod = prop.GetGetMethod();
                    if (getMethod == null)
                        throw Contracts.Except($"GetMethod returns null for type {typeof(TRowValue)} and member '{name}'");
                    throw Contracts.ExceptNotSupp($"Getter must be specified.");
                }
            }
        }
    }

    #endregion
}
