﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Converts a ValueMapper into a IDataTransform.
    /// Similar to a scorer but in a more explicit way.
    /// </summary>
    public class TransformFromValueMapper : IDataTransform, IValueMapper
    {
        #region members

        readonly IDataTransform _transform;
        readonly IDataView _source;
        readonly IHostEnvironment _host;
        readonly IValueMapper _mapper;
        readonly string _inputColumn;
        readonly string _outputColumn;
        readonly Schema _schema;

        #endregion

        #region constructors, transform API

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="mapper">IValueMapper</param>
        /// <param name="source">source to replace</param>
        /// <param name="inputColumn">name of the input column (the last one sharing the same type)</param>
        /// <param name="outputColumn">name of the output column</param>
        public TransformFromValueMapper(IHostEnvironment env, IValueMapper mapper, IDataView source,
                                        string inputColumn, string outputColumn = "output")
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(source);
            _host = env;

            if (string.IsNullOrEmpty(inputColumn))
            {
                var inputType = mapper.InputType;
                for (int i = source.Schema.Count - 1; i >= 0; --i)
                {
                    var ty = source.Schema[i].Type;
                    if (ty.SameSizeAndItemType(inputType))
                    {
                        inputColumn = source.Schema[i].Name;
                        break;
                    }
                }
            }

            _source = source;
            _mapper = mapper;
            int index = SchemaHelper.GetColumnIndex(_source.Schema, inputColumn);
            _inputColumn = inputColumn;
            _outputColumn = outputColumn;
            _schema = ExtendedSchema.Create(new ExtendedSchema(source.Schema, new[] { outputColumn }, new[] { mapper.OutputType }));
            _transform = CreateMemoryTransform();
        }

        public ColumnType InputType { get { return _mapper.InputType; } }
        public ColumnType OutputType { get { return _mapper.OutputType; } }
        public string InputName { get { return _inputColumn; } }
        public string OutputName { get { return _outputColumn; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>() { return _mapper.GetMapper<TSrc, TDst>(); }
        public IDataView Source { get { return _source; } }
        public bool CanShuffle { get { return _source.CanShuffle; } }
        public long? GetRowCount() { return _source.GetRowCount(); }
        public Schema Schema { get { return _schema; } }

        public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            return _transform.GetRowCursor(columnsNeeded, rand);
        }

        public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            return _transform.GetRowCursorSet(columnsNeeded, n, rand);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            throw _host.ExceptNotSupp("Not meant to be serialized. You need to serialize whatever it takes to instantiate it.");
        }

        #endregion

        #region Cast

        IDataTransform CreateMemoryTransform()
        {
            if (InputType.IsVector())
            {
                switch (InputType.AsVector().ItemType().RawKind())
                {
                    case DataKind.R4:
                        return CreateMemoryTransformIn<VBuffer<float>>();
                    default:
                        throw _host.ExceptNotImpl("Input Type '{0}' is not handled yet.", InputType.AsVector().ItemType().RawKind());
                }
            }
            else
            {
                switch (InputType.RawKind())
                {
                    case DataKind.R4:
                        return CreateMemoryTransformIn<float>();
                    default:
                        throw _host.ExceptNotImpl("Input Type '{0}' is not handled yet.", InputType.RawKind());
                }
            }
        }

        IDataTransform CreateMemoryTransformIn<TSrc>()
        {
            if (OutputType.IsVector())
            {
                switch (OutputType.AsVector().ItemType().RawKind())
                {
                    case DataKind.U4:
                        return CreateMemoryTransformInOut<TSrc, VBuffer<uint>>();
                    case DataKind.R4:
                        return CreateMemoryTransformInOut<TSrc, VBuffer<float>>();
                    default:
                        throw _host.ExceptNotImpl("Output Type '{0}' is not handled yet.", OutputType.AsVector().ItemType().RawKind());
                }
            }
            else
            {
                switch (OutputType.RawKind())
                {
                    case DataKind.U4:
                        return CreateMemoryTransformInOut<TSrc, uint>();
                    case DataKind.R4:
                        return CreateMemoryTransformInOut<TSrc, float>();
                    default:
                        throw _host.ExceptNotImpl("Output Type '{0}' is not handled yet.", OutputType.RawKind());
                }
            }
        }

        IDataTransform CreateMemoryTransformInOut<TSrc, TDst>()
        {
            return new MemoryTransform<TSrc, TDst>(_host, this);
        }

        #endregion

        #region memory transform

        class MemoryTransform<TSrc, TDst> : IDataTransform
        {
            readonly IHostEnvironment _host;
            readonly TransformFromValueMapper _parent;

            public MemoryTransform(IHostEnvironment env, TransformFromValueMapper parent)
            {
                _host = env;
                _parent = parent;
            }

            public TransformFromValueMapper Parent { get { return _parent; } }
            public ColumnType InputType { get { return _parent.InputType; } }
            public ColumnType OutputType { get { return _parent.OutputType; } }
            public ValueMapper<TTSrc, TTDst> GetMapper<TTSrc, TTDst>() { return _parent.GetMapper<TTSrc, TTDst>(); }
            public IDataView Source { get { return _parent.Source; } }
            public bool CanShuffle { get { return _parent.CanShuffle; } }
            public long? GetRowCount() { return _parent.GetRowCount(); }
            public Schema Schema { get { return _parent.Schema; } }
            public void Save(ModelSaveContext ctx) { throw Contracts.ExceptNotSupp("Not meant to be serialized. You need to serialize whatever it takes to instantiate it."); }

            public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
            {
                int index = SchemaHelper.GetColumnIndex(Source.Schema, _parent.InputName);
                if (columnsNeeded.Where(c => c.Index == index).Any())
                {
                    var newCols = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, index, Schema.Count);
                    var oldCols = SchemaHelper.ColumnsNeeded(newCols, Source.Schema);
                    var cursor = Source.GetRowCursor(oldCols, rand);
                    return new MemoryCursor<TSrc, TDst>(this, cursor, index);
                }
                else
                    // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                    return new SameCursor(Source.GetRowCursor(columnsNeeded, rand), Schema);
            }

            public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
            {
                int index = SchemaHelper.GetColumnIndex(Source.Schema, _parent.InputName);
                if (columnsNeeded.Where(c => c.Index == index).Any())
                {
                    var newCols = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, index, Schema.Count);
                    var oldCols = SchemaHelper.ColumnsNeeded(newCols, Source.Schema);
                    var cursors = Source.GetRowCursorSet(oldCols, n, rand);
                    return cursors.Select(c => new MemoryCursor<TSrc, TDst>(this, c, index)).ToArray();
                }
                else
                    // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                    return Source.GetRowCursorSet(columnsNeeded, n, rand)
                                 .Select(c => new SameCursor(c, Schema))
                                 .ToArray();
            }
        }

        #endregion

        #region cursor

        class MemoryCursor<TSrc, TDst> : RowCursor
        {
            readonly MemoryTransform<TSrc, TDst> _view;
            readonly RowCursor _inputCursor;
            readonly int _inputCol;

            public MemoryCursor(MemoryTransform<TSrc, TDst> view, RowCursor cursor, int inputCol)
            {
                _view = view;
                _inputCursor = cursor;
                _inputCol = inputCol;
            }

            public override bool IsColumnActive(int col)
            {
                // The column is active if is active in the input view or if it the new vector with the polynomial features.
                return col >= _inputCursor.Schema.Count || _inputCursor.IsColumnActive(col);
            }

            public override ValueGetter<RowId> GetIdGetter()
            {
                // We do not change the ID (row to row transform).
                var getId = _inputCursor.GetIdGetter();
                return (ref RowId pos) =>
                {
                    getId(ref pos);
                };
            }

            public override long Batch { get { return _inputCursor.Batch; } }        // No change.
            public override long Position { get { return _inputCursor.Position; } }  // No change.
            public override Schema Schema { get { return _view.Schema; } }          // No change.

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveNext()
            {
                return _inputCursor.MoveNext();
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                // If the column is part of the input view.
                if (col < _inputCursor.Schema.Count)
                    return _inputCursor.GetGetter<TValue>(col);
                // If it is the added column.
                else if (col == _inputCursor.Schema.Count)
                    return GetGetterMapper() as ValueGetter<TValue>;
                // Otherwise, it is an error.
                else
                    throw Contracts.Except("Unexpected columns {0} > {1}.", col, _inputCursor.Schema.Count);
            }

            ValueGetter<TDst> GetGetterMapper()
            {
                var mapper = _view.Parent.GetMapper<TSrc, TDst>();
                var getter = _inputCursor.GetGetter<TSrc>(_inputCol);
                TSrc input = default(TSrc);
                return (ref TDst output) =>
                {
                    getter(ref input);
                    mapper(in input, ref output);
                };
            }
        }

        #endregion
    }
}
