﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.RandomTransforms
{
    /// <summary>
    /// Shake one input of a predictor and merges all outputs. This can be used to measure the sensitivity to one input.
    /// </summary>
    public class ShakeInputTransform : IDataTransform
    {
        #region identification

        public const string LoaderSignature = "ShakeInputTransform";  // Not more than 24 letters.
        public const string Summary = "Shake one input of a predictor and merges all outputs. This can be used to measure the sensitivity to one input.";
        public const string RegistrationName = LoaderSignature;

        /// <summary>
        /// Identify the object for dynamic instantiation.
        /// This is also used to track versionning when serializing and deserializing.
        /// </summary>
        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SHAKEINP",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        #endregion

        #region parameters / command line

        public enum ShakeInputAlgorithm
        {
            /// <summary>
            /// Try all possible shaking values.
            /// </summary>
            exhaustive = 1
        }

        public enum ShakeAggregation
        {
            /// <summary>
            /// Output vector will be concatenated in a single row
            /// </summary>
            concatenate = 1,

            /// <summary>
            /// Adds vector altogether, produces one single row.
            /// </summary>
            add = 2
        }

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Algorithm to shake rows.", ShortName = "al")]
            public ShakeInputAlgorithm algo = ShakeInputAlgorithm.exhaustive;

            #region columns

            [Argument(ArgumentType.Required, HelpText = "Features columns (a vector)", ShortName = "in")]
            public string inputColumn;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Features indices to shake", ShortName = "c")]
            public string inputFeatures;

            public int[] inputFeaturesInt;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to merge", ShortName = "out")]
            public string[] outputColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Values to shake", ShortName = "w")]
            public string values;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads used to estimate how much a class should resample.", ShortName = "nt")]
            public int? numThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Different ways to aggregate the rows produced by the skaking", ShortName = "agg")]
            public ShakeAggregation aggregation = ShakeAggregation.concatenate;

            #endregion

            public void PostProcess()
            {
                if (outputColumns != null && outputColumns.Length == 1 && outputColumns[0].Contains(","))
                    outputColumns = outputColumns[0].Split(',');
                if (inputFeaturesInt == null && !string.IsNullOrEmpty(inputFeatures))
                    inputFeaturesInt = inputFeatures.Split(',').Select(c => int.Parse(c)).ToArray();
            }

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(inputColumn);
                ctx.Writer.Write(string.Join(",", inputFeatures));
                ctx.Writer.Write(string.Join(",", outputColumns));
                ctx.Writer.Write(values);
                ctx.Writer.Write(numThreads ?? -1);
                ctx.Writer.Write((int)aggregation);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                inputColumn = ctx.Reader.ReadString();
                string sr = ctx.Reader.ReadString();
                inputFeaturesInt = sr.Split(',').Select(c => Convert.ToInt32(c)).ToArray();
                sr = ctx.Reader.ReadString();
                outputColumns = sr.Split(',');
                values = ctx.Reader.ReadString();
                int nb = ctx.Reader.ReadInt32();
                numThreads = nb > 0 ? (int?)nb : null;
                aggregation = (ShakeAggregation)ctx.Reader.ReadInt32();
            }
        }

        #endregion

        #region internal members / accessors

        IValueMapper[] _toShake;
        IDataView _input;
        IDataTransform _transform;          // templated transform (not the serialized version)
        Arguments _args;
        IHost _host;

        public IDataView Source { get { return _input; } }

        #endregion

        #region public constructor / serialization / load / save

        public ShakeInputTransform(IHostEnvironment env, Arguments args, IDataView input, IValueMapper[] toShake)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register("ShakeInputTransform");
            _host.CheckValue(args, "args");
            args.PostProcess();
            _host.CheckValue(toShake, "toShake");
            _host.CheckValue(input, "input");
            _host.CheckValue(args.inputColumn, "inputColumn");
            _host.CheckValue(args.inputFeaturesInt, "inputFeatures");
            _host.CheckValue(args.outputColumns, "outputColumns");

            _toShake = toShake;
            _input = input;

            int ind;
            if (!input.Schema.TryGetColumnIndex(args.inputColumn, out ind))
                throw _host.ExceptParam("inputColumn", "Column '{0}' not found in schema.", args.inputColumn);

            if (toShake.Length != args.outputColumns.Length)
                throw _host.ExceptParam("outputColumns", "toShake and outputColumns must have the same length");

            for (int i = 1; i < _toShake.Length; ++i)
            {
                if (_toShake[i].OutputType.IsVector && _toShake[i - 1].OutputType.IsVector)
                {
                    if (_toShake[i].OutputType.ItemType != _toShake[i - 1].OutputType.ItemType)
                        throw _host.Except("All value mappers must be the same type.");
                }
                else if (_toShake[i].OutputType != _toShake[i - 1].OutputType)
                    throw _host.Except("All value mappers must be the same type.");
            }
            if (_toShake[0].OutputType.IsVector)
            {
                var vec = _toShake[0].OutputType.AsVector;
                if (vec.ItemType.IsVector)
                    throw _host.ExceptNotSupp("Unable to handle vectors of vectors as outputs of the mapper.");
            }

            _args = args;
            _transform = CreateTemplatedTransform();
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

#if false

        private ShakeInputTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            _transform = CreateTemplatedTransform();
        }

        public static ShakeInputTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ShakeInputTransform(h, ctx, input));
        }

#endif

        #endregion

        #region IDataTransform API

        public ISchema Schema { get { return _transform.Schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount(lazy);
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            IDataTransform transform = null;

            int index;
            if (!_input.Schema.TryGetColumnIndex(_args.inputColumn, out index))
                throw _host.Except("Unable to find '{0}'", _args.inputColumn);

            var typeCol = _input.Schema.GetColumnType(index);
            if (!typeCol.IsVector)
                throw _host.Except("Expected a vector as input.");
            typeCol = typeCol.AsVector.ItemType;

            switch (typeCol.RawKind)
            {
                case DataKind.R4:
                    transform = new ShakeInputState<float>(_host, transform ?? Source, _toShake, _args);
                    break;
                case DataKind.BL:
                    transform = new ShakeInputState<bool>(_host, transform ?? Source, _toShake, _args);
                    break;
                case DataKind.U1:
                    transform = new ShakeInputState<Byte>(_host, transform ?? Source, _toShake, _args);
                    break;
                case DataKind.U2:
                    transform = new ShakeInputState<UInt16>(_host, transform ?? Source, _toShake, _args);
                    break;
                case DataKind.U4:
                    transform = new ShakeInputState<UInt32>(_host, transform ?? Source, _toShake, _args);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Type '{0}' is not handled yet.", typeCol.RawKind);
            }
            return transform;
        }

        #endregion

        #region State

        /// <summary>
        /// Templated transform which sorts rows based on one column.
        /// </summary>
        public class ShakeInputState<TInput> : IDataTransform
        {
            IHost _host;
            IDataView _input;
            IValueMapper[] _toShake;

            readonly ISchema _schema;
            readonly Arguments _args;
            readonly int _inputCol;
            TInput[][] _shakingValues;

            object _lock;

            public IDataView Source { get { return _input; } }
            public ISchema Schema { get { return _schema; } }

            public ShakeInputState(IHostEnvironment host, IDataView input, IValueMapper[] toShake, Arguments args)
            {
                _host = host.Register("ShakeInputState");
                _host.CheckValue(input, "input");
                _input = input;
                _lock = new object();
                _args = args;
                _toShake = toShake;

                foreach (var vm in toShake)
                {
                    if (vm.OutputType.IsVector && vm.OutputType.AsVector.DimCount > 1)
                        throw _host.Except("If a ValueMapper return a vector, it should have one dimension or zero.");
                }

                if (!_input.Schema.TryGetColumnIndex(_args.inputColumn, out _inputCol))
                    throw _host.Except("Unable to find '{0}'", _args.inputColumn);
                _shakingValues = ExtractShakingValues();
                if (_shakingValues.Length != _args.inputFeaturesInt.Length)
                    throw _host.Except("Shaking Values and columns to shake do not have the same dimension {0} and '{1}'.", _args.inputFeaturesInt.Length, _args.values);

                var colTypes = new List<ColumnType>();

                switch (_args.aggregation)
                {
                    case ShakeAggregation.concatenate:
                        int m = 1;
                        foreach (var shakeVal in _shakingValues)
                            m *= shakeVal.Length;
                        if (m == 0)
                            throw _host.Except("No shaking values ('{0}')", _args.values);
                        foreach (var c in toShake)
                        {
                            var vt = c.OutputType.IsVector
                                            ? new VectorType(c.OutputType.ItemType.AsPrimitive, c.OutputType.AsVector.DimCount == 0 ? 0 : c.OutputType.AsVector.GetDim(0) * m)
                                            : new VectorType(c.OutputType.AsPrimitive, m);
                            colTypes.Add(vt);
                        }
                        break;
                    case ShakeAggregation.add:
                        foreach (var c in toShake)
                        {
                            var vt = c.OutputType.IsVector
                                            ? new VectorType(c.OutputType.ItemType.AsPrimitive, c.OutputType.AsVector.DimCount == 0 ? 0 : c.OutputType.AsVector.GetDim(0))
                                            : new VectorType(c.OutputType.AsPrimitive, 1);
                            colTypes.Add(vt);
                        }
                        break;
                    default:
                        throw _host.ExceptNotSupp("Unknown aggregatino strategy {0}", _args.aggregation);

                }
                _schema = new ExtendedSchema(input.Schema, args.outputColumns, colTypes.ToArray());
            }

            public void Save(ModelSaveContext ctx)
            {
            }

            TInput[][] ExtractShakingValues()
            {
                bool identity;
                var ty = _input.Schema.GetColumnType(_inputCol);
                var conv = Conversions.Instance.GetStandardConversion<ReadOnlyMemory<char>, TInput>(TextType.Instance, ty.AsVector.ItemType, out identity);
                if (string.IsNullOrEmpty(_args.values))
                    throw _host.ExceptParam("_args.values cannot be null.");
                string[][] values = _args.values.Split(';').Select(c => c.Split(',')).ToArray();
                TInput[][] res = new TInput[values.Length][];
                for (int i = 0; i < res.Length; ++i)
                {
                    res[i] = new TInput[values[i].Length];
                    for (int j = 0; j < res[i].Length; ++j)
                    {
                        var t = new ReadOnlyMemory<char>(values[i][j].ToCharArray());
                        conv(ref t, ref res[i][j]);
                    }
                }
                return res;
            }

            public bool CanShuffle { get { return true; } }

            public long? GetRowCount(bool lazy = true)
            {
                return null;
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                var kind = _toShake[0].OutputType.IsVector
                                ? _toShake[0].OutputType.ItemType.RawKind
                                : _toShake[0].OutputType.RawKind;
                switch (kind)
                {
                    case DataKind.R4:
                        var cursor = _input.GetRowCursor(i => i == _inputCol || predicate(i), rand);
                        return new ShakeInputCursor<TInput, float>(this, cursor, i => i == _inputCol || predicate(i), _args, _inputCol, _toShake, _shakingValues,
                                        (float x, float y) => { return x + y; });
                    default:
                        throw _host.Except("Not supported RawKind {0}", _toShake[0].OutputType.RawKind);
                }
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                DataKind kind;
                if (_toShake[0].OutputType.IsVector)
                    kind = _toShake[0].OutputType.AsVector.ItemType.RawKind;
                else
                    kind = _toShake[0].OutputType.RawKind;

                switch (kind)
                {
                    case DataKind.R4:
                        var cursors = _input.GetRowCursorSet(out consolidator, i => i == _inputCol || predicate(i), n, rand);
                        return cursors.Select(c => new ShakeInputCursor<TInput, float>(this, c, predicate, _args, _inputCol, _toShake, _shakingValues,
                                            (float x, float y) => { return x + y; })).ToArray();
                    default:
                        throw _host.Except("Not supported RawKind {0}", _toShake[0].OutputType.RawKind);
                }
            }
        }

        #endregion

        #region Cursor

        public class ShakeInputCursor<TInput, TOutput> : IRowCursor
        {
            readonly ShakeInputState<TInput> _view;
            readonly IRowCursor _inputCursor;
            readonly Arguments _args;
            readonly TInput[][] _shakingValues;
            readonly IValueMapper[] _toShake;

            VBuffer<TOutput>[] _collected;
            ValueGetter<VBuffer<TInput>> _inputGetter;
            VBuffer<TInput> _inputValue;
            ValueMapper<VBuffer<TInput>, VBuffer<TOutput>>[] _mappersV;
            ValueMapper<VBuffer<TInput>, TOutput>[] _mappers;
            Func<TOutput, TOutput, TOutput> _aggregation;

            public ShakeInputCursor(ShakeInputState<TInput> view, IRowCursor cursor, Func<int, bool> predicate,
                                    Arguments args, int column, IValueMapper[] toShake, TInput[][] shakingValues,
                                    Func<TOutput, TOutput, TOutput> aggregation)
            {
                _view = view;
                _args = args;
                _inputCursor = cursor;
                _toShake = toShake;
                _inputGetter = cursor.GetGetter<VBuffer<TInput>>(column);

                _mappersV = _toShake.Select(c => !c.OutputType.IsVector
                                ? null
                                : c.GetMapper<VBuffer<TInput>, VBuffer<TOutput>>()).ToArray();
                _mappers = _toShake.Select(c => c.OutputType.IsVector
                                ? null
                                : c.GetMapper<VBuffer<TInput>, TOutput>()).ToArray();

                for (int i = 0; i < _mappers.Length; ++i)
                {
                    if (_mappers[i] == null && _mappersV[i] == null)
                        throw Contracts.Except("Type mismatch.");
                }
                _shakingValues = shakingValues;
                _collected = new VBuffer<TOutput>[_toShake.Length];
                _aggregation = aggregation;
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool IsColumnActive(int col)
            {
                return col >= _inputCursor.Schema.ColumnCount || _inputCursor.IsColumnActive(col);
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref UInt128 pos) =>
                {
                    getId(ref pos);
                };
            }

            public CursorState State { get { return _inputCursor.State; } }
            public long Batch { get { return _inputCursor.Batch; } }
            public long Position { get { return _inputCursor.Position; } }
            public ISchema Schema { get { return _view.Schema; } }

            void IDisposable.Dispose()
            {
                _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public bool MoveMany(long count)
            {
                var r = _inputCursor.MoveMany(count);
                if (!r)
                    return r;
                _inputGetter(ref _inputValue);
                switch (_args.algo)
                {
                    case ShakeInputAlgorithm.exhaustive:
                        FillShakingValuesExhaustive();
                        break;
                    default:
                        throw Contracts.Except("Not available algo {0}", _args.algo);
                }
                return true;
            }

            public bool MoveNext()
            {
                var r = _inputCursor.MoveNext();
                if (!r)
                    return r;
                _inputGetter(ref _inputValue);
                switch (_args.algo)
                {
                    case ShakeInputAlgorithm.exhaustive:
                        FillShakingValuesExhaustive();
                        break;
                    default:
                        throw Contracts.Except("Not available algo {0}", _args.algo);
                }
                return true;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (col < _inputCursor.Schema.ColumnCount)
                    return _inputCursor.GetGetter<TValue>(col);
                else if (col - _inputCursor.Schema.ColumnCount >= _collected.Length)
                    throw Contracts.Except("Unexpected columns {0} > {1}.", col, _collected.Length + _inputCursor.Schema.ColumnCount);
                return GetBufferGetter(col) as ValueGetter<TValue>;
            }

            ValueGetter<VBuffer<TOutput>> GetBufferGetter(int col)
            {
                int diff = col - _inputCursor.Schema.ColumnCount;
                return (ref VBuffer<TOutput> output) =>
                {
                    output = _collected[diff];
                };
            }

            void FillShakingValuesExhaustive()
            {
                switch (_args.aggregation)
                {
                    case ShakeAggregation.concatenate:
                        FillShakingValuesConcatenate();
                        break;
                    case ShakeAggregation.add:
                        FillShakingValuesAdd();
                        break;
                    default:
                        throw Contracts.Except("Unkown aggregation strategy {0}", _args.aggregation);

                }
            }

            void FillShakingValuesConcatenate()
            {
                if (_args.numThreads.HasValue && _args.numThreads.Value > 1)
                    throw Contracts.ExceptNotSupp("Multithread not implemented yet.");
                var values = _inputValue.Values;
                var indices = _inputValue.Indices;
                int dim;

                var merges = new List<TOutput>[_collected.Length];
                for (int k = 0; k < merges.Length; ++k)
                    merges[k] = new List<TOutput>();

                TOutput oneRes = default(TOutput);
                VBuffer<TOutput> oneResV = new VBuffer<TOutput>();

                for (int i = 0; i < _shakingValues.Length; ++i)
                {
                    if (indices == null)
                        dim = _args.inputFeaturesInt[i];
                    else
                    {
                        dim = Array.FindIndex(indices, v => v == _args.inputFeaturesInt[i]);
                        if (dim == -1)
                        {
                            // We need to create a new sparse vector.
                            TInput[] newValues = new TInput[_inputValue.Count + 1];
                            int[] newIndices = new int[_inputValue.Count + 1];
                            int d = 0;
                            for (int k = 0; k < _inputValue.Count; ++k)
                            {
                                if (d == 0 && indices[k] > _args.inputFeaturesInt[i])
                                {
                                    newIndices[k] = _args.inputFeaturesInt[i];
                                    ++d;
                                }
                                newIndices[k + d] = indices[k];
                                newValues[k + d] = values[k];
                            }
                            _inputValue = new VBuffer<TInput>(_inputValue.Count + 1, _inputValue.Length, newValues, newIndices);
                            values = _inputValue.Values;
                            indices = _inputValue.Indices;
                        }
                    }

                    for (int k = 0; k < _mappers.Length; ++k)
                    {
                        if (_mappers[k] != null)
                        {
                            for (int j = 0; j < _shakingValues[i].Length; ++j)
                            {
                                values[dim] = _shakingValues[i][j];
                                _mappers[k](ref _inputValue, ref oneRes);
                                merges[k].Add(oneRes);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < _shakingValues[i].Length; ++j)
                            {
                                values[dim] = _shakingValues[i][j];
                                _mappersV[k](ref _inputValue, ref oneResV);
                                merges[k].AddRange(oneResV.DenseValues());
                            }
                        }
                    }
                }
                for (int k = 0; k < _collected.Length; ++k)
                    _collected[k] = new VBuffer<TOutput>(merges[k].Count, merges[k].ToArray());

                #region debug
#if (DEBUG)
                var sch = Schema;
                for (int col = 0; col < _collected.Length; ++col)
                {
                    int icol = col + _inputCursor.Schema.ColumnCount;
                    var type = sch.GetColumnType(icol);
                    if (!type.IsVector)
                        throw Contracts.Except("Incompatible type '{0}' != '{1}'", type, _collected[col].GetType());
                    var v = type.AsVector;
                    if (v.DimCount > 1)
                        throw Contracts.Except("Incompatible type '{0}' != '{1}'", type, _collected[col].GetType());
                    if (v.DimCount != 0 && v.GetDim(0) != 0 && v.GetDim(0) != _collected[col].Length)
                        throw Contracts.Except("Incompatible dimension {0} != {1}", v.GetDim(0), _collected[col].Length);
                }
#endif
                #endregion
            }

            void FillShakingValuesAdd()
            {
                if (_args.numThreads.HasValue && _args.numThreads.Value > 1)
                    throw Contracts.ExceptNotSupp("Multithread not implemented yet.");
                if (_aggregation == null)
                    throw Contracts.Except("Aggregation is null.");
                var values = _inputValue.Values;
                var indices = _inputValue.Indices;
                int dim;

                var merges = new List<TOutput>[_collected.Length];
                for (int k = 0; k < merges.Length; ++k)
                    merges[k] = new List<TOutput>();

                TOutput oneRes = default(TOutput);
                VBuffer<TOutput> oneResV = new VBuffer<TOutput>();

                for (int i = 0; i < _shakingValues.Length; ++i)
                {
                    if (indices == null)
                        dim = _args.inputFeaturesInt[i];
                    else
                    {
                        dim = Array.FindIndex(indices, v => v == _args.inputFeaturesInt[i]);
                        if (dim == -1)
                        {
                            // We need to create a new sparse vector.
                            TInput[] newValues = new TInput[_inputValue.Count + 1];
                            int[] newIndices = new int[_inputValue.Count + 1];
                            int d = 0;
                            for (int k = 0; k < _inputValue.Count; ++k)
                            {
                                if (d == 0 && indices[k] > _args.inputFeaturesInt[i])
                                {
                                    newIndices[k] = _args.inputFeaturesInt[i];
                                    ++d;
                                }
                                newIndices[k + d] = indices[k];
                                newValues[k + d] = values[k];
                            }
                            _inputValue = new VBuffer<TInput>(_inputValue.Count + 1, _inputValue.Length, newValues, newIndices);
                            values = _inputValue.Values;
                            indices = _inputValue.Indices;
                        }
                    }

                    for (int k = 0; k < _mappers.Length; ++k)
                    {
                        if (_mappers[k] != null)
                        {
                            for (int j = 0; j < _shakingValues[i].Length; ++j)
                            {
                                values[dim] = _shakingValues[i][j];
                                _mappers[k](ref _inputValue, ref oneRes);
                                if (j == 0 && i == 0)
                                    merges[k].Add(oneRes);
                                else
                                    merges[k][0] = _aggregation(merges[k][0], oneRes);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < _shakingValues[i].Length; ++j)
                            {
                                values[dim] = _shakingValues[i][j];
                                _mappersV[k](ref _inputValue, ref oneResV);
                                if (j == 0 && i == 0)
                                    merges[k].AddRange(oneResV.DenseValues());
                                else
                                {
                                    var array = oneResV.DenseValues().ToArray();
                                    for (int a = 0; a < merges[k].Count; ++a)
                                        merges[k][a] = _aggregation(merges[k][a], array[a]);
                                }
                            }
                        }
                    }
                }
                for (int k = 0; k < _collected.Length; ++k)
                    _collected[k] = new VBuffer<TOutput>(merges[k].Count, merges[k].ToArray());

                #region debug
#if (DEBUG)
                var sch = Schema;
                for (int col = 0; col < _collected.Length; ++col)
                {
                    int icol = col + _inputCursor.Schema.ColumnCount;
                    var type = sch.GetColumnType(icol);
                    if (!type.IsVector)
                        throw Contracts.Except("Incompatible type '{0}' != '{1}'", type, _collected[col].GetType());
                    var v = type.AsVector;
                    if (v.DimCount > 1)
                        throw Contracts.Except("Incompatible type '{0}' != '{1}'", type, _collected[col].GetType());
                    if (v.DimCount != 0 && v.GetDim(0) != 0 && v.GetDim(0) != _collected[col].Length)
                        throw Contracts.Except("Incompatible dimension {0} != {1}", v.GetDim(0), _collected[col].Length);
                }
#endif
                #endregion
            }
        }
    }

    #endregion
}
