// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using AddRandomTransform = Scikit.ML.PipelineTransforms.AddRandomTransform;


[assembly: LoadableClass(AddRandomTransform.Summary, typeof(AddRandomTransform),
    typeof(AddRandomTransform.Arguments), typeof(SignatureDataTransform),
    "Add Random Transform", AddRandomTransform.LoaderSignature, "AddRandom", "arnd")]

[assembly: LoadableClass(AddRandomTransform.Summary, typeof(AddRandomTransform),
    null, typeof(SignatureLoadDataTransform),
    "Add Random Transform", AddRandomTransform.LoaderSignature, "AddRandom", "arnd")]

namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Multiplies features, build polynomial features x1, x1^2, x1x2, x2, x2^2...
    /// </summary>
    public class AddRandomTransform : IDataTransformSingle
    {
        #region identification

        public const string LoaderSignature = "AddRandomTransform";  // Not more than 24 letters.
        public const string Summary = "Add random noise to each column (mostly for tests)";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ARNDARND",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(AddRandomTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to convert.", ShortName = "col")]
            public Column1x1[] columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "RandomSeed", ShortName = "s")]
            public int seed;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(seed);
                ctx.Writer.Write(Column1x1.ArrayToLine(columns));
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                seed = ctx.Reader.ReadInt32();
                string sr = ctx.Reader.ReadString();
                columns = Column1x1.ParseMulti(sr);
            }
        }

        #endregion

        #region internal members / accessors

        IDataView _input;
        Arguments _args;
        IHost _host;
        DataViewSchema _schema;
        Dictionary<int, DataViewSchema.Column> _columnMapping;

        public IDataView Source { get { return _input; } }

        #endregion

        #region public constructor / serialization / load / save

        public AddRandomTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register("ULabelToR4LabelTransform");
            _host.CheckValue(args, "args");                 // Checks values are valid.
            _host.CheckValue(input, "input");
            _host.CheckValue(args.columns, "columns");

            _input = input;

            var schema = _input.Schema;
            for (int i = 0; i < args.columns.Length; ++i)
                SchemaHelper.GetColumnIndex(_input.Schema, args.columns[i].Source);
            _args = args;
            _schema = BuildSchema();
            _columnMapping = BuildMapping();
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        private AddRandomTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            _schema = BuildSchema();
            _columnMapping = BuildMapping();
        }

        public static AddRandomTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new AddRandomTransform(h, ctx, input));
        }

        #endregion

        #region IDataTransform API

        DataViewSchema BuildSchema()
        {
            var sch = Source.Schema;
            var newNames = _args.columns.Select(c => c.Name).ToArray();
            var newTypes = _args.columns.Select(c => SchemaHelper.GetColumnType(sch, c.Source)).ToArray();
            var extSchema = new ExtendedSchema(Source.Schema, newNames, newTypes);
            return ExtendedSchema.Create(extSchema);
        }

        Dictionary<int, DataViewSchema.Column> BuildMapping()
        {
            var res = new Dictionary<int, DataViewSchema.Column>();
            foreach (var col in _args.columns)
                res[SchemaHelper.GetColumnIndex(_schema, col.Name)] = SchemaHelper.GetColumnIndexDC(_schema, col.Source);
            return res;
        }

        public DataViewSchema Schema { get { return _schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount(); // We do not add or remove any row. Same number of rows as the input.
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            //var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, _args.columns);
            var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, Source.Schema);
            var cur = Source.GetRowCursor(cols);
            return new AddRandomCursor(this, cur);
        }

        public DataViewRowCursor GetRowCursorSingle(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            //var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, _args.columns);
            var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, Source.Schema);
            var cur = CursorHelper.GetRowCursorSingle(Source, cols);
            return new AddRandomCursor(this, cur);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            n = DataViewUtils.GetThreadCount(n);

            if (n <= 1)
                return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
            else
            {
                //var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, _args.columns);
                var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, Source.Schema);
                var cursors = Source.GetRowCursorSet(cols, n, rand);
                for (int i = 0; i < cursors.Length; ++i)
                    cursors[i] = new AddRandomCursor(this, cursors[i]);
                return cursors;
            }
        }

        #endregion

        #region cursor

        public class AddRandomCursor : DataViewRowCursor
        {
            readonly AddRandomTransform _view;
            readonly DataViewSchema _schema;
            readonly DataViewRowCursor _inputCursor;
            Random _rand;

            public AddRandomCursor(AddRandomTransform view, DataViewRowCursor cursor)
            {
                _view = view;
                _inputCursor = cursor;
                _schema = _view.Schema;
                _rand = new Random(_view._args.seed);
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                if (col.Index < _inputCursor.Schema.Count)
                    return _inputCursor.IsColumnActive(col);
                return true;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref DataViewRowId pos) =>
                {
                    getId(ref pos);
                };
            }

            public override long Batch { get { return _inputCursor.Batch; } }
            public override long Position { get { return _inputCursor.Position; } }
            public override DataViewSchema Schema { get { return _view.Schema; } }

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

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                if (col.Index < _view.Source.Schema.Count)
                    return _inputCursor.GetGetter<TValue>(col);
                else if (col.Index < _view.Schema.Count)
                {
                    var colType = _schema[_view._columnMapping[col.Index].Index].Type;
                    if (colType.IsVector())
                    {
                        switch (colType.ItemType().RawKind())
                        {
                            case DataKind.Boolean: return GetGetterVector(col, false) as ValueGetter<TValue>;
                            case DataKind.Int32: return GetGetterVector(col, 0) as ValueGetter<TValue>;
                            case DataKind.UInt32: return GetGetterVector(col, (uint)0) as ValueGetter<TValue>;
                            case DataKind.Int64: return GetGetterVector(col, (Int64)0) as ValueGetter<TValue>;
                            case DataKind.Single: return GetGetterVector(col, 0f) as ValueGetter<TValue>;
                            case DataKind.Double: return GetGetterVector(col, 0.0) as ValueGetter<TValue>;
                            case DataKind.String: return GetGetterVector(col, new ReadOnlyMemory<char>()) as ValueGetter<TValue>;
                            default:
                                throw Contracts.ExceptNotImpl($"Unsupported type '{colType.ItemType().RawKind()}'.");
                        }
                    }
                    else
                    {
                        switch (colType.RawKind())
                        {
                            case DataKind.Boolean: return GetGetter(col, false) as ValueGetter<TValue>;
                            case DataKind.Int32: return GetGetter(col, 0) as ValueGetter<TValue>;
                            case DataKind.UInt32: return GetGetter(col, (uint)0) as ValueGetter<TValue>;
                            case DataKind.Int64: return GetGetter(col, (Int64)0) as ValueGetter<TValue>;
                            case DataKind.Single: return GetGetter(col, 0f) as ValueGetter<TValue>;
                            case DataKind.Double: return GetGetter(col, 0.0) as ValueGetter<TValue>;
                            case DataKind.String: return GetGetter(col, new ReadOnlyMemory<char>()) as ValueGetter<TValue>;
                            default:
                                throw Contracts.ExceptNotImpl($"Unsupported type '{colType.ItemType().RawKind()}'.");
                        }
                    }
                }
                else
                    throw _view._host.Except("Column index {0} does not exist.", col);
            }

            ValueGetter<bool> GetGetter(DataViewSchema.Column col, bool defval)
            {
                return (ref bool value) =>
                {
                    value = _rand.Next() % 2 == 0;
                };
            }

            ValueGetter<int> GetGetter(DataViewSchema.Column col, int defval)
            {
                var getter = _inputCursor.GetGetter<int>(_view._columnMapping[col.Index]);
                return (ref int value) =>
                {
                    getter(ref value);
                    value += (_rand.Next() % 2) * 2 - 1;
                };
            }

            ValueGetter<uint> GetGetter(int col, uint defval)
            {
                var getter = _inputCursor.GetGetter<uint>(_view._columnMapping[col]);
                return (ref uint value) =>
                {
                    getter(ref value);
                    value += (uint)_rand.Next() % 2;
                };
            }

            ValueGetter<Int64> GetGetter(DataViewSchema.Column col, Int64 defval)
            {
                var getter = _inputCursor.GetGetter<Int64>(_view._columnMapping[col.Index]);
                return (ref Int64 value) =>
                {
                    getter(ref value);
                    value += (Int64)(_rand.Next() % 2) * 2 - 1;
                };
            }

            ValueGetter<uint> GetGetter(DataViewSchema.Column col, uint defval)
            {
                var getter = _inputCursor.GetGetter<uint>(_view._columnMapping[col.Index]);
                return (ref uint value) =>
                {
                    getter(ref value);
                    value += (uint)(_rand.Next() % 2) * 2 - 1;
                };
            }

            ValueGetter<float> GetGetter(DataViewSchema.Column col, float defval)
            {
                var getter = _inputCursor.GetGetter<float>(_view._columnMapping[col.Index]);
                return (ref float value) =>
                {
                    getter(ref value);
                    value += (float)_rand.NextDouble();
                };
            }

            ValueGetter<double> GetGetter(DataViewSchema.Column col, double defval)
            {
                var getter = _inputCursor.GetGetter<double>(_view._columnMapping[col.Index]);
                return (ref double value) =>
                {
                    getter(ref value);
                    value += _rand.NextDouble();
                };
            }

            ValueGetter<ReadOnlyMemory<char>> GetGetter(DataViewSchema.Column col, ReadOnlyMemory<char> defval)
            {
                string alpha = "abcdefghijklmnopqrstuvwxyz";
                var getter = _inputCursor.GetGetter<ReadOnlyMemory<char>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to get a getter for column {_view._columnMapping[col.Index]} and schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref ReadOnlyMemory<char> value) =>
                {
                    getter(ref value);
                    var cs = value.ToString();
                    cs = cs + alpha[_rand.Next() % alpha.Length];
                    value = new ReadOnlyMemory<char>(cs.ToCharArray());
                };
            }

            ValueGetter<VBuffer<bool>> GetGetterVector(DataViewSchema.Column col, bool defval)
            {
                var getter = _inputCursor.GetGetter<VBuffer<bool>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref VBuffer<bool> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                        value.Values[i] = _rand.Next() % 2 == 0;
                };
            }

            ValueGetter<VBuffer<int>> GetGetterVector(DataViewSchema.Column col, int defval)
            {
                var getter = _inputCursor.GetGetter<VBuffer<int>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref VBuffer<int> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                        value.Values[i] += (_rand.Next() % 2) * 2 - 1;
                };
            }

            ValueGetter<VBuffer<uint>> GetGetterVector(DataViewSchema.Column col, uint defval)
            {
                var getter = _inputCursor.GetGetter<VBuffer<uint>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref VBuffer<uint> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                        value.Values[i] += (uint)_rand.Next() % 2;
                };
            }

            ValueGetter<VBuffer<Int64>> GetGetterVector(DataViewSchema.Column col, Int64 defval)
            {
                var getter = _inputCursor.GetGetter<VBuffer<Int64>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref VBuffer<Int64> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                        value.Values[i] += (Int64)(_rand.Next() % 2) * 2 - 1;
                };
            }

            ValueGetter<VBuffer<float>> GetGetterVector(DataViewSchema.Column col, float defval)
            {
                var getter = _inputCursor.GetGetter<VBuffer<float>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref VBuffer<float> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                        value.Values[i] += (float)_rand.NextDouble();
                };
            }

            ValueGetter<VBuffer<double>> GetGetterVector(DataViewSchema.Column col, double defval)
            {
                var getter = _inputCursor.GetGetter<VBuffer<double>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                return (ref VBuffer<double> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                        value.Values[i] += _rand.NextDouble();
                };
            }

            ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetGetterVector(DataViewSchema.Column col, ReadOnlyMemory<char> defval)
            {
                string alpha = "abcdefghijklmnopqrstuvwxyz";
                var getter = _inputCursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(_view._columnMapping[col.Index]);
                if (getter == null)
                    throw _view._host.Except($"Unable to create a getter for column {_view._columnMapping[col.Index]} from schema\n{SchemaHelper.ToString(_inputCursor.Schema)}.");
                string cs;
                return (ref VBuffer<ReadOnlyMemory<char>> value) =>
                {
                    getter(ref value);
                    for (int i = 0; i < value.Length; ++i)
                    {
                        cs = value.Values[i].ToString();
                        cs = cs + alpha[_rand.Next() % alpha.Length];
                        value.Values[i] = new ReadOnlyMemory<char>(cs.ToCharArray());
                    }
                };
            }
        }

        #endregion
    }
}
