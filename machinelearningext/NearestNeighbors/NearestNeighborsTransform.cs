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
using NearestNeighborsTransform = Scikit.ML.NearestNeighbors.NearestNeighborsTransform;

[assembly: LoadableClass(NearestNeighborsTransform.Summary, typeof(NearestNeighborsTransform),
    typeof(NearestNeighborsTransform.Arguments), typeof(SignatureDataTransform),
    NearestNeighborsTransform.LongName, NearestNeighborsTransform.LoaderSignature,
    NearestNeighborsTransform.ShortName)]

[assembly: LoadableClass(NearestNeighborsTransform.Summary, typeof(NearestNeighborsTransform),
    null, typeof(SignatureLoadDataTransform), NearestNeighborsTransform.LongName,
    NearestNeighborsTransform.LoaderSignature, NearestNeighborsTransform.ShortName)]


namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsTransform : IDataTransformSingle
    {
        public const string LoaderSignature = "NearNeighborsTransform";  // Not more than 24 letters.
        public const string Summary = "Retrieves the closest neighbors among a set of points.";
        public const string RegistrationName = LoaderSignature;
        public const string LongName = "Nearest Neighbors Transform";
        public const string ShortName = "knntr";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NEARNEST",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NearestNeighborsTransform).Assembly.FullName);
        }

        public class Arguments : NearestNeighborsArguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Feature column", ShortName = "col")]
            public string column = "Features";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Distance columns (output)", ShortName = "dist")]
            public string distColumn = "Distances";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Id of the neighbors (output)", ShortName = "idn")]
            public string idNeighborsColumn = "idNeighbors";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Label (unused) in this transform but could be leveraged later.", ShortName = "l")]
            public string labelColumn = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weights columns.", ShortName = "colw")]
            public string weightColumn = null;

            public override void Write(ModelSaveContext ctx, IHost host)
            {
                base.Write(ctx, host);
                ctx.Writer.Write(column);
                ctx.Writer.Write(distColumn);
                ctx.Writer.Write(idNeighborsColumn);
                ctx.Writer.Write(labelColumn == null ? "" : labelColumn);
                ctx.Writer.Write(weightColumn == null ? "" : weightColumn);
            }

            public override void Read(ModelLoadContext ctx, IHost host)
            {
                base.Read(ctx, host);
                column = ctx.Reader.ReadString();
                distColumn = ctx.Reader.ReadString();
                idNeighborsColumn = ctx.Reader.ReadString();
                labelColumn = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(labelColumn))
                    labelColumn = null;
                weightColumn = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(weightColumn))
                    weightColumn = null;
            }

            public override void PostProcess()
            {
                base.PostProcess();
            }
        }

        IDataView _input;
        Arguments _args;
        IHost _host;
        DataViewSchema _extendedSchema;
        NearestNeighborsTrees _trees;
        object _lock;

        public IDataView Source { get { return _input; } }
        public NearestNeighborsTrees Trees { get { return _trees; } }

        public NearestNeighborsTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            args.PostProcess();
            _host.CheckValue(args.column, "column");

            _input = input;
            _trees = null;
            _args = args;
            _lock = new object();
            _extendedSchema = ComputeExtendedSchema();
        }

        public static NearestNeighborsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NearestNeighborsTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.Writer.Write((byte)(_trees != null ? 1 : 0));
            if (_trees != null)
                // If _trees is null, this means the pipeline was never run once.
                _trees.Save(ctx);
            _extendedSchema = ComputeExtendedSchema();
        }

        private NearestNeighborsTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _lock = new object();
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            bool run = ctx.Reader.ReadByte() == 1;
            _trees = run ? new NearestNeighborsTrees(host, ctx) : null;
            _extendedSchema = ComputeExtendedSchema();
        }

        DataViewSchema ComputeExtendedSchema()
        {
            return ExtendedSchema.Create(new ExtendedSchema(_input.Schema, new string[] { _args.distColumn, _args.idNeighborsColumn },
                                       new DataViewType[] { new VectorDataViewType(NumberDataViewType.Single, _args.k),
                                       new VectorDataViewType(NumberDataViewType.Int64, _args.k) }));
        }

        public DataViewSchema Schema { get { return _extendedSchema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount();
        }

        /// <summary>
        /// When the last column is requested, we also need the column used to compute it.
        /// This function ensures that this column is requested when the last one is.
        /// </summary>
        IEnumerable<DataViewSchema.Column> PredicatePropagation(IEnumerable<DataViewSchema.Column> columnsNeeded)
        {
            var cols = columnsNeeded.ToList();
            if (cols.Where(c => c.Index == _input.Schema.Count).Any())
            {
                if (_args.column != null)
                    cols.Add(Schema.Where(c => c.Name == _args.column).First());
                if (_args.idNeighborsColumn != null)
                    cols.Add(Schema.Where(c => c.Name == _args.idNeighborsColumn).First());
                if (_args.labelColumn != null)
                    cols.Add(Schema.Where(c => c.Name == _args.labelColumn).First());
                if (_args.weightColumn != null)
                    cols.Add(Schema.Where(c => c.Name == _args.weightColumn).First());
            }
            return cols;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return GetRowCursor(columnsNeeded, rand, (c, r) => _input.GetRowCursor(c, r));
        }

        public DataViewRowCursor GetRowCursorSingle(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return GetRowCursor(columnsNeeded, rand, (c, r) => CursorHelper.GetRowCursorSingle(_input, c, r));
        }

        private DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand,
                                               DelegateGetRowCursor getterCursor)
        {
            ComputeNearestNeighbors();
            _host.AssertValue(_input, "_input");
            var schema = _input.Schema;

            if (columnsNeeded.Where(c => c.Index == _input.Schema.Count).Any())
            {
                var newColumns = PredicatePropagation(columnsNeeded);
                var oldCols = SchemaHelper.ColumnsNeeded(newColumns, schema);
                var featureIndex = SchemaHelper.GetColumnIndexDC(Schema, _args.column);
                return new NearestNeighborsCursor(getterCursor(oldCols, rand), this, newColumns, featureIndex);
            }
            else
            {
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                var oldCols = SchemaHelper.ColumnsNeeded(columnsNeeded, schema);
                return new SameCursor(getterCursor(oldCols, rand), Schema);
            }
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            ComputeNearestNeighbors();
            _host.AssertValue(_input, "_input");
            var schema = _input.Schema;

            if (columnsNeeded.Where(c => c.Index == _input.Schema.Count).Any())
            {
                var newColumns = PredicatePropagation(columnsNeeded);
                var oldCols = SchemaHelper.ColumnsNeeded(newColumns, schema);
                var featureIndex = SchemaHelper.GetColumnIndexDC(Schema, _args.column);
                var res = _input.GetRowCursorSet(oldCols, n, rand)
                                .Select(c => new NearestNeighborsCursor(c, this, newColumns, featureIndex)).ToArray();
                return res;
            }
            else
            {
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                var oldCols = SchemaHelper.ColumnsNeeded(columnsNeeded, schema);
                return _input.GetRowCursorSet(oldCols, n, rand).Select(c => new SameCursor(c, Schema)).ToArray();
            }
        }

        int GetColumnIndex(IExceptionContext ch, string name)
        {
            if (string.IsNullOrEmpty(name))
                return -1;
            return SchemaHelper.GetColumnIndex(Schema, name);
        }

        void ComputeNearestNeighbors()
        {
            lock (_lock)
            {
                if (_trees != null)
                    return;

                using (var ch = _host.Start("Build k-d tree"))
                {
                    ch.Info(MessageSensitivity.None, "ComputeNearestNeighbors: build a k-d tree.");
                    int featureIndex, labelIndex, idIndex, weightIndex;
                    featureIndex = GetColumnIndex(ch, _args.column);
                    if (featureIndex == -1)
                        throw ch.Except($"Unable to find column '{_args.column}' in {SchemaHelper.ToString(Schema)}.");
                    labelIndex = GetColumnIndex(ch, _args.labelColumn);
                    weightIndex = GetColumnIndex(ch, _args.weightColumn);
                    idIndex = GetColumnIndex(ch, _args.colId);

                    Dictionary<long, Tuple<long, float>> merged;
                    _trees = NearestNeighborsBuilder.NearestNeighborsBuild<long>(ch, _input, featureIndex, labelIndex,
                                        idIndex, weightIndex, out merged, _args);
                    ch.Info(MessageSensitivity.UserData, "Done. Tree size: {0} points.", _trees.Count());
                }
            }
        }

        #region Cursor

        public class NearestNeighborsCursor : DataViewRowCursor
        {
            readonly DataViewRowCursor _inputCursor;
            readonly NearestNeighborsTransform _parent;
            readonly ValueGetter<VBuffer<float>> _getterFeatures;
            readonly NearestNeighborsTrees _trees;
            readonly int _k;

            VBuffer<float> _tempFeatures;
            VBuffer<float> _distance;
            VBuffer<long> _idn;

            public NearestNeighborsCursor(DataViewRowCursor cursor, NearestNeighborsTransform parent,
                                          IEnumerable<DataViewSchema.Column> columnsNeeded,
                                          DataViewSchema.Column colFeatures)
            {
                _inputCursor = cursor;
                _parent = parent;
                _trees = parent._trees;
                _k = parent._args.k;
                _getterFeatures = _inputCursor.GetGetter<VBuffer<float>>(colFeatures);
                _tempFeatures = new VBuffer<float>();
                _distance = new VBuffer<float>(_k, new float[_k]);
                _idn = new VBuffer<long>(_k, new long[_k]);
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                return col.Index >= _inputCursor.Schema.Count || _inputCursor.IsColumnActive(col);
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
            public override DataViewSchema Schema { get { return _parent.Schema; } }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveNext()
            {
                var res = _inputCursor.MoveNext();
                if (!res)
                    return res;
                RetrieveNeighbors();
                return true;
            }

            void RetrieveNeighbors()
            {
                _getterFeatures(ref _tempFeatures);
                var res = _trees.NearestNNeighbors(_tempFeatures, _k);
                if (res.Length > _distance.Length || res.Length > _distance.Count ||
                    res.Length > _distance.Values.Length || _distance.Values == null)
                {
                    _distance = new VBuffer<float>(res.Length, new float[res.Length]);
                    _idn = new VBuffer<long>(res.Length, new long[res.Length]);
                }
                else if (res.Length > _distance.Count)
                {
                    _distance = new VBuffer<float>(res.Length, _distance.Values);
                    _idn = new VBuffer<long>(res.Length, _idn.Values);
                }
                int pos = 0;
                foreach (var pair in res.OrderBy(c => c.Key))
                {
                    _distance.Values[pos] = pair.Key;
                    _idn.Values[pos++] = pair.Value;
                }
                Contracts.Assert(_distance.IsDense);
                Contracts.Assert(_idn.IsDense);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                var schema = _inputCursor.Schema;
                ValueGetter<TValue> res;
                if (col.Index < schema.Count)
                    res = _inputCursor.GetGetter<TValue>(col);
                else if (col.Index == schema.Count)
                    res = GetGetterDistance(col) as ValueGetter<TValue>;
                else if (col.Index == schema.Count + 1)
                    res = GetGetterIdNeighbors(col) as ValueGetter<TValue>;
                else
                    throw Contracts.Except("Unexpected column position:{0}.", col);
#if(DEBUG)
                if (res == null)
                    throw _parent._host.Except("Unable to retrieve a getter for col={0} type={1} schema={2}", col, typeof(TValue), SchemaHelper.ToString(Schema));
#endif
                return res;
            }

            ValueGetter<VBuffer<float>> GetGetterDistance(DataViewSchema.Column col)
            {
                if (col.Index == _inputCursor.Schema.Count)
                    return (ref VBuffer<float> distance) =>
                    {
                        distance = new VBuffer<float>(_distance.Count, _distance.Values);
                    };
                else
                    throw Contracts.Except("Unexpected column for distance (position:{0})", col);
            }

            ValueGetter<VBuffer<long>> GetGetterIdNeighbors(DataViewSchema.Column col)
            {
                if (col.Index == _inputCursor.Schema.Count + 1)
                    return (ref VBuffer<long> distance) =>
                    {
                        distance = new VBuffer<long>(_idn.Count, _idn.Values);
                    };
                else
                    throw Contracts.Except("Unexpected column for neighbors ids (position:{0})", col);
            }
        }

        #endregion
    }
}

