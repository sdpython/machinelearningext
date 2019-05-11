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
using ScalerTransform = Scikit.ML.FeaturesTransforms.ScalerTransform;

[assembly: LoadableClass(ScalerTransform.Summary, typeof(ScalerTransform),
    typeof(ScalerTransform.Arguments), typeof(SignatureDataTransform),
    ScalerTransform.LongName, ScalerTransform.LoaderSignature, ScalerTransform.ShortName)]

[assembly: LoadableClass(ScalerTransform.Summary, typeof(ScalerTransform),
    null, typeof(SignatureLoadDataTransform),
    ScalerTransform.LongName, ScalerTransform.LoaderSignature, ScalerTransform.ShortName)]


namespace Scikit.ML.FeaturesTransforms
{
    /// <summary>
    /// Normalizes columns with various stategies.
    /// </summary>
    public class ScalerTransform : IDataTransformSingle, ITrainableTransform
    {
        public const string LoaderSignature = "ScalerTransform";  // Not more than 24 letters.
        public const string Summary = "Rescales a column (only float).";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "Scaler";
        public const string LongName = "Scaler Transform";

        /// <summary>
        /// Identify the object for dynamic instantiation.
        /// This is also used to track versionning when serializing and deserializing.
        /// </summary>
        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SCALETNS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ScalerTransform).Assembly.FullName);
        }

        public enum ScalerStrategy
        {
            meanVar = 0,
            minMax = 1
        }

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments
        {
            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to normalize.", ShortName = "col")]
            public Column1x1[] columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling strategy.", ShortName = "scale")]
            public ScalerStrategy scaling = ScalerStrategy.meanVar;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(Column1x1.ArrayToLine(columns));
                ctx.Writer.Write((int)scaling);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                string sr = ctx.Reader.ReadString();
                columns = Column1x1.ParseMulti(sr);
                scaling = (ScalerStrategy)ctx.Reader.ReadInt32();
            }

            public void PostProcess()
            {
            }
        }

        IDataView _input;
        Arguments _args;
        Dictionary<string, List<ColumnStatObs>> _scalingStat;
        Dictionary<int, ScalingFactor> _scalingFactors;
        Dictionary<int, int> _revIndex;
        IHost _host;
        DataViewSchema _extendedSchema;
        object _lock;

        public IDataView Source { get { return _input; } }

        public ScalerTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            args.PostProcess();
            _host.CheckValue(args.columns, "columns");

            _input = input;
            _args = args;
            _lock = new object();
            _scalingStat = null;
            _scalingFactors = null;
            _revIndex = null;
            _extendedSchema = ComputeExtendedSchema();
        }

        public static ScalerTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ScalerTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.Writer.Write(_scalingStat == null ? 0 : _scalingStat.Count);
            if (_scalingFactors != null)
            {
                foreach (var pair in _scalingStat)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value.Count);
                    foreach (var val in pair.Value)
                        val.Write(ctx);
                }
            }
        }

        private ScalerTransform(IHost host, ModelLoadContext ctx, IDataView input)
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
            int nbStat = ctx.Reader.ReadInt32();
            _extendedSchema = ComputeExtendedSchema();
            if (nbStat == 0)
            {
                _scalingFactors = null;
                _scalingStat = null;
                _revIndex = null;
            }
            else
            {
                _scalingStat = new Dictionary<string, List<ColumnStatObs>>();
                for (int i = 0; i < nbStat; ++i)
                {
                    string key = ctx.Reader.ReadString();
                    int nb = ctx.Reader.ReadInt32();
                    var li = new List<ColumnStatObs>();
                    for (int k = 0; k < nb; ++k)
                        li.Add(new ColumnStatObs(ColumnStatObs.StatKind.sum));
                    _scalingStat[key] = li;
                }
                _scalingFactors = GetScalingParameters();
                _revIndex = ComputeRevIndex();
            }
        }

        DataViewSchema ComputeExtendedSchema()
        {
            int index;
            Func<string, DataViewType> getType = (string col) =>
            {
                var schema = _input.Schema;
                index = SchemaHelper.GetColumnIndex(schema, col);
                return schema[index].Type;
            };
            var iterCols = _args.columns.Where(c => c.Name != c.Source);
            return iterCols.Any()
                        ? ExtendedSchema.Create(new ExtendedSchema(_input.Schema,
                                                iterCols.Select(c => c.Name).ToArray(),
                                                iterCols.Select(c => getType(c.Source)).ToArray()))
                        : _input.Schema;
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

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            ComputeStatistics();
            _host.AssertValue(_input, "_input");
            var newColumnsNeeded = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, _args.columns);
            var oldCols = SchemaHelper.ColumnsNeeded(newColumnsNeeded, _input.Schema);
            var cursor = _input.GetRowCursor(oldCols, rand);
            return new ScalerCursor(cursor, this, newColumnsNeeded);
        }

        public DataViewRowCursor GetRowCursorSingle(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            ComputeStatistics();
            _host.AssertValue(_input, "_input");
            var newColumnsNeeded = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, _args.columns);
            var oldCols = SchemaHelper.ColumnsNeeded(newColumnsNeeded, _input.Schema);
            var cursor = CursorHelper.GetRowCursorSingle(_input, oldCols, rand);
            return new ScalerCursor(cursor, this, newColumnsNeeded);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            ComputeStatistics();
            _host.AssertValue(_input, "_input");
            var newColumnsNeeded = SchemaHelper.ColumnsNeeded(columnsNeeded, Schema, _args.columns);
            var oldCols = SchemaHelper.ColumnsNeeded(newColumnsNeeded, _input.Schema);
            var cursors = _input.GetRowCursorSet(oldCols, n, rand);
            var res = cursors.Select(c => new ScalerCursor(c, this, newColumnsNeeded)).ToArray();
            return res;
        }

        public void Estimate()
        {
            ComputeStatistics();
        }

        private DataViewSchema.Column _dc(int i)
        {
            return new DataViewSchema.Column(null, i, false, null, null);
        }

        void ComputeStatistics()
        {
            lock (_lock)
            {
                if (_scalingStat == null)
                {
                    using (var ch = _host.Start("ScalerTransform"))
                    {
                        var sch = _input.Schema;
                        var indexesCol = new List<int>();

                        var textCols = _args.columns.Select(c => c.Source).ToArray();
                        _scalingStat = new Dictionary<string, List<ColumnStatObs>>();

                        for (int i = 0; i < textCols.Length; ++i)
                        {
                            int index = SchemaHelper.GetColumnIndex(sch, textCols[i]);
                            var ty = sch[index].Type;
                            if (!(ty == NumberDataViewType.Single || ty == NumberDataViewType.UInt32 || ty == TextDataViewType.Instance || ty == BooleanDataViewType.Instance ||
                                (ty.IsKey() && ty.AsKey().RawKind() == DataKind.UInt32) || (ty.IsVector() && ty.AsVector().ItemType() == NumberDataViewType.Single)))
                                throw ch.Except("Only a float or a vector of floats or a uint or a text or a bool is allowed for column {0} (schema={1}).", _args.columns[i], SchemaHelper.ToString(sch));
                            indexesCol.Add(index);
                        }

                        // Computation
                        var required = new HashSet<int>(indexesCol);
                        var requiredIndexes = required.OrderBy(c => c).ToArray();
                        using (var cur = _input.GetRowCursor(Schema.Where(c => required.Contains(c.Index))))
                        {
                            bool[] isText = requiredIndexes.Select(c => sch[c].Type == TextDataViewType.Instance).ToArray();
                            bool[] isBool = requiredIndexes.Select(c => sch[c].Type == BooleanDataViewType.Instance).ToArray();
                            bool[] isFloat = requiredIndexes.Select(c => sch[c].Type == NumberDataViewType.Single).ToArray();
                            bool[] isUint = requiredIndexes.Select(c => sch[c].Type == NumberDataViewType.UInt32 || sch[c].Type.RawKind() == DataKind.UInt32).ToArray();
                            ValueGetter<bool>[] boolGetters = requiredIndexes.Select(i => sch[i].Type == BooleanDataViewType.Instance || sch[i].Type.RawKind() == DataKind.Boolean ? cur.GetGetter<bool>(_dc(i)) : null).ToArray();
                            ValueGetter<uint>[] uintGetters = requiredIndexes.Select(i => sch[i].Type == NumberDataViewType.UInt32 || sch[i].Type.RawKind() == DataKind.UInt32 ? cur.GetGetter<uint>(_dc(i)) : null).ToArray();
                            ValueGetter<ReadOnlyMemory<char>>[] textGetters = requiredIndexes.Select(i => sch[i].Type == TextDataViewType.Instance ? cur.GetGetter<ReadOnlyMemory<char>>(_dc(i)) : null).ToArray();
                            ValueGetter<float>[] floatGetters = requiredIndexes.Select(i => sch[i].Type == NumberDataViewType.Single ? cur.GetGetter<float>(_dc(i)) : null).ToArray();
                            ValueGetter<VBuffer<float>>[] vectorGetters = requiredIndexes.Select(i => sch[i].Type.IsVector() ? cur.GetGetter<VBuffer<float>>(_dc(i)) : null).ToArray();

                            var schema = _input.Schema;
                            for (int i = 0; i < schema.Count; ++i)
                            {
                                string name = schema[i].Name;
                                if (!required.Contains(i))
                                    continue;
                                _scalingStat[name] = new List<ColumnStatObs>();
                                var t = _scalingStat[name];
                                switch (_args.scaling)
                                {
                                    case ScalerStrategy.meanVar:
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.sum));
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.sum2));
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.nb));
                                        break;
                                    case ScalerStrategy.minMax:
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.min));
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.max));
                                        break;
                                    default:
                                        throw _host.ExceptNotSupp($"Unsupported scaling strategy: {_args.scaling}.");
                                }
                            }

                            float value = 0;
                            var tvalue = new ReadOnlyMemory<char>();
                            VBuffer<float> vector = new VBuffer<float>();
                            uint uvalue = 0;
                            bool bvalue = true;
                            var curschema = cur.Schema;

                            while (cur.MoveNext())
                            {
                                for (int i = 0; i < requiredIndexes.Length; ++i)
                                {
                                    string name = curschema[requiredIndexes[i]].Name;
                                    if (!_scalingStat.ContainsKey(name))
                                        continue;
                                    if (isFloat[i])
                                    {
                                        floatGetters[i](ref value);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(value);
                                    }
                                    else if (isBool[i])
                                    {
                                        boolGetters[i](ref bvalue);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(bvalue);
                                    }
                                    else if (isText[i])
                                    {
                                        textGetters[i](ref tvalue);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(tvalue.ToString());
                                    }
                                    else if (isUint[i])
                                    {
                                        uintGetters[i](ref uvalue);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(uvalue);
                                    }
                                    else
                                    {
                                        vectorGetters[i](ref vector);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(vector);
                                    }
                                }
                            }
                        }

                        _scalingFactors = GetScalingParameters();
                        _revIndex = ComputeRevIndex();
                    }
                }
            }
        }

        Dictionary<int, int> ComputeRevIndex()
        {
            var revIndex = new Dictionary<int, int>();
            foreach (var pair in _scalingFactors)
                revIndex[pair.Value.columnId] = pair.Key;
            return revIndex;
        }

        public enum ScalingMethod
        {
            Affine = 0
        };

        public class ScalingFactor
        {
            public ScalingMethod scalingMethod;

            // Y = scale (X - mean)
            public int columnId;
            public VBuffer<float> mean;
            public VBuffer<float> scale;

            public ScalingFactor(int colid, ScalingMethod method, VBuffer<float> mean, VBuffer<float> scale)
            {
                scalingMethod = method;
                columnId = colid;
                this.mean = mean;
                this.scale = scale;
            }

            public ScalingFactor(IHost host, int colid, ScalerStrategy strategy, List<ColumnStatObs> obs)
            {
                columnId = colid;
                switch (strategy)
                {
                    case ScalerStrategy.meanVar:
                        scalingMethod = ComputeMeanVar(host, obs, out mean, out scale);
                        break;
                    case ScalerStrategy.minMax:
                        scalingMethod = ComputeMinMax(host, obs, out mean, out scale);
                        break;
                    default:
                        throw host.ExceptNotSupp($"Unknown scaling strategy {strategy}.");
                }
            }

            ScalingMethod ComputeMeanVar(IHost host, List<ColumnStatObs> stats,
                                         out VBuffer<float> mean, out VBuffer<float> variance)
            {
                var nb = stats.Where(c => c.kind == ColumnStatObs.StatKind.nb).ToArray();
                var sum = stats.Where(c => c.kind == ColumnStatObs.StatKind.sum).ToArray();
                var sum2 = stats.Where(c => c.kind == ColumnStatObs.StatKind.sum2).ToArray();
                if (nb.Length != 1)
                    throw host.Except("nb is null");
                if (sum.Length != 1)
                    throw host.Except("sum is null");
                if (sum2.Length != 1)
                    throw host.Except("sum2 is null");
                var dnb = nb[0].stat.DenseValues().ToArray();
                var dsum = sum[0].stat.DenseValues().ToArray();
                var dsum2 = sum2[0].stat.DenseValues().ToArray();
                if (dnb.Length != dsum.Length)
                    throw host.Except("{0} != {1}", dnb.Length, dsum.Length);
                if (dnb.Length != dsum2.Length)
                    throw host.Except("{0} != {1}", dnb.Length, dsum2.Length);
                var dmean = new float[dnb.Length];
                var dvar = new float[dnb.Length];
                for (int i = 0; i < dmean.Length; ++i)
                {
                    dmean[i] = (float)(dnb[i] == 0 ? 0 : dsum[i] / dnb[i]);
                    dvar[i] = dnb[i] == 0 ? 0 : (float)Math.Sqrt(dsum2[i] / dnb[i] - dmean[i] * dmean[i]);
                    if (dvar[i] != 0)
                        dvar[i] = 1f / dvar[i];
                }
                mean = new VBuffer<float>(dmean.Length, dmean);
                variance = new VBuffer<float>(dvar.Length, dvar);
                return ScalingMethod.Affine;
            }

            ScalingMethod ComputeMinMax(IHost host, List<ColumnStatObs> stats,
                                        out VBuffer<float> mean, out VBuffer<float> scale)
            {
                var min = stats.Where(c => c.kind == ColumnStatObs.StatKind.min).ToArray();
                var max = stats.Where(c => c.kind == ColumnStatObs.StatKind.max).ToArray();
                if (min.Length != 1)
                    throw host.Except("sum is null");
                if (max.Length != 1)
                    throw host.Except("sum2 is null");
                var dmin = min[0].stat.DenseValues().ToArray();
                var dmax = max[0].stat.DenseValues().ToArray();
                if (dmin.Length != dmax.Length)
                    throw host.Except("{0} != {1}", dmin.Length, dmax.Length);
                var dmean = new float[dmin.Length];
                var dscale = new float[dmin.Length];
                double delta;
                for (int i = 0; i < dmean.Length; ++i)
                {
                    dmean[i] = (float)(dmin[i]);
                    delta = dmax[i] - dmin[i];
                    dscale[i] = (float)(delta == 0 ? 1.0 : 1.0 / delta);
                }
                mean = new VBuffer<float>(dmean.Length, dmean);
                scale = new VBuffer<float>(dscale.Length, dscale);
                return ScalingMethod.Affine;
            }

            public void Update(ref VBuffer<float> dst)
            {
                switch (scalingMethod)
                {
                    case ScalingMethod.Affine:
                        if (dst.IsDense)
                        {
                            for (int i = 0; i < dst.Count; ++i)
                            {
                                dst.Values[i] -= mean.Values[i];
                                if (scale.Values[i] != 0f)
                                    dst.Values[i] *= scale.Values[i];
                            }
                        }
                        else
                        {
                            for (int i = 0; i < dst.Count; ++i)
                            {
                                dst.Values[i] -= mean.Values[dst.Indices[i]];
                                if (scale.Values[dst.Indices[i]] != 0f)
                                    dst.Values[i] *= scale.Values[dst.Indices[i]];
                            }
                        }
                        break;
                    default:
                        throw Contracts.ExceptNotSupp($"Unknown scaling method: {scalingMethod}.");
                }
            }
        }

        Dictionary<int, ScalingFactor> GetScalingParameters()
        {
            var res = new Dictionary<int, ScalingFactor>();
            int index, index2;
            var thisSchema = Schema;
            var schema = _input.Schema;
            for (int i = 0; i < _args.columns.Length; ++i)
            {
                index = SchemaHelper.GetColumnIndex(schema, _args.columns[i].Source);
                string name = thisSchema[index].Name;
                var stats = _scalingStat[name];

                if (_args.columns[i].Source == _args.columns[i].Name)
                    res[index] = new ScalingFactor(_host, index, _args.scaling, stats);
                else
                {
                    index2 = SchemaHelper.GetColumnIndex(Schema, _args.columns[i].Name);
                    res[index2] = new ScalingFactor(_host, index, _args.scaling, stats);
                }
            }
            return res;
        }

        #region Cursor

        public class ScalerCursor : DataViewRowCursor
        {
            readonly DataViewRowCursor _inputCursor;
            readonly ScalerTransform _parent;
            readonly Dictionary<int, ScalingFactor> _scalingFactors;
            readonly IEnumerable<DataViewSchema.Column> _columnsNeeded;

            public ScalerCursor(DataViewRowCursor cursor, ScalerTransform parent, IEnumerable<DataViewSchema.Column> columnsNeeded)
            {
                _inputCursor = cursor;
                _parent = parent;
                _scalingFactors = parent._scalingFactors;
                if (_scalingFactors == null)
                    throw parent._host.ExceptValue("The transform was never trained. Predictions cannot be computed.");
                _columnsNeeded = columnsNeeded;
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                bool active = col.Index >= _inputCursor.Schema.Count || _inputCursor.IsColumnActive(col);
                if (active)
                    return active;
#if(DEBUG)
                if (_columnsNeeded.Where(c => c.Index == col.Index).Any())
                    throw Contracts.ExceptNotImpl("Caught in debug mode.");
#endif
                return false;
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
                return _inputCursor.MoveNext();
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                var schema = _inputCursor.Schema;
                if (_scalingFactors.ContainsKey(col.Index))
                {
                    var type = schema[_scalingFactors[col.Index].columnId].Type;
                    if (type.IsVector())
                        return GetGetterVector(_scalingFactors[col.Index]) as ValueGetter<TValue>;
                    else
                        return GetGetter(_scalingFactors[col.Index]) as ValueGetter<TValue>;
                }
                else if (col.Index < schema.Count)
                    return _inputCursor.GetGetter<TValue>(col);
                else
                    throw Contracts.Except("Unexpected columns {0}.", col);
            }

            private DataViewSchema.Column _dc(int i)
            {
                return new DataViewSchema.Column(null, i, false, null, null);
            }

            ValueGetter<VBuffer<float>> GetGetter(ScalingFactor scales)
            {
                var getter = _inputCursor.GetGetter<float>(_dc(scales.columnId));
                float value = 0f;
                return (ref VBuffer<float> dst) =>
                {
                    getter(ref value);
                    if (1 != scales.mean.Length)
                        throw _parent._host.Except("Mismatch dimension {0} for destination != {1} for scaling vectors.", dst.Length, scales.mean.Length);
                    if (dst.Length != 1)
                        dst = new VBuffer<float>(1, new[] { value });
                    else
                        dst.Values[0] = value;
                    scales.Update(ref dst);
                };
            }

            ValueGetter<VBuffer<float>> GetGetterVector(ScalingFactor scales)
            {
                var getter = _inputCursor.GetGetter<VBuffer<float>>(_dc(scales.columnId));
                return (ref VBuffer<float> dst) =>
                {
                    getter(ref dst);
                    if (dst.Length != scales.mean.Length)
                        throw _parent._host.Except("Mismatch dimension {0} for destination != {1} for scaling vectors.", dst.Length, scales.mean.Length);
                    scales.Update(ref dst);
                };
            }
        }

        #endregion
    }
}

