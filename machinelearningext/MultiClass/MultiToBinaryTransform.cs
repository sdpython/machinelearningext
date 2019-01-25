// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Data.Conversion;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using MultiToBinaryTransform = Scikit.ML.MultiClass.MultiToBinaryTransform;

[assembly: LoadableClass(MultiToBinaryTransform.Summary, typeof(MultiToBinaryTransform),
    typeof(MultiToBinaryTransform.Arguments), typeof(SignatureDataTransform),
    "MultiClass to Binary classification", MultiToBinaryTransform.LoaderSignature,
    "MultiToBinary", "M2B")]

[assembly: LoadableClass(MultiToBinaryTransform.Summary, typeof(MultiToBinaryTransform),
    null, typeof(SignatureLoadDataTransform),
    "MultiClass to Binary classification", MultiToBinaryTransform.LoaderSignature)]


namespace Scikit.ML.MultiClass
{
    /// <summary>
    /// Multiplies rows to tranform a multi-class problem into a binary classification problem.
    /// </summary>
    public class MultiToBinaryTransform : IDataTransform
    {
        #region identification

        public const string LoaderSignature = "MultiToBinaryTransform";  // Not more than 24 letters.
        public const string Summary = "Converts a multi-class classification problem into a binary classification problem.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULTIBIN",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiToBinaryTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public enum MultiplicationAlgorithm
        {
            Default = 1,
            Reweight = 2,
            Ranking = 3
        }

        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Algorithm to duplicate rows.", ShortName = "al")]
            public MultiplicationAlgorithm algo = MultiplicationAlgorithm.Default;

            #region parameters

            [Argument(ArgumentType.Required, HelpText = "Multi-class label column", ShortName = "lab")]
            public string label;

            [Argument(ArgumentType.Required, HelpText = "New labels column", ShortName = "name")]
            public string newColumn = "binaryLabel";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weight column", ShortName = "w")]
            public string weight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of time an example can be multiplied", ShortName = "m")]
            public float maxMulti = 5f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed to multiply randomly the label.", ShortName = "s")]
            public int seed = 42;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads used to estimate how much a class should resample.", ShortName = "nt")]
            public int? numThreads;

            #endregion

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write((byte)algo);
                ctx.Writer.Write(label);
                ctx.Writer.Write(newColumn);
                ctx.Writer.Write(weight);
                ctx.Writer.Write(maxMulti);
                ctx.Writer.Write(seed);
                ctx.Writer.Write(numThreads ?? -1);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                algo = (MultiplicationAlgorithm)ctx.Reader.ReadByte();
                host.Check(algo >= 0, "algo");
                label = ctx.Reader.ReadString();
                host.CheckValue(label, "label");
                newColumn = ctx.Reader.ReadString();
                host.CheckValue(newColumn, "newColumn");
                weight = ctx.Reader.ReadString();
                maxMulti = ctx.Reader.ReadInt32();
                host.Check(maxMulti >= 0, "maxMulti");
                seed = ctx.Reader.ReadInt32();
                int nb = ctx.Reader.ReadInt32();
                host.Check(nb > -2, "numThreads");
                numThreads = nb > 0 ? (int?)nb : null;
            }
        }

        #endregion

        #region internal members / accessors

        IDataView _input;
        IDataTransform _transform;          // templated transform (not the serialized version)
        IHost _host;
        Arguments _args;

        #endregion

        #region public constructor / serialization / load / save

        public MultiToBinaryTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, "args");
            _input = input;

            SchemaHelper.GetColumnIndex(input.Schema, args.label);
            _args = args;
            _transform = CreateTemplatedTransform();
        }

        public static MultiToBinaryTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new MultiToBinaryTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            if (_transform != null)
            {
                ctx.Writer.Write(true);
                _transform.Save(ctx);
            }
            else
                ctx.Writer.Write(false);
        }

        private MultiToBinaryTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            bool tr = ctx.Reader.ReadBoolean();
            if (tr)
                _transform = CreateTemplatedTransform(ctx);
            else
                _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        public IDataView Source { get { return _input; } }
        public Schema Schema { get { return _transform.Schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        public long? GetRowCount()
        {
            _host.AssertValue(_input, "_input");
            return null;
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(columnsNeeded, rand);
        }

        public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(columnsNeeded, n, rand);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            int labelIndex = SchemaHelper.GetColumnIndex(_input.Schema, _args.label);
            var typeLabel = _input.Schema[labelIndex].Type;

            switch (typeLabel.RawKind())
            {
                case DataKind.R4:
                    return new MultiToBinaryState<float, float>(_host, _input, _args);
                case DataKind.BL:
                    return new MultiToBinaryState<bool, bool>(_host, _input, _args);
                case DataKind.U1:
                    return new MultiToBinaryState<byte, byte>(_host, _input, _args);
                case DataKind.U2:
                    return new MultiToBinaryState<ushort, ushort>(_host, _input, _args);
                case DataKind.U4:
                    return new MultiToBinaryState<uint, uint>(_host, _input, _args);
                default:
                    throw Contracts.ExceptNotSupp("Type '{0}' is not handle for a multi-class label. Try unsigned int.", typeLabel.RawKind());
            };
        }

        private IDataTransform CreateTemplatedTransform(ModelLoadContext ctx)
        {
            int labelIndex = SchemaHelper.GetColumnIndex(_input.Schema, _args.label);
            var typeLabel = _input.Schema[labelIndex].Type;

            switch (typeLabel.RawKind())
            {
                case DataKind.R4:
                    return new MultiToBinaryState<float, float>(ctx, _host, _input, _args);
                case DataKind.BL:
                    return new MultiToBinaryState<bool, bool>(ctx, _host, _input, _args);
                case DataKind.U1:
                    return new MultiToBinaryState<byte, byte>(ctx, _host, _input, _args);
                case DataKind.U2:
                    return new MultiToBinaryState<ushort, ushort>(ctx, _host, _input, _args);
                case DataKind.U4:
                    return new MultiToBinaryState<uint, uint>(ctx, _host, _input, _args);
                default:
                    throw Contracts.ExceptNotSupp("Type '{0}' is not handle for a multi-class label. Try unsigned int.", typeLabel.RawKind());
            }
        }

        public VBuffer<TYPE> GetClasses<TYPE>()
            where TYPE : IEquatable<TYPE>
        {
            _host.Assert(_transform != null);
            var tr = _transform as MultiToBinaryState<TYPE, TYPE>;
            _host.CheckValue(tr, "Unexpected type.");
            return tr.GetClasses();
        }

        /// <summary>
        /// Steal the data from a trained transform.
        /// </summary>
        public bool Steal(MultiToBinaryTransform toSteal)
        {
            var p1 = _transform as MultiToBinaryState<float, float>;
            if (p1 != null)
                return p1.Steal(toSteal._transform);
            var p2 = _transform as MultiToBinaryState<bool, bool>;
            if (p2 != null)
                return p2.Steal(toSteal._transform);
            var p3 = _transform as MultiToBinaryState<byte, byte>;
            if (p3 != null)
                return p3.Steal(toSteal._transform);
            var p4 = _transform as MultiToBinaryState<ushort, ushort>;
            if (p4 != null)
                return p4.Steal(toSteal._transform);

            var p5 = _transform as MultiToBinaryState<uint, uint>;
            if (p5 != null)
            {
                // We deal with the case where the label is R4 and the transform is uint.
                var p6 = toSteal._transform as MultiToBinaryState<float, float>;
                if (p6 != null)
                    return p5.StealConvert(p6);
                else
                    return p5.Steal(toSteal._transform);
            }
            throw _host.Except("Unable to cast '{0}'", _transform.GetType());
        }

        #endregion

        #region State

        /// <summary>
        /// Templated transform which sorts rows based on one column.
        /// </summary>
        public class MultiToBinaryState<TLabelC, TLabel> : IDataTransform
            where TLabel : IEquatable<TLabel>
        {
            IHost _host;
            IDataView _input;
            Schema _schema;
            Arguments _args;
            Dictionary<TLabel, float> _labelDistribution;
            float _averageMultiplication;
            int _maxReplica;
            int _colLabel;
            int _colWeight;

            public Dictionary<TLabel, float> LabelDistribution { get { return _labelDistribution; } }
            public float AverageMultiplication { get { return _averageMultiplication; } }
            public VBuffer<TLabel> GetClasses()
            {
                Contracts.Check(_labelDistribution != null, "The transform was not trained. The classes cannot be returned yet.");
                return new VBuffer<TLabel>(_labelDistribution.Count, _labelDistribution.Keys.OrderBy(c => c).ToArray());
            }

            public bool Steal(IDataTransform transform)
            {
                var tr = transform as MultiToBinaryState<TLabelC, TLabel>;
                if (tr == null)
                    throw _host.Except("Type mismatch '{0}' != '{1}'", this.GetType(), transform.GetType());
                return Steal(tr);
            }

            protected bool Steal(MultiToBinaryState<TLabelC, TLabel> transform)
            {
                _labelDistribution = transform._labelDistribution;
                _averageMultiplication = transform._averageMultiplication;
                _maxReplica = transform._maxReplica;
                _colLabel = transform._colLabel;
                _colWeight = transform._colWeight;
                return true;
            }

            public bool StealConvert(MultiToBinaryState<float, float> transform)
            {
                var thisuint = this as MultiToBinaryState<uint, uint>;
                if (thisuint == null)
                    throw _host.Except("Unable to proceed. Type mismatch.");

                var newDist = new Dictionary<uint, float>();
                foreach (var pair in transform._labelDistribution)
                    newDist[(uint)pair.Key + 1] = pair.Value;
                thisuint._labelDistribution = newDist;
                _averageMultiplication = transform._averageMultiplication;
                _maxReplica = transform._maxReplica;
                _colLabel = transform._colLabel;
                _colWeight = transform._colWeight;
                return true;
            }

            object _lock;

            public IDataView Source { get { return _input; } }
            public Schema Schema { get { return _schema; } }

            public MultiToBinaryState(IHostEnvironment host, IDataView input, Arguments args)
            {
                _host = host.Register("MultiToBinaryState");
                _host.CheckValue(input, "input");
                _input = input;
                _lock = new object();
                _args = args;
                BuildSchema(input);
                _labelDistribution = null;
                _averageMultiplication = -1;
                _colLabel = SchemaHelper.GetColumnIndex(_input.Schema, _args.label);
                if (string.IsNullOrEmpty(_args.weight))
                    _colWeight = -1;
                else SchemaHelper.GetColumnIndex(_input.Schema, _args.weight);
            }

            void BuildSchema(IDataView input)
            {
                Contracts.AssertValue(_args);
                Contracts.AssertValue(input);
                switch (_args.algo)
                {
                    case MultiplicationAlgorithm.Default:
                    case MultiplicationAlgorithm.Reweight:
                        _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema,
                                                new string[] { _args.newColumn },
                                                new ColumnType[] { BoolType.Instance },
                                                true, true));
                        break;
                    case MultiplicationAlgorithm.Ranking:
                        _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema,
                                                new string[] { _args.newColumn },
                                                new ColumnType[] { NumberType.U4 },
                                                true, true));
                        break;
                    default:
                        throw _host.ExceptNotSupp("Unsupported algorithm {0}", _args.algo);
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                if (_labelDistribution == null)
                    ctx.Writer.Write(0);
                else
                {
                    ctx.Writer.Write(_labelDistribution.Count);
                    foreach (var pair in _labelDistribution)
                    {
                        ctx.Writer.Write(Convert.ToInt64(pair.Key));
                        ctx.Writer.Write(pair.Value);
                    }
                }
            }

            public MultiToBinaryState(ModelLoadContext ctx, IHostEnvironment host, IDataView input, Arguments args)
            {
                _host = host.Register("MultiToBinaryState");
                _host.CheckValue(input, "input");
                _input = input;
                _lock = new object();
                _args = args;
                BuildSchema(input);
                _colLabel = SchemaHelper.GetColumnIndex(_input.Schema, _args.label);
                if (string.IsNullOrEmpty(_args.weight))
                    _colWeight = -1;
                else _colWeight = SchemaHelper.GetColumnIndex(_input.Schema, _args.weight);

                int nb = ctx.Reader.ReadInt32();
                if (nb == 0)
                    _labelDistribution = null;
                else
                {
                    var typeLabel = _input.Schema[_colLabel].Type;
                    bool identity;
                    ValueMapper<long, TLabel> mapper = Conversions.Instance.GetStandardConversion<long, TLabel>(NumberType.I8, typeLabel, out identity);

                    _labelDistribution = new Dictionary<TLabel, float>();
                    for (int i = 0; i < nb; ++i)
                    {
                        var key = ctx.Reader.ReadInt64();
                        var value = ctx.Reader.ReadInt64();

                        TLabel tkey = default(TLabel);
                        mapper(in key, ref tkey);
                        _labelDistribution[tkey] = value;
                    }
                    using (var ch = _host.Start("Finalize MultiToBinaryState"))
                        Finalize(ch);
                }
            }

            void TrainTransform(Random rand)
            {
                lock (_lock)
                {
                    if (_labelDistribution != null)
                        return;

                    using (var ch = _host.Start("MultiToBinary Training"))
                        ComputeLabelDistribution(ch, rand);
                }
            }

            void ComputeLabelDistribution(IChannel ch, Random rand)
            {
                int nt = DataViewUtils.GetThreadCount(_host, _args.numThreads ?? 0);
                var cols = _input.Schema.Where(col => col.Index == _colLabel || col.Index == _colWeight).ToArray();
                var cursors = nt <= 1 ? null : _input.GetRowCursorSet(cols, nt, rand);
                if (cursors != null && cursors.Length == 1)
                {
                    cursors[0].Dispose();
                    cursors = null;
                }

                if (cursors != null)
                {
                    var hists = new Dictionary<TLabel, float>[cursors.Length];
                    Action[] ops = new Action[cursors.Length];
                    long[] positions = new long[ops.Length];
                    for (int i = 0; i < ops.Length; i++)
                    {
                        int chunkId = i;
                        hists[i] = new Dictionary<TLabel, float>();
                        ops[i] = new Action(() =>
                        {
                            var cursor = cursors[chunkId];
                            var labelGetter = cursor.GetGetter<TLabel>(_colLabel);
                            var weightGetter = _colWeight == -1 ? null : cursor.GetGetter<float>(_colWeight);
                            TLabel value = default(TLabel);
                            float weight = 0;
                            var hist = hists[chunkId];
                            while (cursor.MoveNext())
                            {
                                labelGetter(ref value);
                                if (weightGetter != null)
                                    weightGetter(ref weight);
                                else
                                    weight = 1;
                                if (hist.ContainsKey(value))
                                    hist[value] += weight;
                                else
                                    hist[value] = weight;
                            }
                            cursor.Dispose();
                        });
                    }

                    // parallel processing
                    Parallel.Invoke(new ParallelOptions() { }, ops);

                    // finalization
                    _labelDistribution = new Dictionary<TLabel, float>();
                    foreach (var hist in hists)
                    {
                        foreach (var pair in hist)
                        {
                            if (_labelDistribution.ContainsKey(pair.Key))
                                _labelDistribution[pair.Key] += pair.Value;
                            else
                                _labelDistribution[pair.Key] = pair.Value;
                        }
                    }
                }
                else
                {
                    using (var cursor = _input.GetRowCursor(cols))
                    {
                        var labelGetter = cursor.GetGetter<TLabel>(_colLabel);
                        var weightGetter = _colWeight == -1 ? null : cursor.GetGetter<float>(_colWeight);
                        TLabel value = default(TLabel);
                        float weight = 0;
                        _labelDistribution = new Dictionary<TLabel, float>();
                        var hist = _labelDistribution;
                        while (cursor.MoveNext())
                        {
                            labelGetter(ref value);
                            if (weightGetter != null)
                                weightGetter(ref weight);
                            else
                                weight = 1;
                            if (hist.ContainsKey(value))
                                hist[value] += weight;
                            else
                                hist[value] = weight;
                        }
                    }
                }

                if (!_labelDistribution.Any())
                    throw ch.Except("View is empty.");

                if (!_labelDistribution.Any())
                    throw ch.Except("View is empty.");

                Finalize(ch);
            }

            private Func<TYPE, int, TYPE> LabelConverter<TYPE>()
            {
                TYPE[] pointer = new TYPE[0];
                if ((pointer as float[]) != null)
                    return FloatLabelConverter() as Func<TYPE, int, TYPE>;
                if ((pointer as byte[]) != null)
                    return ByteLabelConverter() as Func<TYPE, int, TYPE>;
                if ((pointer as ushort[]) != null)
                    return UShortLabelConverter() as Func<TYPE, int, TYPE>;
                if ((pointer as uint[]) != null)
                    return UIntLabelConverter() as Func<TYPE, int, TYPE>;
                throw Contracts.ExceptNotSupp("Unable to convert Type {0} into int.", default(TYPE).GetType().ToString());
            }

            private Func<float, int, float> FloatLabelConverter() { return (float x, int d) => x + d; }
            private Func<byte, int, byte> ByteLabelConverter() { throw Contracts.ExceptNotSupp(); }
            private Func<ushort, int, ushort> UShortLabelConverter() { throw Contracts.ExceptNotSupp(); }
            private Func<uint, int, uint> UIntLabelConverter() { return (uint x, int d) => x + (uint)d; }

            private Func<TYPE, int, bool> LabelFilter<TYPE>()
            {
                TYPE[] pointer = new TYPE[0];
                if ((pointer as uint[]) != null)
                    return UIntLabelFilter() as Func<TYPE, int, bool>;
                return (x, d) => true;
            }

            private Func<uint, int, bool> UIntLabelFilter() { return (x, d) => x >= -d; }

            void Finalize(IChannel ch)
            {
                var max = _labelDistribution.Count;
                float maxRepl = Math.Max(0.001f, Math.Min(Math.Max(2, max), _args.maxMulti));
                if (maxRepl > 1000)
                    throw ch.Except("The initial datasets cannot be multiplied by more than 1000 (x{0}). Change the parameter maxMulti", maxRepl);
                _maxReplica = (int)maxRepl;

                switch (_args.algo)
                {
                    case MultiplicationAlgorithm.Default:
                        _averageMultiplication = _maxReplica;
                        break;
                    case MultiplicationAlgorithm.Reweight:
                        // REVIEW: to be improved or propopose another and better strategy.
                        _averageMultiplication = _maxReplica;
                        break;
                    case MultiplicationAlgorithm.Ranking:
                        _averageMultiplication = _maxReplica;
                        break;
                    default:
                        throw ch.ExceptParam("algo", "Unrecognized algo {0}", _args.algo);
                }
                ch.Info("    Classes distribution:");
                var rows = new List<string>();
                foreach (var pair in _labelDistribution.OrderBy(c => c.Key))
                {
                    if (rows.Count >= 5)
                    {
                        ch.Info("        {0}", string.Join(" ", rows));
                        rows.Clear();
                    }
                    else
                        rows.Add(string.Format("c{0}:{1}", pair.Key, pair.Value));
                }
                if (rows.Any())
                    ch.Info("        {0}", string.Join(" ", rows));
            }

            public bool CanShuffle { get { return true; } }
            public long? GetRowCount() { return null; }

            public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
            {
                var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, _input.Schema, new[] { _colLabel, _colWeight }).ToArray();
                var oldCols = SchemaHelper.ColumnsNeeded(cols, _input.Schema);
                TrainTransform(rand);
                _host.AssertValue(_labelDistribution, "_labelDistribution");
                var cursor = _input.GetRowCursor(oldCols, rand);
                switch (_args.algo)
                {
                    case MultiplicationAlgorithm.Default:
                    case MultiplicationAlgorithm.Reweight:
                        return new MultiToBinaryCursor<TLabelC, TLabel, bool>(this, cursor, _colLabel, _colWeight, _maxReplica, _args.algo, _args.seed);
                    case MultiplicationAlgorithm.Ranking:
                        return new MultiToBinaryCursor<TLabelC, TLabel, uint>(this, cursor, _colLabel, _colWeight, _maxReplica, _args.algo, _args.seed);
                    default:
                        throw _host.ExceptNotSupp("Unkown algorithm '{0}'", _args.algo);
                }
            }

            public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
            {
                var cols = SchemaHelper.ColumnsNeeded(columnsNeeded, _input.Schema, new[] { _colLabel, _colWeight }).ToArray();
                var oldCols = SchemaHelper.ColumnsNeeded(cols, _input.Schema);
                TrainTransform(rand);
                _host.AssertValue(_labelDistribution, "_labelDistribution");
                var cursors = _input.GetRowCursorSet(oldCols, n, rand);
                switch (_args.algo)
                {
                    case MultiplicationAlgorithm.Default:
                    case MultiplicationAlgorithm.Reweight:
                        return cursors.Select(c => new MultiToBinaryCursor<TLabelC, TLabel, bool>(this, c, _colLabel, _colWeight, _maxReplica, _args.algo, _args.seed)).ToArray();
                    case MultiplicationAlgorithm.Ranking:
                        return cursors.Select(c => new MultiToBinaryCursor<TLabelC, TLabel, uint>(this, c, _colLabel, _colWeight, _maxReplica, _args.algo, _args.seed)).ToArray();
                    default:
                        throw _host.ExceptNotSupp("Unkown algorithm '{0}'", _args.algo);
                }
            }
        }

        #endregion

        #region Cursor

        public class MultiToBinaryCursor<TFeatures, TLabel, TLabelInter> : RowCursor
            where TLabel : IEquatable<TLabel>
        {
            readonly MultiToBinaryState<TFeatures, TLabel> _view;
            readonly RowCursor _inputCursor;
            readonly int _colLabel;
            readonly int _colWeight;
            readonly int _colName;
            readonly int _maxReplica;
            readonly int _shift;
            readonly MultiplicationAlgorithm _algo;
            readonly TLabel[] _labels;

            TLabel _label;
            float _weight;
            Tuple<TLabel, TLabelInter>[] _copies;
            int _copy;
            ValueGetter<TLabel> _labelGetter;
            ValueGetter<float> _weightGetter;
            float _maxFreq, _minFreq;
            Random _rand;

            public MultiToBinaryCursor(MultiToBinaryState<TFeatures, TLabel> view, RowCursor cursor, int colLabel, int colWeight, int maxReplica, MultiplicationAlgorithm algo, int seed)
            {
                _view = view;
                _colName = view.Source.Schema.Count;
                _colLabel = colLabel;
                _colWeight = colWeight;
                _inputCursor = cursor;
                _algo = algo;
                _labelGetter = null;
                _weightGetter = null;
                _maxReplica = maxReplica;
                _shift = 0;
                ++maxReplica;
                while (maxReplica > 0)
                {
                    _shift += 1;
                    maxReplica >>= 1;
                }
                _maxFreq = view.LabelDistribution.Select(c => c.Value).Max();
                _minFreq = Math.Max(1, Math.Max(view.LabelDistribution.Select(c => c.Value).Min(), _maxFreq / _maxReplica));
                _rand = new Random(seed);
                _labels = view.LabelDistribution.Select(c => c.Key).ToArray();
                _copies = null;
                _copy = -1;
            }

            public override ValueGetter<RowId> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref RowId pos) =>
                {
                    if (_shift > 0)
                    {
                        Contracts.Assert(_copy >= 0 && _copy <= _maxReplica);
                        getId(ref pos);
                        ulong left = pos.Low << _shift;
                        left >>= _shift;
                        left = pos.Low - left;
                        ulong lo = pos.Low << _shift;
                        ulong hi = pos.High << _shift;
                        hi += left >> (64 - _shift);
                        pos = new RowId(lo + (ulong)_copy, hi);
                    }
                    else
                        Contracts.Assert(_copy == 0);
                };
            }

            public override bool IsColumnActive(int col)
            {
                if (col < _inputCursor.Schema.Count)
                {
                    Contracts.Assert(_inputCursor.IsColumnActive(_colLabel) &&
                                     (_colWeight == -1 || _inputCursor.IsColumnActive(_colWeight)));
                    return _inputCursor.IsColumnActive(col);
                }
                return true;
            }

            public override long Batch { get { return _inputCursor.Batch; } }
            public override long Position { get { return _inputCursor.Position; } }
            public override Schema Schema { get { return _view.Schema; } }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveNext()
            {
                if (_labelGetter == null)
                {
                    _labelGetter = _inputCursor.GetGetter<TLabel>(_colLabel);
                    _weightGetter = _colWeight >= 0 ? _inputCursor.GetGetter<float>(_colWeight) : null;
                    _copies = new Tuple<TLabel, TLabelInter>[0];
                    _copy = -1;
                }

                ++_copy;

                if (_copy >= _copies.Length)
                {
                    var r = _inputCursor.MoveNext();
                    if (!r)
                    {
                        _labelGetter = null;
                        return r;
                    }
                    _labelGetter(ref _label);
                    if (_weightGetter != null)
                        _weightGetter(ref _weight);
                    else
                        _weight = 1;
                    _copies = GetMultiplicator();
                    _copy = 0;
                }
                return true;
            }

            Tuple<TLabel, TLabelInter>[] GetMultiplicator()
            {
                switch (_algo)
                {
                    case MultiplicationAlgorithm.Default:
                    case MultiplicationAlgorithm.Reweight:
                        return GetMultiplicatorBool() as Tuple<TLabel, TLabelInter>[];
                    case MultiplicationAlgorithm.Ranking:
                        return GetMultiplicatorUint() as Tuple<TLabel, TLabelInter>[];
                    default:
                        throw Contracts.ExceptNotSupp("Algorithm not supported for algo={0}", _algo);
                }
            }

            Tuple<TLabel, bool>[] GetMultiplicatorBool()
            {
                int nb;
                switch (_algo)
                {
                    case MultiplicationAlgorithm.Default:
                        if (_labels.Length <= _maxReplica)
                            return _view.LabelDistribution.Select(c => new Tuple<TLabel, bool>(c.Key, c.Key.Equals(_label)))
                                                                 .ToArray();
                        else
                        {
                            nb = _maxReplica;
                            break;
                        }
                    case MultiplicationAlgorithm.Reweight:
                        float val = _view.LabelDistribution[_label];
                        nb = (int)(((long)_maxReplica * _minFreq / val) + 0.0001f);
                        break;
                    default:
                        throw Contracts.ExceptNotSupp("Unexpected '{0}'", _algo);
                }

                // REVIEW: think about sampling around difficult borders...
                if (nb <= 0)
                    return new Tuple<TLabel, bool>[0];
                else if (nb <= 1)
                {
                    var h = _rand.Next() % 2;
                    if (h == 0)
                        return new Tuple<TLabel, bool>[1] { new Tuple<TLabel, bool>(_label, true) };
                    else
                    {
                        h = _rand.Next() % _labels.Length;
                        return new Tuple<TLabel, bool>[1] { new Tuple<TLabel, bool>(_label, _labels[h].Equals(_label)) };
                    }
                }
                else
                {
                    nb = Math.Min(nb, _maxReplica);
                    var res = new Tuple<TLabel, bool>[nb];
                    var h = _rand.Next() % 2;
                    if (h == 1)
                        res[0] = new Tuple<TLabel, bool>(_label, true);
                    for (int i = h; i < nb; ++i)
                    {
                        h = _rand.Next() % _labels.Length;
                        res[i] = new Tuple<TLabel, bool>(_labels[h], _labels[h].Equals(_label));
                    }
                    return res;
                }
            }

            Tuple<TLabel, uint>[] GetMultiplicatorUint()
            {
                int nb;
                switch (_algo)
                {
                    case MultiplicationAlgorithm.Ranking:
                        // Label 3 or 4 means Excellent or Perfect.
                        if (_labels.Length <= _maxReplica)
                            return _view.LabelDistribution.Select(c => new Tuple<TLabel, uint>(c.Key, c.Key.Equals(_label) ? (uint)4 : (uint)0))
                                                                 .ToArray();
                        else
                        {
                            nb = _maxReplica;
                            break;
                        }
                    default:
                        throw Contracts.ExceptNotSupp("Unexpected '{0}'", _algo);
                }

                if (nb <= 0)
                    return new Tuple<TLabel, uint>[0];
                else if (nb <= 1)
                {
                    var h = _rand.Next() % 2;
                    if (h == 0)
                        // Label 3 or 4 means Excellent or Perfect.
                        return new Tuple<TLabel, uint>[1] { new Tuple<TLabel, uint>(_label, 4) };
                    else
                    {
                        h = _rand.Next() % _labels.Length;
                        // Label 3 or 4 means Excellent or Perfect.
                        return new Tuple<TLabel, uint>[1] { new Tuple<TLabel, uint>(_label, _labels[h].Equals(_label) ? (uint)4 : 0) };
                    }
                }
                else
                {
                    nb = Math.Min(nb, _maxReplica);
                    var res = new Tuple<TLabel, uint>[nb];
                    var h = _rand.Next() % 2;
                    if (h == 1)
                        // Label 3 or 4 means Excellent or Perfect.
                        res[0] = new Tuple<TLabel, uint>(_label, 4);
                    for (int i = h; i < nb; ++i)
                    {
                        h = _rand.Next() % _labels.Length;
                        // Label 3 or 4 means Excellent or Perfect.
                        res[i] = new Tuple<TLabel, uint>(_labels[h], _labels[h].Equals(_label) ? (uint)4 : (uint)0);
                    }
                    return res;
                }
            }

            static public Func<TYPE, int> LabelConverter<TYPE>()
            {
                TYPE[] pointer = new TYPE[1];
                if ((pointer as float[]) != null)
                    return FloatLabelConverter() as Func<TYPE, int>;
                if ((pointer as byte[]) != null)
                    return ByteLabelConverter() as Func<TYPE, int>;
                if ((pointer as ushort[]) != null)
                    return UShortLabelConverter() as Func<TYPE, int>;
                if ((pointer as uint[]) != null)
                    return UIntLabelConverter() as Func<TYPE, int>;
                throw Contracts.ExceptNotSupp("Unable to convert Type {0} into int.", default(TYPE).GetType().ToString());
            }

            static private Func<float, int> FloatLabelConverter() { return x => (int)x; }
            static private Func<byte, int> ByteLabelConverter() { return x => (int)x; }
            static private Func<ushort, int> UShortLabelConverter() { return x => (int)x; }
            static private Func<uint, int> UIntLabelConverter() { return x => (int)x; }

            public ValueGetter<TValue> GetGetterLabelAsVector<TValue>(int col, bool useFeatures)
            {
                if (col == _colLabel)
                {
                    var lablc = LabelConverter<TLabel>();
                    int nb = lablc(_view.LabelDistribution.Select(c => c.Key).Max()) + 1;

                    if (useFeatures)
                    {
#if (DEBUG)
                        string sch = SchemaHelper.ToString(_inputCursor.Schema);
                        Contracts.Assert(!string.IsNullOrWhiteSpace(sch));
#endif
                        ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                        {
                            if (value.Length != 1 || value.Count != nb)
                                value = new VBuffer<float>(nb, 1, new float[1], new int[1]);
                            value.Values[0] = 1;
                            value.Indices[0] = lablc(_copies[_copy].Item1);
                        };
                        return getter as ValueGetter<TValue>;
                    }
                    else
                    {
                        ValueGetter<VBuffer<bool>> getter = (ref VBuffer<bool> value) =>
                        {
                            if (value.Length != 1 || value.Count != nb)
                                value = new VBuffer<bool>(nb, 1, new bool[1], new int[1]);
                            value.Values[0] = true;
                            value.Indices[0] = lablc(_copies[_copy].Item1);
                        };
                        return getter as ValueGetter<TValue>;
                    }
                }
                else
                    throw Contracts.ExceptNotSupp("Outside of the scope of this function. Use GetGetter.");
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (col == _colLabel)
                {
                    if (((new TValue[0]) as VBuffer<float>[]) != null)
                        return GetGetterLabelAsVector<TValue>(col, true);
                    else if (((new TValue[0]) as VBuffer<bool>[]) != null)
                        return GetGetterLabelAsVector<TValue>(col, false);
                    else if (new TLabel[0] as float[] != null)
                    {
                        var lablc = LabelConverter<TLabel>();
                        ValueGetter<float> getter = (ref float value) =>
                        {
                            Contracts.AssertValue(_copies);
                            Contracts.Check(lablc(_copies[_copy].Item1) >= 0, "Negative label");
                            value = (float)(lablc(_copies[_copy].Item1));
                        };
                        return getter as ValueGetter<TValue>;
                    }
                    else if (new TLabel[0] as uint[] != null)
                    {
                        var lablc = LabelConverter<TLabel>();
                        if (new TValue[0] as float[] != null)
                        {
                            ValueGetter<float> getter = (ref float value) =>
                            {
                                Contracts.AssertValue(_copies);
                                Contracts.Check(lablc(_copies[_copy].Item1) >= 0, "Negative label");
                                value = (float)(lablc(_copies[_copy].Item1));
                            };
                            return getter as ValueGetter<TValue>;
                        }
                        else if (new TValue[0] as uint[] != null)
                        {
                            ValueGetter<uint> getter = (ref uint value) =>
                            {
                                Contracts.AssertValue(_copies);
                                Contracts.Check(lablc(_copies[_copy].Item1) >= 0, "Negative label");
                                value = (uint)(lablc(_copies[_copy].Item1));
                            };
                            return getter as ValueGetter<TValue>;
                        }
                        else
                            throw Contracts.Except("Unable to produce a getter for type {0}", default(TValue).GetType());
                    }
                    else
                    {
                        ValueGetter<TLabel> getter = (ref TLabel value) =>
                        {
                            Contracts.AssertValue(_copies);
                            value = _copies[_copy].Item1;
                        };
                        return getter as ValueGetter<TValue>;
                    }
                }
                else if (col < _view.Source.Schema.Count)
                {
                    try
                    {
                        var getter = _inputCursor.GetGetter<TValue>(col);
                        if (getter == null)
                            throw Contracts.Except($"Unable to create a getter of type '{typeof(TValue)}' for column {col}:{_view.Source.Schema[col].Name} of type {_view.Source.Schema[col].Type}.");
                        return getter;
                    }
                    catch (Exception e)
                    {
                        throw Contracts.Except($"Unable to create a getter of type '{typeof(TValue)}' for column {col}:{_view.Source.Schema[col].Name} of type {_view.Source.Schema[col].Type} due to {e}");
                    }
                }
                else if (col == _view.Source.Schema.Count)
                {
                    ValueGetter<TLabelInter> getter = (ref TLabelInter value) =>
                    {
                        Contracts.AssertValue(_copies);
                        value = _copies[_copy].Item2;
                    };
                    return getter as ValueGetter<TValue>;
                }
                else
                    throw Contracts.Except("Unexpected columns {0} > {1}.", col, _view.Source.Schema.Count);
            }
        }

        #endregion
    }
}
