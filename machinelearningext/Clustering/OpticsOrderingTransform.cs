// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;
using Scikit.ML.NearestNeighbors;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using OpticsOrderingTransform = Scikit.ML.Clustering.OpticsOrderingTransform;


[assembly: LoadableClass(OpticsOrderingTransform.Summary, typeof(OpticsOrderingTransform),
    typeof(OpticsOrderingTransform.Arguments), typeof(SignatureDataTransform),
    "OPTICS Ordering Transform", OpticsOrderingTransform.LoaderSignature,
    "OPTICSOrdering", "OPTICSOrd")]

[assembly: LoadableClass(OpticsOrderingTransform.Summary, typeof(OpticsOrderingTransform),
    null, typeof(SignatureLoadDataTransform), "OPTICS Ordering Transform",
    "OPTICS Ordering Transform", OpticsOrderingTransform.LoaderSignature)]


namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Transform which applies the Optics ordering algorithm.
    /// </summary>
    public class OpticsOrderingTransform : TransformBase
    {
        #region identification

        public const string LoaderSignature = "OpticsOrderingTransform";
        public const string Summary = "Orders data using OPTICS algorithm.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "OPTORDME",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OpticsOrderingTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Column which contains the features.", ShortName = "col")]
            public string features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Radius of the sample areas. If null, the transform will give it a default value based on the data.", ShortName = "eps")]
            public double epsilon = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of points in the sample area to be considered a cluster.", ShortName = "mps")]
            public int minPoints = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Ordering results.", ShortName = "outc")]
            public string outOrdering = "Ordering";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reachability distance", ShortName = "outr")]
            public string outReachabilityDistance = "Reachability";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Core distance", ShortName = "outcd")]
            public string outCoreDistance = "Core";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the number generators.", ShortName = "s")]
            public int? seed = 42;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(features);
                ctx.Writer.Write(epsilon);
                ctx.Writer.Write(minPoints);
                ctx.Writer.Write(outOrdering);
                ctx.Writer.Write(outReachabilityDistance);
                ctx.Writer.Write(outCoreDistance);
                ctx.Writer.Write(seed ?? -1);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                features = ctx.Reader.ReadString();
                epsilon = ctx.Reader.ReadDouble();
                minPoints = ctx.Reader.ReadInt32();
                outOrdering = ctx.Reader.ReadString();
                outReachabilityDistance = ctx.Reader.ReadString();
                outCoreDistance = ctx.Reader.ReadString();
                int s = ctx.Reader.ReadInt32();
                seed = s < 0 ? (int?)null : s;
            }
        }

        #endregion

        #region internal members / accessors

        IDataTransform _transform;      // templated transform (not the serialized version)
        Arguments _args;                // parameters
        DataViewSchema _schema;                 // We need the schema the transform outputs.

        public override DataViewSchema OutputSchema { get { return _schema; } }

        #endregion

        #region public constructor / serialization / load / save

        public OpticsOrderingTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");

            if (args.epsilon < 0)
                Contracts.Check(false, "Parameter epsilon must be positive or null.");

            if (args.minPoints <= 0)
                Contracts.Check(false, "Parameter minPoints must be positive.");

            _args = args;
            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema, new string[] { args.outOrdering, args.outReachabilityDistance, args.outCoreDistance },
                                                       new DataViewType[] { NumberDataViewType.Int64, NumberDataViewType.Single, NumberDataViewType.Single }));
            _transform = CreateTemplatedTransform();
        }
        public static OpticsOrderingTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new OpticsOrderingTransform(h, ctx, input));
        }

        protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, Host);
        }

        private OpticsOrderingTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, Host);
            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema, new string[] { _args.outOrdering, _args.outReachabilityDistance, _args.outCoreDistance },
                                                       new DataViewType[] { NumberDataViewType.Int64, NumberDataViewType.Single, NumberDataViewType.Single }));
            _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        public override bool CanShuffle { get { return _transform.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public override long? GetRowCount()
        {
            Host.AssertValue(Source, "_input");
            return Source.GetRowCount();
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return false;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(columnsNeeded, rand);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(columnsNeeded, n, rand);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            int index = SchemaHelper.GetColumnIndex(Source.Schema, _args.features);
            var type = Source.Schema[index].Type;
            if (!type.IsVector())
                throw Host.Except("Features must be a vector.");
            switch (type.AsVector().ItemType().RawKind())
            {
                case DataKind.R4:
                    return new OpticsOrderingState(Host, this, Source, _args);
                default:
                    throw Host.Except("Features must be a vector a floats.");
            }
        }

        public class OpticsOrderingState : IDataTransform
        {
            IHost _host;
            IDataView _input;
            Arguments _args;
            OpticsOrderingTransform _parent;
            OpticsOrderingResult[] _Results;              // OptictsOrderingResult: result of a point
            Dictionary<long, long> _reversedMapping;      // long: index of a point

            /// <summary>
            /// Array of cluster.
            /// </summary>
            public OpticsOrderingResult[] OpticsOrderingResults { get { return _Results; } }

            /// <summary>
            /// To retrieve the index within the ordering of one observation.
            /// </summary>
            public long GetMappedIndex(long vertexId) { return _reversedMapping[vertexId]; }

            object _lock;

            public IDataView Source { get { return _input; } }
            public DataViewSchema Schema { get { return _parent.OutputSchema; } }

            public OpticsOrderingState(IHostEnvironment host, OpticsOrderingTransform parent, IDataView input, Arguments args)
            {
                _host = host.Register("OpticsOrderingState");
                _host.CheckValue(input, "input");
                _input = input;
                _lock = new object();
                _args = args;
                _Results = null;
                _parent = parent;
            }

            void TrainTransform()
            {
                lock (_lock)
                {
                    if (_Results != null)
                        return;

                    using (var ch = _host.Start("Starting Optics"))
                    {
                        var sw = Stopwatch.StartNew();
                        sw.Start();
                        var points = new List<IPointIdFloat>();
                        int index = SchemaHelper.GetColumnIndex(_input.Schema, _args.features);

                        // Caching data.
                        ch.Info("Caching the data.");
                        using (var cursor = _input.GetRowCursor(_input.Schema.Where(c => c.Index == index)))
                        {
                            var getter = cursor.GetGetter<VBuffer<float>>(index);
                            var getterId = cursor.GetIdGetter();
                            DataViewRowId id = new DataViewRowId();

                            VBuffer<float> tmp = new VBuffer<float>();

                            for (int i = 0; cursor.MoveNext(); i++)
                            {
                                getter(ref tmp);
                                getterId(ref id);
                                if (id > long.MaxValue)
                                    ch.Except("An id is outside the range for long {0}", id);
                                points.Add(new PointIdFloat((long)id, tmp.DenseValues().Select(c => (float)c)));
                            }
                        }

                        // Mapping.
                        // long: index in the ordering
                        // long: index of a point
                        var mapping = new long[points.Count];
                        var mapprev = new Dictionary<long, long>();

                        var distance = (float)_args.epsilon;
                        if (distance <= 0)
                        {
                            float mind, maxd;
                            distance = EstimateDistance(ch, points, out mind, out maxd);
                            ch.Info(MessageSensitivity.UserData, "epsilon (=Radius) was estimating on random couples of points: {0} in [{1}, {2}]", distance, mind, maxd);
                        }

                        Optics opticsAlgo = new Optics(points, _args.seed);
                        //Ordering
                        ch.Info(MessageSensitivity.UserData, "Generating OPTICS ordering for {0} points.", points.Count);
                        int nPoints = points.Count;
                        int cyclesBetweenLogging = Math.Min(1000, nPoints / 10);
                        int currentIteration = 0;

                        Action progressLogger = () =>
                        {
                            if (++currentIteration % cyclesBetweenLogging == 0)
                                ch.Info(MessageSensitivity.None, "Processing {0}/{1}", currentIteration, nPoints);
                        };

                        OpticsOrdering opticsOrdering = opticsAlgo.Ordering(
                            distance,
                            _args.minPoints,
                            seed: _args.seed,
                            onShuffle: msg => ch.Info(MessageSensitivity.UserData, msg),
                            onPointProcessing: progressLogger);
                        IReadOnlyDictionary<long, long> results = opticsOrdering.orderingMapping;
                        var reachabilityDs = opticsOrdering.reachabilityDistances;
                        var coreDs = opticsOrdering.coreDistancesCache;

                        for (int i = 0; i < results.Count; ++i)
                        {
                            var p = points[i];
                            mapprev[results[i]] = i;
                            mapping[i] = results[i];
                        }
                        _reversedMapping = mapprev;

                        // Cleaning.
                        ch.Info(MessageSensitivity.None, "Cleaning.");
                        // We replace by the original labels.
                        _Results = new OpticsOrderingResult[results.Count];

                        for (int i = 0; i < results.Count; ++i)
                        {
                            long pId = points[i].id;
                            float? rd;
                            float? cd;

                            reachabilityDs.TryGetValue(pId, out rd);
                            coreDs.TryGetValue(pId, out cd);

                            _Results[i] = new OpticsOrderingResult()
                            {
                                id = results[i] != DBScan.NOISE ? results[i] : -1,
                                reachability = (float)rd.GetValueOrDefault(float.PositiveInfinity),
                                core = (float)cd.GetValueOrDefault(float.PositiveInfinity)
                            };
                        }
                        ch.Info(MessageSensitivity.UserData, "Ordered {0} points.", _Results.Count());
                        sw.Stop();
                        ch.Info(MessageSensitivity.None, "'OpticsOrdering' finished in {0}.", sw.Elapsed);
                    }
                }
            }

            public float EstimateDistance(IChannel ch, List<IPointIdFloat> points, out float minDistance, out float maxDistance)
            {
                ch.Info("Estimating epsilon based on the data. We pick up two random random computes the average distance.");
                var rand = _args.seed.HasValue ? new Random(_args.seed.Value) : new Random();
                var stack = new List<float>();
                float sum = 0, sum2 = 0;
                float d;
                int nb = 0;
                float ave = 0, last = 0;
                int i, j;
                while (stack.Count < 10000)
                {
                    i = rand.Next(0, points.Count - 1);
                    j = rand.Next(0, points.Count - 1);
                    if (i == j)
                        continue;
                    d = points[i].DistanceTo(points[j]);
                    sum += d;
                    sum2 += d * d;
                    stack.Add(d);
                    ++nb;
                    if (nb > 10)
                    {
                        ave = sum2 / nb;
                        if (Math.Abs(ave - last) < 1e-5)
                            break;
                    }
                    if (nb > 9)
                        last = ave;
                }
                if (stack.Count == 0)
                    throw ch.Except("The radius cannot be estimated.");
                stack.Sort();
                if (!stack.Where(c => !float.IsNaN(c)).Any())
                    throw ch.Except("All distances are NaN. Check your datasets.");
                minDistance = stack.Where(c => !float.IsNaN(c)).First();
                maxDistance = stack.Last();
                return stack[Math.Min(stack.Count - 1, Math.Max(stack.Count / 20, 2))];
            }

            public bool CanShuffle { get { return true; } }

            public long? GetRowCount()
            {
                return _input.GetRowCount();
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                TrainTransform();
                _host.AssertValue(_Results, "_Results");
                var cursor = _input.GetRowCursor(columnsNeeded, rand);
                return new OpticsOrderingCursor(this, cursor);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                TrainTransform();
                _host.AssertValue(_Results, "_Results");
                var cursors = _input.GetRowCursorSet(columnsNeeded, n, rand);
                return cursors.Select(c => new OpticsOrderingCursor(this, c)).ToArray();
            }

            public void Save(ModelSaveContext ctx)
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        public class OpticsOrderingCursor : DataViewRowCursor
        {
            readonly OpticsOrderingState _view;
            readonly DataViewRowCursor _inputCursor;
            readonly int _colOrdering;
            readonly int _colReachability;
            readonly int _colCore;
            readonly int _colName;

            public OpticsOrderingCursor(OpticsOrderingState view, DataViewRowCursor cursor)
            {
                _view = view;
                _colOrdering = view.Source.Schema.Count;
                _colReachability = _colOrdering + 1;
                _colCore = _colReachability + 1;
                _colName = _colCore + 1;
                _inputCursor = cursor;
            }

            public override bool IsColumnActive(int col)
            {
                if (col < _inputCursor.Schema.Count)
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

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (col < _view.Source.Schema.Count)
                    return _inputCursor.GetGetter<TValue>(col);
                else if (col == _view.Source.Schema.Count) // Ordering
                    return GetGetterOrdering() as ValueGetter<TValue>;
                else if (col == _view.Source.Schema.Count + 1) // Reachability Distance
                    return GetGetterReachabilityDistance() as ValueGetter<TValue>;
                else if (col == _view.Source.Schema.Count + 2) // Core Distance
                    return GetGetterCoreDistance() as ValueGetter<TValue>;
                else
                    throw new IndexOutOfRangeException();
            }

            ValueGetter<long> GetGetterOrdering()
            {
                return (ref long orderingId) =>
                {
                    orderingId = _view.GetMappedIndex(_inputCursor.Position);
                };
            }

            ValueGetter<float> GetGetterReachabilityDistance()
            {
                return (ref float rDist) =>
                {
                    long rowIndex = _view.GetMappedIndex(_inputCursor.Position);
                    rDist = _view.OpticsOrderingResults[rowIndex].reachability;
                };
            }

            ValueGetter<float> GetGetterCoreDistance()
            {
                return (ref float cDist) =>
                {
                    long rowIndex = _view.GetMappedIndex(_inputCursor.Position);
                    cDist = _view.OpticsOrderingResults[rowIndex].core;
                };
            }
        }

        #endregion
    }
}
