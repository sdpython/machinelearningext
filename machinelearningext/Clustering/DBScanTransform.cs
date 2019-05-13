// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;
using Scikit.ML.NearestNeighbors;


// The following files makes the object visible to maml.
// This way, it can be added to any pipeline.
using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using DBScanTransform = Scikit.ML.Clustering.DBScanTransform;


[assembly: LoadableClass(DBScanTransform.Summary, typeof(DBScanTransform),
    typeof(DBScanTransform.Arguments), typeof(SignatureDataTransform),
    DBScanTransform.LoaderSignature, "DBScan")]

[assembly: LoadableClass(DBScanTransform.Summary, typeof(DBScanTransform),
    null, typeof(SignatureLoadDataTransform), "DBScan Transform", "DBScan", DBScanTransform.LoaderSignature)]


namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Transform which applies the DBScan clustering algorithm.
    /// </summary>
    public class DBScanTransform : TransformBase
    {
        #region identification

        public const string LoaderSignature = "DBScanTransform";
        public const string Summary = "Clusters data using DBScan algorithm.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DBSCANME",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DBScanTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Column which contains the features.", ShortName = "col")]
            public string features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Radius of the sample areas. If null, the transform will give it a default value based on the data.", ShortName = "eps")]
            public float epsilon = 0f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of points in the sample area to be considered a cluster.", ShortName = "mps")]
            public int minPoints = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Cluster results.", ShortName = "outc")]
            public string outCluster = "ClusterId";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scores", ShortName = "outs")]
            public string outScore = "Score";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the number generators.", ShortName = "s")]
            public int? seed = 42;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(features);
                ctx.Writer.Write(epsilon);
                ctx.Writer.Write(minPoints);
                ctx.Writer.Write(outCluster);
                ctx.Writer.Write(outScore);
                ctx.Writer.Write(seed ?? -1);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                features = ctx.Reader.ReadString();
                epsilon = ctx.Reader.ReadSingle();
                minPoints = ctx.Reader.ReadInt32();
                outCluster = ctx.Reader.ReadString();
                outScore = ctx.Reader.ReadString();
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

        public DBScanTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");
            if (args.epsilon < 0)
                Contracts.Check(false, "Parameter epsilon must be positive or null.");
            if (args.minPoints <= 0)
                Contracts.Check(false, "Parameter minPoints must be positive.");
            _args = args;
            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema, new string[] { args.outCluster, args.outScore },
                                                       new DataViewType[] { NumberDataViewType.Int32, NumberDataViewType.Single }));
            _transform = CreateTemplatedTransform();
        }

        public static DBScanTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new DBScanTransform(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, Host);
        }

        private DBScanTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, Host);
            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema, new string[] { _args.outCluster, _args.outScore },
                                                       new DataViewType[] { NumberDataViewType.Int32, NumberDataViewType.Single }));
            _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        public override bool CanShuffle { get { return _transform.CanShuffle; } }

        public override long? GetRowCount()
        {
            Host.AssertValue(Source, "_input");
            return Source.GetRowCount();
        }

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
                case DataKind.Single:
                    return new DBScanState(Host, this, Source, _args);
                default:
                    throw Host.Except("Features must be a vector a floats.");
            }
        }

        public class DBScanState : ADataTransform, IDataTransform
        {
            IHost _host;
            Arguments _args;
            DBScanTransform _parent;
            Dictionary<long, Tuple<int, float>> _reversedMapping;      // long: index of a point

            /// <summary>
            /// To retrieve the cluster of one observation.
            /// </summary>
            public Tuple<int, float> GetMappedIndex(int vertexId) { return _reversedMapping[vertexId]; }

            object _lock;

            public DataViewSchema Schema { get { return _parent.OutputSchema; } }

            public DBScanState(IHostEnvironment host, DBScanTransform parent, IDataView input, Arguments args)
            {
                _host = host.Register("DBScanState");
                _host.CheckValue(input, "input");
                _input = input;
                _lock = new object();
                _args = args;
                _reversedMapping = null;
                _parent = parent;
            }

            void TrainTransform()
            {
                lock (_lock)
                {
                    if (_reversedMapping != null)
                    {
                        return;
                    }

                    using (var ch = _host.Start("DBScan"))
                    {
                        var sw = Stopwatch.StartNew();
                        sw.Start();
                        var points = new List<IPointIdFloat>();
                        var index = SchemaHelper.GetColumnIndexDC(_input.Schema, _args.features);

                        // Caching data.
                        ch.Info(MessageSensitivity.None, "Caching the data.");
                        using (var cursor = _input.GetRowCursor(_input.Schema.Where(c => c.Index == index.Index)))
                        {
                            var getter = cursor.GetGetter<VBuffer<float>>(index);
                            var getterId = cursor.GetIdGetter();
                            DataViewRowId id = new DataViewRowId();

                            VBuffer<float> tmp = new VBuffer<float>();

                            for (int i = 0; cursor.MoveNext(); ++i)
                            {
                                getter(ref tmp);
                                getterId(ref id);
                                points.Add(new PointIdFloat((long)id.Low, tmp.DenseValues()));
                            }
                        }

                        // Mapping.
                        // int: index of a cluster
                        // long: index of a point
                        var mapping = new int[points.Count];
                        var mapprev = new Dictionary<long, int>();

                        float distance = _args.epsilon;
                        if (distance <= 0)
                        {
                            float mind, maxd;
                            distance = EstimateDistance(ch, points, out mind, out maxd);
                            ch.Info(MessageSensitivity.UserData, "epsilon (=Radius) was estimating on random couples of points: {0} in [{1}, {2}]", distance, mind, maxd);
                        }

                        DBScan dbscanAlgo = new DBScan(points, _args.seed);
                        // Clustering.
                        ch.Info(MessageSensitivity.UserData, "Clustering {0} points.", points.Count);

                        int nPoints = points.Count;
                        int cyclesBetweenLogging = Math.Min(1000, nPoints / 10);
                        int currentIteration = 0;
                        Action<int> progressLogger = nClusters =>
                        {
                            if (++currentIteration % cyclesBetweenLogging == 0)
                                ch.Info(MessageSensitivity.UserData, "Processing  {0}/{1} - NbClusters={2}", currentIteration, nPoints, nClusters);
                        };

                        Dictionary<long, int> results = dbscanAlgo.Cluster(
                            distance,
                            _args.minPoints,
                            seed: _args.seed,
                            onShuffle: msg => ch.Info(MessageSensitivity.UserData, msg),
                            onPointProcessing: progressLogger);

                        // Cleaning small clusters.
                        ch.Info(MessageSensitivity.UserData, "Removing clusters with less than {0} points.", _args.minPoints);
                        var finalCounts_ = results.GroupBy(c => c.Value, (key, g) => new { key = key, nb = g.Count() });
                        var finalCounts = finalCounts_.ToDictionary(c => c.key, d => d.nb);
                        results = results.Select(c => new KeyValuePair<long, int>(c.Key, finalCounts[c.Value] < _args.minPoints ? -1 : c.Value))
                                         .ToDictionary(c => c.Key, c => c.Value);

                        _reversedMapping = new Dictionary<long, Tuple<int, float>>();

                        ch.Info(MessageSensitivity.None, "Compute scores.");
                        HashSet<int> clusterIds = new HashSet<int>();
                        for (int i = 0; i < results.Count; ++i)
                        {
                            IPointIdFloat p = points[i];

                            int cluster = results[p.id];
                            mapprev[p.id] = cluster;
                            if (cluster >= 0)  // -1 is noise
                                mapping[cluster] = cluster;
                            mapping[i] = cluster;
                            if (cluster != DBScan.NOISE)
                            {
                                clusterIds.Add(cluster);
                            }
                        }
                        foreach (var p in points)
                        {
                            if (mapprev[p.id] < 0)
                                continue;
                            _reversedMapping[p.id] = new Tuple<int, float>(mapprev[p.id],
                                        dbscanAlgo.Score(p, _args.epsilon, mapprev));
                        }

                        // Adding points with no clusters.
                        foreach (var p in points)
                        {
                            if (!_reversedMapping.ContainsKey(p.id))
                                _reversedMapping[p.id] = new Tuple<int, float>(-1, float.PositiveInfinity);
                        }

                        if (_reversedMapping.Count != points.Count)
                            throw ch.Except("Mismatch between the number of points. This means some ids are not unique {0} != {1}.", _reversedMapping.Count, points.Count);

                        ch.Info(MessageSensitivity.UserData, "Found {0} clusters.", mapprev.Select(c => c.Value).Where(c => c >= 0).Distinct().Count());
                        sw.Stop();
                        ch.Info(MessageSensitivity.UserData, "'DBScan' finished in {0}.", sw.Elapsed);
                    }
                }
            }

            public float EstimateDistance(IChannel ch, List<IPointIdFloat> points,
                                           out float minDistance, out float maxDistance)
            {
                ch.Info(MessageSensitivity.UserData, "Estimating epsilon based on the data. We pick up two random random computes the average distance.");
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
                if (!stack.Where(c => !double.IsNaN(c)).Any())
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
                _host.AssertValue(_reversedMapping, "_reversedMapping");
                var cursor = _input.GetRowCursor(columnsNeeded, rand);
                return new DBScanCursor(this, cursor);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                TrainTransform();
                _host.AssertValue(_reversedMapping, "_reversedMapping");
                var cursors = _input.GetRowCursorSet(columnsNeeded, n, rand);
                return cursors.Select(c => new DBScanCursor(this, c)).ToArray();
            }

            public void Save(ModelSaveContext ctx)
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        public class DBScanCursor : DataViewRowCursor
        {
            readonly DBScanState _view;
            readonly DataViewRowCursor _inputCursor;
            readonly int _colCluster;
            readonly int _colScore;
            readonly int _colName;

            public DBScanCursor(DBScanState view, DataViewRowCursor cursor)
            {
                _view = view;
                _colCluster = view.Source.Schema.Count;
                _colScore = _colCluster + 1;
                _colName = _colScore + 1;
                _inputCursor = cursor;
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
                else if (col.Index == _view.Source.Schema.Count) // Cluster
                    return GetGetterCluster() as ValueGetter<TValue>;
                else if (col.Index == _view.Source.Schema.Count + 1) // Score
                    return GetGetterScore() as ValueGetter<TValue>;
                else
                    throw new IndexOutOfRangeException();
            }

            ValueGetter<int> GetGetterCluster()
            {
                return (ref int cluster) =>
                {
                    cluster = _view.GetMappedIndex((int)_inputCursor.Position).Item1;
                };
            }

            ValueGetter<float> GetGetterScore()
            {
                return (ref float score) =>
                {
                    score = _view.GetMappedIndex((int)_inputCursor.Position).Item2;
                };
            }
        }

        #endregion
    }
}
