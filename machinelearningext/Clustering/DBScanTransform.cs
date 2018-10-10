﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;
using Scikit.ML.NearestNeighbors;


// The following files makes the object visible to maml.
// This way, it can be added to any pipeline.
using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
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
        ISchema _schema;                // We need the schema the transform outputs.

        public override ISchema Schema { get { return _schema; } }

        #endregion

        #region public constructor / serialization / load / save

        public DBScanTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");

            if (args.epsilon < 0)
            {
                Contracts.Check(false, "Parameter epsilon must be positive or null.");
            }

            if (args.minPoints <= 0)
            {
                Contracts.Check(false, "Parameter minPoints must be positive.");
            }

            _args = args;
            _schema = new ExtendedSchema(input.Schema, new string[] { args.outCluster, args.outScore },
                                    new ColumnType[] { NumberType.I4, NumberType.R4 });
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

        public override void Save(ModelSaveContext ctx)
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
            _schema = new ExtendedSchema(input.Schema, new string[] { _args.outCluster, _args.outScore },
                                    new ColumnType[] { NumberType.I4, NumberType.R4 });
            _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        public override bool CanShuffle { get { return _transform.CanShuffle; } }

        public override long? GetRowCount(bool lazy = true)
        {
            Host.AssertValue(Source, "_input");
            return Source.GetRowCount(lazy);
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return false;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> needCol, IRandom rand = null)
        {
            Host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(needCol, rand);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            Host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            int index;
            if (!Source.Schema.TryGetColumnIndex(_args.features, out index))
                throw Host.Except("Features does not belong the input schema.");
            var type = Source.Schema.GetColumnType(index);
            if (!type.IsVector)
                throw Host.Except("Features must be a vector.");
            switch (type.AsVector.ItemType.RawKind)
            {
                case DataKind.R4:
                    return new DBScanState(Host, this, Source, _args);
                default:
                    throw Host.Except("Features must be a vector a floats.");
            }
        }

        public class DBScanState : IDataTransform
        {
            IHost _host;
            IDataView _input;
            Arguments _args;
            DBScanTransform _parent;
            Dictionary<long, Tuple<int, float>> _reversedMapping;      // long: index of a point

            /// <summary>
            /// To retrieve the cluster of one observation.
            /// </summary>
            public Tuple<int, float> GetMappedIndex(int vertexId) { return _reversedMapping[vertexId]; }

            object _lock;

            public IDataView Source { get { return _input; } }
            public ISchema Schema { get { return _parent.Schema; } }

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

                        int index;
                        if (!_input.Schema.TryGetColumnIndex(_args.features, out index))
                        {
                            ch.Except("Unable to find column '{0}'", _args.features);
                        }

                        // Caching data.
                        ch.Info("Caching the data.");
                        using (var cursor = _input.GetRowCursor(i => i == index))
                        {
                            var getter = cursor.GetGetter<VBuffer<float>>(index);
                            var getterId = cursor.GetIdGetter();
                            UInt128 id = new UInt128();

                            VBuffer<float> tmp = new VBuffer<float>();

                            for (int i = 0; cursor.MoveNext(); ++i)
                            {
                                getter(ref tmp);
                                getterId(ref id);
                                if (id > long.MaxValue)
                                {
                                    ch.Except("An id is outside the range for long {0}", id);
                                }
                                points.Add(new PointIdFloat((long)id, tmp.DenseValues()));
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
                            ch.Info("epsilon (=Radius) was estimating on random couples of points: {0} in [{1}, {2}]", distance, mind, maxd);
                        }

                        DBScan dbscanAlgo = new DBScan(points, _args.seed);
                        // Clustering.
                        ch.Info("Clustering {0} points.", points.Count);

                        int nPoints = points.Count;
                        int cyclesBetweenLogging = Math.Min(1000, nPoints / 10);
                        int currentIteration = 0;
                        Action<int> progressLogger = nClusters =>
                        {
                            if (++currentIteration % cyclesBetweenLogging == 0)
                            {
                                ch.Info("Processing  {0}/{1} - NbClusters={2}", currentIteration, nPoints, nClusters);
                            }
                        };

                        Dictionary<long, int> results = dbscanAlgo.Cluster(
                            distance,
                            _args.minPoints,
                            seed: _args.seed,
                            onShuffle: msg => ch.Info(msg),
                            onPointProcessing: progressLogger);

                        // Cleaning small clusters.
                        ch.Info("Removing clusters with less than {0} points.", _args.minPoints);
                        var finalCounts_ = results.GroupBy(c => c.Value, (key, g) => new { key = key, nb = g.Count() });
                        var finalCounts = finalCounts_.ToDictionary(c => c.key, d => d.nb);
                        results = results.Select(c => new KeyValuePair<long, int>(c.Key, finalCounts[c.Value] < _args.minPoints ? -1 : c.Value))
                                         .ToDictionary(c => c.Key, c => c.Value);

                        _reversedMapping = new Dictionary<long, Tuple<int, float>>();

                        ch.Info("Compute scores.");
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

                        ch.Info("Found {0} clusters.", mapprev.Select(c => c.Value).Where(c => c >= 0).Distinct().Count());
                        sw.Stop();
                        ch.Info("'DBScan' finished in {0}.", sw.Elapsed);
                    }
                }
            }

            public float EstimateDistance(IChannel ch, List<IPointIdFloat> points,
                                           out float minDistance, out float maxDistance)
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
                if (!stack.Where(c => !double.IsNaN(c)).Any())
                    throw ch.Except("All distances are NaN. Check your datasets.");
                minDistance = stack.Where(c => !float.IsNaN(c)).First();
                maxDistance = stack.Last();
                return stack[Math.Min(stack.Count - 1, Math.Max(stack.Count / 20, 2))];
            }

            public bool CanShuffle { get { return true; } }

            public long? GetRowCount(bool lazy = true)
            {
                return _input.GetRowCount(lazy);
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                TrainTransform();
                _host.AssertValue(_reversedMapping, "_reversedMapping");
                var cursor = _input.GetRowCursor(predicate, rand);
                return new DBScanCursor(this, cursor);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                TrainTransform();
                _host.AssertValue(_reversedMapping, "_reversedMapping");
                var cursors = _input.GetRowCursorSet(out consolidator, predicate, n, rand);
                return cursors.Select(c => new DBScanCursor(this, c)).ToArray();
            }

            public void Save(ModelSaveContext ctx)
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        public class DBScanCursor : IRowCursor
        {
            readonly DBScanState _view;
            readonly IRowCursor _inputCursor;
            readonly int _colCluster;
            readonly int _colScore;
            readonly int _colName;

            public DBScanCursor(DBScanState view, IRowCursor cursor)
            {
                _view = view;
                _colCluster = view.Source.Schema.ColumnCount;
                _colScore = _colCluster + 1;
                _colName = _colScore + 1;
                _inputCursor = cursor;
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool IsColumnActive(int col)
            {
                if (col < _inputCursor.Schema.ColumnCount)
                    return _inputCursor.IsColumnActive(col);
                return true;
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
                return _inputCursor.MoveMany(count);
            }

            public bool MoveNext()
            {
                return _inputCursor.MoveNext();
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (col < _view.Source.Schema.ColumnCount)
                {
                    return _inputCursor.GetGetter<TValue>(col);
                }
                else if (col == _view.Source.Schema.ColumnCount) // Cluster
                {
                    return GetGetterCluster() as ValueGetter<TValue>;
                }
                else if (col == _view.Source.Schema.ColumnCount + 1) // Score
                {
                    return GetGetterScore() as ValueGetter<TValue>;
                }
                else
                {
                    throw new IndexOutOfRangeException();
                }
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
