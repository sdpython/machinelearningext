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
using OpticsTransform = Scikit.ML.Clustering.OpticsTransform;


[assembly: LoadableClass(OpticsTransform.Summary, typeof(OpticsTransform),
    typeof(OpticsTransform.Arguments), typeof(SignatureDataTransform),
    OpticsTransform.LoaderSignature, "OPTICS")]

[assembly: LoadableClass(OpticsTransform.Summary, typeof(OpticsTransform),
    null, typeof(SignatureLoadDataTransform), "OPTICS Transform", "OPTICS", OpticsTransform.LoaderSignature)]


namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Transform which applies the Optics clustering algorithm.
    /// </summary>
    public class OpticsTransform : TransformBase
    {
        #region identification

        public const string LoaderSignature = "OpticsTransform";
        public const string Summary = "Clusters data using OPTICS algorithm.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "OPTICSME",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OpticsTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Column which contains the features.", ShortName = "col")]
            public string features;

            [Argument(ArgumentType.Multiple, HelpText = "Radius of the sample areas. If null, the transform will give it a default value based on the data.", ShortName = "eps")]
            public string epsilons = null;

            public float[] epsilonsDouble = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of points in the sample area to be considered a cluster.", ShortName = "mps")]
            public int minPoints = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Cluster results.", ShortName = "outc")]
            public string outCluster = "ClusterId";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scores", ShortName = "outs")]
            public string outScore = "Score";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the number generators.", ShortName = "s")]
            public int? seed = 42;

            public int newColumnsNumber;

            public void PostProcess()
            {
                if (epsilonsDouble == null && !string.IsNullOrEmpty(epsilons))
                    epsilonsDouble = epsilons.Split(',').Select(c => float.Parse(c)).ToArray();
            }

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(features);
                ctx.Writer.Write(epsilonsDouble == null ? "" : String.Join("'", epsilonsDouble.Select(eps => eps.ToString())));
                ctx.Writer.Write(newColumnsNumber);
                ctx.Writer.Write(minPoints);
                ctx.Writer.Write(outCluster);
                ctx.Writer.Write(outScore);
                ctx.Writer.Write(seed ?? -1);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                features = ctx.Reader.ReadString();
                epsilons = ctx.Reader.ReadString();
                epsilonsDouble = string.IsNullOrEmpty(epsilons) ? null : epsilons.Split(',').Select(eps => float.Parse(eps)).ToArray();
                newColumnsNumber = ctx.Reader.ReadInt32();
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

        public OpticsTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");
            args.PostProcess();

            if (args.epsilonsDouble != null && args.epsilonsDouble.Any(eps => eps < 0))
            {
                Contracts.Check(false, "Parameter epsilon, if passed, must be positive.");
            }

            if (args.minPoints <= 0)
            {
                Contracts.Check(false, "Parameter minPoints must be positive.");
            }

            _args = args;
            string[] newColumnNames = null;
            DataViewType[] newColumnTypes = null;

            int epsilonsCount = args.epsilonsDouble == null ? 0 : args.epsilonsDouble.Count();
            if (args.epsilonsDouble == null || epsilonsCount <= 1)
            {
                newColumnNames = new string[] { args.outCluster, args.outScore };
                newColumnTypes = new DataViewType[] { NumberDataViewType.Int32, NumberDataViewType.Single };
            }
            else
            {
                //Adding 2 columns, ClusterId + Score, for each value of epsilon 
                newColumnNames = new string[2 * epsilonsCount];
                newColumnTypes = new DataViewType[2 * epsilonsCount];

                for (int i = 0; i < epsilonsCount; i += 1)
                {
                    newColumnNames[2 * i] = String.Format("{0}_{1}", args.outCluster, i);
                    newColumnNames[2 * i + 1] = String.Format("{0}_{1}", args.outScore, i); ;
                    newColumnTypes[2 * i] = NumberDataViewType.Int32;
                    newColumnTypes[2 * i + 1] = NumberDataViewType.Single;
                }
            }

            args.newColumnsNumber = newColumnNames.Count() / 2;
            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema, newColumnNames, newColumnTypes));
            _transform = CreateTemplatedTransform();
        }

        public static OpticsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new OpticsTransform(h, ctx, input));
        }

        protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, Host);
        }

        private OpticsTransform(IHost host, ModelLoadContext ctx, IDataView input) :
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

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public override long? GetRowCount()
        {
            Host.AssertValue(Source, "Source");
            return Source.GetRowCount();
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected override bool? ShouldUseParallelCursors(Func<int, bool> columnsNeeded)
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
                    return new OpticsState(Host, this, Source, _args);
                default:
                    throw Host.Except("Features must be a vector a floats.");
            }
        }

        public class OpticsState : IDataTransform
        {
            IHost _host;
            IDataView _input;
            Arguments _args;
            OpticsTransform _parent;
            IList<Dictionary<int, ClusteringResult>> _Results;  // ClusteringResult: cluster of a point
            IList<Dictionary<long, int>> _reversedMapping;      // long: cluster of a point (id)

            /// <summary>
            /// Array of cluster.
            /// </summary>
            public IList<Dictionary<int, ClusteringResult>> ClusteringResults { get { return _Results; } }

            /// <summary>
            /// To retrieve the cluster of one observation.
            /// </summary>
            public int GetMappedIndex(int runId, int vertexId) { return _reversedMapping[runId][vertexId]; }

            object _lock;

            public IDataView Source { get { return _input; } }
            public DataViewSchema Schema { get { return _parent.OutputSchema; } }

            public OpticsState(IHostEnvironment host, OpticsTransform parent, IDataView input, Arguments args)
            {
                _host = host.Register("OpticsState");
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
                    {
                        return;
                    }

                    using (var ch = _host.Start("Optics"))
                    {
                        var sw = Stopwatch.StartNew();
                        sw.Start();
                        var points = new List<IPointIdFloat>();
                        int index = SchemaHelper.GetColumnIndex(_input.Schema, _args.features);

                        // Caching data.
                        ch.Info(MessageSensitivity.None, "Caching the data.");
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
                                points.Add(new PointIdFloat((long)id.Low, tmp.DenseValues().Select(c => (float)c)));
                            }
                        }

                        // Mapping.
                        // int: index of a cluster
                        // long: index of a point
                        var mapping = new int[points.Count];
                        var mapprev = new Dictionary<long, int>();

                        float[] distances = null;
                        if (_args.epsilons == null || _args.epsilons.Count() == 0)
                        {
                            float mind, maxd;
                            distances = new[] { EstimateDistance(ch, points, out mind, out maxd) };
                            ch.Info(MessageSensitivity.UserData, "epsilon (=Radius) was estimating on random couples of points: {0} in [{1}, {2}]", distances.First(), mind, maxd);
                        }
                        else
                            distances = _args.epsilonsDouble;

                        var maxEpsilon = distances.Max();
                        _Results = new List<Dictionary<int, ClusteringResult>>();
                        _reversedMapping = new List<Dictionary<long, int>>();

                        Optics opticsAlgo = new Optics(points, _args.seed);
                        //Ordering
                        ch.Info(MessageSensitivity.UserData, "Generating OPTICS ordering for {0} points.", points.Count);
                        int nPoints = points.Count;
                        int cyclesBetweenLogging = Math.Min(1000, nPoints / 10);
                        int currentIteration = 0;
                        Action progressLogger = () =>
                        {
                            if (++currentIteration % cyclesBetweenLogging == 0)
                                ch.Info(MessageSensitivity.UserData, "Processing {0}/{1}", currentIteration, nPoints);
                        };

                        OpticsOrdering opticsOrdering = opticsAlgo.Ordering(
                            maxEpsilon,
                            _args.minPoints,
                            seed: _args.seed,
                            onShuffle: msg => ch.Info(MessageSensitivity.UserData, msg),
                            onPointProcessing: progressLogger);

                        // Clustering.
                        foreach (var epsilon in distances)
                        {
                            ch.Info(MessageSensitivity.UserData, "Clustering {0} points using epsilon={1}.", points.Count, epsilon);
                            Dictionary<long, int> results = opticsOrdering.Cluster(epsilon);

                            HashSet<int> clusterIds = new HashSet<int>();

                            for (int i = 0; i < results.Count; ++i)
                            {
                                var p = points[i];
                                int cluster = results[p.id];
                                mapprev[p.id] = cluster;
                                mapping[i] = cluster;
                                if (cluster != DBScan.NOISE)
                                    clusterIds.Add(cluster);
                            }

                            _reversedMapping.Add(mapprev);

                            // Cleaning small clusters.
                            ch.Info(MessageSensitivity.UserData, "Removing clusters with less than {0} points.", _args.minPoints);
                            var finalCounts_ = results.GroupBy(c => c.Value, (key, g) => new { key = key, nb = g.Count() });
                            var finalCounts = finalCounts_.ToDictionary(c => c.key, d => d.nb);
                            results = results.Select(c => new KeyValuePair<long, int>(c.Key, finalCounts[c.Value] < _args.minPoints ? -1 : c.Value))
                                             .ToDictionary(c => c.Key, c => c.Value);

                            // Cleaning.
                            ch.Info(MessageSensitivity.None, "Cleaning.");
                            // We replace by the original labels.
                            var runResults = new Dictionary<int, ClusteringResult>();
                            for (int i = 0; i < results.Count; ++i)
                            {
                                runResults[i] = new ClusteringResult()
                                {
                                    cl = results[i] != DBScan.NOISE ? results[i] : -1,
                                    score = results[i] != DBScan.NOISE ? 1f : 0f
                                };
                            }

                            _Results.Add(runResults);
                            ch.Info(MessageSensitivity.UserData, "Found {0} clusters.", clusterIds.Count);
                        }
                        sw.Stop();
                        ch.Info(MessageSensitivity.UserData, "'Optics' finished in {0}.", sw.Elapsed);
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
                return new OpticsCursor(this, cursor, _args.newColumnsNumber);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                TrainTransform();
                _host.AssertValue(_Results, "_Results");
                var cursors = _input.GetRowCursorSet(columnsNeeded, n, rand);
                return cursors.Select(c => new OpticsCursor(this, c, _args.newColumnsNumber)).ToArray();
            }

            public void Save(ModelSaveContext ctx)
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        public class OpticsCursor : DataViewRowCursor
        {
            readonly OpticsState _view;
            readonly DataViewRowCursor _inputCursor;
            readonly int _newColNumber;

            public OpticsCursor(OpticsState view, DataViewRowCursor cursor, int newColNumber)
            {
                _view = view;
                _inputCursor = cursor;
                _newColNumber = newColNumber;
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
                {
                    return _inputCursor.GetGetter<TValue>(col);
                }
                else                                                // New columns
                {
                    int runIndex = (col - _view.Source.Schema.Count) / 2;
                    if (runIndex < _newColNumber)
                    {
                        int colIndex = (col - _view.Source.Schema.Count) % 2;

                        if (colIndex == 0)                          // Cluster Column
                        {
                            return GetGetterCluster(runIndex) as ValueGetter<TValue>;
                        }
                        else
                        {                                           // Score
                            return GetGetterScore(runIndex) as ValueGetter<TValue>;
                        }
                    }
                    else
                    {
                        throw new IndexOutOfRangeException();
                    }
                }
            }

            ValueGetter<int> GetGetterCluster(int runIndex)
            {
                return (ref int cluster) =>
                {
                    cluster = _view.GetMappedIndex(runIndex, (int)_inputCursor.Position);
                };
            }

            ValueGetter<float> GetGetterScore(int runIndex)
            {
                return (ref float score) =>
                {
                    int cl = _view.GetMappedIndex(runIndex, (int)_inputCursor.Position);
                    if (cl != -1)
                        score = _view.ClusteringResults[runIndex][cl].score;
                    else
                        score = 0f;
                };
            }
        }

        #endregion
    }
}
