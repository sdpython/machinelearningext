// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;
using Scikit.ML.DataManipulation;

// This indicates where to find objects in ML.net assemblies.
using ArgumentAttribute = Microsoft.ML.CommandLine.ArgumentAttribute;
using ArgumentType = Microsoft.ML.CommandLine.ArgumentType;

using DataKind = Microsoft.ML.Data.DataKind;
using IDataTransform = Microsoft.ML.Data.IDataTransform;
using IDataView = Microsoft.ML.IDataView;
using DataViewRowCursor = Microsoft.ML.DataViewRowCursor;
using DataViewSchema = Microsoft.ML.DataViewSchema;
using TransformBase = Microsoft.ML.Data.TransformBase;

using ModelLoadContext = Microsoft.ML.ModelLoadContext;
using ModelSaveContext = Microsoft.ML.ModelSaveContext;
using VersionInfo = Microsoft.ML.VersionInfo;

using DvText = Scikit.ML.PipelineHelper.DvText;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using SortInDataFrameTransform = Scikit.ML.PipelineTransforms.SortInDataFrameTransform;

[assembly: LoadableClass(SortInDataFrameTransform.Summary, typeof(SortInDataFrameTransform),
    typeof(SortInDataFrameTransform.Arguments), typeof(SignatureDataTransform),
    "Sort In DataFrame Transform", SortInDataFrameTransform.LoaderSignature, "SortInDataFrame", "SortMem", "SortDf")]

[assembly: LoadableClass(SortInDataFrameTransform.Summary, typeof(SortInDataFrameTransform),
    null, typeof(SignatureLoadDataTransform),
    "Sort In DataFrame Transform", SortInDataFrameTransform.LoaderSignature)]


namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Serialized transform which sorts a view in memory.
    /// The key must be a column. The output schema is not changed.
    /// The difficulty here is about the type of the sorting column type.
    /// We want this transform to handle many types.
    /// 
    /// In this particular case, we need a class SortInDataFrameTransform which 
    /// serializes the parameters of the transform (sorting column and sorting order)
    /// and another one which will is inserted into the pipeline. 
    /// This second one is type dependant (template).
    /// The first class creates an instance of the second based on the data view
    /// it receives as input.
    /// 
    /// The data can be cached into a dataframe and the data can be modified in the cache.
    /// This can be used to test the sensibility of the predictions.
    /// </summary>
    public class SortInDataFrameTransform : TransformBase
    {
        #region identification

        public const string LoaderSignature = "SortInDataFrameTransform";
        public const string Summary = "Sorts a data view in memory (all the data must hold in memory).";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SORTINDF",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SortInDataFrameTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Columns used to sort the view.", ShortName = "col")]
            public string sortColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Sorting order.", ShortName = "r")]
            public bool reverse = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Filling the cache with or without multithreading.", ShortName = "nt")]
            public int? numThreads = null;
        }

        #endregion

        #region internal members / accessors

        IDataTransform _transform;          // templated transform (not the serialized version)
        readonly string _sortColumn;        // sorting column
        readonly bool _reverse;             // sorting order
        readonly int? _numThreads;           // filling the cache with or without multithreading

        public override DataViewSchema OutputSchema { get { return Source.Schema; } }

        #endregion

        #region public constructor / serialization / load / save

        public SortInDataFrameTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");
            Host.CheckUserArg(!args.numThreads.HasValue || args.numThreads.Value > 0, "numThreads cannot be negative.");

            if (!string.IsNullOrEmpty(args.sortColumn))
            {
                var schema = input.Schema;
                int index = SchemaHelper.GetColumnIndex(schema, args.sortColumn);
                var type = schema[index].Type;
                Host.Check(!type.IsVector(), "sortColumn cannot be a vector.");
            }

            _reverse = args.reverse;
            _sortColumn = args.sortColumn;
            _numThreads = args.numThreads;
            _transform = CreateTemplatedTransform();
        }

        public static SortInDataFrameTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new SortInDataFrameTransform(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_sortColumn);
            ctx.Writer.Write(_reverse);
            ctx.Writer.Write(_numThreads.HasValue ? _numThreads.Value : -1);
        }

        private SortInDataFrameTransform(IHost host, ModelLoadContext ctx, IDataView input) : base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");

            _sortColumn = ctx.Reader.ReadString();
            Host.AssertValue(_sortColumn);
            var schema = input.Schema;
            int index = SchemaHelper.GetColumnIndex(schema, _sortColumn);
            var type = schema[index].Type;
            Host.Check(!type.IsVector(), "sortColumn cannot be a vector.");
            _reverse = ctx.Reader.ReadBoolean();
            _numThreads = ctx.Reader.ReadInt32();
            if (_numThreads < 0)
                _numThreads = null;
            _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        /// <summary>
        /// Shuffling would destroy the sorting effect.
        /// </summary>
        public override bool CanShuffle { get { return false; } }

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
        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return false;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.Check(string.IsNullOrEmpty(_sortColumn) || rand == null, "Random access is not allowed on sorted data. (5)");
            Host.AssertValue(_transform, "_transform");
            int sortColumn = -1;
            if (!string.IsNullOrEmpty(_sortColumn))
                sortColumn = SchemaHelper.GetColumnIndex(Source.Schema, _sortColumn);
            return _transform.GetRowCursor(SchemaHelper.ColumnsNeeded(columnsNeeded, _transform.Schema, _sortColumn), rand);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.Check(string.IsNullOrEmpty(_sortColumn) || rand == null, "Random access is not allowed on sorted data. (6)");
            Host.AssertValue(_transform, "_transform");
            if (string.IsNullOrEmpty(_sortColumn))
                return _transform.GetRowCursorSet(columnsNeeded, n, rand);
            else
            {
                int sortColumn = SchemaHelper.GetColumnIndex(Source.Schema, _sortColumn);
                return _transform.GetRowCursorSet(SchemaHelper.ColumnsNeeded(columnsNeeded, _transform.Schema, _sortColumn), n, rand);
            }
        }

        #endregion

        #region transform own logic

        /// <summary>
        /// We do not insert an instance of class SortInDataFrameTransform.
        /// We need to instantiate a templated class specific to the type of the sorting column.
        /// </summary>
        private IDataTransform CreateTemplatedTransform()
        {
            // Get the type of the sorting columns.
            var schema = Source.Schema;
            if (string.IsNullOrEmpty(_sortColumn))
                return CreateTransformNoSort();
            else
            {
                int index = SchemaHelper.GetColumnIndex(schema, _sortColumn);
                var ct = schema[index].Type;
                DataKind dk = ct.RawKind();

                // Instantiate the associated instance.
                switch (dk)
                {
                    case DataKind.Byte:
                        return CreateTransform<Byte>();
                    case DataKind.UInt16:
                        return CreateTransform<UInt16>();
                    case DataKind.UInt32:
                        return CreateTransform<UInt32>();
                    case DataKind.UInt64:
                        return CreateTransform<UInt64>();
                    case DataKind.SByte:
                        return CreateTransform<char>();
                    case DataKind.Int16:
                        return CreateTransform<short>();
                    case DataKind.Int32:
                        return CreateTransform<int>();
                    case DataKind.Int64:
                        return CreateTransform<long>();
                    case DataKind.Single:
                        return CreateTransform<float>();
                    case DataKind.Double:
                        return CreateTransform<double>();
                    case DataKind.Boolean:
                        return CreateTransform<bool>();
                    case DataKind.String:
                        return CreateTransform<DvText>();
                    case DataKind.DateTime:
                        return CreateTransform<DateTime>();
                    case DataKind.TimeSpan:
                        return CreateTransform<TimeSpan>();
                    default:
                        throw Host.Except("Unexpected raw type for a sortColumn. It cannot be an array.");
                }
            }
        }

        /// <summary>
        /// Creates the transform.
        /// </summary>
        private IDataTransform CreateTransform<TValue>()
            where TValue : IComparable<TValue>
        {
            int col = SchemaHelper.GetColumnIndex(Source.Schema, _sortColumn);
            return new SortInDataFrameState<TValue>(Host, Source, col, _reverse, _numThreads);
        }

        /// <summary>
        /// Creates the transform without any sorting.
        /// </summary>
        private IDataTransform CreateTransformNoSort()
        {
            return new SortInDataFrameState<byte>(Host, Source, -1, _reverse, _numThreads);
        }

        /// <summary>
        /// Templated transform which sorts rows based on one column.
        /// </summary>
        public class SortInDataFrameState<TValue> : IDataTransformSingle
            where TValue : IComparable<TValue>
        {
            DataFrame _autoView;
            IHost _host;
            IDataView _source;
            readonly bool _reverse;
            readonly bool _canShuffle;
            readonly int? _numThreads;
            readonly int _sortColumn;

            object _lock;

            public IDataView Source { get { return _source; } }
            public DataViewSchema Schema { get { return _source.Schema; } }

            public SortInDataFrameState(IHostEnvironment host, IDataView input, int sortColumn, bool reverse, int? numThreads)
            {
                _host = host.Register("SortInDataFrameState");
                _host.CheckValue(input, "input");
                _source = input;
                _reverse = reverse;
                _lock = new object();
                _autoView = null;
                _canShuffle = sortColumn < 0;
                _numThreads = numThreads;
                _sortColumn = sortColumn;
            }

            private DataViewSchema.Column _dc(int i)
            {
                return new DataViewSchema.Column(null, i, false, null, null);
            }

            void FillCacheIfNotFilled()
            {
                lock (_lock)
                {
                    if (!(_autoView is null))
                        return;

                    _autoView = DataFrameIO.ReadView(_source, keepVectors: true, numThreads: _numThreads);

                    if (_sortColumn >= 0)
                    {
                        var sortedPosition = new List<KeyValuePair<TValue, long>>();
                        long position = 0;
                        TValue got = default(TValue);

                        // We could use multithreading here but the cost of sorting
                        // might be higher than going through an array in memory.
                        using (var cursor = _autoView.GetRowCursor(_autoView.Schema.Where(c => c.Index == _sortColumn)))
                        {
                            var sortColumnGetter = cursor.GetGetter<TValue>(_dc(_sortColumn));
                            while (cursor.MoveNext())
                            {
                                sortColumnGetter(ref got);
                                sortedPosition.Add(new KeyValuePair<TValue, long>(got, position));
                                ++position;
                            }
                        }
                        sortedPosition.Sort(CompareTo);
                        _autoView.Order(sortedPosition.Select(c => (int)c.Value).ToArray());
                    }
                }
            }

            public bool CanShuffle { get { return _canShuffle; } }

            public long? GetRowCount()
            {
                lock (_lock)
                {
                    if (_autoView == null)
                        return null;
                }
                return _autoView.Length;
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                FillCacheIfNotFilled();
                _host.Check(_canShuffle || rand == null, "Random access is not allowed on sorted data (1).");
                _host.AssertValue(_autoView, "_autoView");
                return _autoView.GetRowCursor(columnsNeeded, rand);
            }

            public DataViewRowCursor GetRowCursorSingle(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                FillCacheIfNotFilled();
                _host.Check(_canShuffle || rand == null, "Random access is not allowed on sorted data (1).");
                _host.AssertValue(_autoView, "_autoView");
                return CursorHelper.GetRowCursorSingle(_autoView, columnsNeeded, rand);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                FillCacheIfNotFilled();
                _host.Check(_canShuffle || rand == null, "Random access is not allowed on sorted data (2).");
                _host.AssertValue(_autoView, "_autoView");
                return _autoView.GetRowCursorSet(columnsNeeded, n, rand);
            }

            private int CompareTo(KeyValuePair<TValue, long> a, KeyValuePair<TValue, long> b)
            {
                Contracts.Assert(!_canShuffle, "Sorting is not allowed as the key is not set up.");
                var r = a.Key.CompareTo(b.Key);
                return _reverse ? -r : r;
            }

            public void Save(ModelSaveContext ctx)
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        #endregion
    }
}
