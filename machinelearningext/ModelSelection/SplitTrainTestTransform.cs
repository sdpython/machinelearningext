// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.PipelineGraphTransforms;


using ArgumentAttribute = Microsoft.ML.CommandLine.ArgumentAttribute;
using ArgumentType = Microsoft.ML.CommandLine.ArgumentType;

using SplitTrainTestTransform = Scikit.ML.ModelSelection.SplitTrainTestTransform;
[assembly: LoadableClass(SplitTrainTestTransform.Summary, typeof(SplitTrainTestTransform),
    typeof(SplitTrainTestTransform.Arguments), typeof(SignatureDataTransform),
    "Split Train Test Transform", SplitTrainTestTransform.LoaderSignature, "SplitTrainTest")]

[assembly: LoadableClass(SplitTrainTestTransform.Summary, typeof(SplitTrainTestTransform),
    null, typeof(SignatureLoadDataTransform),
    "Split Train Test Transform", SplitTrainTestTransform.LoaderSignature)]


namespace Scikit.ML.ModelSelection
{
    /// <summary>
    /// The transform randomly splits a datasets into two by adding
    /// a column which tells in which part the data belongs to.
    /// The transform can save the result if requested in that case, the added
    /// column will be removed before the data is saved.
    /// </summary>
    public class SplitTrainTestTransform : TransformBase, ITaggedDataView
    {
        #region identification

        public const string LoaderSignature = "SplitTrainTestTransform";
        public const string Summary = "Splits a datasets into train / test.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SPLTTRTE",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SplitTrainTestTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the added column", ShortName = "col")]
            public string newColumn = "splitTrain0Test1";

            [Argument(ArgumentType.Multiple, HelpText = "Array of Ratios", ShortName = "a")]
            public string[] ratios = new string[] { string.Format("{0},{1}", 2.0f / 3.0f, 1.0f / 3.0f) };

            public float[] fratios;

            [Argument(ArgumentType.AtMostOnce, HelpText = "File name of the cache if stored on disk.", ShortName = "c")]
            public string cacheFile = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reuse the previous cache.", ShortName = "r")]
            public bool reuse = false;

            [Argument(ArgumentType.Multiple, HelpText = "Names of saved datasets (idv only), null for none.", ShortName = "f")]
            public string[] filename = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random seed used to split the datasets.")]
            public uint? seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random seed used to shuffle. If unspecified random state will be instead derived from the environment.", ShortName = "shuffle")]
            public int? seedShuffle;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should attempt to shuffle the source data. By default on, but can be turned off for efficiency.", ShortName = "si")]
            public bool shuffleInput = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "When shuffling the output, the number of output rows to keep in that pool. Note that shuffling of output is completely distinct from shuffling of input.", ShortName = "pool")]
            public int poolRows = 1000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads used to fill the cache.", ShortName = "nt")]
            public int? numThreads = null;

            [Argument(ArgumentType.MultipleUnique, HelpText = "To tag the split views (one tag per view).", ShortName = "tag")]
            public string[] tag = null;

            [Argument(ArgumentType.Multiple, HelpText = "Saver settings if data is saved on disk (default is binary).", ShortName = "saver",
                      SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver> saverSettings = new ScikitSubComponent<IDataSaver, SignatureDataSaver>("binary");

            public void PostProcess()
            {
                if (tag != null && tag.Length == 1 && tag[0].Contains(","))
                    tag = tag[0].Split(',');
                if (filename != null && filename.Length == 1 && filename[0].Contains(","))
                    filename = filename[0].Split(',');
                if (ratios != null && ratios.Length == 1 && ratios[0].Contains(","))
                    ratios = ratios[0].Split(',');
                try
                {
                    fratios = ratios.Select(c => float.Parse(c)).ToArray();
                }
                catch (Exception)
                {
                    throw Contracts.Except("Unable to parse '{0}'.", string.Join(",", ratios));
                }
            }
        }

        #endregion

        #region internal members / accessors

        readonly string _newColumn;
        readonly float[] _ratios;
        readonly string[] _filenames;
        readonly int? _seedShuffle;
        readonly uint? _seed;
        readonly bool _shuffleInput;
        readonly int _poolRows;
        readonly int? _numThreads;
        readonly string _cacheFile;
        readonly bool _reuse;
        readonly string[] _tags;
        IDataTransform _pipedTransform;
        readonly string _saverSettings;

        public override Schema OutputSchema { get { return _pipedTransform.Schema; } }
        public IPredictor TaggedPredictor { get { return null; } }

        #endregion

        #region public constructor / serialization / load / save

        /// <summary>
        /// Create a SplitTrainTestTransform transform.
        /// </summary>
        public SplitTrainTestTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");
            args.PostProcess();
            Host.CheckUserArg(args.poolRows >= 0, "poolRows must be > 0");
            Host.CheckUserArg(!string.IsNullOrEmpty(args.newColumn), "newColumn cannot be empty");
            Host.CheckUserArg(args.ratios != null, "ratios cannot be null");
            Host.CheckUserArg(args.ratios.Length > 1, "Number of ratios must be > 1");
            Host.CheckUserArg(args.filename == null || args.tag != null || args.filename.Length == args.ratios.Length, "filenames must be either empty either an array of the same size as ratios");
            Host.CheckUserArg(args.tag == null || args.filename != null || args.tag.Length == args.ratios.Length, "filenames must be either empty either an array of the same size as ratios");
            Host.CheckUserArg(!args.numThreads.HasValue || args.numThreads.Value > 0, "numThreads cannot be negative.");
            var sum = args.fratios.Sum();
            Host.CheckUserArg(Math.Abs(sum - 1f) < 1e-5, "Sum of ratios must be 1.");
            int col = SchemaHelper.GetColumnIndex(input.Schema, args.newColumn, true);
            Host.Check(col == -1, $"Column '{args.newColumn}' should not exist.");

            _newColumn = args.newColumn;
            _shuffleInput = args.shuffleInput;
            _poolRows = args.poolRows;
            _filenames = args.filename;
            _seed = args.seed;
            _seedShuffle = args.seedShuffle;
            _ratios = args.fratios;
            _cacheFile = args.cacheFile;
            _reuse = args.reuse;
            _tags = args.tag;

            var saveSettings = args.saverSettings as ICommandLineComponentFactory;
            Host.CheckValue(saveSettings, nameof(saveSettings));
            _saverSettings = string.Format("{0}{{{1}}}", saveSettings.Name, saveSettings.GetSettingsString());
            _saverSettings = _saverSettings.Replace("{}", "");

            var saver = ComponentCreation.CreateSaver(Host, _saverSettings);
            if (saver == null)
                throw Host.Except("Cannot parse '{0}'", _saverSettings);

            _pipedTransform = AppendToPipeline(input);
        }

        public IEnumerable<Tuple<string, ITaggedDataView>> EnumerateTaggedView(bool recursive = true)
        {
            var view = _pipedTransform as ITaggedDataView;
            if (view != null)
                return view.EnumerateTaggedView(recursive);
            return null;
        }

        public IEnumerable<Tuple<string, ITaggedDataView>> ParallelViews
        {
            get
            {
                var view = _pipedTransform as ITaggedDataView;
                if (view != null)
                    return view.ParallelViews;
                else
                    return null;
            }
        }

        public void AddRange(IEnumerable<Tuple<string, ITaggedDataView>> tagged)
        {
            // We do nothing here.
        }

        public static SplitTrainTestTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");

            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new SplitTrainTestTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_newColumn);
            ctx.Writer.WriteSingleArray(_ratios);
            ctx.Writer.Write(_cacheFile ?? string.Empty);
            ctx.Writer.Write(_reuse);
            ctx.Writer.Write(_filenames == null ? string.Empty : string.Join(";", _filenames));
            ctx.Writer.Write(_seed.HasValue ? (long)_seed : (long)-1);
            ctx.Writer.Write(_seedShuffle.HasValue ? (int)_seedShuffle : (int)-1);
            ctx.Writer.Write(_shuffleInput);
            ctx.Writer.Write(_poolRows);
            ctx.Writer.Write(_numThreads.HasValue ? _numThreads.Value : -1);
            ctx.Writer.Write(_tags == null ? "" : string.Join(",", _tags));
            ctx.Writer.Write(_saverSettings);
        }

        private SplitTrainTestTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");

            _newColumn = ctx.Reader.ReadString();
            _ratios = ctx.Reader.ReadSingleArray();
            _cacheFile = ctx.Reader.ReadString();
            if (string.IsNullOrEmpty(_cacheFile))
                _cacheFile = null;
            _reuse = ctx.Reader.ReadBoolean();
            var filenames = ctx.Reader.ReadString();
            _filenames = string.IsNullOrEmpty(filenames) ? null : filenames.Split(';');
            var seed = ctx.Reader.ReadInt64();
            _seed = seed < 0 ? (uint?)null : (uint?)seed;
            var seedShuffle = ctx.Reader.ReadInt32();
            _seedShuffle = seedShuffle < 0 ? (int?)null : (int?)seedShuffle;
            _shuffleInput = ctx.Reader.ReadBoolean();
            _poolRows = ctx.Reader.ReadInt32();
            _numThreads = ctx.Reader.ReadInt32();
            if (_numThreads < 0)
                _numThreads = null;
            string tags = ctx.Reader.ReadString();
            _tags = string.IsNullOrEmpty(tags) ? null : tags.Split(',');
            _saverSettings = ctx.Reader.ReadString();

            var saver = ComponentCreation.CreateSaver(Host, _saverSettings);
            if (saver == null)
                throw Host.Except("Cannot parse '{0}'", _saverSettings);

            _pipedTransform = AppendToPipeline(input);
        }

        #endregion

        #region IDataTransform API

        /// <summary>
        /// Shuffling is allowed.
        /// </summary>
        public override bool CanShuffle { get { return true; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public override long? GetRowCount()
        {
            Host.AssertValue(Source, "Source");
            return _pipedTransform.GetRowCount();
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        protected override RowCursor GetRowCursorCore(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValue(_pipedTransform, "_pipedTransform");
            return _pipedTransform.GetRowCursor(columnsNeeded, rand);
        }

        public override RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.AssertValue(_pipedTransform, "_pipedTransform");
            return _pipedTransform.GetRowCursorSet(columnsNeeded, n, rand);
        }

        #endregion

        #region transform own logic

        IDataTransform AppendToPipeline(IDataView input)
        {
            IDataView current = input;
            if (_shuffleInput)
            {
                var args1 = new RowShufflingTransformer.Arguments()
                {
                    ForceShuffle = false,
                    ForceShuffleSeed = _seedShuffle,
                    PoolRows = _poolRows,
                    PoolOnly = false,
                };
                current = new RowShufflingTransformer(Host, args1, current);
            }

            // We generate a random number.
            var columnName = current.Schema.GetTempColumnName();
            var args2 = new GenerateNumberTransform.Arguments()
            {
                Columns = new GenerateNumberTransform.Column[] { new GenerateNumberTransform.Column() { Name = columnName } },
                Seed = _seed ?? 42
            };
            IDataTransform currentTr = new GenerateNumberTransform(Host, args2, current);

            // We convert this random number into a part.
            var cRatios = new float[_ratios.Length];
            cRatios[0] = 0;
            for (int i = 1; i < _ratios.Length; ++i)
                cRatios[i] = cRatios[i - 1] + _ratios[i - 1];

            ValueMapper<float, int> mapper = (in float src, ref int dst) =>
            {
                for (int i = cRatios.Length - 1; i > 0; --i)
                {
                    if (src >= cRatios[i])
                    {
                        dst = i;
                        return;
                    }
                }
                dst = 0;
            };

            // Get location of columnName

            int index = SchemaHelper.GetColumnIndex(currentTr.Schema, columnName);
            var ct = currentTr.Schema[index].Type;
            var view = LambdaColumnMapper.Create(Host, "Key to part mapper", currentTr,
                                    columnName, _newColumn, ct, NumberType.I4, mapper);

            // We cache the result to avoid the pipeline to change the random number.
            var args3 = new ExtendedCacheTransform.Arguments()
            {
                inDataFrame = string.IsNullOrEmpty(_cacheFile),
                numTheads = _numThreads,
                cacheFile = _cacheFile,
                reuse = _reuse,
            };
            currentTr = new ExtendedCacheTransform(Host, args3, view);

            // Removing the temporary column.
            var finalTr = ColumnSelectingTransformer.CreateDrop(Host, currentTr, new string[] { columnName });
            var taggedViews = new List<Tuple<string, ITaggedDataView>>();

            // filenames
            if (_filenames != null || _tags != null)
            {
                int nbf = _filenames == null ? 0 : _filenames.Length;
                if (nbf > 0 && nbf != _ratios.Length)
                    throw Host.Except("Differen number of filenames and ratios.");
                int nbt = _tags == null ? 0 : _tags.Length;
                if (nbt > 0 && nbt != _ratios.Length)
                    throw Host.Except("Differen number of filenames and ratios.");
                int nb = Math.Max(nbf, nbt);

                using (var ch = Host.Start("Split the datasets and stores each part."))
                {
                    for (int i = 0; i < nb; ++i)
                    {
                        if (_filenames == null || !_filenames.Any())
                            ch.Info("Create part {0}: {1} (tag: {2})", i + 1, _ratios[i], _tags[i]);
                        else
                            ch.Info("Create part {0}: {1} (file: {2})", i + 1, _ratios[i], _filenames[i]);
                        var ar1 = new RangeFilter.Arguments() { Column = _newColumn, Min = i, Max = i, IncludeMax = true };
                        int pardId = i;
                        var filtView = LambdaFilter.Create<int>(Host, string.Format("Select part {0}", i), currentTr,
                                                                   _newColumn, NumberType.I4,
                                                                   (in int part) => { return part.Equals(pardId); });
#if (DEBUG)
                        long count = DataViewUtils.ComputeRowCount(filtView);
                        if (count == 0)
                            throw Host.Except("Part {0} is empty.", i);
#endif
                        filtView = ColumnSelectingTransformer.CreateDrop(Host, filtView, new string[] { columnName, _newColumn });

                        if (_filenames != null && _filenames.Any())
                        {
                            var saver = ComponentCreation.CreateSaver(Host, _saverSettings);
                            using (var fs0 = Host.CreateOutputFile(_filenames[i]))
                                DataSaverUtils.SaveDataView(ch, saver, filtView, fs0, true);
                        }

                        if (_tags != null && _tags.Any())
                        {
                            var ar = new TagViewTransform.Arguments { tag = _tags[i] };
                            var tar = new TagViewTransform(Host, ar, filtView);
                            taggedViews.Add(new Tuple<string, ITaggedDataView>(_tags[i], tar));
                        }
                    }
                }
            }

            if (taggedViews == null)
                return finalTr;
            else
            {
#if (DEBUG)
                for (int ii = 0; ii < taggedViews.Count; ++ii)
                {
                    long count = DataViewUtils.ComputeRowCount(taggedViews[ii].Item2);
                    if (count == 0)
                        throw Host.Except("Tagged part {0}:{1} is empty.", ii, taggedViews[ii].Item1);
                }
#endif
                foreach (var item in taggedViews)
                    item.Item2.AddRange(taggedViews);
                return new TagViewTransform(Host, finalTr, taggedViews);
            }
        }

        #endregion
    }
}
