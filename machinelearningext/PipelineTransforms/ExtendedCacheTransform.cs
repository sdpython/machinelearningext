﻿// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Scikit.ML.PipelineHelper;


// This indicates where to find objects in ML.net assemblies.
using ComponentCreation = Microsoft.ML.Runtime.Api.ComponentCreation;
using DataSaverUtils = Microsoft.ML.Runtime.Data.DataSaverUtils;
using IDataTransform = Microsoft.ML.Runtime.Data.IDataTransform;
using IDataSaver = Microsoft.ML.Runtime.Data.IDataSaver;
using IDataView = Microsoft.ML.Runtime.Data.IDataView;
using RowCursor = Microsoft.ML.Runtime.Data.RowCursor;
using IRowCursorConsolidator = Microsoft.ML.Runtime.Data.IRowCursorConsolidator;
using Schema = Microsoft.ML.Data.Schema;
using TransformBase = Microsoft.ML.Runtime.Data.TransformBase;
using MultiFileSource = Microsoft.ML.Runtime.Data.MultiFileSource;
using CacheDataView = Microsoft.ML.Data.CacheDataView;
using SignatureDataSaver = Microsoft.ML.Runtime.Data.SignatureDataSaver;
using ModelLoadContext = Microsoft.ML.Runtime.Model.ModelLoadContext;
using ModelSaveContext = Microsoft.ML.Runtime.Model.ModelSaveContext;
using VersionInfo = Microsoft.ML.Runtime.Model.VersionInfo;
using ICanSaveOnnx = Microsoft.ML.Runtime.Model.Onnx.ICanSaveOnnx;
using OnnxContext = Microsoft.ML.Runtime.Model.Onnx.OnnxContext;
using SchemaHelper = Scikit.ML.PipelineHelper.SchemaHelper;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using ExtendedCacheTransform = Scikit.ML.PipelineTransforms.ExtendedCacheTransform;

[assembly: LoadableClass(ExtendedCacheTransform.Summary, typeof(ExtendedCacheTransform),
    typeof(ExtendedCacheTransform.Arguments), typeof(SignatureDataTransform),
    "Extended Cache Transform", ExtendedCacheTransform.LoaderSignature, "CacheDF")]

[assembly: LoadableClass(ExtendedCacheTransform.Summary, typeof(ExtendedCacheTransform),
    null, typeof(SignatureLoadDataTransform),
    "Extended Cache Transform", ExtendedCacheTransform.LoaderSignature, "CacheDF")]


namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Cache data in memory or on disk. If async is true, the cache is asynchronous
    /// (different thread) and relies for some scenarios on class DataFrame.
    /// This transform can be used to overwrite some values in the middle of the pipeline
    /// while doing prediction.
    /// </summary>
    public class ExtendedCacheTransform : TransformBase, ICanSaveOnnx
    {
        #region identification

        public const string LoaderSignature = "ExtendedCacheTransform";
        public const string Summary = "Caches data in memory or on disk. If async is true, the cache is asynchronous (different thread). " +
                                      "The transform can be used to overwrite some values in the middle of a pipeline while doing predictions.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXTCACHT",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ExtendedCacheTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Cache in dataframe (in memory).", ShortName = "df")]
            public bool inDataFrame = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Asynchronous (use CacheView).", ShortName = "as")]
            public bool async = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads used to fill the cache.", ShortName = "nt")]
            public int? numTheads = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "File name of the cache if stored on disk.", ShortName = "f")]
            public string cacheFile = "tempCacheFile.idv";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reuse the previous cache.", ShortName = "r")]
            public bool reuse = false;

            [Argument(ArgumentType.Multiple, HelpText = "Saver settings if data is saved on disk (default is binary).", ShortName = "saver",
                      SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver> saverSettings = new ScikitSubComponent<IDataSaver, SignatureDataSaver>("binary");
        }

        #endregion

        #region internal members / accessors

        readonly bool _inDataFrame;
        readonly string _cacheFile;
        readonly bool _reuse;
        readonly bool _async;
        readonly int? _numThreads;
        readonly string _saverSettings;
        readonly IDataTransform _pipedTransform;

        public override Schema OutputSchema { get { return Source.Schema; } }

        #endregion

        #region public constructor / serialization / load / save

        public ExtendedCacheTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");
            Host.CheckUserArg(args.inDataFrame || !string.IsNullOrEmpty(args.cacheFile), "cacheFile cannot be empty if inDataFrame is false.");
            Host.CheckUserArg(!args.async || args.inDataFrame, "inDataFrame must be true if async is true.");
            Host.CheckUserArg(!args.numTheads.HasValue || args.numTheads > 0, "numThread must be > 0 if specified.");
            var saverSettings = args.saverSettings as ICommandLineComponentFactory;
            Host.CheckValue(saverSettings, nameof(saverSettings));
            _saverSettings = string.Format("{0}{{{1}}}", saverSettings.Name, saverSettings.GetSettingsString());
            _saverSettings = _saverSettings.Replace("{}", "");
            if (!_saverSettings.ToLower().StartsWith("binary"))
                throw env.ExceptNotSupp("Only binary format is supported.");
            _inDataFrame = args.inDataFrame;
            _cacheFile = args.cacheFile;
            _reuse = args.reuse;
            _async = args.async;
            _numThreads = args.numTheads;

            var saver = ComponentCreation.CreateSaver(Host, _saverSettings);
            if (saver == null)
                throw Host.Except("Cannot parse '{0}'", _saverSettings);

            _pipedTransform = CreatePipeline(env, input);
        }

        public static ExtendedCacheTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ExtendedCacheTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_inDataFrame);
            ctx.Writer.Write(_async);
            ctx.Writer.Write(_numThreads.HasValue ? _numThreads.Value : -1);
            ctx.Writer.Write(_saverSettings);
            if (!_inDataFrame)
            {
                ctx.Writer.Write(_cacheFile);
                ctx.Writer.Write(_reuse);
            }
        }

        private ExtendedCacheTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");

            _inDataFrame = ctx.Reader.ReadBoolean();
            _async = ctx.Reader.ReadBoolean();
            _numThreads = ctx.Reader.ReadInt32();
            host.Check(_numThreads > -2, "_numThreads");
            if (_numThreads < 0)
                _numThreads = null;
            _saverSettings = ctx.Reader.ReadString();
            if (_inDataFrame)
            {
                _cacheFile = null;
                _reuse = false;
            }
            else
            {
                _cacheFile = ctx.Reader.ReadString();
                _reuse = ctx.Reader.ReadBoolean();
                host.CheckValue(_cacheFile, "_cacheFile");
            }

            var saver = ComponentCreation.CreateSaver(Host, _saverSettings);
            if (saver == null)
                throw Host.Except("Cannot parse '{0}'", _saverSettings);

            _pipedTransform = CreatePipeline(host, input);
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

        protected override RowCursor GetRowCursorCore(Func<int, bool> needCol, Random rand = null)
        {
            Host.AssertValue(_pipedTransform, "_pipedTransform");
            return _pipedTransform.GetRowCursor(needCol, rand);
        }

        public override RowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, Random rand = null)
        {
            Host.AssertValue(_pipedTransform, "_pipedTransform");
            return _pipedTransform.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        #endregion

        #region transform own logic

        /// <summary>
        /// Creation of the pipeline knowing parameters _inDataFrame, _cacheFile, _reuse.
        /// </summary>
        protected IDataTransform CreatePipeline(IHostEnvironment env, IDataView input)
        {
            if (_inDataFrame)
            {
                if (_async)
                {
                    var view = new CacheDataView(env, input, null);
                    var tr = new PassThroughTransform(env, new PassThroughTransform.Arguments(), view);
                    return tr;
                }
                else
                {
                    var args = new SortInDataFrameTransform.Arguments() { numThreads = _numThreads, sortColumn = null };
                    var tr = new SortInDataFrameTransform(env, args, input);
                    return tr;
                }
            }
            else
            {
                string nt = _numThreads > 0 ? string.Format("{{t={0}}}", _numThreads) : string.Empty;
                using (var ch = Host.Start("Caching data..."))
                {
                    if (_reuse && File.Exists(_cacheFile))
                        ch.Info(MessageSensitivity.UserData, "Reusing cache '{0}'", _cacheFile);
                    else
                    {
                        ch.Info(MessageSensitivity.UserData, "Building cache '{0}'", _cacheFile);
                        var saver = ComponentCreation.CreateSaver(env, _saverSettings);
                        using (var fs0 = Host.CreateOutputFile(_cacheFile))
                            DataSaverUtils.SaveDataView(ch, saver, input, fs0, true);
                    }
                }
                var loader = ComponentCreation.CreateLoader(env, string.Format("binary{{{0}}}", nt),
                                                            new MultiFileSource(_cacheFile));
                SchemaHelper.CheckSchema(Host, input.Schema, loader.Schema);
                var copy = ComponentCreation.CreateTransform(env, "skip{s=0}", loader);
                return copy;
            }
        }

        #endregion

        #region onnx

        public bool CanSaveOnnx(OnnxContext ctx)
        {
            return true;
        }

        public void SaveAsOnnx(OnnxContext ctx)
        {
            // Nothing to do.
        }

        #endregion
    }
}
