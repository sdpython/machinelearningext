// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using TagViewTransform = Scikit.ML.PipelineGraphTransforms.TagViewTransform;


[assembly: LoadableClass(TagViewTransform.Summary, typeof(TagViewTransform),
    typeof(TagViewTransform.Arguments), typeof(SignatureDataTransform),
    "Tag View", TagViewTransform.LoaderSignature, "Tag")]

[assembly: LoadableClass(TagViewTransform.Summary, typeof(TagViewTransform),
    null, typeof(SignatureLoadDataTransform),
    "Tag View", TagViewTransform.LoaderSignature, "Tag")]


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Tags the source data view in order to reuse it later in the pipeline.
    /// </summary>
    public class TagViewTransform : IDataTransform, ITaggedDataView
    {
        #region identification

        public const string LoaderSignature = "TagViewTransform";  // Not more than 24 letters.
        public const string Summary = "Tags the source data view in order to reuse it later in the pipeline.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TAGSVIEW",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TagViewTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Tag for the source of this transform.", ShortName = "t")]
            public string tag;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(tag);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                tag = ctx.Reader.ReadString();
            }
        }

        #endregion

        #region internal members / accessors

        readonly Arguments _args;
        readonly IHost _host;
        readonly IDataView _source;
        readonly List<Tuple<string, ITaggedDataView>> _parallelViews;
        readonly IPredictor _taggedPredictor;

        #endregion

        #region API DataTransform

        public IDataView Source { get { return _source; } }
        public IPredictor TaggedPredictor { get { return _taggedPredictor; } }

        /// <summary>
        /// Create a TagViewTransform transform.
        /// </summary>
        public TagViewTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var tagged = input as ITaggedDataView;
            if (tagged != null)
                throw env.Except("The input view is already tagged. Don't tag again with '{0}'.", args.tag);
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, "args");
            _args = args;
            _source = input;
            _host.CheckValue(args.tag, "Tag cannot be empty.");
            if (TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.tag).Any())
                throw _host.Except("Tag '{0}' is already used.", args.tag);
            _parallelViews = new List<Tuple<string, ITaggedDataView>>();
            _parallelViews.Add(new Tuple<string, ITaggedDataView>(_args.tag, this));
            _taggedPredictor = null;
        }

        public TagViewTransform(IHostEnvironment env, Arguments args, IDataView input, IPredictor predictor)
        {
            Contracts.CheckValue(env, "env");
            var tagged = input as ITaggedDataView;
            if (tagged != null)
                throw env.Except("The input view is already tagged. Don't tag again with '{0}'.", args.tag);
            if (predictor == null)
                throw env.Except("Predictor is null, it cannot be tagged with '{0}'.", args.tag);
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, "args");
            _args = args;
            _source = input;
            _host.CheckValue(args.tag, "Tag cannot be empty.");
            if (TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.tag).Any())
                throw _host.Except("Tag '{0}' is already used.", args.tag);
            _parallelViews = new List<Tuple<string, ITaggedDataView>>();
            _parallelViews.Add(new Tuple<string, ITaggedDataView>(_args.tag, this));
            _taggedPredictor = predictor;
        }

        public TagViewTransform(IHostEnvironment env, IDataView input, IEnumerable<Tuple<string, ITaggedDataView>> addition)
        {
            _host = env.Register(RegistrationName);
            _args = new Arguments { tag = "" };
            _source = input;
            _parallelViews = TagHelper.Reconcile(addition);
            _taggedPredictor = null;
        }

        public static TagViewTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var tagged = input as ITaggedDataView;
            if (tagged != null)
                throw env.Except("The input view is already tagged. Don't tag again.");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new TagViewTransform(h, ctx, input));
        }

        /// <summary>
        /// Serializes the transform. 
        /// </summary>
        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        /// <summary>
        /// Reading serialized transform.
        /// </summary>
        private TagViewTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            _host = host;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _source = input;
            _args = new Arguments();
            _args.Read(ctx, _host);
            _host.CheckValue(_args.tag, "Tag cannot be empty.");
            if (TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == _args.tag).Any())
                throw _host.Except("Tag '{0}' is already used.", _args.tag);
            _parallelViews = new List<Tuple<string, ITaggedDataView>>();
            _parallelViews.Add(new Tuple<string, ITaggedDataView>(_args.tag, this));
        }

        public DataViewSchema Schema { get { return _source.Schema; } }
        public bool CanShuffle { get { return _source.CanShuffle; } }
        public long? GetRowCount()
        {
            _host.AssertValue(_source, "_input");
            return _source.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValue(_source, "_source");
            return _source.GetRowCursor(columnsNeeded, rand);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            _host.AssertValue(_source, "_source");
            return _source.GetRowCursorSet(columnsNeeded, n, rand);
        }

        #endregion

        #region tagged interface

        public IEnumerable<Tuple<string, ITaggedDataView>> EnumerateTaggedView(bool recursive = true)
        {
            return TagHelper.EnumerateTaggedView(recursive, this);
        }

        public IEnumerable<Tuple<string, ITaggedDataView>> ParallelViews { get { return _parallelViews; } }

        public void AddRange(IEnumerable<Tuple<string, ITaggedDataView>> tagged)
        {
            _parallelViews.AddRange(tagged);
        }

        #endregion
    }
}
