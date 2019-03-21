// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data.IO;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using SelectTaggedViewTransform = Scikit.ML.PipelineGraphTransforms.SelectTaggedViewTransform;

[assembly: LoadableClass(SelectTaggedViewTransform.Summary, typeof(SelectTaggedViewTransform),
    typeof(SelectTaggedViewTransform.Arguments), typeof(SignatureDataTransform),
    "Select Tagged View", SelectTaggedViewTransform.LoaderSignature,
    "SelectTaggedViewTransform", "SelectTag", "SelTag")]

[assembly: LoadableClass(SelectTaggedViewTransform.Summary, typeof(SelectTaggedViewTransform),
    null, typeof(SignatureLoadDataTransform),
    "Select Tagged View", SelectTaggedViewTransform.LoaderSignature,
    "SelectTaggedViewTransform", "SelectTag", "SelTag")]


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Tag the source data view in order to reuse it later in the pipeline and select another one.
    /// </summary>
    public class SelectTaggedViewTransform : AbstractSimpleTransformTemplate
    {
        public const string LoaderSignature = "SelTaggedViewTransform";  // Not more than 24 letters.
        internal const string Summary = "Tag the source data view in order to reuse it later in the pipeline and select another one.";
        public const string RegistrationName = "SelectTaggedViewTransform";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SELTAGVI",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SelectTaggedViewTransform).Assembly.FullName);
        }

        public new class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Tag for the source of this transform.", ShortName = "t")]
            public string tag;

            [Argument(ArgumentType.Required, HelpText = "Tag of the view to select.", ShortName = "s")]
            public string selectTag;

            [Argument(ArgumentType.AtMostOnce, HelpText = "In that case, the selected view is an idv file or a text file.", ShortName = "f")]
            public string filename;

            [Argument(ArgumentType.Multiple, HelpText = "Loader settings if data is loaded from disk (default is binary).", 
                ShortName = "loader", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<ILegacyDataLoader> loaderSettings = 
                new ScikitSubComponent<ILegacyDataLoader, SignatureDataLoader>("binary");

            public void Read(ModelLoadContext ctx, IHost host)
            {
                tag = ctx.Reader.ReadString();
                selectTag = ctx.Reader.ReadString();
                filename = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(filename))
                    filename = null;
                var sloader = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(sloader))
                    sloader = "binary";
                loaderSettings = new ScikitSubComponent<ILegacyDataLoader, SignatureDataLoader>(sloader);
            }

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(tag);
                ctx.Writer.Write(selectTag);
                ctx.Writer.Write(string.IsNullOrEmpty(filename) ? "" : filename);

                var loadSettings = loaderSettings as ICommandLineComponentFactory;
                Contracts.CheckValue(loadSettings, nameof(loaderSettings));

                var sloaderSettings = string.Format("{0}{{{1}}}", loadSettings.Name, loadSettings.GetSettingsString());
                sloaderSettings = sloaderSettings.Replace("{}", "");
                ctx.Writer.Write(sloaderSettings);
            }
        }

        readonly Arguments _args;

        public SelectTaggedViewTransform(IHostEnvironment env, Arguments args, IDataView input) :
            base(env, input, RegistrationName)
        {
            _host.CheckValue(args, "args");
            _host.CheckValue(args.tag, "tag");
            _host.CheckValue(args.selectTag, "selectTag");
            _args = args;
            _sourcePipe = Create(_host, args, input, out _sourceCtx);
        }

        protected override void DelayedInitialisationLockFree()
        {
            // nothing 
        }

        protected override void SaveModel(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        public SelectTaggedViewTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, ctx, input, RegistrationName)
        {
            _args = new Arguments();
            _args.Read(ctx, _host);
            _sourcePipe = Create(_host, _args, input, out _sourceCtx);
        }

        public static SelectTaggedViewTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new SelectTaggedViewTransform(h, ctx, input));
        }

        static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input, out IDataView sourceCtx)
        {
            sourceCtx = input;
            env.CheckValue(args.tag, "Tag cannot be empty.");
            if (TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.tag).Any())
                throw env.Except("Tag '{0}' is already used.", args.tag);
            env.CheckValue(args.selectTag, "Selected tag cannot be empty.");

            if (string.IsNullOrEmpty(args.filename))
            {
                var selected = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.selectTag);
                if (!selected.Any())
                    throw env.Except("Unable to find a view to select with tag '{0}'. Did you forget to specify a filename?", args.selectTag);
                var first = selected.First();
                if (selected.Skip(1).Any())
                    throw env.Except("Tag '{0}' is ambiguous, {1} views were found.", args.selectTag, selected.Count());
                var tagged = input as ITaggedDataView;
                if (tagged == null)
                {
                    var ag = new TagViewTransform.Arguments { tag = args.tag };
                    tagged = new TagViewTransform(env, ag, input);
                }
                first.Item2.AddRange(new[] { new Tuple<string, ITaggedDataView>(args.tag, tagged) });
                tagged.AddRange(new[] { new Tuple<string, ITaggedDataView>(args.selectTag, first.Item2) });
#if (DEBUG_TIP)
                long count = DataViewUtils.ComputeRowCount(tagged);
                if (count == 0)
                    throw env.Except("Replaced view is empty.");
                count = DataViewUtils.ComputeRowCount(first.Item2);
                if (count == 0)
                    throw env.Except("Selected view is empty.");
#endif
                var tr = first.Item2 as IDataTransform;
                env.AssertValue(tr);
                return tr;
            }
            else
            {
                if (!File.Exists(args.filename))
                    throw env.Except("Unable to find file '{0}'.", args.filename);
                var selected = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.selectTag);
                if (selected.Any())
                    throw env.Except("Tag '{0}' was already given. It cannot be assigned to the new file.", args.selectTag);
                var loaderArgs = new BinaryLoader.Arguments();
                var file = new MultiFileSource(args.filename);
                var loadSettings = ScikitSubComponent<ILegacyDataLoader, SignatureDataLoader>.AsSubComponent(args.loaderSettings);
                IDataView loader = loadSettings.CreateInstance(env, file);

                var ag = new TagViewTransform.Arguments { tag = args.selectTag };
                var newInput = new TagViewTransform(env, ag, loader);
                var tagged = input as ITaggedDataView;
                if (tagged == null)
                {
                    ag = new TagViewTransform.Arguments { tag = args.tag };
                    tagged = new TagViewTransform(env, ag, input);
                }

                newInput.AddRange(new[] { new Tuple<string, ITaggedDataView>(args.tag, tagged) });
                tagged.AddRange(new[] { new Tuple<string, ITaggedDataView>(args.selectTag, newInput) });

                var schema = loader.Schema;
                if (schema.Count == 0)
                    throw env.Except("The loaded view '{0}' is empty (empty schema).", args.filename);
                return newInput;
            }
        }
    }
}
