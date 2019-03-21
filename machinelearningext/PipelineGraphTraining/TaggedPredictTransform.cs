// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.PipelineGraphTransforms;
using Scikit.ML.PipelineLambdaTransforms;


using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using TaggedPredictTransform = Scikit.ML.PipelineGraphTraining.TaggedPredictTransform;


[assembly: LoadableClass(TaggedPredictTransform.Summary, typeof(TaggedPredictTransform),
    typeof(TaggedPredictTransform.Arguments), typeof(SignatureDataTransform),
    "Run prediction for a transform", TaggedPredictTransform.LoaderSignature, "TagPredict")]

[assembly: LoadableClass(TaggedPredictTransform.Summary, typeof(TaggedPredictTransform),
    null, typeof(SignatureLoadDataTransform),
    "Run prediction for a transform", TaggedPredictTransform.LoaderSignature, "TagPredict")]


namespace Scikit.ML.PipelineGraphTraining
{
    /// <summary>
    /// Scores a predictor hosted by a tagged view.
    /// </summary>
    public class TaggedPredictTransform : PredictTransform
    {
        public new const string LoaderSignature = "TaggedPredictTr";  // Not more than 24 letters.
        internal const string Summary = "Run a previously trained predictor on the data. Output only the prediction.";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TAGPREDA",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TaggedPredictTransform).Assembly.FullName);
        }

        public new class Arguments : PredictTransform.Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Tag of the view which holds the predictor.", ShortName = "in")]
            public string taggedPredictor;

            public override void Write(ModelSaveContext ctx, IHost host)
            {
                base.Write(ctx, host);
                ctx.Writer.Write(taggedPredictor);
            }

            public override void Read(ModelLoadContext ctx, IHost host)
            {
                base.Read(ctx, host);
                taggedPredictor = ctx.Reader.ReadString();
            }
        }

        public TaggedPredictTransform(IHostEnvironment env, Arguments args, IDataView input) :
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            args.PostProcess();
            _args = args;
            _host.CheckValue(args.taggedPredictor, "taggedPredictor");
            _sourcePipe = Create(_host, args, input, out _sourceCtx, null);
            _host.Check(_predictor != null, "_predictor is null. It should not.");
        }

        public TaggedPredictTransform(IHostEnvironment env, Arguments args, IDataView input, IPredictor predictor) :
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            args.PostProcess();
            _args = args;
            _host.Check(predictor != null || !string.IsNullOrEmpty(args.taggedPredictor), "taggedPredictor");
            _sourcePipe = Create(_host, args, input, out _sourceCtx, predictor);
            _host.Check(_predictor != null, "_predictor is null. It should not.");
        }

        protected override void SaveModel(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            Write(ctx);
        }

        public TaggedPredictTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, ctx, input, LoaderSignature)
        {
            _args = new Arguments();
            _args.Read(ctx, _host);
            byte b = ctx.Reader.ReadByte();
            if (b != 177)
                throw _host.Except("Corrupt file.");
            if (_args.serialize)
                ctx.LoadModel<IPredictor, SignatureLoadModel>(_host, out _predictor, "predictor");
            else
                _predictor = null;
            _sourcePipe = Create(_host, _args, input, out _sourceCtx, _predictor);
        }

        public new static TaggedPredictTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new TaggedPredictTransform(h, ctx, input));
        }

        protected override IDataTransform Create(IHostEnvironment env, PredictTransform.Arguments args_, IDataView input, out IDataView sourceCtx, IPredictor overwritePredictor)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(args_, "args_");
            var args = args_ as Arguments;
            env.CheckValue(args, "args");
            env.CheckValue(input, "input");

            IPredictor predictor;
            if (overwritePredictor == null)
            {
                env.CheckUserArg(!string.IsNullOrWhiteSpace(args.taggedPredictor), "taggedPredictor",
                    "The input tag is required.");
                predictor = TagHelper.GetTaggedPredictor(env, input, args.taggedPredictor);
            }
            else
                predictor = overwritePredictor;

            return base.Create(env, args, input, out sourceCtx, predictor);
        }
    }
}
