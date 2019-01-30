// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;


using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using PredictTransform = Scikit.ML.PipelineLambdaTransforms.PredictTransform;


[assembly: LoadableClass(PredictTransform.Summary, typeof(PredictTransform),
    typeof(PredictTransform.Arguments), typeof(SignatureDataTransform),
    "Run prediction for a transform", PredictTransform.LoaderSignature, "Predict")]

[assembly: LoadableClass(PredictTransform.Summary, typeof(PredictTransform),
    null, typeof(SignatureLoadDataTransform),
    "Run prediction for a transform", PredictTransform.LoaderSignature, "Predict")]


namespace Scikit.ML.PipelineLambdaTransforms
{
    /// <summary>
    /// Scores a predictor.
    /// </summary>
    public class PredictTransform : AbstractSimpleTransformTemplate
    {
        public const string LoaderSignature = "PredictTransform";  // Not more than 24 letters.
        internal const string Summary = "Run a previously trained predictor on the data. Output only the prediction.";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PREDAPPL",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PredictTransform).Assembly.FullName);
        }

        public new class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for features", ShortName = "feat")]
            public string featureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Output column", ShortName = "col")]
            public string outputColumn = "Predictions";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output probabilities instead of score", ShortName = "p")]
            public bool useProb = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Saves the model with this transform", ShortName = "s")]
            public bool serialize = true;

            public virtual void PostProcess()
            {
            }

            public virtual void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(featureColumn);
                ctx.Writer.Write(outputColumn);
                ctx.Writer.Write(useProb ? (byte)1 : (byte)0);
                ctx.Writer.Write(serialize ? (byte)1 : (byte)0);
            }

            public virtual void Read(ModelLoadContext ctx, IHost host)
            {
                featureColumn = ctx.Reader.ReadString();
                outputColumn = ctx.Reader.ReadString();
                useProb = ctx.Reader.ReadByte() == 1;
                serialize = ctx.Reader.ReadByte() == 1;
            }
        }

        protected Arguments _args;
        protected IPredictor _predictor;

        protected PredictTransform(IHostEnvironment env, IDataView input, string loaderSignature) :
            base(env, input, loaderSignature)
        {
        }

        protected PredictTransform(IHost host, ModelLoadContext ctx, IDataView input, string loaderSignature) :
            base(host, ctx, input, loaderSignature)
        {
        }

        public PredictTransform(IHostEnvironment env, Arguments args, IDataView input) :
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            args.PostProcess();
            _args = args;
            _sourcePipe = Create(_host, args, input, out _sourceCtx, null);
            _host.Check(_predictor != null, "_predictor is null. It should not.");
        }

        public PredictTransform(IHostEnvironment env, Arguments args, IDataView input, IPredictor predictor) :
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            args.PostProcess();
            _host.Check(predictor != null);
            _sourcePipe = Create(_host, args, input, out _sourceCtx, predictor);
            _host.Check(_predictor != null, "_predictor is null. It should not.");
        }

        public override void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            Write(ctx);
        }

        protected void Write(ModelSaveContext ctx)
        {
            _args.Write(ctx, _host);
            ctx.Writer.Write((byte)177);
            if (_args.serialize)
            {
                if (_predictor == null)
                    throw _host.Except("_predictor cannot be null.");
                ctx.SaveModel(_predictor, "predictor");
            }
        }

        public PredictTransform(IHost host, ModelLoadContext ctx, IDataView input) :
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

        public static PredictTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new PredictTransform(h, ctx, input));
        }

        protected virtual IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input, out IDataView sourceCtx, IPredictor overwritePredictor)
        {
            sourceCtx = input;
            Contracts.CheckValue(env, "env");
            env.CheckValue(args, "args");
            env.CheckValue(input, "input");

            IPredictor predictor;
            if (overwritePredictor == null)
                throw env.Except("No defined predictor.");
            else
                predictor = overwritePredictor;

            // The function is returning something and modifying a member of the class. Not very fancy.
            _predictor = predictor;

            string feat = TrainUtils.MatchNameOrDefaultOrNull(env, input.Schema,
                    "featureColumn", args.featureColumn, DefaultColumnNames.Features);
            int index = SchemaHelper.GetColumnIndex(input.Schema, feat);
            var type = input.Schema[index].Type;
            if (!type.IsVector() || type.AsVector().ItemType().RawKind() != DataKind.R4)
                throw env.Except("Features must a vector of floats");

            if (args.useProb)
            {
                var valueMapper = predictor as IValueMapperDist;
                if (valueMapper == null)
                    throw env.Except("Predictor must be a IValueMapper.");
                var output = valueMapper.DistType;
                if (output.IsVector())
                    return CreateTransformValueMapperDist<VBuffer<float>, VBuffer<float>, VBuffer<float>>(valueMapper, feat, args.outputColumn);
                else
                    return CreateTransformValueMapperDist<VBuffer<float>, VBuffer<float>, float>(valueMapper, feat, args.outputColumn);
            }
            else
            {
                var valueMapper = predictor as IValueMapper;
                if (valueMapper == null)
                    throw env.Except("Predictor must be a IValueMapper.");
                var output = valueMapper.OutputType;
                if (output.IsVector())
                    return CreateTransformValueMapper<VBuffer<float>, VBuffer<float>>(valueMapper, feat, args.outputColumn);
                else
                    return CreateTransformValueMapper<VBuffer<float>, float>(valueMapper, feat, args.outputColumn);
            }
        }

        IDataTransform CreateTransformValueMapper<TSrc, TDst>(IValueMapper valueMapper, string inputColumn, string outputColumn)
        {
            var mapper = valueMapper.GetMapper<TSrc, TDst>();
            IDataTransform transform;

            // The lambda transform should be replaced by something which does not creates
            // multiple threads. The mapper calls a predictor and for some reason, the context is not 
            // well preserved (rare bug).
            var view = LambdaColumnHelper.Create(_host, "PredictTransform", Source, inputColumn, outputColumn,
                                valueMapper.InputType, valueMapper.OutputType,
                                (in TSrc src, ref TDst dst) =>
                                {
                                    mapper(in src, ref dst);
                                });

            var args = new PassThroughTransform.Arguments();
            transform = new PassThroughTransform(_host, args, view);
            return transform;
        }

        IDataTransform CreateTransformValueMapperDist<TSrc, TDst, TDist>(IValueMapperDist valueMapper, string inputColumn, string outputColumn)
        {
            var mapper = valueMapper.GetMapper<TSrc, TDst, TDist>();
            IDataTransform transform;
            var args = new PassThroughTransform.Arguments();
            TDst temp = default(TDst);

            // The lambda transform should be replaced by something which does not creates
            // multiple threads. The mapper calls a predictor and for some reason, the context is not 
            // well preserved (rare bug).
            var view = LambdaColumnHelper.Create(_host, "PredictTransform", Source, inputColumn, outputColumn,
                                valueMapper.InputType, valueMapper.DistType,
                                (in TSrc src, ref TDist dst) =>
                                {
                                    mapper(in src, ref temp, ref dst);
                                });

            transform = new PassThroughTransform(_host, args, view);
            return transform;
        }
    }
}
