// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using MultiToRankerPredictor = Scikit.ML.MultiClass.MultiToRankerPredictor;

[assembly: LoadableClass(typeof(MultiToRankerPredictor), null, typeof(SignatureLoadModel),
    "iOVA Multi To Binary Ranker Predictor", MultiToRankerPredictor.LoaderSignature, "iOVA Ranker Executor")]


namespace Scikit.ML.MultiClass
{
    using TScalarPredictor = IPredictor; //IPredictorProducing<float>;

    /// <summary>
    /// Defines a predictor which does what OVA does but produces only one model.
    /// It adds the label to the features and produces a ranking answer which tells whether
    /// or not the input vecor belongs the class label which was added to the features.
    /// </summary>
    public class MultiToRankerPredictor : MultiToPredictorCommon
    {
        #region identification

        public const string LoaderSignature = "MultiToRankerPredictor";
        public const string RegistrationName = "MultiToRankerPredictor";

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULRNKPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiToRankerPredictor).Assembly.FullName);
        }

        #endregion

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        internal static MultiToRankerPredictor Create<TLabel>(IHost host, VBuffer<TLabel> classes,
                            TScalarPredictor[] predictors, IPredictor reclassPredictor, bool singleColumn, bool labelKey)
        {
            IImplBase impl;
            using (var ch = host.Start("Creating MultiToRanker predictor"))
                impl = new ImplRaw<TLabel>(classes, predictors, reclassPredictor, singleColumn, labelKey);
            return new MultiToRankerPredictor(host, impl);
        }

        private MultiToRankerPredictor(IHostEnvironment env, IImplBase impl)
            : base(env, impl, RegistrationName)
        {
        }

        public static MultiToRankerPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiToRankerPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.Writer.Write((byte)_impl.LabelType.RawKind());
            _impl.SaveCore(ctx, Host, GetVersionInfo());
        }

        private MultiToRankerPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, RegistrationName)
        {
            byte bkind = ctx.Reader.ReadByte();
            env.Check(bkind >= 0 && bkind <= 100, "kind");
            var kind = (DataKind)bkind;
            switch (kind)
            {
                case DataKind.Single:
                    _impl = new ImplRawRanker<float>(ctx, env);
                    break;
                case DataKind.SByte:
                    _impl = new ImplRawRanker<byte>(ctx, env);
                    break;
                case DataKind.UInt16:
                    _impl = new ImplRawRanker<ushort>(ctx, env);
                    break;
                case DataKind.UInt32:
                    _impl = new ImplRawRanker<uint>(ctx, env);
                    break;
                default:
                    throw env.ExceptNotSupp("Not supported label type.");
            }
        }

        protected class ImplRawRanker<TLabel> : ImplRaw<TLabel>
        {
            internal ImplRawRanker(ModelLoadContext ctx, IHostEnvironment env) : base(ctx, env)
            {
            }

            internal ImplRawRanker(VBuffer<TLabel> classes, TScalarPredictor[] predictors, TScalarPredictor reclassPredicgtor,
                                   bool singleColumn, bool labelKey) :
                base(classes, predictors, reclassPredicgtor, singleColumn, labelKey)
            {
            }
        }
    }
}
