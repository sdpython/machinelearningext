// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using MultiToBinaryPredictor = Scikit.ML.MultiClass.MultiToBinaryPredictor;

[assembly: LoadableClass(typeof(MultiToBinaryPredictor), null, typeof(SignatureLoadModel),
    "iOVA Multi To Binary MC Predictor", MultiToBinaryPredictor.LoaderSignature, "iOVA MC Executor")]


namespace Scikit.ML.MultiClass
{
    using TScalarPredictor = IPredictor; //IPredictorProducing<float>;

    /// <summary>
    /// Defines a predictor which does what OVA does but produces only one model.
    /// It adds the label to the features and produces a binary answer which tells whether
    /// or not the input vecor belongs the class label which was added to the features.
    /// </summary>
    public class MultiToBinaryPredictor : MultiToPredictorCommon
    {
        #region identification

        public const string LoaderSignature = "MultiToBinaryPredictor";
        public const string RegistrationName = "MultiToBinaryPredictor";

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULBINPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiToBinaryPredictor).Assembly.FullName);
        }

        #endregion

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        internal static MultiToBinaryPredictor Create<TLabel>(IHost host, VBuffer<TLabel> classes,
                            TScalarPredictor[] predictors, IPredictor reclassPredictor, bool singleColumn, bool labelKey)
        {
            IImplBase impl;
            using (var ch = host.Start("Creating MultiToBinary predictor"))
                impl = new ImplRaw<TLabel>(classes, predictors, reclassPredictor, singleColumn, labelKey);
            return new MultiToBinaryPredictor(host, impl);
        }

        private MultiToBinaryPredictor(IHostEnvironment env, IImplBase impl)
            : base(env, impl, RegistrationName)
        {
        }

        public static MultiToBinaryPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiToBinaryPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.Writer.Write((byte)_impl.LabelType.RawKind());
            _impl.SaveCore(ctx, Host, GetVersionInfo());
        }

        private MultiToBinaryPredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, ctx, RegistrationName)
        {
            byte bkind = ctx.Reader.ReadByte();
            env.Check(bkind >= 0 && bkind <= 100, "kind");
            var kind = (DataKind)bkind;
            switch (kind)
            {
                case DataKind.R4:
                    _impl = new ImplRawBinary<float>(ctx, env);
                    break;
                case DataKind.I1:
                    _impl = new ImplRawBinary<byte>(ctx, env);
                    break;
                case DataKind.U2:
                    _impl = new ImplRawBinary<ushort>(ctx, env);
                    break;
                case DataKind.U4:
                    _impl = new ImplRawBinary<uint>(ctx, env);
                    break;
                default:
                    throw env.ExceptNotSupp("Not supported label type.");
            }
        }

        protected class ImplRawBinary<TLabel> : ImplRaw<TLabel>
        {
            internal ImplRawBinary(ModelLoadContext ctx, IHostEnvironment env) : base(ctx, env)
            {
            }

            internal ImplRawBinary(VBuffer<TLabel> classes, TScalarPredictor[] predictors, TScalarPredictor reclassPredictor,
                                   bool singleColumn, bool labelKey) :
                base(classes, predictors, reclassPredictor, singleColumn, labelKey)
            {
            }
        }
    }
}
