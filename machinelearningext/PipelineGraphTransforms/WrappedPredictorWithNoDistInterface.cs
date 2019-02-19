// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using WrappedPredictorWithNoDistInterface = Scikit.ML.PipelineGraphTransforms.WrappedPredictorWithNoDistInterface;


[assembly: LoadableClass(typeof(WrappedPredictorWithNoDistInterface), null, typeof(SignatureLoadModel),
    "Wrapped Predictor With No Dist Interface",
    WrappedPredictorWithNoDistInterface.LoaderSignature, "WrappedPredictorWithNoDistInterface")]


namespace Scikit.ML.PipelineGraphTransforms
{
    public class WrappedPredictorWithNoDistInterface : IPredictor, IValueMapper, ICanSaveModel
    {
        #region identification

        public const string LoaderSignature = "WrappedPWithNoDistI";
        public const string RegistrationName = "WrappedPWithNoDistI";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "WRAPNODI",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(WrappedPredictorWithNoDistInterface).Assembly.FullName);
        }

        #endregion

        IPredictor _predictor;

        public WrappedPredictorWithNoDistInterface(IPredictor pred)
        {
            _predictor = pred;
        }
        public PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }
        public IPredictor Predictor { get { return _predictor; } }
        public DataViewType InputType { get { return (_predictor as IValueMapper).InputType; } }
        public DataViewType OutputType { get { return (_predictor as IValueMapper).OutputType; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>() { return (_predictor as IValueMapper).GetMapper<TSrc, TDst>(); }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            Contracts.CheckValue(_predictor, "_predictor");
            ctx.SaveModel(_predictor, "predictor");
        }

        private WrappedPredictorWithNoDistInterface(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.LoadModel<IPredictor, SignatureLoadModel>(env, out _predictor, "predictor");
            Contracts.CheckValue(_predictor, "_predictor");
        }

        public static WrappedPredictorWithNoDistInterface Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new WrappedPredictorWithNoDistInterface(env, ctx);
        }
    }
}

