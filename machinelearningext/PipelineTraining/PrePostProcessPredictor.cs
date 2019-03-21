// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;
using Scikit.ML.ProductionPrediction;

using PrePostProcessPredictor = Scikit.ML.PipelineTraining.PrePostProcessPredictor;

[assembly: LoadableClass(typeof(PrePostProcessPredictor), null, typeof(SignatureLoadModel),
    "Preprocess Postprocess Predictor", PrePostProcessPredictor.LoaderSignature)]


namespace Scikit.ML.PipelineTraining
{
    public class PrePostProcessPredictor : IPredictor, ICanSaveModel, IValueMapper
    {
        public const string LoaderSignature = "PrePostProcessPredictor";
        public const string RegistrationName = "PrePostProcessPredictor";
        public const string Summary = "Appends optional transforms to preprocess and/or postprocess a trainer. " +
                                      "The pipeline will execute the following pipeline pre-predictor-score-post.";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PPCTSPOP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PrePostProcessPredictor).Assembly.FullName);
        }

        public PredictionKind PredictionKind { get { return _predictor.PredictionKind; } }

        private readonly IDataTransform _preProcess;
        private readonly IDataTransform _postProcess;
        private readonly IDataTransform _transformFromPredictor;
        private readonly IPredictor _predictor;
        private readonly IHost _host;
        private readonly string _inputColumn;
        private readonly string _outputColumn;

        public PrePostProcessPredictor(IHostEnvironment env, IDataTransform preProcess, IPredictor predictor,
                                       string inputColumn, string outputColumn, IDataTransform postProcess)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register("PrePostProcessPredictor");
            _host.CheckValue(predictor, "predictor");
            var val = predictor as IValueMapper;
            if (val == null)
                throw env.ExceptNotSupp("Predictor must implemented IValueMapper interface.");
            _preProcess = preProcess;
            _inputColumn = inputColumn;
            _outputColumn = outputColumn;
            _transformFromPredictor = new TransformFromValueMapper(env, predictor as IValueMapper, _preProcess, inputColumn, outputColumn);
            _postProcess = postProcess;
            _predictor = predictor;
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            SaveCore(ctx);
        }

        protected void SaveCore(ModelSaveContext ctx)
        {
            ctx.Writer.Write(_inputColumn);
            SchemaHelper.WriteType(ctx, InputType);
            ctx.Writer.Write(_outputColumn);
            ctx.SaveModel(_preProcess, "_preProcess");
            ctx.SaveModel(_predictor, "_predictor");
            ctx.Writer.Write(_postProcess != null);
            if (_postProcess != null)
                ctx.SaveModel(_postProcess, "_postProcess");
        }

        private PrePostProcessPredictor(IHost host, ModelLoadContext ctx)
        {
            Contracts.CheckValue(host, nameof(host));
            _host = host;
            _inputColumn = ctx.Reader.ReadString();
            var type = SchemaHelper.ReadType(ctx);
            _outputColumn = ctx.Reader.ReadString();

            DataViewSchema schema;
            IDataView data;
            if (type.IsVector())
            {
                switch (type.AsVector().ItemType().RawKind())
                {
                    case DataKind.Single:
                        schema = ExtendedSchema.Create(new ExtendedSchema((ISchema)null, new[] { _inputColumn }, new[] { new VectorType(NumberDataViewType.Single) }));
                        data = new TemporaryViewCursorColumn<VBuffer<float>>(default(VBuffer<float>), 0, schema);
                        break;
                    default:
                        throw Contracts.Except("Unable to create a temporary view from type '{0}'", type);
                }
            }
            else
            {
                switch (type.RawKind())
                {
                    case DataKind.Single:
                        schema = ExtendedSchema.Create(new ExtendedSchema((ISchema)null, new[] { _inputColumn }, new[] { NumberDataViewType.Single }));
                        data = new TemporaryViewCursorColumn<float>(default(float), 0, schema);
                        break;
                    default:
                        throw Contracts.Except("Unable to create a temporary view from type '{0}'", type);
                }
            }

            ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(_host, out _preProcess, "_preProcess", data);
            ctx.LoadModel<IPredictor, SignatureLoadModel>(_host, out _predictor, "_predictor");
            var hasPost = ctx.Reader.ReadBoolByte();
            if (hasPost)
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(_host, out _postProcess, "_postProcess", _transformFromPredictor);
            else
                _postProcess = null;
            _transformFromPredictor = new TransformFromValueMapper(_host, _predictor as IValueMapper, _preProcess, _inputColumn, _outputColumn);
        }

        public static PrePostProcessPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new PrePostProcessPredictor(h, ctx);
        }

        #region IValueMapper interface

        public DataViewType InputType
        {
            get
            {
                if (_preProcess == null)
                    return (_predictor as IValueMapper).InputType;
                else
                {
                    var val = _preProcess as IValueMapper;
                    if (val != null)
                        return val.InputType;
                    else
                    {
                        int index = SchemaHelper.GetColumnIndex(_preProcess.Source.Schema, _inputColumn);
                        return _preProcess.Source.Schema[index].Type;
                    }
                }
            }
        }

        public DataViewType OutputType
        {
            get
            {
                if (_postProcess == null)
                    return (_predictor as IValueMapper).OutputType;
                else
                {
                    var val = _postProcess as IValueMapper;
                    if (val != null)
                        return val.OutputType;
                    else
                        throw Contracts.ExceptNotSupp("postProcess transform is not null and does not implement IValueMapper interface. We cannot transform this predictor into a IValueMapper.");
                }
            }
        }

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            if (_preProcess == null)
            {
                if (_postProcess == null)
                {
                    return (_predictor as IValueMapper).GetMapper<TSrc, TDst>();
                }
                else
                {
                    throw _host.ExceptNotImpl();
                }
            }
            else
            {
                var valuemapper = _preProcess as IValueMapper;
                DataViewType outType;
                if (valuemapper != null)
                    outType = valuemapper.OutputType;
                else
                {
                    int index = SchemaHelper.GetColumnIndex(_preProcess.Source.Schema, _inputColumn);
                    outType = _preProcess.Source.Schema[index].Type;
                }

                if (outType.IsVector())
                {
                    switch (outType.AsVector().ItemType().RawKind())
                    {
                        case DataKind.Single:
                            return GetMapperWithTransform<TSrc, VBuffer<float>, TDst>(_preProcess);
                        default:
                            throw _host.ExceptNotSupp("Type '{0}' is not handled yet.", outType);
                    }
                }
                else
                {
                    switch (valuemapper.OutputType.RawKind())
                    {
                        case DataKind.Single:
                            return GetMapperWithTransform<TSrc, float, TDst>(_preProcess);
                        default:
                            throw _host.ExceptNotSupp("Type '{0}' is not handled yet.", outType);
                    }
                }
                throw _host.ExceptNotImpl();
            }
        }

        ValueMapper<TSrc, TDst> GetMapperWithTransform<TSrc, TMiddle, TDst>(IDataTransform trans)
        {
            var mapperPreVM = new ValueMapperFromTransformFloat<TMiddle>(_host, trans, _inputColumn, _inputColumn, trans.Source);
            var mapperPre = mapperPreVM.GetMapper<TSrc, TMiddle>();
            var mapperPred = (_predictor as IValueMapper).GetMapper<TMiddle, TDst>();
            TMiddle middle = default(TMiddle);
            return (in TSrc src, ref TDst dst) =>
            {
                mapperPre(in src, ref middle);
                mapperPred(in middle, ref dst);
            };
        }

        #endregion
    }
}
