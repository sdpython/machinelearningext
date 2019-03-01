// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using MultiToBinaryTrainer = Scikit.ML.MultiClass.MultiToBinaryTrainer;

[assembly: LoadableClass(MultiToBinaryTrainer.Summary, typeof(MultiToBinaryTrainer), typeof(MultiToBinaryTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MultiToBinaryTrainer.LoaderSignature, "InternalOVA", "iOVA")]


namespace Scikit.ML.MultiClass
{
    using TScalarTrainer = ITrainer<IPredictor>; // ITrainer<IPredictorProducing<float>>;
    using TScalarPredictor = IPredictor; //IPredictorProducing<float>;
    using TVectorPredictor = IPredictor; //IPredictorProducing<VBuffer<float>>;

    /// <summary>
    /// Train a MultiToBinary predictor. It multiplies the rows by the number of classes to predict.
    /// (multi class problem).
    /// </summary>
    public class MultiToBinaryTrainer : MultiToTrainerCommon
    {
        #region identification

        public const string LoaderSignature = "MultiToBinary";  // Not more than 24 letters.
        public const string Summary = "Converts a multi-class classification problem into a binary classification problem.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULBINTR",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiToBinaryTrainer).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public new class Arguments : MultiToTrainerCommon.Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 1,
                SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<TScalarTrainer> predictorType =
                new ScikitSubComponent<TScalarTrainer, SignatureBinaryClassifierTrainer>("ft");
        }

        #endregion

        #region internal members / accessors

        private readonly Arguments _args;

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        public override TrainerInfo Info
        {
            get
            {
                return new TrainerInfo(normalization: false, calibration: false, caching: false,
                                       supportValid: false, supportIncrementalTrain: false);
            }
        }

        #endregion

        #region public constructor / serialization / load / save

        public MultiToBinaryTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, LoaderSignature)
        {
            Host.CheckValue(args, "args");
            Contracts.CheckValue(args.predictorType, "predictorType", "Must specify a base learner type");
            _args = args;
        }

        protected override TVectorPredictor CreatePredictor()
        {
            Host.Check(_predictors.Length == 1);
            var pred = _predictors[0] as MultiToBinaryPredictor;
            Host.Assert(pred != null);
            return pred;
        }

        protected override TVectorPredictor Train(TrainContext context)
        {
            return Train(context.TrainingSet);
        }

        TVectorPredictor Train(RoleMappedData data)
        {
            Contracts.CheckValue(data, "data");
            data.CheckFeatureFloatVector();

            int count;
            data.CheckMultiClassLabel(out count);

            using (var ch = Host.Start("Training"))
            {
                // Train one-vs-all models.
                _predictors = new TVectorPredictor[1];
                for (int i = 0; i < _predictors.Length; i++)
                {
                    ch.Info("Training learner {0}", i);
                    Contracts.CheckValue(_args.predictorType, "predictorType", "Must specify a base learner type");

                    TScalarTrainer trainer;
                    if (_trainer != null)
                        trainer = _trainer;
                    else
                    {
                        var temp = ScikitSubComponent<ITrainer, SignatureBinaryClassifierTrainer>.AsSubComponent(_args.predictorType);
                        var inst = temp.CreateInstance(Host);
                        trainer = inst as TScalarTrainer;
                        if (trainer == null)
                        {
                            var allTypes = TrainerHelper.GetParentTypes(inst.GetType()).ToArray();
                            var allTypesStr = string.Join("\n", allTypes.Select(c => c.ToString()));
                            throw ch.ExceptNotSupp($"Unable to cast {inst.GetType()} into {typeof(TScalarTrainer)}.\n-TYPES-\n{allTypesStr}");
                        }
                    }

                    _trainer = null;
                    _predictors[i] = TrainPredictor(ch, trainer, data, count);
                }
            }

            return CreatePredictor();
        }

        protected override TVectorPredictor TrainPredictor(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int count)
        {
            var data0 = data;
            string dstName, labName;
            var trans = MapLabelsAndInsertTransform(ch, data, out dstName, out labName, count, true, _args);
            var newFeatures = trans.Schema.GetTempColumnName() + "NF";

            var args3 = new DescribeTransform.Arguments { columns = new string[] { labName, dstName }, oneRowPerColumn = true };
            var desc = new DescribeTransform(Host, args3, trans);

            IDataView viewI;
            if (_args.singleColumn && data.Schema.Label.Value.Type.RawKind() == DataKind.Single)
                viewI = desc;
            else if (_args.singleColumn)
            {
                var sch = new TypeReplacementSchema(desc.Schema, new[] { labName }, new[] { NumberDataViewType.Single });
                viewI = new TypeReplacementDataView(desc, sch);
                #region debug
#if (DEBUG)
                DebugChecking0(viewI, labName, false);
#endif
                #endregion
            }
            else if (data.Schema.Label.Value.Type.IsKey())
            {
                ulong nb = data.Schema.Label.Value.Type.AsKey().GetKeyCount();
                var sch = new TypeReplacementSchema(desc.Schema, new[] { labName }, new[] { new VectorType(NumberDataViewType.Single, (int)nb) });
                viewI = new TypeReplacementDataView(desc, sch);
                #region debug
#if (DEBUG)
                int nb_;
                MinMaxLabelOverDataSet(trans, labName, out nb_);
                int count3;
                data.CheckMultiClassLabel(out count3);
                if ((ulong)count3 != nb)
                    throw ch.Except("Count mismatch (KeyCount){0} != {1}", nb, count3);
                DebugChecking0(viewI, labName, true);
                DebugChecking0Vfloat(viewI, labName, nb);
#endif
                #endregion
            }
            else
            {
                int nb;
                if (count <= 0)
                    MinMaxLabelOverDataSet(trans, labName, out nb);
                else
                    nb = count;
                var sch = new TypeReplacementSchema(desc.Schema, new[] { labName }, new[] { new VectorType(NumberDataViewType.Single, nb) });
                viewI = new TypeReplacementDataView(desc, sch);
                #region debug
#if (DEBUG)
                DebugChecking0(viewI, labName, true);
#endif
                #endregion
            }

            ch.Info("Merging column label '{0}' with features '{1}'", labName, data.Schema.Feature.Value.Name);
            var args = string.Format("Concat{{col={0}:{1},{2}}}", newFeatures, data.Schema.Feature.Value.Name, labName);
            IDataView after_concatenation = ComponentCreation.CreateTransform(Host, args, viewI);

            var roles = data.Schema.GetColumnRoleNames()
                .Where(kvp => kvp.Key.Value != RoleMappedSchema.ColumnRole.Label.Value)
                .Where(kvp => kvp.Key.Value != RoleMappedSchema.ColumnRole.Feature.Value)
                .Prepend(RoleMappedSchema.ColumnRole.Feature.Bind(newFeatures))
                .Prepend(RoleMappedSchema.ColumnRole.Label.Bind(dstName));
            var trainer_input = new RoleMappedData(after_concatenation, roles);

            ch.Info("New Features: {0}:{1}", trainer_input.Schema.Feature.Value.Name, trainer_input.Schema.Feature.Value.Type);
            ch.Info("New Label: {0}:{1}", trainer_input.Schema.Label.Value.Name, trainer_input.Schema.Label.Value.Type);

            // We train the unique binary classifier.
            var trainedPredictor = trainer.Train(trainer_input);
            var predictors = new TScalarPredictor[] { trainedPredictor };

            // We train the reclassification classifier.
            if (_args.reclassicationPredictor != null)
            {
                var pred = CreateFinalPredictor(ch, data, trans, count, _args, predictors, null);
                TrainReclassificationPredictor(data0, pred, ScikitSubComponent<ITrainer, SignatureTrainer>.AsSubComponent(_args.reclassicationPredictor));
            }
            return CreateFinalPredictor(ch, data, trans, count, _args, predictors, _reclassPredictor);
        }

        protected TVectorPredictor CreateFinalPredictor(IChannel ch, RoleMappedData data,
                                    MultiToBinaryTransform trans, int count, Arguments args,
                                    TScalarPredictor[] predictors, IPredictor reclassPredictor)
        {
            // We create the final predictor. We remove every unneeded transform.
            string dstName, labName;
            var trans_ = trans;
            trans = MapLabelsAndInsertTransform(ch, data, out dstName, out labName, count, false, args);
            trans.Steal(trans_);

            int indexLab = SchemaHelper.GetColumnIndex(trans.Schema, labName);
            var labType = trans.Schema[indexLab].Type;
            var initialLabKind = data.Schema.Label.Value.Type.RawKind();

            TVectorPredictor predictor;
            switch (initialLabKind)
            {
                case DataKind.Single:
                    var p4 = MultiToBinaryPredictor.Create(Host, trans.GetClasses<float>(), predictors, reclassPredictor, args.singleColumn, false);
                    predictor = p4 as TVectorPredictor;
                    break;
                case DataKind.SByte:
                    var pu1 = MultiToBinaryPredictor.Create(Host, trans.GetClasses<byte>(), predictors, reclassPredictor, args.singleColumn, true);
                    predictor = pu1 as TVectorPredictor;
                    break;
                case DataKind.UInt16:
                    var pu2 = MultiToBinaryPredictor.Create(Host, trans.GetClasses<ushort>(), predictors, reclassPredictor, args.singleColumn, true);
                    predictor = pu2 as TVectorPredictor;
                    break;
                case DataKind.UInt32:
                    var pu4 = MultiToBinaryPredictor.Create(Host, trans.GetClasses<uint>(), predictors, reclassPredictor, args.singleColumn, true);
                    predictor = pu4 as TVectorPredictor;
                    break;
                default:
                    throw ch.ExceptNotSupp("Unsupported type for a multi class label.");
            }

            Host.Assert(predictor != null);
            return predictor;
        }

        #endregion
    }
}
