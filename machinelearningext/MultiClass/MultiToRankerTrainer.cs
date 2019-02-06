// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Training;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using MultiToRankerTrainer = Scikit.ML.MultiClass.MultiToRankerTrainer;

[assembly: LoadableClass(MultiToRankerTrainer.Summary, typeof(MultiToRankerTrainer), typeof(MultiToRankerTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MultiToRankerTrainer.LoaderSignature, "InternalOVARanker", "iOVArk")]


namespace Scikit.ML.MultiClass
{
    using TScalarTrainer = ITrainer<IPredictor>; // ITrainer<IPredictorProducing<float>>;
    using TScalarPredictor = IPredictor; //IPredictorProducing<float>;
    using TVectorPredictor = IPredictor; //IPredictorProducing<VBuffer<float>>;

    /// <summary>
    /// Train a MultiToRanking predictor. It multiplies the rows by the number of classes to predict.
    /// (multi class problem). It turns a multi-class problem into a raking problem.
    /// </summary>
    public class MultiToRankerTrainer : MultiToTrainerCommon
    {
        #region identification

        public const string LoaderSignature = "MultiToRanker";  // Not more than 24 letters.
        public const string Summary = "Converts a multi-class classification problem into a ranking problem.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULRNKTR",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiToRankerTrainer).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public new class Arguments : MultiToTrainerCommon.Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use U4 for the temporary group column instead of U8 (required by model encoding group on 4 bytes).", ShortName = "u4")]
            public bool groupIsU4 = false;

            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 1,
                SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<TScalarTrainer> predictorType =
                new ScikitSubComponent<TScalarTrainer, SignatureRankerTrainer>("ftrank");
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

        public MultiToRankerTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, LoaderSignature)
        {
            Host.CheckValue(args, "args");
            Host.CheckValue(args.predictorType, "predictorType", "Must specify a base learner type");
            _args = args;
            if (_args.algo == MultiToBinaryTransform.MultiplicationAlgorithm.Default)
                _args.algo = MultiToBinaryTransform.MultiplicationAlgorithm.Ranking;

            if (_args.algo == MultiToBinaryTransform.MultiplicationAlgorithm.Default ||
                _args.algo == MultiToBinaryTransform.MultiplicationAlgorithm.Reweight)
                Host.Except("Algorithm must be Ranking.");
        }

        protected override TVectorPredictor CreatePredictor()
        {
            Host.Check(_predictors.Length == 1);
            var pred = _predictors[0] as MultiToRankerPredictor;
            Host.Assert(pred != null);
            return pred;
        }

        protected override TVectorPredictor Train(TrainContext context)
        {
            var data = context.TrainingSet;
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
                        var trSett = ScikitSubComponent<ITrainer, SignatureTrainer>.AsSubComponent(_args.predictorType);
                        var tr = trSett.CreateInstance(Host);
                        trainer = tr as TScalarTrainer;
                        Contracts.AssertValue(trainer);
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

            #region adding group ID

            // We insert a group Id.
            string groupColumnTemp = DataViewUtils.GetTempColumnName(data.Schema.Schema) + "GR";
            var groupArgs = new GenerateNumberTransform.Arguments
            {
                Columns = new[] { GenerateNumberTransform.Column.Parse(groupColumnTemp) },
                UseCounter = true
            };

            var withGroup = new GenerateNumberTransform(Host, groupArgs, data.Data);
            data = new RoleMappedData(withGroup, data.Schema.GetColumnRoleNames());

            #endregion

            #region preparing the training dataset

            string dstName, labName;
            var trans = MapLabelsAndInsertTransform(ch, data, out dstName, out labName, count, true, _args);
            var newFeatures = trans.Schema.GetTempColumnName() + "NF";

            // We check the label is not boolean.
            int indexLab = SchemaHelper.GetColumnIndex(trans.Schema, dstName);
            var typeLab = trans.Schema[indexLab].Type;
            if (typeLab.RawKind() == DataKind.BL)
                throw Host.Except("Column '{0}' has an unexpected type {1}.", dstName, typeLab.RawKind());

            var args3 = new DescribeTransform.Arguments { columns = new string[] { labName, dstName }, oneRowPerColumn = true };
            var desc = new DescribeTransform(Host, args3, trans);

            IDataView viewI;
            if (_args.singleColumn && data.Schema.Label.Value.Type.RawKind() == DataKind.R4)
                viewI = desc;
            else if (_args.singleColumn)
            {
                var sch = new TypeReplacementSchema(desc.Schema, new[] { labName }, new[] { NumberType.R4 });
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
                var sch = new TypeReplacementSchema(desc.Schema, new[] { labName }, new[] { new VectorType(NumberType.R4, (int)nb) });
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
                var sch = new TypeReplacementSchema(desc.Schema, new[] { labName }, new[] { new VectorType(NumberType.R4, nb) });
                viewI = new TypeReplacementDataView(desc, sch);
                #region debug
#if (DEBUG)
                DebugChecking0(viewI, labName, true);
#endif
                #endregion
            }

            ch.Info("Merging column label '{0}' with features '{1}'", labName, data.Schema.Feature.Value.Name);
            var args = string.Format("Concat{{col={0}:{1},{2}}}", newFeatures, data.Schema.Feature.Value.Name, labName);
            var after_concatenation_ = ComponentCreation.CreateTransform(Host, args, viewI);

            #endregion

            #region converting label and group into keys

            // We need to convert the label into a Key.
            var convArgs = new MultiClassConvertTransform.Arguments
            {
                column = new[] { MultiClassConvertTransform.Column.Parse(string.Format("{0}k:{0}", dstName)) },
                keyCount = new KeyCount(4),
                resultType = DataKind.U4
            };
            IDataView after_concatenation_key_label = new MultiClassConvertTransform(Host, convArgs, after_concatenation_);

            // The group must be a key too!
            convArgs = new MultiClassConvertTransform.Arguments
            {
                column = new[] { MultiClassConvertTransform.Column.Parse(string.Format("{0}k:{0}", groupColumnTemp)) },
                keyCount = new KeyCount(),
                resultType = _args.groupIsU4 ? DataKind.U4 : DataKind.U8
            };
            after_concatenation_key_label = new MultiClassConvertTransform(Host, convArgs, after_concatenation_key_label);

            #endregion

            #region preparing the RoleMapData view

            string groupColumn = groupColumnTemp + "k";
            dstName += "k";

            var roles = data.Schema.GetColumnRoleNames();
            var rolesArray = roles.ToArray();
            roles = roles
                .Where(kvp => kvp.Key.Value != RoleMappedSchema.ColumnRole.Label.Value)
                .Where(kvp => kvp.Key.Value != RoleMappedSchema.ColumnRole.Feature.Value)
                .Where(kvp => kvp.Key.Value != groupColumn)
                .Where(kvp => kvp.Key.Value != groupColumnTemp);
            rolesArray = roles.ToArray();
            if (rolesArray.Any() && rolesArray[0].Value == groupColumnTemp)
                throw ch.Except("Duplicated group.");
            roles = roles
                .Prepend(RoleMappedSchema.ColumnRole.Feature.Bind(newFeatures))
                .Prepend(RoleMappedSchema.ColumnRole.Label.Bind(dstName))
                .Prepend(RoleMappedSchema.ColumnRole.Group.Bind(groupColumn));
            var trainer_input = new RoleMappedData(after_concatenation_key_label, roles);

            #endregion

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
            int indexLab;
            var trans_ = trans;
            trans = MapLabelsAndInsertTransform(ch, data, out dstName, out labName, count, false, _args);
            trans.Steal(trans_);
            indexLab = SchemaHelper.GetColumnIndex(trans.Schema, labName);
            var labType = trans.Schema[indexLab].Type;
            var initialLabKind = data.Schema.Label.Value.Type.RawKind();

            TVectorPredictor predictor;
            switch (initialLabKind)
            {
                case DataKind.R4:
                    var p4 = MultiToRankerPredictor.Create(Host, trans.GetClasses<float>(), predictors, _reclassPredictor, _args.singleColumn, false);
                    predictor = p4 as TVectorPredictor;
                    break;
                case DataKind.U1:
                    var pu1 = MultiToRankerPredictor.Create(Host, trans.GetClasses<byte>(), predictors, _reclassPredictor, _args.singleColumn, true);
                    predictor = pu1 as TVectorPredictor;
                    break;
                case DataKind.U2:
                    var pu2 = MultiToRankerPredictor.Create(Host, trans.GetClasses<ushort>(), predictors, _reclassPredictor, _args.singleColumn, true);
                    predictor = pu2 as TVectorPredictor;
                    break;
                case DataKind.U4:
                    var pu4 = MultiToRankerPredictor.Create(Host, trans.GetClasses<uint>(), predictors, _reclassPredictor, _args.singleColumn, true);
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
