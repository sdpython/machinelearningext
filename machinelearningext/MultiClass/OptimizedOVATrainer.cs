// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Training;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Transforms;
using Scikit.ML.RandomTransforms;
using Scikit.ML.PipelineHelper;

using OptimizedOVATrainer = Scikit.ML.MultiClass.OptimizedOVATrainer;

[assembly: LoadableClass(OptimizedOVATrainer.Summary, typeof(OptimizedOVATrainer), typeof(OptimizedOVATrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    OptimizedOVATrainer.UserNameValue, OptimizedOVATrainer.LoadNameValue, "OOVA")]


namespace Scikit.ML.MultiClass
{
    using TScalarTrainer = ITrainer<IPredictor>; //ITrainer<IPredictorProducing<float>>;
    using TScalarPredictor = IPredictor; //IPredictorProducing<float>;
    using TVectorPredictor = IPredictor; //IPredictorProducing<VBuffer<float>>;
    using CR = RoleMappedSchema.ColumnRole;

    /// <summary>
    /// Trainer for an OptimizedOVAPredictor.
    /// </summary>
    public sealed class OptimizedOVATrainer : TrainerBase<TVectorPredictor>
    {
        internal const string LoadNameValue = "OptimizedOVA";
        internal const string UserNameValue = "Optimized One-vs-All";
        internal const string Summary = "Optimized OVA, playground for experimentation";

        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use probability or margins to determine max", ShortName = "useprob")]
            public bool useProbabilities = true;

            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 1,
                SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<TScalarTrainer> predictorType = null;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>",
                SignatureType = typeof(SignatureCalibrator))]
            public IComponentFactory<ICalibratorTrainer> calibratorType = new ScikitSubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int maxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Downsample the zero class. Training a multi-class leads to imbalanced data. Why downsampling it?", ShortName = "ds")]
            public float downsampling = 0f;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Drop missing labels.", ShortName = "na")]
            public bool dropNALabel = true;

            [Argument(ArgumentType.Multiple, HelpText = "Add a cache transform before training. That might required if cursor happen to be in an unstable state",
                ShortName = "cache", NullName = "<None>", SignatureType = typeof(SignatureDataTransform))]
            public IComponentFactory<IDataTransform> cacheTransform = null;
        }

        private readonly Arguments _args;
        private readonly bool _needNorm;
        private TScalarPredictor[] _predictors;
        private TScalarTrainer _trainer;

        public OptimizedOVATrainer(IHostEnvironment env, Arguments args) : base(env, LoadNameValue)
        {
            _args = args = args ?? new Arguments();
            Contracts.CheckValue(_args.predictorType, "predictorType", "Must specify a base learner type");
            var predSett = ScikitSubComponent<ITrainer, SignatureBinaryClassifierTrainer>.AsSubComponent(_args.predictorType);
            var trainer = predSett.CreateInstance(env);
            _trainer = trainer as TScalarTrainer;
            Contracts.CheckValue(_trainer, nameof(_trainer));
            _needNorm = _trainer.Info.NeedNormalization;
        }

        // No matter what the internal predictor, we're performing many passes
        // simply by virtue of this being one-versus-all.
        public override TrainerInfo Info => new TrainerInfo(_needNorm, false, true, false, false);

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        protected override TVectorPredictor Train(TrainContext ctx)
        {
            var data = ctx.TrainingSet;
            Contracts.CheckValue(data, "data");
            data.CheckFeatureFloatVector();

            int count;
            data.CheckMultiClassLabel(out count);
            Contracts.Assert(count > 0);

            using (var ch = Host.Start("Training"))
            {
                // Train one-vs-all models.
                _predictors = new TScalarPredictor[count];
                for (int i = 0; i < _predictors.Length; i++)
                {
                    ch.Info("Training learner {0}", i);

                    // We may have instantiated the first trainer to use already. If so capture it;
                    // otherwise create a new one.
                    TScalarTrainer trainer;
                    if (_trainer != null)
                        trainer = _trainer;
                    else
                    {
                        var temp = ScikitSubComponent<ITrainer, SignatureBinaryClassifierTrainer>.AsSubComponent(_args.predictorType);
                        trainer = temp.CreateInstance(Host) as TScalarTrainer;
                    }
                    _trainer = null;
                    _predictors[i] = TrainOne(ch, trainer, data, i);
                }
            }
            return CreatePredictor();
        }

        // cls is the "class id", zero-based.
        private TScalarPredictor TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls)
        {
            string dstName;
            var view = MapLabels(data, cls, out dstName, ch);

            if (_args.cacheTransform != null)
            {
                var sub = ScikitSubComponent<IDataTransform, SignatureDataTransform>.AsSubComponent(_args.cacheTransform);
                view = sub.CreateInstance(Host, view);
            }

            var roles = data.Schema.GetColumnRoleNames()
                .Where(kvp => kvp.Key.Value != CR.Label.Value)
                .Prepend(CR.Label.Bind(dstName));
            var td = new RoleMappedData(view, roles);

            var predictor = trainer.Train(td);

            if (_args.useProbabilities)
            {
                var calSett = ScikitSubComponent<ICalibratorTrainer, SignatureCalibrator>.AsSubComponent(_args.calibratorType);
                var calibrator = calSett.CreateInstance(Host);
                var res = CalibratorUtils.TrainCalibratorIfNeeded(Host, ch,
                                        calibrator, _args.maxCalibrationExamples,
                                        trainer, predictor, td);
                predictor = res as TScalarPredictor;
                Host.Check(predictor != null, "Calibrated predictor does not implement the expected interface");
            }
            return predictor;
        }

        private IDataView FilterNA(IDataView view, string label)
        {
            if (_args.dropNALabel)
            {
                var args = new NAFilter.Arguments { Columns = new[] { label } };
                return new NAFilter(Host, args, view);
            }
            else
                return view;
        }

        private IDataView MapLabels(RoleMappedData data, int cls, out string dstName, IChannel ch)
        {
            var lab = data.Schema.Label.Value;
            Host.Assert(!data.Schema.Schema[lab.Index].IsHidden);
            Host.Assert(lab.Type.GetKeyCount() > 0 || lab.Type == NumberDataViewType.Single || lab.Type == NumberDataViewType.Double);

            // Get the destination label column name.
            dstName = data.Schema.Schema.GetTempColumnName();

            // Key values are 1-based.
            if (lab.Type.GetKeyCount() > 0)
            {
                uint key = (uint)(cls + 1);
                if (_args.downsampling > 0)
                    return CreateTrainingView(data, key, 1f, -1f, 0f, NumberDataViewType.UInt32, NumberDataViewType.Single, ch);
                else
                    return LambdaColumnMapper.Create(Host, "LabelColumnMapper in oOVA (1)", FilterNA(data.Data, lab.Name),
                        lab.Name, dstName, NumberDataViewType.UInt32, NumberDataViewType.Single,
                        (in uint src, ref float dst) => { dst = src == key ? 1 : default(float); });
            }
            if (lab.Type == NumberDataViewType.Single)
            {
                float key = cls;
                if (_args.downsampling > 0)
                    return CreateTrainingView(data, key, 1f, -1f, 0f, NumberDataViewType.Single, NumberDataViewType.Single, ch);
                else
                    return LambdaColumnMapper.Create(Host, "LabelColumnMapper in oOVA (2)", FilterNA(data.Data, lab.Name),
                        lab.Name, dstName, NumberDataViewType.Single, NumberDataViewType.Single,
                        (in float src, ref float dst) => { dst = src == key ? 1 : default(float); });
            }
            if (lab.Type == NumberDataViewType.Double)
            {
                double key = cls;
                if (_args.downsampling > 0)
                    return CreateTrainingView(data, key, 1f, -1f, 0f, NumberDataViewType.Double, NumberDataViewType.Single, ch);
                else
                    return LambdaColumnMapper.Create(Host, "LabelColumnMapper in oOVA (3)", FilterNA(data.Data, lab.Name),
                        lab.Name, dstName, NumberDataViewType.Double, NumberDataViewType.Single,
                        (in double src, ref float dst) => { dst = src == key ? 1 : default(float); });
            }

            throw Host.ExceptNotSupp("Label column type is not supported by OVA: {0}", lab.Type);
        }

        IDataView CreateTrainingView<T1, T2>(RoleMappedData data, T1 cls, T2 one, T2 mone, T2 zero, DataViewType c1, DataViewType c2, IChannel ch)
            where T1 : IEquatable<T1>
            where T2 : IEquatable<T2>
        {
            var dstName = data.Schema.Schema.GetTempColumnName();
            var lab = data.Schema.Label.Value;
            T1 key = cls;
            var labelMapper = LambdaColumnMapper.Create<T1, T2>(Host, "LabelColumnMapper in oOVA (4)", FilterNA(data.Data, lab.Name),
                lab.Name, dstName, c1, c2,
                (in T1 src, ref T2 dst) =>
                {
                    dst = src.Equals(key) ? one : zero;
                });

            ch.Info("[OptimizedOVATrainer] downsampling classes != {0} by {1}%", key, _args.downsampling * 100);
            var rarg = new ResampleTransform.Arguments
            {
                column = dstName,
                cache = true,
                lambda = 1 - _args.downsampling,
                classValue = "0"
            };
            var resample = new ResampleTransform(Host, rarg, labelMapper);
            return resample;
        }

        TVectorPredictor CreatePredictor()
        {
            return OptimizedOVAPredictor.Create(Host, _args.useProbabilities, _predictors);
        }
    }
}
