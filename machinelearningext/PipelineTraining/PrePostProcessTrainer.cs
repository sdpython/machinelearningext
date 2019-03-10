// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;

using PrePostProcessTrainer = Scikit.ML.PipelineTraining.PrePostProcessTrainer;

[assembly: LoadableClass(PrePostProcessTrainer.Summary, typeof(PrePostProcessTrainer), typeof(PrePostProcessTrainer.Arguments),
    new[] { typeof(SignatureTrainer) }, PrePostProcessTrainer.UserNameValue,
    PrePostProcessTrainer.LoadNameValue)]


namespace Scikit.ML.PipelineTraining
{
    using CR = RoleMappedSchema.ColumnRole;

    /// <summary>
    /// Append optional transforms to preprocess and/or postprocess a trainer.
    /// </summary>
    public class PrePostProcessTrainer : TrainerBase<IPredictor>
    {
        internal const string LoadNameValue = "PrePost";
        internal const string UserNameValue = "PPTSP";
        internal const string Summary = "Append optional transforms to preprocess and/or postprocess a trainer. " +
                                        "The pipeline will execute the following pipeline pret-pre-predictor-post.";

        /// <summary>
        /// Arguments passed
        /// </summary>
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Transform which preprocesses the data (can be null)", ShortName = "pre",
                SignatureType = typeof(SignatureDataTransform))]
            public IComponentFactory<IDataTransform> preType = null;

            [Argument(ArgumentType.Multiple, HelpText = "Transform which preprocesses the data before training and only before training (can be null)",
                ShortName = "pret", SignatureType = typeof(SignatureDataTransform))]
            public IComponentFactory<IDataTransform> preTrainType = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Cache the data preprocessed only for training (", ShortName = "c")]
            public bool cache = false;

            [Argument(ArgumentType.Multiple, HelpText = "Transform which postprocesses the data (can be null)", ShortName = "post",
                SignatureType = typeof(SignatureDataTransform))]
            public IComponentFactory<IDataTransform> postType = null;

            [Argument(ArgumentType.Multiple, HelpText = "Predictor", ShortName = "p",
                SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<ITrainer> predictorType = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output column for the predictor", ShortName = "col")]
            public string outputColumn = "output";
        }

        private readonly Arguments _args;
        private ITrainer _trainer;
        private IDataTransform _preProcess;
        private IDataTransform _postProcess;
        private IDataTransform _preTrainProcess;
        private IDataTransform _predictorAsTransform;
        private IPredictor _predictor;
        private string _outputColumn;
        private string _inputColumn;
        private bool _cache;

        public PrePostProcessTrainer(IHostEnvironment env, Arguments args) : base(env, LoadNameValue)
        {
            _args = args = args ?? new Arguments();
            Contracts.CheckValue(_args.predictorType, "predictorType", "Must specify a base learner type");
            Contracts.CheckValue(args.outputColumn, "outputColumn", "outputColumn cannot be empty");
            var tempSett = ScikitSubComponent<ITrainer, SignatureTrainer>.AsSubComponent(_args.predictorType);
            var temp = tempSett.CreateInstance(env);
            _trainer = temp as ITrainer;
            if (_trainer == null)
                env.Except(temp == null ? "Trainer cannot be cast: {0}." : "Trainer cannot be instantiated: {0}.", _args.predictorType);
            _preProcess = null;
            _postProcess = null;
            _preTrainProcess = null;
            _predictorAsTransform = null;
            _predictor = null;
            _outputColumn = _args.outputColumn;
            _cache = args.cache;
        }

        public override TrainerInfo Info => new TrainerInfo(false, false, _cache);
        public override PredictionKind PredictionKind { get { Contracts.Assert(_trainer != null); return _trainer.PredictionKind; } }

        protected override IPredictor Train(TrainContext ctx)
        {
            var data = ctx.TrainingSet;
            Contracts.CheckValue(data, "data");
            Contracts.CheckValue(_trainer, "_trainer");

            IDataView view = data.Data;

            // Preprocess only for training.
            if (_args.preTrainType != null)
            {
                using (var ch2 = Host.Start("PreProcessTraining"))
                {
                    ch2.Info("Applies a preprocess only for training: {0}", _args.preTrainType);
                    var trSett = ScikitSubComponent<IDataTransform, SignatureDataTransform>.AsSubComponent(_args.preTrainType);
                    _preTrainProcess = trSett.CreateInstance(Host, view);
                }
                view = _preTrainProcess;
            }

            // Preprocess.
            if (_args.preType != null)
            {
                using (var ch2 = Host.Start("PreProcess"))
                {
                    ch2.Info("Applies a preprocess: {0}", _args.preType);
                    var trSett = ScikitSubComponent<IDataTransform, SignatureDataTransform>.AsSubComponent(_args.preType);
                    _preProcess = trSett.CreateInstance(Host, view);
                }
            }
            else
                _preProcess = new PassThroughTransform(Host, new PassThroughTransform.Arguments { }, view);
            view = _preProcess;

            // New RoleDataMapping
            var roles = data.Schema.GetColumnRoleNames()
                .Where(kvp => kvp.Key.Value != CR.Feature.Value)
                .Where(kvp => kvp.Key.Value != CR.Group.Value)
                .Where(kvp => kvp.Key.Value != CR.Label.Value)
                .Where(kvp => kvp.Key.Value != CR.Name.Value)
                .Where(kvp => kvp.Key.Value != CR.Weight.Value);
            if (data.Schema.Feature != null)
                roles = roles.Prepend(CR.Feature.Bind(data.Schema.Feature.Value.Name));
            if (data.Schema.Group != null)
                roles = roles.Prepend(CR.Group.Bind(data.Schema.Group.Value.Name));
            if (data.Schema.Label != null)
                roles = roles.Prepend(CR.Label.Bind(data.Schema.Label.Value.Name));
            if (data.Schema.Weight != null)
                roles = roles.Prepend(CR.Weight.Bind(data.Schema.Weight.Value.Name));
            var td = new RoleMappedData(view, roles);

            // Train.
            if (_args.predictorType != null)
            {
                using (var ch2 = Host.Start("Training"))
                {
                    var sch1 = SchemaHelper.ToString(data.Schema.Schema);
                    var sch2 = SchemaHelper.ToString(td.Schema.Schema);
                    ch2.Info("Initial schema: {0}", sch1);
                    ch2.Info("DataViewSchema before training: {0}", sch2);
                    ch2.Info("Train a predictor: {0}", _args.predictorType);
                    _predictor = _trainer.Train(td);
                }
            }

            // Predictor as a transform.
            {
                using (var ch2 = Host.Start("Predictor as Transform"))
                {
                    ch2.Info("Creates a transfrom from a predictor");
                    _inputColumn = td.Schema.Feature.Value.Name;
                    _predictorAsTransform = new TransformFromValueMapper(Host, _predictor as IValueMapper,
                                                              view, td.Schema.Feature.Value.Name, _outputColumn);
                }
                view = _predictorAsTransform;
            }

            // Postprocess.
            if (_args.postType != null)
            {
                using (var ch2 = Host.Start("PostProcess"))
                {
                    ch2.Info("Applies a postprocess: {0}", _args.postType);
                    var postSett = ScikitSubComponent<IDataTransform, SignatureDataTransform>.AsSubComponent(_args.postType);
                    _postProcess = postSett.CreateInstance(Host, view);
                }
            }
            else
                _postProcess = null;
            return CreatePredictor();
        }

        IPredictor CreatePredictor()
        {
            if (_preProcess == null)
                throw Host.Except("preProcess should not be null even if it is a PassThroughTransform.");
            return new PrePostProcessPredictor(Host, _preProcess, _predictor, _inputColumn, _outputColumn, _postProcess);
        }
    }
}
