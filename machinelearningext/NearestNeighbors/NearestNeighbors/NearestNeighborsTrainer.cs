// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Scikit.ML.PipelineHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;


namespace Scikit.ML.NearestNeighbors
{
    public interface INearestNeighborsPredictor : IPredictor
    {
    }

    /// <summary>
    /// Train a MultiToBinary predictor. It multiplies the rows by the number of classes to predict.
    /// (multi class problem).
    /// </summary>
    public abstract class NearestNeighborsTrainer : TrainerBase<INearestNeighborsPredictor>
    {
        #region parameters / command line

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments : NearestNeighborsArguments
        {
            [Argument(ArgumentType.Required, ShortName = "data", HelpText = "The data to be used for training", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public IDataView TrainingData;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for features", ShortName = "feat",
                      Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels", ShortName = "lab",
                      Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string WeightColumn = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize option for the feature column", ShortName = "norm",
                      Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

            [Argument(ArgumentType.LastOccurrenceWins, HelpText = "Whether learner should cache input training data", ShortName = "cache",
                      Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public CachingOptions Caching = CachingOptions.Auto;
        }

        #endregion

        #region internal members / accessors

        protected readonly Arguments _args;
        private INearestNeighborsPredictor _predictor;

        public override PredictionKind PredictionKind { get { return _predictor.PredictionKind; } }

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

        /// <summary>
        /// Create a NearestNeighborsTrainer transform.
        /// </summary>
        public NearestNeighborsTrainer(IHostEnvironment env, Arguments args, string loaderSignature)
            : base(env, loaderSignature)
        {
            Host.CheckValue(args, "args");
            Host.Check(args.k > 0, "k must be > 0.");
            _args = args;
        }

        protected override INearestNeighborsPredictor Train(TrainContext context) 
        {
            return Train(context.TrainingSet);
        }

        protected virtual INearestNeighborsPredictor Train(RoleMappedData data)
        {
            Contracts.CheckValue(data, "data");
            data.CheckFeatureFloatVector();

            using (var ch = Host.Start("Training kNN"))
            {
                // Train one-vs-all models.
                _predictor = TrainPredictor(ch, data);
            }

            return _predictor;
        }

        /// <summary>
        /// Train the predictor.
        /// </summary>
        protected INearestNeighborsPredictor TrainPredictor(IChannel ch, RoleMappedData data)
        {
            var labType = data.Schema.Label.Value.Type;
            var initialLabKind = labType.RawKind();
            INearestNeighborsPredictor predictor;

            switch (initialLabKind)
            {
                case DataKind.Boolean:
                    predictor = TrainPredictorLabel<bool>(ch, data);
                    break;
                case DataKind.Single:
                    predictor = TrainPredictorLabel<float>(ch, data);
                    break;
                case DataKind.SByte:
                    predictor = TrainPredictorLabel<byte>(ch, data);
                    break;
                case DataKind.UInt16:
                    predictor = TrainPredictorLabel<ushort>(ch, data);
                    break;
                case DataKind.UInt32:
                    predictor = TrainPredictorLabel<uint>(ch, data);
                    break;
                default:
                    throw ch.ExceptNotSupp("Unsupported type for a label.");
            }

            Host.Assert(predictor != null);
            return predictor;
        }

        private INearestNeighborsPredictor TrainPredictorLabel<TLabel>(IChannel ch, RoleMappedData data)
            where TLabel : IComparable<TLabel>
        {
            int featureIndex = data.Schema.Feature.Value.Index;
            int labelIndex = data.Schema.Label.Value.Index;
            int idIndex = SchemaHelper.GetColumnIndex(data.Schema.Schema, _args.colId, true);
            int weightIndex = data.Schema.Weight == null ? -1 : data.Schema.Weight.Value.Index;
            var indexes = new HashSet<int>() { featureIndex, labelIndex, weightIndex };
            if (!string.IsNullOrEmpty(_args.colId) && idIndex != -1)
                indexes.Add(idIndex);
            if (idIndex != -1)
            {
                var colType = data.Schema.Schema[idIndex].Type;
                if (colType.IsVector() || colType.RawKind() != DataKind.Int64)
                    throw ch.Except("Column '{0}' must be of type '{1}' not '{2}'", _args.colId, DataKind.Int64, colType);
            }

            Dictionary<long, Tuple<TLabel, float>> merged;
            var kdtrees = NearestNeighborsBuilder.NearestNeighborsBuild<TLabel>(ch, data.Data, featureIndex, labelIndex,
                                idIndex, weightIndex, out merged, _args);

            // End.
            return CreateTrainedPredictor(kdtrees.Trees, merged);
        }

        protected virtual INearestNeighborsPredictor CreateTrainedPredictor<TLabel>(KdTree[] kdtrees,
            Dictionary<long, Tuple<TLabel, float>> labelsWeights)
            where TLabel : IComparable<TLabel>
        {
            throw new NotImplementedException("This function is different for each kind of classifier.");
        }

        #endregion
    }
}
