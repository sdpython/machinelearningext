﻿// See the LICENSE file in the project root for more information.
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Calibration;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// The transform is trainable: it must be trained on the training data.
    /// </summary>
    public interface ITrainableTransform
    {
        void Estimate();
    }

    /// <summary>
    /// Extended interface for trainers.
    /// </summary>
    public interface ITrainerExtended
    {
        /// <summary>
        /// Returns the inner trainer.
        /// </summary>
        ITrainer Trainer { get; }

        /// <summary>
        /// Returns the loading name of the trainer.
        /// </summary>
        string LoadName { get; }

        /// <summary>
        /// Trains a model.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="ch">channel</param>
        /// <param name="data">traing data</param>
        /// <param name="validData">validation data</param>
        /// <param name="calibrator">calibrator</param>
        /// <param name="maxCalibrationExamples">number of examples used to calibrate</param>
        /// <param name="cacheData">cache training data</param>
        /// <param name="inpPredictor">for continuous training, initial state</param>
        /// <returns>predictor</returns>
        IPredictor Train(IHostEnvironment env, IChannel ch, RoleMappedData data, RoleMappedData validData = null,
                         SubComponent<ICalibratorTrainer, SignatureCalibrator> calibrator = null, int maxCalibrationExamples = 0,
                         bool? cacheData = null, IPredictor inpPredictor = null);
    }

    public interface IPredictorExtended : IPredictor
    {
        /// <summary>
        /// Computes the prediction for a predictor.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="data">data + role</param>
        /// <returns></returns>
        IDataScorerTransform Predict(IHostEnvironment env, RoleMappedData data);
    }
}
