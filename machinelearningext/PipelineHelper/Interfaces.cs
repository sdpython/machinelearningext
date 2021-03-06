﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    public delegate DataViewRowCursor DelegateGetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand);

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
    internal interface ITrainerExtended
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
                         ICalibratorTrainer calibrator = null, int maxCalibrationExamples = 0,
                         bool? cacheData = null, IPredictor inpPredictor = null);
    }

    internal interface IPredictorExtended : IPredictor
    {
        /// <summary>
        /// Computes the prediction for a predictor.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="data">data + role</param>
        /// <returns>IDataScorerTransform</returns>
        IDataScorerTransform Predict(IHostEnvironment env, RoleMappedData data);
    }

    public abstract class ADataTransform : IDataTransformPropagateSingleThread, IDataTransformSource
    {
        protected IDataView _input = null;
        public IDataView Source => _input;

        public virtual bool SingleThread()
        {
            var tf = Source as TransformBase;
            if (tf != null)
                return tf.SingleThread();
            var tf2 = Source as IDataTransformPropagateSingleThread;
            if (tf2 != null)
                return tf2.SingleThread();
            return (Source as IDataViewSingleThreaded) != null;
        }
    }
}
