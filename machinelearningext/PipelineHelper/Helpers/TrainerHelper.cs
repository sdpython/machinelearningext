// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Wrapper for a trainer with
    /// extra functionalities.
    /// </summary>
    public class ExtendedTrainer : ITrainerExtended, ITrainer
    {
        ITrainer _trainer;
        string _loadName;

        #region ITrainer API

        public string LoadName => _loadName;
        public ITrainer Trainer => _trainer;
        public PredictionKind PredictionKind => _trainer.PredictionKind;
        public TrainerInfo Info => _trainer.Info;

        #endregion

        public ExtendedTrainer(ITrainer trainer, string loadName)
        {
            _loadName = loadName;
            _trainer = trainer;
        }

        /// <summary>
        /// Create a trainer.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="settings">trainer description as a string such <pre>ova{p=lr}</pre></param>
        /// <param name="extraArgs">additional arguments</param>
        public static ITrainerExtended CreateTrainer(IHostEnvironment env, string settings, params object[] extraArgs)
        {
            var sc = ScikitSubComponent.Parse<ITrainer, SignatureTrainer>(settings);
            var inst = sc.CreateInstance(env, extraArgs);
            return new ExtendedTrainer(inst, sc.Kind);
        }

        public IPredictor Train(TrainContext context)
        {
            return _trainer.Train(context);
        }

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
        /// <param name="inputPredictor">for continuous training, initial state</param>
        /// <returns>predictor</returns>
        public IPredictor Train(IHostEnvironment env, IChannel ch, RoleMappedData data, RoleMappedData validData = null,
                                ICalibratorTrainer calibrator = null, int maxCalibrationExamples = 0,
                                bool? cacheData = null, IPredictor inputPredictor = null)
        {
            /*
            return TrainUtils.Train(env, ch, data, Trainer, LoadName, validData, calibrator, maxCalibrationExamples,
                                    cacheData, inpPredictor);
                                    */

            var trainer = Trainer;
            var name = LoadName;

            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(data, nameof(data));
            ch.CheckValue(trainer, nameof(trainer));
            ch.CheckNonEmpty(name, nameof(name));
            ch.CheckValueOrNull(validData);
            ch.CheckValueOrNull(inputPredictor);

            AddCacheIfWanted(env, ch, trainer, ref data, cacheData);
            ch.Trace(MessageSensitivity.None, "Training");
            if (validData != null)
                AddCacheIfWanted(env, ch, trainer, ref validData, cacheData);

            if (inputPredictor != null && !trainer.Info.SupportsIncrementalTraining)
            {
                ch.Warning(MessageSensitivity.None, "Ignoring " + nameof(TrainCommand.Arguments.InputModelFile) +
                    ": Trainer does not support incremental training.");
                inputPredictor = null;
            }
            ch.Assert(validData == null || trainer.Info.SupportsValidation);
            var predictor = trainer.Train(new TrainContext(data, validData, null, inputPredictor));
            return CalibratorUtils.TrainCalibratorIfNeeded(env, ch, calibrator, maxCalibrationExamples, trainer, predictor, data);
        }

        public static bool AddCacheIfWanted(IHostEnvironment env, IChannel ch, ITrainer trainer, ref RoleMappedData data, bool? cacheData)
        {
            Contracts.AssertValue(env, nameof(env));
            env.AssertValue(ch, nameof(ch));
            ch.AssertValue(trainer, nameof(trainer));
            ch.AssertValue(data, nameof(data));

            bool shouldCache = cacheData ?? !(data.Data is BinaryLoader) && trainer.Info.WantCaching;

            if (shouldCache)
            {
                ch.Trace(MessageSensitivity.None, "Caching");
                var prefetch = data.Schema.GetColumnRoles().Select(kc => kc.Value.Index).ToArray();
                var cacheView = new CacheDataView(env, data.Data, prefetch);
                // Because the prefetching worked, we know that these are valid columns.
                data = new RoleMappedData(cacheView, data.Schema.GetColumnRoleNames());
            }
            else
                ch.Trace(MessageSensitivity.None, "Not caching");
            return shouldCache;
        }
    }

    public static class TrainerHelper
    {
        /// <summary>
        /// Create a trainer.
        /// </summary>
        /// <param name="env">host</param>
        /// <param name="settings">trainer description as a string such <pre>ova{p=lr}</pre></param>
        /// <param name="extraArgs">additional arguments</param>
        public static ITrainerExtended CreateTrainer(this IHostEnvironment env, string settings, params object[] extraArgs)
        {
            return ExtendedTrainer.CreateTrainer(env, settings, extraArgs);
        }

        /// <summary>
        /// Returns all base types for a given type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static IEnumerable<Type> GetParentTypes(Type type)
        {
            // is there any base type?
            if (type == null)
                yield break;

            // return all implemented or inherited interfaces
            foreach (var i in type.GetInterfaces())
                yield return i;

            // return all inherited types
            var currentBaseType = type.BaseType;
            while (currentBaseType != null)
            {
                yield return currentBaseType;
                currentBaseType = currentBaseType.BaseType;
            }
        }
    }
}
