﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Runtime.Sweeper;
using Google.Protobuf;
using Scikit.ML.Clustering;
using Scikit.ML.DataManipulation;
using Scikit.ML.TimeSeries;
using Scikit.ML.FeaturesTransforms;
using Scikit.ML.PipelineLambdaTransforms;
using Scikit.ML.ModelSelection;
using Scikit.ML.MultiClass;
using Scikit.ML.NearestNeighbors;
using Scikit.ML.OnnxHelper;
using Scikit.ML.PipelineGraphTraining;
using Scikit.ML.PipelineGraphTransforms;
using Scikit.ML.PipelineTraining;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.RandomTransforms;

using TensorFlowTransform = Microsoft.ML.Transforms.TensorFlowTransform;


namespace Scikit.ML.ScikitAPI
{
    public static class ComponentHelper
    {
        /// <summary>
        /// Register one assembly.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="a">assembly</param>
        public static void AddComponent(IHostEnvironment env, Assembly a)
        {
            try
            {
                env.ComponentCatalog.RegisterAssembly(a);
            }
            catch (Exception e)
            {
                throw new Exception($"Unable to register assembly '{a.FullName}' due to '{e}'.");
            }
        }

        public static Assembly[] GetAssemblies()
        {
            var res = new List<Assembly>();
            res.Add(typeof(TextLoader).Assembly);
            res.Add(typeof(LinearModelStatistics).Assembly);
            res.Add(typeof(Categorical).Assembly);
            res.Add(typeof(FastTreeRankingTrainer).Assembly);
            res.Add(typeof(EnsemblePredictor).Assembly);
            res.Add(typeof(KMeansPlusPlusTrainer).Assembly);
            res.Add(typeof(LightGbm).Assembly);
            res.Add(typeof(OlsLinearRegressionTrainer).Assembly);
            res.Add(typeof(PcaPredictor).Assembly);
            res.Add(typeof(SlidingWindowTransform).Assembly);
            res.Add(typeof(TextFeaturizingEstimator).Assembly);
            res.Add(typeof(TensorFlowTransform).Assembly);
            res.Add(typeof(TrainCommand).Assembly);
            res.Add(typeof(ICanSaveOnnx).Assembly);
            res.Add(typeof(SweeperBase).Assembly);
            res.Add(typeof(VectorTypeAttribute).Assembly);
            res.Add(typeof(JsonParser).Assembly);
            // ext
            res.Add(typeof(DataFrame).Assembly);
            res.Add(typeof(DBScan).Assembly);
            res.Add(typeof(DeTrendTransform).Assembly);
            res.Add(typeof(PolynomialTransform).Assembly);
            res.Add(typeof(PredictTransform).Assembly);
            res.Add(typeof(NearestNeighborsBinaryClassificationTrainer).Assembly);
            res.Add(typeof(MultiToBinaryPredictor).Assembly);
            res.Add(typeof(TaggedPredictTransform).Assembly);
            res.Add(typeof(AppendViewTransform).Assembly);
            res.Add(typeof(PrePostProcessPredictor).Assembly);
            res.Add(typeof(PassThroughTransform).Assembly);
            res.Add(typeof(ResampleTransform).Assembly);
            res.Add(typeof(SplitTrainTestTransform).Assembly);
            res.Add(typeof(ValueMapperPredictionEngineFloat).Assembly);
            res.Add(typeof(Convert2Onnx).Assembly);
            res.Add(typeof(ScikitPipeline).Assembly);
            return res.ToArray();
        }

        /// <summary>
        /// Register standard assemblies from Microsoft.ML and Scikit.ML.
        /// </summary>
        /// <param name="env">environment</param>
        public static void AddStandardComponents(IHostEnvironment env)
        {
            var res = GetAssemblies();
            foreach (var a in res)
                AddComponent(env, a);
        }
    }
}
