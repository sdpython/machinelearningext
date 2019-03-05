// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Sweeper;
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

using TensorFlowTransformer = Microsoft.ML.Transforms.TensorFlowTransformer;


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
            res.Add(typeof(OneHotEncodingEstimator).Assembly);
            res.Add(typeof(FastTreeRankingTrainer).Assembly);
            res.Add(typeof(KMeansPlusPlusTrainer).Assembly);
            res.Add(typeof(LightGbmBinaryTrainer).Assembly);
            res.Add(typeof(PcaModelParameters).Assembly);
            res.Add(typeof(PredictionFunctionExtensions).Assembly);
            res.Add(typeof(TextFeaturizingEstimator).Assembly);
            res.Add(typeof(TensorFlowTransformer).Assembly);
            res.Add(typeof(TrainCommand).Assembly);
            res.Add(typeof(ICanSaveOnnx).Assembly);
            res.Add(typeof(OrdinaryLeastSquaresRegressionTrainer).Assembly);
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
