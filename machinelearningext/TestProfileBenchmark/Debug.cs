

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
//using Microsoft.ML.Ensemble;
using Microsoft.ML.Trainers;
using Microsoft.ML.Model.OnnxConverter;
//using Microsoft.ML.Sweeper;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML.Tools;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
//using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Transforms;
using Scikit.ML.Clustering;
using Scikit.ML.FeaturesTransforms;
using Scikit.ML.ModelSelection;
using Scikit.ML.Multiclass;
using Scikit.ML.NearestNeighbors;
using Scikit.ML.OnnxHelper;
using Scikit.ML.PipelineGraphTraining;
using Scikit.ML.PipelineGraphTransforms;
using Scikit.ML.PipelineLambdaTransforms;
using Scikit.ML.PipelineTraining;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.RandomTransforms;
using Scikit.ML.TimeSeries;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DataManipulation;


namespace TestProfileBenchmark

{
    public static class DynamicCSFunctions_example_diabetes
    {
        public class TrainTestDiabetesRF
        {
            string _dataset;
            ScikitPipeline _pipeline;

            public TrainTestDiabetesRF(string ds)
            {
                _dataset = ds;
            }

            public void Train()
            {
                /*using (*/
                var env = new ConsoleEnvironment();
                {
                    var df = DataFrameIO.ReadCsv(_dataset, sep: ',',
                                                 dtypes: new DataViewType[] { NumberDataViewType.Single });
                    var concat = "Concat{col=Features:F0,F1,F2,F3,F4,F5,F6,F7,F8,F9}";
                    var pipe = new ScikitPipeline(new[] { concat }, "ftr{iter=10}");
                    pipe.Train(df, "Features", "Label");
                    _pipeline = pipe;
                }
            }

            public DataFrame Predict(double[] features)
            {
                DataFrame pred = null;
                var df = new DataFrame();
                df.AddColumn("Label", new float[] { 0f });
                for (int i = 0; i < features.Length; ++i)
                    df.AddColumn(string.Format("F{0}", i), new float[] { (float)features[i] });
                _pipeline.Predict(df, ref pred);
                return pred;
            }

            public DataFrame PredictBatch(int nf, double[] features)
            {
                DataFrame pred = null;
                var df = new DataFrame();
                int N = features.Length / nf;
                df.AddColumn("Label", Enumerable.Range(0, N).Select(i => (float)features[nf * i]).ToArray());
                for (int i = 0; i < nf; ++i)
                    df.AddColumn(string.Format("F{0}", i),
                                 Enumerable.Range(0, N).Select(k => (float)features[nf * k + i]).ToArray());
                _pipeline.Predict(df, ref pred);
                return pred;
            }

            public void Read(string name)
            {
                _pipeline = new ScikitPipeline(name);
            }

            public void Save(string name)
            {
                _pipeline.Save(name, true);
            }
        }

        public static TrainTestDiabetesRF ReturnMLClassRF(string ds)
        {
            return new TrainTestDiabetesRF(ds);
        }
    }
}
