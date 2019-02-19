// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestQuestion
    {
        public class IrisObservation
        {
            [Column("0")]
            [ColumnName("Label")]
            public string Label;

            [Column("1")]
            public float Sepal_length;

            [Column("2")]
            public float Sepal_width;

            [Column("3")]
            public float Petal_length;

            [Column("4")]
            public float Petal_width;
        }

        public class IrisPrediction
        {
            public uint PredictedLabel;

            [VectorType(4)]
            public float[] Score;
        }     

        public static ITrainer CreateTrainer(IHostEnvironment env, string settings, params object[] extraArgs)
        {
            var sc = ScikitSubComponent.Parse<ITrainer, SignatureTrainer>(settings);
            var inst = sc.CreateInstance(env, extraArgs);
            return inst;
        }

        [TestMethod]
        public void TestI_Q_KMeansInnerAPIWithDataFrame()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var iris = FileHelper.GetTestFile("iris.txt");
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                ComponentHelper.AddStandardComponents(env);

                var df = Scikit.ML.DataManipulation.DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new DataViewType[] { NumberDataViewType.Single });
                var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
                var roleMap = env.CreateExamples(conc, "Feature", label: "Label");
                var trainer = CreateTrainer(env, "km");
                IPredictor model;
                using (var ch = env.Start("test"))
                    model = TrainUtils.Train(env, ch, roleMap, trainer, null, 0);

                using (var ch = env.Start("Save"))
                using (var fs = File.Create(outModelFilePath))
                    TrainUtils.SaveModel(env, ch, fs, model, roleMap);

                IPredictor ipred;
                using (var fs = File.OpenRead(outModelFilePath))
                    ipred = ModelFileUtils.LoadPredictorOrNull(env, fs);

                var scorer = ScoreUtils.GetScorer(ipred, roleMap, env, null);
                var dfout = Scikit.ML.DataManipulation.DataFrameIO.ReadView(scorer);
                Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 13));
            }
        }
    }
}
