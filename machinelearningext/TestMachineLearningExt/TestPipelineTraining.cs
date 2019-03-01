// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineTraining
    {
        #region none

        static void TrainPrePostProcessTrainer(string modelName, bool checkError, int threads, bool addpre)
        {
            var methodName = string.Format("{0}-{1}-T{2}", System.Reflection.MethodBase.GetCurrentMethod().Name, modelName, threads);
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: threads == 1 ? 1 : 0);
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));
                var xf = env.CreateTransform("shuffle{force=+}", loader); // We shuffle because Iris is order by label.
                xf = env.CreateTransform("concat{col=RawFeatures:Slength,Swidth}", xf);
                var roles = env.CreateExamples(xf, "RawFeatures", "Label");

                string pred = addpre ? "PrePost{pre=poly{col=Feature:RawFeatures} p=___ pret=Take{n=80}}" : "PrePost{p=___ pret=Take{n=80}}";
                pred = pred.Replace("___", modelName);
                var trainer = env.CreateTrainer(pred);
                using (var ch = env.Start("Train"))
                {
                    var predictor = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                                PredictionKind.MultiClassClassification, checkError, ratio: 0.15f);
                }
            }
        }

        [TestMethod]
        public void TrainPrePostProcessTrainerPre()
        {
            TrainPrePostProcessTrainer("mlr", true, 1, true);
        }

        [TestMethod]
        public void TrainPrePostProcessTrainerNoPre()
        {
            TrainPrePostProcessTrainer("mlr", true, 1, false);
        }

        #endregion
    }
}
