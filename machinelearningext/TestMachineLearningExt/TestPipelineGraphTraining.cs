// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineGraphTraining
    {
        [TestMethod]
        public void TrainTestSingleTrainingPipelineWithTags()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var pipe = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                pipe = env.CreateTransform("SplitTrainTest{col=base tag=train tag=test}", pipe);
                pipe = env.CreateTransform("SelectTag{tag=unused selectTag=train}", pipe);
                pipe = env.CreateTransform(string.Format("TagTrainScore{{tag=trainP out={0} tr=mlr}}", outModelFilePath), pipe);
                pipe = env.CreateTransform("SelectTag{tag=scoredTrain selectTag=test}", pipe);
                pipe = env.CreateTransform("TagScore{in=trainP}", pipe);

                var cursor = pipe.GetRowCursor(pipe.Schema);
                string schema = SchemaHelper.ToString(pipe.Schema);
                string schema2 = SchemaHelper.ToString(cursor.Schema);
                if (schema != schema2)
                    throw new Exception("DataViewSchema mismatch.");
                long count = DataViewUtils.ComputeRowCount(pipe);
                if (count != 49)
                    throw new Exception(string.Format("Unexpected number of rows {0}", count));

                // Checks the outputs.
                var saver = env.CreateSaver("Text");
                var columns = new string[pipe.Schema.Count];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = pipe.Schema[i].Name;
                using (var fs2 = File.Create(outData))
                    saver.SaveData(fs2, pipe, StreamHelper.GetColumnsIndex(pipe.Schema));

                var lines = File.ReadAllLines(outData);
                if (lines.Length < 40)
                    throw new Exception("Something is missing:" + string.Join("\n", lines));
                if (lines.Length > 70)
                    throw new Exception("Too much data:" + string.Join("\n", lines));

                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, pipe, loader, outData, outData2);
            }
        }

        [TestMethod]
        public void TrainTestPipelinePredictTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var pipe = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                pipe = env.CreateTransform("SplitTrainTest{col=base tag=train tag=test}", pipe);
                pipe = env.CreateTransform("SelectTag{tag=unused selectTag=train}", pipe);
                pipe = env.CreateTransform(string.Format("TagTrainScore{{tag=trainP out={0} tr=mlr}}", outModelFilePath), pipe);
                pipe = env.CreateTransform("SelectTag{tag=scoredTrain selectTag=test}", pipe);
                pipe = env.CreateTransform("TagPredict{in=trainP}", pipe);

                string schema = SchemaHelper.ToString(pipe.Schema);
                var cursor = pipe.GetRowCursor(pipe.Schema);
                string schema2 = SchemaHelper.ToString(cursor.Schema);
                if (schema != schema2)
                    throw new Exception("DataViewSchema mismatch.");
                long count = DataViewUtils.ComputeRowCount(pipe);
                if (count != 49)
                    throw new Exception(string.Format("Unexpected number of rows {0}", count));

                // Checks the outputs.
                var saver = env.CreateSaver("Text");
                var columns = new string[pipe.Schema.Count];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = pipe.Schema[i].Name;
                using (var fs2 = File.Create(outData))
                    saver.SaveData(fs2, pipe, StreamHelper.GetColumnsIndex(pipe.Schema));

                var lines = File.ReadAllLines(outData);
                if (lines.Length < 40)
                    throw new Exception("Something is missing:" + string.Join("\n", lines));
                if (lines.Length > 70)
                    throw new Exception("Too much data:" + string.Join("\n", lines));

                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, pipe, loader, outData, outData2);
            }
        }  
    }
}
