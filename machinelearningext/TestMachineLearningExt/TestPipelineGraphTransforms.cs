// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineGraphTransforms
    {
        #region Chain, SelectTag, Tagged

        [TestMethod]
        public void TestTagViewTransform()
        {
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var inputs = new[] {
                    new ExampleA() { X = new float[] { 0, 1 } },
                    new ExampleA() { X = new float[] { 2, 3 } }
                };

                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                var data = host.CreateTransform("Scaler{col=X1:X}", loader);
                data = host.CreateTransform("tag{t=memory}", data);

                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
                var outData = FileHelper.GetOutputFile("outData.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
            }
        }

        [TestMethod]
        public void TestSelectTagContactViewTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var firstData = FileHelper.GetOutputFile("first.idv", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment())
            {
                var inputs = new[] {
                    new ExampleA() { X = new float[] { 0, 1, 4 } },
                    new ExampleA() { X = new float[] { 2, 3, 7 } }
                };

                // Create IDV
                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(env, inputs);
                var saver = ComponentCreation.CreateSaver(env, "binary");
                using (var ch = env.Start("save"))
                {
                    using (var fs0 = env.CreateOutputFile(firstData))
                        DataSaverUtils.SaveDataView(ch, saver, loader, fs0, true);

                    // Create parallel pipeline
                    loader = DataViewConstructionUtils.CreateFromEnumerable(env, inputs);
                    var data = env.CreateTransform("Scaler{col=X1:X}", loader);
                    data = env.CreateTransform(string.Format("selecttag{{t=first s=second f={0}}}", firstData), data);
                    data = env.CreateTransform("Scaler{col=X1:X}", data);
                    var merged = env.CreateTransform("append{t=first}", data);

                    // Save the outcome
                    var text = env.CreateSaver("Text");
                    var columns = new int[merged.Schema.Count];
                    for (int i = 0; i < columns.Length; ++i)
                        columns[i] = i;
                    using (var fs2 = File.Create(outData))
                        text.SaveData(fs2, merged, columns);

                    // Final checking
                    var lines = File.ReadAllLines(outData);
                    if (!lines.Any())
                        throw new Exception("Empty file.");
                    if (lines.Length != 9)
                        throw new Exception("Some lines are missing.");
                }
            }
        }

        [TestMethod]
        public void TestChainTransform()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            using (var host = EnvHelper.NewTestEnvironment()) { 
                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
            var chained = host.CreateTransform("ChainTrans{ xf1=Scaler{col=X2:X} xf2=Scaler{col=X3:X2} }", loader);
            var schStr = SchemaHelper.ToString(chained.Schema);
            Assert.AreEqual(schStr, "X:Vec<R4,3>:0; X2:Vec<R4,3>:1; X3:Vec<R4,3>:2");

                using (var cursor = chained.GetRowCursor(chained.Schema))
                {
                    var outValues0 = new List<float[]>();
                    var outValues1 = new List<float[]>();
                    var outValues2 = new List<float[]>();
                    var colGetter0 = cursor.GetGetter<VBuffer<float>>(0);
                    var colGetter1 = cursor.GetGetter<VBuffer<float>>(1);
                    var colGetter2 = cursor.GetGetter<VBuffer<float>>(2);
                    while (cursor.MoveNext())
                    {
                        VBuffer<float> got = new VBuffer<float>();
                        colGetter0(ref got);
                        outValues0.Add(got.DenseValues().ToArray());
                        colGetter1(ref got);
                        outValues1.Add(got.DenseValues().ToArray());
                        colGetter2(ref got);
                        outValues2.Add(got.DenseValues().ToArray());
                    }
                    if (outValues0.Count != 2)
                        throw new Exception("expected 2");
                    if (outValues1.Count != 2)
                        throw new Exception("expected 2");
                    if (outValues2.Count != 2)
                        throw new Exception("expected 2");
                    for (int i = 0; i < outValues0.Count; ++i)
                    {
                        if (outValues0[i].Length != outValues1[i].Length)
                            throw new Exception("mismatch");
                        if (outValues2[i].Length != outValues1[i].Length)
                            throw new Exception("mismatch");
                        for (int j = 0; j < outValues0[i].Length; ++j)
                        {
                            if (outValues0[i][j] == outValues1[i][j])
                                throw new Exception("mismatch");
                            if (outValues2[i][j] != outValues1[i][j])
                                throw new Exception("mismatch");
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void TestChainTransformSerialize()
        {
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var inputs = new[] {
                    new ExampleA() { X = new float[] { 1, 10, 100 } },
                    new ExampleA() { X = new float[] { 2, 3, 5 } }
                };

                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                IDataTransform data = host.CreateTransform("Scaler{col=X4:X}", loader);
                data = host.CreateTransform("ChainTrans{ xf1=Scaler{col=X2:X} xf2=Poly{col=X3:X2} }", data);

                // We create a specific folder in build/UnitTest which will contain the output.
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
                var outData = FileHelper.GetOutputFile("outData.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
            }
        }

        [TestMethod]
        public void TestChainTransformSerializeWithKMeans()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris_binary.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment())
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Features:R4:1-4 header=+}",
                                    new MultiFileSource(dataFilePath));

                var xf = env.CreateTransform("ChainTrans{xf1=Scaler{col=Features} xf2=Scaler{col=Features}}", loader);
                var roles = env.CreateExamples(xf, "Features");
                var trainer = env.CreateTrainer("km{k=5}");
                using (var ch = env.Start("Train"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                         PredictionKind.Clustering, false);
                }
            }
        }

        #endregion

        #region TagTrainOrScoreTransform

        [TestMethod]
        public void TestTagTrainOrScoreTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment())
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=-}",
                    new MultiFileSource(dataFilePath));

                using (var pipe = new ScikitPipeline(new[] {
                        "Concat{col=Feature:Slength,Swidth}",
                        "TagTrainScore{tr=iova{p=ft{nl=10 iter=1}} lab=Label feat=Feature tag=model}" }, host: env))
                {
                    pipe.Train(loader);
                    var pred = pipe.Predict(loader);
                    var df = DataFrameIO.ReadView(pred);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 11));
                    var dfs = df.Head().ToString();
                    Assert.IsTrue(dfs.StartsWith("Label,Slength,Swidth,Plength,Pwidth,Feature.0,Feature.1,PredictedLabel,Score.0,Score.1,Score.2"));
                }
            }
        }

        [TestMethod]
        public void TestTagTrainOrScoreTransformCustomScorer()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment())
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=-}",
                    new MultiFileSource(dataFilePath));

                using (var pipe = new ScikitPipeline(new[] {
                        "Concat{col=Feature:Slength,Swidth}",
                        "TagTrainScore{tr=iova{p=ft{nl=10 iter=1}} lab=Label feat=Feature tag=model scorer=MultiClassClassifierScorer{ex=AA}}" }, host: env))
                {
                    pipe.Train(loader);
                    var pred = pipe.Predict(loader);
                    var df = DataFrameIO.ReadView(pred);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 11));
                    var dfs = df.Head().ToString();
                    Assert.IsTrue(dfs.StartsWith("Label,Slength,Swidth,Plength,Pwidth,Feature.0,Feature.1,PredictedLabelAA,ScoreAA.0,ScoreAA.1,ScoreAA.2"));
                }
            }
        }

        #endregion
    }
}
