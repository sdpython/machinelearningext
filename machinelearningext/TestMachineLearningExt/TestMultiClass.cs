// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;
using Scikit.ML.Multiclass;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestMulticlass
    {
        #region MultiToBinary transform

        public static void TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm algo, int max)
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new InputOutputU[] {
                    new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                    new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                    new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new MultiToBinaryTransform.Arguments { label = "Y", algo = algo, maxMulti = max };
                var multiplied = new MultiToBinaryTransform(host, args, data);

                using (var cursor = multiplied.GetRowCursor(multiplied.Schema))
                {
                    var labelGetter = cursor.GetGetter<uint>(SchemaHelper._dc(1, cursor));
                    var binGetter = cursor.GetGetter<bool>(SchemaHelper._dc(2, cursor));
                    var cont = new List<Tuple<uint, bool>>();
                    bool bin = false;
                    while (cursor.MoveNext())
                    {
                        uint got = 0;
                        labelGetter(ref got);
                        binGetter(ref bin);
                        cont.Add(new Tuple<uint, bool>(got, bin));
                    }

                    if (max >= 3)
                    {
                        if (cont.Count != 9)
                            throw new Exception("It should be 9.");
                        if (algo == MultiToBinaryTransform.MultiplicationAlgorithm.Default)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                var co = cont.Where(c => c.Item1 == (uint)i && c.Item2);
                                if (co.Count() != 1)
                                    throw new Exception(string.Format("Unexpected number of true labels for class {0} - algo={1} - max={2}", i, algo, max));
                            }
                        }
                    }
                    else
                    {
                        if (cont.Count != 3 * max)
                            throw new Exception(string.Format("It should be {0}.", 3 * max));
                    }
                }
            }
        }

        public static void TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm algo, int max)
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new InputOutputU[] {
                    new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                    new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                    new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new MultiToBinaryTransform.Arguments { label = "Y", algo = algo, maxMulti = max };
                var multiplied = new MultiToBinaryTransform(host, args, data);

                using (var cursor = multiplied.GetRowCursor(multiplied.Schema))
                {
                    var labelGetter = cursor.GetGetter<uint>(SchemaHelper._dc(1, cursor));
                    var labelVectorGetter = cursor.GetGetter<VBuffer<bool>>(SchemaHelper._dc(1, cursor));
                    var labelVectorFloatGetter = cursor.GetGetter<VBuffer<float>>(SchemaHelper._dc(1, cursor));
                    var binGetter = cursor.GetGetter<bool>(SchemaHelper._dc(2, cursor));
                    Contracts.CheckValue(binGetter, "Type mismatch.");
                    var cont = new List<Tuple<uint, bool>>();
                    bool bin = false;
                    uint got = 0;
                    var gotv = new VBuffer<bool>();
                    var gotvf = new VBuffer<float>();
                    while (cursor.MoveNext())
                    {
                        labelGetter(ref got);
                        labelVectorGetter(ref gotv);
                        labelVectorFloatGetter(ref gotvf);
                        binGetter(ref bin);
                        cont.Add(new Tuple<uint, bool>(got, bin));
                        if (gotv.Length != 3) throw new Exception("Bad dimension (Length)");
                        if (gotv.Count != 1) throw new Exception("Bad dimension (Count)");
                        if (!gotv.Values[0]) throw new Exception("Bad value (Count)");
                        if (gotv.Indices[0] != got) throw new Exception("Bad index (Count)");
                        var ar = gotv.DenseValues().ToArray();
                        if (ar.Length != 3) throw new Exception("Bad dimension (dense)");

                        if (gotvf.Length != 3) throw new Exception("Bad dimension (Length)f");
                        if (gotvf.Count != 1) throw new Exception("Bad dimension (Count)f");
                        if (gotvf.Values[0] != 1) throw new Exception("Bad value (Count)f");
                        if (gotvf.Indices[0] != got) throw new Exception("Bad index (Count)f");
                        var ar2 = gotv.DenseValues().ToArray();
                        if (ar2.Length != 3) throw new Exception("Bad dimension (dense)f");
                    }

                    if (max >= 3)
                    {
                        if (cont.Count != 9)
                            throw new Exception("It should be 9.");
                        if (algo == MultiToBinaryTransform.MultiplicationAlgorithm.Default)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                var co = cont.Where(c => c.Item1 == (uint)i && c.Item2);
                                if (co.Count() != 1)
                                    throw new Exception(string.Format("Unexpected number of true labels for class {0} - algo={1} - max={2}", i, algo, max));
                            }
                        }
                    }
                    else
                    {
                        if (cont.Count != 3 * max)
                            throw new Exception(string.Format("It should be {0}.", 3 * max));
                    }
                }
            }
        }

        [TestMethod]
        public void TestTransMultiToBin()
        {
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 5);
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 2);
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 5);
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 2);
        }

        [TestMethod]
        public void TestTransMultiToBinVector()
        {
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 5);
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 2);
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 5);
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 2);
        }

        #endregion

        #region MultiToBinary Predictors

        static void TrainMultiToBinaryPredictorDense(string modelName, int threads, bool checkError,
                                                            bool singleColumn, bool shift, bool useUint,
                                                            string reclassPredictor = null)
        {
            var methodName = string.Format("{0}-{1}-V{2}-T{3}-S{4}-RP{5}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    modelName, singleColumn ? "C" : "Vec", threads, shift ? "shift" : "std",
                                    string.IsNullOrEmpty(reclassPredictor) ? "0" : reclassPredictor.Replace("{", "")
                                            .Replace("}", "").Replace("+", "").Replace("-", "").Replace(" ", "").Replace("=", ""));
            var dataFilePath = shift
                ? FileHelper.GetTestFile("mc_iris_shift.txt")
                : FileHelper.GetTestFile("mc_iris.txt");
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: threads == 1 ? 1 : 0);
            {
                string labelType = useUint ? "U4[0-2]" : "R4";
                string loadSettings = string.Format("Text{{col=Label:{0}:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}}", labelType);
                var loader = env.CreateLoader(loadSettings, new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                var roles = env.CreateExamples(concat, "Features", "Label");
                string modelDef = threads <= 0 ? modelName : string.Format("{0}{{t={1}}}", modelName, threads);
                if (!string.IsNullOrEmpty(reclassPredictor))
                    reclassPredictor = " rp=" + reclassPredictor;
                string iova = string.Format("iova{{p={0} sc={1}{2}}}", modelDef, singleColumn ? "+" : "-", reclassPredictor);
                var trainer = env.CreateTrainer(iova);
                using (var ch = env.Start("train"))
                {
                    var predictor = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                         PredictionKind.MulticlassClassification, checkError, ratio: 0.1f);
                }
            }
        }

        static void TrainMultiToRankerPredictorDense(string modelName, int threads, bool checkError,
                                                     bool singleColumn, bool shift, bool useUint)
        {
            var methodName = string.Format("{0}-{1}-V{2}-T{3}-S{4}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    modelName, singleColumn ? "C" : "Vec", threads, shift ? "shift" : "std");
            var dataFilePath = shift
                ? FileHelper.GetTestFile("mc_iris_shift.txt")
                : FileHelper.GetTestFile("mc_iris.txt");
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: threads == 1 ? 1 : 0);
            {
                string labelType = useUint ? "U4[0-2]" : "R4";
                string loadSettings = string.Format("Text{{col=Label:{0}:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}}", labelType);
                var loader = env.CreateLoader(loadSettings, new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                var roles = env.CreateExamples(concat, "Features", "Label");
                string modelDef = threads <= 0 ? modelName : string.Format("{0}{{t={1}}}", modelName, threads);
                string additionnal = modelName.Contains("xgbrk") ? " u4=+" : "";
                string iova = string.Format("iovark{{p={0} sc={1}{2}}}", modelDef, singleColumn ? "+" : "-", additionnal);
                var trainer = env.CreateTrainer(iova);
                using (var ch = env.Start("train"))
                {
                    var predictor = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                         PredictionKind.MulticlassClassification, checkError, ratio: 0.1f);
                }
            }
        }

        static void TrainMultiToBinaryPredictorSparse(bool singleColumn, bool checkError)
        {
            var methodName = string.Format("{0}-{1}-V{2}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    "lr", singleColumn ? "C" : "Vec");
            var trainFile = FileHelper.GetTestFile("Train-28x28_small.txt");
            var testFile = FileHelper.GetTestFile("Test-28x28_small.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Features:R4:1-784}", new MultiFileSource(trainFile));
                var roles = env.CreateExamples(loader, "Features", "Label");
                var df = DataFrameIO.ReadView(roles.Data);
                Assert.IsTrue(df.Shape[0] > 0);
                var iova = string.Format("iova{{p=lr sc={0} nt=1}}", singleColumn ? "+" : "-");
                loader = env.CreateLoader("Text{col=Label:R4:0 col=Features:R4:1-784}", new MultiFileSource(testFile));
                var trainer = env.CreateTrainer(iova);
                using (var ch = env.Start("train"))
                {
                    var predictor = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                         PredictionKind.MulticlassClassification, checkError, ratio: 0.1f);
                }
            }
        }

        public static void TrainMultiToRankerPredictorSparse(bool singleColumn, bool checkError)
        {
            var methodName = string.Format("{0}-{1}-V{2}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    "lr", singleColumn ? "C" : "Vec");
            var trainFile = FileHelper.GetTestFile("Train-28x28_small.txt");
            var testFile = FileHelper.GetTestFile("Test-28x28_small.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            /*using (*/var env = EnvHelper.NewTestEnvironment();
            {
                var loader = env.CreateLoader("Text", new MultiFileSource(trainFile));
                var roles = env.CreateExamples(loader, "Features", "Label");
                var iova = string.Format("iovark{{p=ftrank sc={0}}}", singleColumn ? "+" : "-");
                loader = env.CreateLoader("Text", new MultiFileSource(testFile));
                var trainer = env.CreateTrainer(iova);
                using (var ch = env.Start("train"))
                {
                    var predictor = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                         PredictionKind.MulticlassClassification, checkError, ratio: 0.1f);
                }
            }
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseLR_SingleColumn()
        {
            TrainMultiToBinaryPredictorDense("lr", -1, false, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseLR_ReclassPredictor()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, false, false, "mlr");
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFTRK_SingleColumn()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseLR_Vector()
        {
            TrainMultiToBinaryPredictorDense("lr", -1, false, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFTRK_Vector()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_SingleColumn()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumn()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false /*true*/, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_SingleColumnUint()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, false, true);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumnUint()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false /*true*/, true, false, true);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_Vector()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_Vector()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false /*true*/, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_VectorUint()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, false, false, true);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_VectorUint()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false /*true*/, false, false, true);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_SingleColumn_Shifted()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, true, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumn_Shifted()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false /*true*/, true, true, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_Vector_Shifted()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, false, true, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_Vector_Shifted()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false /*true*/, false, true, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T10()
        {
            TrainMultiToBinaryPredictorDense("ft", 10, true, true, false, false);
        }

        [TestMethod]
        [Ignore] // parser fails on sparse data
        public void TrainMultiToBinaryPredictorSparseSingleColumn()
        {
            TrainMultiToBinaryPredictorSparse(true, false);
        }

        [TestMethod]
        [Ignore] // parser fails on sparse data
        public void TrainMultiToRankerPredictorSparseSingleColumn()
        {
            TrainMultiToRankerPredictorSparse(true, false);
        }

        [TestMethod]
        [Ignore] // parser fails on sparse data
        public void TrainMultiToBinaryPredictorSparseVector()
        {
            TrainMultiToBinaryPredictorSparse(false, false);
        }

        #endregion

        #region OptimizedOVA

        [TestMethod]
        public void TestTrainMultiToBinaryPredictorIrisTypesR4_SC()
        {
            TrainMultiToBinaryPredictorIris(1, true, "ova", "R4");
            TrainMultiToBinaryPredictorIris(1, true, "iova", "R4");
        }

        [TestMethod]
        public void TestTrainMultiToBinaryPredictorIrisTypesU4_SC()
        {
            TrainMultiToBinaryPredictorIris(1, true, "iova", "U4");
            TrainMultiToBinaryPredictorIris(1, true, "ova", "U4");
        }

        [TestMethod]
        public void TestTrainMultiToBinaryPredictorIrisTypesR4_MC()
        {
            TrainMultiToBinaryPredictorIris(1, false, "iova", "R4");
        }

        [TestMethod]
        public void TestTrainMultiToBinaryPredictorIrisTypesU4_MC()
        {
            TrainMultiToBinaryPredictorIris(1, false, "iova", "U4");
        }

        static void TrainMultiToBinaryPredictorIris(int th, bool singleColumn, string model, string type)
        {
            var methodName = string.Format("{0}-T{1}-{2}-{3}-{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, th, singleColumn ? "asvec" : "asR4", model, type);
            string trainFile, testFile;
            if (type == "R4")
            {
                trainFile = FileHelper.GetTestFile("types/iris_train.idv");
                testFile = FileHelper.GetTestFile("types/iris_test.idv");
            }
            else if (type == "U4")
            {
                trainFile = FileHelper.GetTestFile("types/iris_train_u4.idv");
                testFile = FileHelper.GetTestFile("types/iris_test_u4.idv");
            }
            else if (type == "U43")
            {
                trainFile = FileHelper.GetTestFile("types/iris_train_u43.idv");
                testFile = FileHelper.GetTestFile("types/iris_test_u43.idv");
            }
            else
                throw new NotSupportedException();

            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: th == 1 ? 1 : 0);
            {
                var loaderSettings = "Binary";
                var loader = env.CreateLoader(loaderSettings, new MultiFileSource(trainFile));
                var xf = env.CreateTransform("concat{col=Features:Slength,Swidth}", loader);
                var roles = env.CreateExamples(xf, "Features", "Label");
                ITrainerExtended trainer;
                if (model.ToLower() == "ova" || model.ToLower() == "oova")
                {
                    if (th > 0)
                        trainer = env.CreateTrainer(string.Format("oova{{ p=ft{{t={0}}} }}", th, singleColumn ? "+" : "-"));
                    else
                        trainer = env.CreateTrainer(string.Format("oova{{p=ft{{t=1}} }}", singleColumn ? "+" : "-"));
                }
                else
                {
                    if (th > 0)
                        trainer = env.CreateTrainer(string.Format("iova{{ p=ft{{t={0}}} sc={1} }}", th, singleColumn ? "+" : "-"));
                    else
                        trainer = env.CreateTrainer(string.Format("iova{{p=ft{{t=1}} sc={0} }}", singleColumn ? "+" : "-"));
                }

                using (var ch = env.Start("Train"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    loader = env.CreateLoader(loaderSettings, new MultiFileSource(testFile));
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                         PredictionKind.MulticlassClassification, true,
                                                         ratio: type.StartsWith("U4") && model.ToLower() == "iova" ? 1f : 0.1f);
                }
            }
        }

        [TestMethod]
        public void TestOptimizedOVA()
        {
            OptimizedOVA(0f, "R4", "lr");
            OptimizedOVA(0f, "U4", "lr");
        }

        [TestMethod]
        public void TestOptimizedOVA02()
        {
            OptimizedOVA(0.2f, "R4", "lr");
            OptimizedOVA(0.2f, "U4", "lr");
        }

        static void OptimizedOVA(float downsampling, string type, string model)
        {
            var methodName = string.Format("{0}-D{1}-{2}-{3}", System.Reflection.MethodBase.GetCurrentMethod().Name, downsampling, type, model);
            string trainFile, testFile;
            if (type == "R4")
            {
                trainFile = FileHelper.GetTestFile("types/iris_train.idv");
                testFile = FileHelper.GetTestFile("types/iris_test.idv");
            }
            else if (type == "U4")
            {
                trainFile = FileHelper.GetTestFile("types/iris_train_u4.idv");
                testFile = FileHelper.GetTestFile("types/iris_test_u4.idv");
            }
            else if (type == "U43")
            {
                trainFile = FileHelper.GetTestFile("types/iris_train_u43.idv");
                testFile = FileHelper.GetTestFile("types/iris_test_u43.idv");
            }
            else
                throw new NotSupportedException();

            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            StringWriter sout, serr;
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(out sout, out serr, verbose: false);
            {
                var loaderSettings = "Binary";
                var loader = env.CreateLoader(loaderSettings, new MultiFileSource(trainFile));
                var xf = env.CreateTransform("concat{col=Features:Slength,Swidth}", loader);
                var roles = env.CreateExamples(xf, "Features", "Label");
                var trainer = env.CreateTrainer(string.Format("oova{{p={1} ds={0}}}", downsampling, model));
                using (var ch = env.Start("Train"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    var sbout = sout.GetStringBuilder().ToString();
                    var sbrr = serr.GetStringBuilder().ToString();
                    loader = env.CreateLoader(loaderSettings, new MultiFileSource(testFile));
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                                PredictionKind.MulticlassClassification, true, ratio: 0.8f);
                }
            }
        }

        #endregion
    }
}
