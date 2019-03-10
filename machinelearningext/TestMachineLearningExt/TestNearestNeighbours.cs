// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineHelper;
using Scikit.ML.NearestNeighbors;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestNearestNeighbors
    {
        private static DataViewSchema.Column _dc(int i)
        {
            return new DataViewSchema.Column(null, i, false, null, null);
        }

        static void TrainkNNBinaryClassification(int k, NearestNeighborsWeights weight, int threads, float ratio = 0.2f,
                                                 string distance = "L2", int conc = 0)
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris_binary.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment(conc: conc);
            //using (env)
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                if (distance == "cosine")
                    concat = env.CreateTransform("Scaler{col=Features}", concat);
                var roles = env.CreateExamples(concat, "Features", "Label");
                string modelDef;
                modelDef = string.Format("knn{{k={0} weighting={1} nt={2} distance={3} seed=1}}", k,
                                         weight == NearestNeighborsWeights.distance ? "distance" : "uniform", threads, distance);
                var trainer = env.CreateTrainer(modelDef);
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                               PredictionKind.BinaryClassification, true, ratio: ratio);
                }
            }
        }

        public static void TrainkNNBinaryClassificationId(int k, NearestNeighborsWeights weight, int threads, float ratio = 0.2f,
                                                        string distance = "L2")
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris_binary_id.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment();
            //using (env)
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 col=Uid:I8:5 header=+}",
                    new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                if (distance == "cosine")
                    concat = env.CreateTransform("Scaler{col=Features}", concat);
                var roles = env.CreateExamples(concat, "Features", "Label");
                string modelDef;
                modelDef = string.Format("knn{{k={0} weighting={1} nt={2} distance={3} id=Uid}}", k,
                                         weight == NearestNeighborsWeights.distance ? "distance" : "uniform", threads, distance);
                var trainer = env.CreateTrainer(modelDef);
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                              PredictionKind.BinaryClassification, true, ratio: ratio);
                }
            }
        }

        public static void TrainkNNMultiClassification(int k, NearestNeighborsWeights weight, int threads, float ratio = 0.2f,
                                                       string distance = "L2")
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment();
            //using (env)
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                var roles = env.CreateExamples(concat, "Features", "Label");
                string modelDef;
                modelDef = string.Format("knnmc{{k={0} weighting={1} nt={2} distance={3}}}", k,
                                         weight == NearestNeighborsWeights.distance ? "distance" : "uniform", threads, distance);
                var trainer = env.CreateTrainer(modelDef);
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                           PredictionKind.MultiClassClassification, true, ratio: ratio);
                }
            }
        }

        public static void TrainkNNTransformId(int k, NearestNeighborsWeights weight, int threads, string distance = "L2")
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris_binary_id.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment();
            //using (env)
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 col=Uid:I8:5 header=+}",
                    new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                if (distance == "cosine")
                    concat = env.CreateTransform("Scaler{col=Features}", concat);
                concat = env.CreateTransform("knntr{k=5 id=Uid}", concat);
                long nb = DataViewUtils.ComputeRowCount(concat);
                if (nb == 0)
                    throw new System.Exception("Empty pipeline.");

                using (var cursor = concat.GetRowCursor(concat.Schema))
                {
                    var getdist = cursor.GetGetter<VBuffer<float>>(_dc(7));
                    var getid = cursor.GetGetter<VBuffer<long>>(_dc(8));
                    var ddist = new VBuffer<float>();
                    var did = new VBuffer<long>();
                    while (cursor.MoveNext())
                    {
                        getdist(ref ddist);
                        getid(ref did);
                        if (!ddist.IsDense || !did.IsDense)
                            throw new System.Exception("not dense");
                        if (ddist.Count != did.Count)
                            throw new System.Exception("not the same dimension");
                        for (int i = 1; i < ddist.Count; ++i)
                        {
                            if (ddist.Values[i - 1] > ddist.Values[i])
                                throw new System.Exception("not sorted");
                            if (did.Values[i] % 2 != 1)
                                throw new System.Exception("wrong id");
                        }
                    }
                }

                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, concat, loader, outData, outData2, false);
            }
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassification1()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f);
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassification2()
        {
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 1);
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassification5()
        {
            TrainkNNBinaryClassification(5, NearestNeighborsWeights.uniform, 1);
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassificationId()
        {
            TrainkNNBinaryClassificationId(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f);
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassificationCosine()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f, distance: "cosine", conc: 1);
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 1, distance: "cosine", conc: 1);
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassificationL1()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f, distance: "L1");
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 1, distance: "L1");
            TrainkNNBinaryClassification(10, NearestNeighborsWeights.uniform, 1, distance: "L1");
        }

        [TestMethod]
        public void TestI_TrainkNNBinaryClassificationMultiThread()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 2, ratio: 0.05f);
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 2);
            TrainkNNBinaryClassification(10, NearestNeighborsWeights.uniform, 2);
        }

        [TestMethod]
        public void TestI_TrainkNNMultiClassification()
        {
            TrainkNNMultiClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f);
            TrainkNNMultiClassification(2, NearestNeighborsWeights.uniform, 1);
            TrainkNNMultiClassification(10, NearestNeighborsWeights.uniform, 1);
        }

        [TestMethod]
        public void TestI_TrainkNNMultiClassificationMultiThread()
        {
            TrainkNNMultiClassification(1, NearestNeighborsWeights.uniform, 2, ratio: 0.05f);
            TrainkNNMultiClassification(2, NearestNeighborsWeights.uniform, 2);
            TrainkNNMultiClassification(10, NearestNeighborsWeights.uniform, 2);
        }

        [TestMethod]
        public void TestI_TrainkNNTransformId()
        {
            TrainkNNTransformId(1, NearestNeighborsWeights.uniform, 2);
            TrainkNNTransformId(2, NearestNeighborsWeights.uniform, 2);
            TrainkNNTransformId(10, NearestNeighborsWeights.uniform, 2);
        }                
    }
}
