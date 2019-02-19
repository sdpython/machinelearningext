// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;
using Scikit.ML.ModelSelection;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestModelSelection
    {
        #region SplitTrainTestTransform

        static void TestSplitTrainTestTransform(string option, int numThreads = 1)
        {
            using (var host = EnvHelper.NewTestEnvironment(conc: numThreads == 1 ? 1 : 0))
            {
                var inputsl = new List<InputOutput>();
                for (int i = 0; i < 100; ++i)
                    inputsl.Add(new InputOutput { X = new float[] { 0, 1 }, Y = i });
                var inputs = inputsl.ToArray();
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new SplitTrainTestTransform.Arguments { newColumn = "Part", numThreads = numThreads };
                if (option == "2")
                {
                    var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                    var cacheFile = FileHelper.GetOutputFile("cacheFile.idv", methodName);
                    args.cacheFile = cacheFile;
                }

                var transformedData = new SplitTrainTestTransform(host, args, data);

                var counter1 = new Dictionary<int, List<int>>();
                using (var cursor = transformedData.GetRowCursor(transformedData.OutputSchema))
                {
                    int index = SchemaHelper.GetColumnIndex(cursor.Schema, "Y");
                    var sortColumnGetter = cursor.GetGetter<int>(index);
                    index = SchemaHelper.GetColumnIndex(cursor.Schema, args.newColumn);
                    var partGetter = cursor.GetGetter<int>(index);
                    var schema = SchemaHelper.ToString(cursor.Schema);
                    if (string.IsNullOrEmpty(schema))
                        throw new Exception("null");
                    if (!schema.Contains("Part:I4"))
                        throw new Exception(schema);
                    var schema2 = SchemaHelper.ToString(transformedData.OutputSchema);
                    SchemaHelper.CheckSchema(host, transformedData.OutputSchema, cursor.Schema);
                    int got = 0;
                    int part = 0;
                    while (cursor.MoveNext())
                    {
                        sortColumnGetter(ref got);
                        partGetter(ref part);
                        if (!counter1.ContainsKey(part))
                            counter1[part] = new List<int>();
                        if (counter1[part].Any() && got.Equals(counter1[part][counter1[part].Count - 1]))
                            throw new Exception("Unexpected value, they should be all different.");
                        counter1[part].Add(got);
                    }
                }

                // Check than there is no overlap.
                if (counter1.Count != 2)
                    throw new Exception(string.Format("Too many or not enough parts: {0}", counter1.Count));
                var nb = counter1.Select(c => c.Value.Count).Sum();
                if (inputs.Length != nb)
                    throw new Exception(string.Format("Length mismath: {0} != {1}", inputs.Length, nb));
                foreach (var part in counter1)
                {
                    var hash = part.Value.ToDictionary(c => c, d => d);
                    if (hash.Count != part.Value.Count)
                        throw new Exception(string.Format("Not identical id for part {0}", part));
                }
                var part0 = new HashSet<int>(counter1[0]);
                var part1 = new HashSet<int>(counter1[1]);
                if (part0.Intersect(part1).Any())
                    throw new Exception("Intersection is not null.");

                // Check sizes.
                if (part0.Count > part1.Count * 2 + 15)
                    throw new Exception("Size are different from ratios.");
                if (part0.Count < part1.Count + 5)
                    throw new Exception("Size are different from ratios.");

                // We check a second run brings the same results (CacheView).
                var counter2 = new Dictionary<int, List<int>>();
                using (var cursor = transformedData.GetRowCursor(transformedData.OutputSchema))
                {
                    var schema = SchemaHelper.ToString(cursor.Schema);
                    if (string.IsNullOrEmpty(schema))
                        throw new Exception("null");
                    if (!schema.Contains("Part:I4"))
                        throw new Exception(schema);
                    var schema2 = SchemaHelper.ToString(transformedData.OutputSchema);
                    SchemaHelper.CheckSchema(host, transformedData.OutputSchema, cursor.Schema);
                    int index = SchemaHelper.GetColumnIndex(cursor.Schema, "Y");
                    var sortColumnGetter = cursor.GetGetter<int>(index);
                    index = SchemaHelper.GetColumnIndex(cursor.Schema, args.newColumn);
                    var partGetter = cursor.GetGetter<int>(index);
                    int got = 0;
                    int part = 0;
                    while (cursor.MoveNext())
                    {
                        sortColumnGetter(ref got);
                        partGetter(ref part);
                        if (!counter2.ContainsKey(part))
                            counter2[part] = new List<int>();
                        counter2[part].Add(got);
                    }
                }

                if (counter1.Count != counter2.Count)
                    throw new Exception("Not the same number of parts.");
                foreach (var pair in counter1)
                {
                    var list1 = pair.Value;
                    var list2 = counter2[pair.Key];
                    var difList = list1.Where(a => !list2.Any(a1 => a1 == a))
                        .Union(list2.Where(a => !list1.Any(a1 => a1 == a)));
                    if (difList.Any())
                        throw new Exception("Not the same results for a part.");
                }
            }
        }

        [TestMethod]
        public void TestDataSplitTrainTestSerialization()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var cacheFile = FileHelper.GetOutputFile("outputDataFilePath.idv", methodName);
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment())
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var args = new SplitTrainTestTransform.Arguments
                {
                    newColumn = "Part",
                    cacheFile = cacheFile,
                    filename = new string[] { trainFile, testFile },
                    reuse = true
                };

                var transformedData = new SplitTrainTestTransform(env, args, loader);
                StreamHelper.SaveModel(env, transformedData, outModelFilePath);

                using (var fs = File.OpenRead(outModelFilePath))
                {
                    var deserializedData = ModelFileUtils.LoadTransforms(env, loader, fs);
                    var saver = env.CreateSaver("Text");
                    var columns = new int[deserializedData.Schema.Count];
                    for (int i = 0; i < columns.Length; ++i)
                        columns[i] = i;
                    using (var fs2 = File.Create(outData))
                        saver.SaveData(fs2, deserializedData, columns);
                }

                if (!File.Exists(cacheFile))
                    throw new FileNotFoundException(cacheFile);
                if (!File.Exists(trainFile))
                    throw new FileNotFoundException(trainFile);
                if (!File.Exists(testFile))
                    throw new FileNotFoundException(testFile);
            }
        }

        public static void TestDataSplitTrainTestSerializationIris(string saverSetting)
        {
            var methodName = string.Format("{0}-{1}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                        saverSetting.Replace("{", "").Replace("}", "").Replace(" ", "").Replace("=", "").Replace("+", "Y").Replace("-", "N"));
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            string ext = saverSetting.Contains("bin") ? "idv" : "txt";
            var trainFile = FileHelper.GetOutputFile(string.Format("iris_train.{0}", ext), methodName);
            var testFile = FileHelper.GetOutputFile(string.Format("iris_test.{0}", ext), methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment())
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var trans = string.Format("SplitTrainTest{{f={0} f={1} saver={2} }}", trainFile, testFile, saverSetting);
                var transformedData = env.CreateTransform(trans, loader);

                var saver = env.CreateSaver("Text");
                var columns = new int[transformedData.Schema.Count];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = i;
                using (var fs2 = File.Create(outData))
                    saver.SaveData(fs2, transformedData, columns);

                if (!File.Exists(trainFile))
                    throw new FileNotFoundException(trainFile);
                if (!File.Exists(testFile))
                    throw new FileNotFoundException(testFile);
            }
        }

        [TestMethod]
        public void TestDataSplitTrainTestSerializationIrisAndResample()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment(conc: 4))
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var resample = env.CreateTransform("Resample{lambda=3}", loader);
                var trans = string.Format("SplitTrainTest{{f={0} f={1} }}", trainFile, testFile);
                var transformedData = env.CreateTransform(trans, resample);

                var saver = env.CreateSaver("Text");
                var columns = new int[transformedData.Schema.Count];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = i;
                using (var fs2 = File.Create(outData))
                    saver.SaveData(fs2, transformedData, columns);

                if (!File.Exists(trainFile))
                    throw new FileNotFoundException(trainFile);
                if (!File.Exists(testFile))
                    throw new FileNotFoundException(testFile);
            }
        }

        [TestMethod]
        public void TestTransSplitTrainTestTransform()
        {
            TestSplitTrainTestTransform(null);
        }

        [TestMethod]
        public void TestTransSplitTrainTestTransform2()
        {
            TestSplitTrainTestTransform("2");
        }

        [TestMethod]
        public void TestTransSplitTrainTestSerializationIrisBinary()
        {
            TestDataSplitTrainTestSerializationIris("binary");
        }

        [TestMethod]
        public void TestTransSplitTrainTestSerializationIrisText()
        {
            TestDataSplitTrainTestSerializationIris("text{schema=- header=+}");
        }

        #endregion
    }
}
