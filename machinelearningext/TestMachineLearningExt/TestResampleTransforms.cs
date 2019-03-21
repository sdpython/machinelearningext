// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.RandomTransforms;
using Scikit.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestResampleTransforms
    {
        private DataViewSchema.Column _dc(int i)
        {
            return new DataViewSchema.Column(null, i, false, null, null);
        }

        #region ResampleTransform

        private void TestResampleTransform(float ratio)
        {
            /*using (*/var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var inputs = new InputOutput[] {
                    new InputOutput() { X = new float[] { 0, 1 }, Y = 1 },
                    new InputOutput() { X = new float[] { 0, 1 }, Y = 0 },
                    new InputOutput() { X = new float[] { 0, 1 }, Y = 2 },
                    new InputOutput() { X = new float[] { 0, 1 }, Y = 3 },
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(env, inputs);
                var args = new ResampleTransform.Arguments { lambda = ratio, cache = false };
                var tr = new ResampleTransform(env, args, data);
                var values = new List<int>();
                using (var cursor = tr.GetRowCursor(tr.Schema))
                {
                    var columnGetter = cursor.GetGetter<int>(_dc(1));
                    while (cursor.MoveNext())
                    {
                        int got = 0;
                        columnGetter(ref got);
                        values.Add((int)got);
                    }
                }
                if (ratio < 1 && values.Count > 8)
                    throw new Exception("ResampleTransform did not work.");
                if (ratio > 1 && values.Count < 1)
                    throw new Exception("ResampleTransform did not work.");
            }
        }

        [TestMethod]
        public void TestI_ResampleSerializationRatio()
        {
            TestResampleTransform(0.5f);
            TestResampleTransform(1.5f);
        }

        [TestMethod]
        public void TestI_ResampleSerialization()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);

            /*using (*/var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 " +
                                              "col=Pwidth:R4:4 header=+ sep=tab}",
                                              new MultiFileSource(dataFilePath));
                var sorted = env.CreateTransform("resample{lambda=1 c=-}", loader);
                DataViewHelper.ToCsv(env, sorted, outputDataFilePath);

                var lines = File.ReadAllLines(outputDataFilePath);
                int begin = 0;
                for (; begin < lines.Length; ++begin)
                {
                    if (lines[begin].StartsWith("Label"))
                        break;
                }
                lines = lines.Skip(begin).ToArray();
                var linesSorted = lines.OrderBy(c => c).ToArray();
                for (int i = 1; i < linesSorted.Length; ++i)
                {
                    if (linesSorted[i - 1][0] > linesSorted[i][0])
                        throw new Exception("The output is not sorted.");
                }
            }
        }

        #endregion

        #region Shake Input Transform

        [TestMethod]
        public void Testl_ShakeInputTransform()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new[] {
                    new SHExampleA() { X = new float[] { 0, 1 } },
                    new SHExampleA() { X = new float[] { 2, 3 } }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new ShakeInputTransform.Arguments
                {
                    inputColumn = "X",
                    inputFeaturesInt = new[] { 0, 1 },
                    outputColumns = new[] { "yo" },
                    values = "-10,10;-100,100"
                };

                var trv = new SHExampleValueMapper();
                if (trv == null)
                    throw new Exception("Invalid");
                var shake = new ShakeInputTransform(host, args, data, new IValueMapper[] { trv });

                using (var cursor = shake.GetRowCursor(shake.Schema))
                {
                    var outValues = new List<float>();
                    var colGetter = cursor.GetGetter<VBuffer<float>>(_dc(1));
                    while (cursor.MoveNext())
                    {
                        VBuffer<float> got = new VBuffer<float>();
                        colGetter(ref got);
                        outValues.AddRange(got.DenseValues());
                    }
                    if (outValues.Count != 8)
                        throw new Exception("expected 8");
                }
            }
        }

        [TestMethod]
        public void Testl_ShakeInputTransformAdd()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new[] {
                    new SHExampleA() { X = new float[] { 0, 1 } },
                    new SHExampleA() { X = new float[] { 2, 3 } }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new ShakeInputTransform.Arguments
                {
                    inputColumn = "X",
                    inputFeaturesInt = new[] { 0, 1 },
                    outputColumns = new[] { "yo" },
                    values = "-10,10;-100,100",
                    aggregation = ShakeInputTransform.ShakeAggregation.add
                };

                var trv = new SHExampleValueMapper();
                if (trv == null)
                    throw new Exception("Invalid");
                var shake = new ShakeInputTransform(host, args, data, new IValueMapper[] { trv });

                using (var cursor = shake.GetRowCursor(shake.Schema))
                {
                    var outValues = new List<float>();
                    var colGetter = cursor.GetGetter<VBuffer<float>>(_dc(1));
                    while (cursor.MoveNext())
                    {
                        VBuffer<float> got = new VBuffer<float>();
                        colGetter(ref got);
                        outValues.AddRange(got.DenseValues());
                    }
                    if (outValues.Count != 2)
                        throw new Exception("expected 2");
                }
            }
        }

        [TestMethod]
        public void Testl_ShakeInputTransformVector()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new[] {
                    new SHExampleA() { X = new float[] { 0, 1 } },
                    new SHExampleA() { X = new float[] { 2, 3 } }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new ShakeInputTransform.Arguments
                {
                    inputColumn = "X",
                    inputFeaturesInt = new[] { 0, 1 },
                    outputColumns = new[] { "yo" },
                    values = "-10,10;-100,100"
                };

                var trv = new ExampleValueMapperVector();
                if (trv == null)
                    throw new Exception("Invalid");
                var shake = new ShakeInputTransform(host, args, data, new IValueMapper[] { trv });

                using (var cursor = shake.GetRowCursor(shake.Schema))
                {
                    var outValues = new List<float>();
                    var colGetter = cursor.GetGetter<VBuffer<float>>(_dc(1));
                    while (cursor.MoveNext())
                    {
                        VBuffer<float> got = new VBuffer<float>();
                        colGetter(ref got);
                        outValues.AddRange(got.DenseValues());
                    }
                    if (outValues.Count != 16)
                        throw new Exception("expected 16");
                }
            }
        }

        [TestMethod]
        public void Testl_ShakeInputTransformVectorAdd()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new[] {
                    new SHExampleA() { X = new float[] { 0, 1 } },
                    new SHExampleA() { X = new float[] { 2, 3 } }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new ShakeInputTransform.Arguments
                {
                    inputColumn = "X",
                    inputFeaturesInt = new[] { 0, 1 },
                    outputColumns = new[] { "yo" },
                    values = "-10,10;-100,100",
                    aggregation = ShakeInputTransform.ShakeAggregation.add
                };

                var trv = new ExampleValueMapperVector();
                if (trv == null)
                    throw new Exception("Invalid");
                var shake = new ShakeInputTransform(host, args, data, new IValueMapper[] { trv });

                using (var cursor = shake.GetRowCursor(shake.Schema))
                {
                    var outValues = new List<float>();
                    var colGetter = cursor.GetGetter<VBuffer<float>>(_dc(1));
                    while (cursor.MoveNext())
                    {
                        VBuffer<float> got = new VBuffer<float>();
                        colGetter(ref got);
                        outValues.AddRange(got.DenseValues());
                    }
                    if (outValues.Count != 4)
                        throw new Exception("expected 4");
                }
            }
        }

        #endregion
    }
}

