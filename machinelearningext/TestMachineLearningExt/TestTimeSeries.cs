// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;
using Scikit.ML.DataManipulation;
using Scikit.ML.TimeSeries;
using Scikit.ML.ScikitAPI;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestTimeSeries
    {
        class InputOutput
        {
            public float X;
            public float time;
            public float one;
        }

        [TestMethod]
        public void TestTimeSeriesFloatRegression()
        {
            var inputs = new[] {
                new InputOutput() { X = 5f, time=0f, one=1f },
                new InputOutput() { X = 7f, time=1f, one=1f },
                new InputOutput() { X = 9f, time=2f, one=1f },
                new InputOutput() { X = 11f, time=3f, one=1f },
                new InputOutput() { X = 5f, time=0f, one=1f },
                new InputOutput() { X = 7f, time=1f, one=1f },
                new InputOutput() { X = 9f, time=2f, one=1f },
                new InputOutput() { X = 11f, time=3f, one=1f },
                new InputOutput() { X = 5f, time=0f, one=1f },
                new InputOutput() { X = 7f, time=1f, one=1f },
                new InputOutput() { X = 9f, time=2f, one=1f },
                new InputOutput() { X = 11f, time=3f, one=1f },
            };
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "concat{col=xt:time,one}" }, "sasdcar{iter=50}", host))
                {
                    pipe.Train(data, feature: "xt", label: "X");
                    var view = pipe.Predict(data);
                    var df = DataFrameIO.ReadView(view).Head(4).Copy();
                    df["diff"] = df["Score"] - df["X"];
                    var exp = DataFrameIO.ReadStr("null\n0\n0\n0\n0");
                    df["diff"].AssertAlmostEqual(exp["null"].AsType(NumberType.R4), precision: 1e-1);
                }
            }
        }

        [TestMethod]
        public void TestTimeSeriesFloatPerfectTrended()
        {
            var inputs = new[] {
                new InputOutput() { X = 5f, time=0f },
                new InputOutput() { X = 7f, time=1f },
                new InputOutput() { X = 9f, time=2f },
                new InputOutput() { X = 11f, time=3f },
            };
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new DeTrendTransform.Arguments
                {
                    columns = new[] { new Scikit.ML.PipelineHelper.Column1x1() { Source = "X", Name = "Y" } },
                    timeColumn = "time"
                };

                var scaled = new DeTrendTransform(host, args, data);

                using (var cursor = scaled.GetRowCursor(scaled.OutputSchema))
                {
                    var outValues = new List<float>();
                    int pos = SchemaHelper.GetColumnIndex(cursor.Schema, "Y");
                    var type = cursor.Schema[pos].Type;
                    if (type != NumberType.R4)
                        throw new Exception("Unexpected type");
                    var colGetter = cursor.GetGetter<float>(pos);
                    float got = -1f;
                    while (cursor.MoveNext())
                    {
                        colGetter(ref got);
                        outValues.Add(got);
                    }
                    if (outValues.Count != 4)
                        throw new Exception("expected 4");
                    for (int i = 0; i < outValues.Count; ++i)
                        if (Math.Abs(outValues[i]) > 5e-2)
                            throw new Exception(string.Format("Unexpected value {0}!={1}", outValues[i], 0));
                }
            }
        }

        [TestMethod]
        public void TestTimeSeriesFloatUnPerfectTrended()
        {
            var inputs = new[] {
                new InputOutput() { X = 7f, time=0f },
                new InputOutput() { X = 7f, time=1f },
                new InputOutput() { X = 9f, time=2f },
                new InputOutput() { X = 9f, time=3f },
                new InputOutput() { X = 8f, time=4f },
            };
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new DeTrendTransform.Arguments
                {
                    columns = new[] { new Scikit.ML.PipelineHelper.Column1x1() { Source = "X", Name = "Y" } },
                    timeColumn = "time"
                };

                var scaled = new DeTrendTransform(host, args, data);

                using (var cursor = scaled.GetRowCursor(scaled.OutputSchema))
                {
                    var outValues = new List<float>();
                    int pos = SchemaHelper.GetColumnIndex(cursor.Schema, "Y");
                    var type = cursor.Schema[pos].Type;
                    if (type != NumberType.R4)
                        throw new Exception("Unexpected type");
                    var colGetter = cursor.GetGetter<float>(pos);
                    float got = -1f;
                    while (cursor.MoveNext())
                    {
                        colGetter(ref got);
                        outValues.Add(got);
                    }
                    if (outValues.Count != 5)
                        throw new Exception("expected 5");
                    for (int i = 0; i < outValues.Count; ++i)
                        if (Math.Abs(outValues[i]) > 2)
                            throw new Exception(string.Format("Unexpected value {0}", outValues[i]));
                }
            }
        }

        [TestMethod]
        public void TestTimeSeriesFloatPerfectTrended2()
        {
            var inputs = new[] {
                new InputOutput() { X = 3f, time=0f },
                new InputOutput() { X = 2f, time=1f },
                new InputOutput() { X = 1f, time=2f },
                new InputOutput() { X = 0f, time=3f },
            };
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new DeTrendTransform.Arguments
                {
                    columns = new[] { new Scikit.ML.PipelineHelper.Column1x1() { Source = "X", Name = "Y" } },
                    timeColumn = "time"
                };

                var scaled = new DeTrendTransform(host, args, data);

                using (var cursor = scaled.GetRowCursor(scaled.OutputSchema))
                {
                    var outValues = new List<float>();
                    int pos = SchemaHelper.GetColumnIndex(cursor.Schema, "Y");
                    var type = cursor.Schema[pos].Type;
                    if (type != NumberType.R4)
                        throw new Exception("Unexpected type");
                    var colGetter = cursor.GetGetter<float>(pos);
                    float got = -1f;
                    while (cursor.MoveNext())
                    {
                        colGetter(ref got);
                        outValues.Add(got);
                    }
                    if (outValues.Count != 4)
                        throw new Exception("expected 4");
                    for (int i = 0; i < outValues.Count; ++i)
                        if (Math.Abs(outValues[i]) > 1e-2)
                            throw new Exception(string.Format("Unexpected value {0}", outValues[i]));
                }
            }
        }

        [TestMethod]
        public void TestTimeSeriesFloatUnPerfectTrended2()
        {
            var inputs = new[] {
                new InputOutput() { X = 3740f, time=1f },
                new InputOutput() { X = 2880f, time=2f },
                new InputOutput() { X = 2620f, time=3f },
                new InputOutput() { X = 2860f, time=4f },
                new InputOutput() { X = 2640f, time=5f },
                new InputOutput() { X = 2740f, time=6f },
                new InputOutput() { X = 2940f, time=7f },
                new InputOutput() { X = 4040f, time=8f },
                new InputOutput() { X = 3800f, time=9f },
                new InputOutput() { X = 3640f, time=10f },
            };
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var args = new DeTrendTransform.Arguments
                {
                    columns = new[] { new Scikit.ML.PipelineHelper.Column1x1() { Source = "X", Name = "Y" } },
                    timeColumn = "time"
                };

                var scaled = new DeTrendTransform(host, args, data);

                using (var cursor = scaled.GetRowCursor(scaled.OutputSchema))
                {
                    var outValues = new List<float>();
                    int pos = SchemaHelper.GetColumnIndex(cursor.Schema, "Y");
                    var type = cursor.Schema[pos].Type;
                    if (type != NumberType.R4)
                        throw new Exception("Unexpected type");
                    var colGetter = cursor.GetGetter<float>(pos);
                    float got = -1f;
                    while (cursor.MoveNext())
                    {
                        colGetter(ref got);
                        outValues.Add(got);
                    }
                    if (outValues.Count != inputs.Length)
                        throw new Exception("unexpected size");
                    for (int i = 0; i < outValues.Count; ++i)
                        if (Math.Abs(outValues[i]) > 2000)
                            throw new Exception(string.Format("Unexpected value {0}", outValues[i]));
                }
            }
        }

        [TestMethod]
        public void TestTimeSeriesDeTrendSerialize()
        {
            using (var host = EnvHelper.NewTestEnvironment())
            {
                var inputs = new[] {
                    new InputOutput() { X = 7f, time=0f },
                    new InputOutput() { X = 7f, time=1f },
                    new InputOutput() { X = 9f, time=2f },
                    new InputOutput() { X = 9f, time=3f },
                    new InputOutput() { X = 8f, time=4f },
                };

                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                var data = host.CreateTransform("detrend{col=Y:X time=time optim=sasdcar{iter=50}}", loader);

                // To train the model.
                using (var cursor = data.GetRowCursor(data.Schema)) { }

                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
                var outData = FileHelper.GetOutputFile("outData.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
            }
        }
    }
}

