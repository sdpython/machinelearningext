// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Model;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineLambdaTransforms;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.DataManipulation;
using Scikit.ML.DocHelperMlExt;
using Scikit.ML.PipelineHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestProductionPrediction
    {
        #region TransformValueMapper

        [TestMethod]
        public void TestTransform2ValueMapperMultiThread()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment();
            {
                var host = env.Register("unittest");

                var inputs = new[] {
                    new InputOutput { X = new float[] { 0, 1 }, Y=10 },
                    new InputOutput { X = new float[] { 2, 3 }, Y=100 }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var trv = LambdaTransform.CreateMap(host, data,
                                            (InputOutput src, InputOutput dst, EnvHelper.EmptyState state) =>
                                            {
                                                dst.X = new float[] { src.X[0] + 1f, src.X[1] - 1f };
                                            }, (EnvHelper.EmptyState state) => { });

                var ino = new InputOutput { X = new float[] { -5, -5 }, Y = 3 };
                var inob = new VBuffer<float>(2, ino.X);
                var ans = new VBuffer<float>();

                using (var valueMapper = new ValueMapperFromTransformFloat<VBuffer<float>>(host, trv, "X", "X", ignoreOtherColumn: true))
                {
                    var mapper = valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();

                    var listy = new List<int>();
                    var listx = new List<float>();
                    int y = 0;
                    for (int i = 0; i < inputs.Length; ++i)
                    {
                        mapper(in inob, ref ans);
                        y = inputs[i].Y;
                        if (ans.Count != 2)
                            throw new Exception("Issue with dimension.");
                        listx.AddRange(ans.GetValues().ToArray());
                        listy.Add((int)y);
                    }
                    if (listy.Count != 2)
                        throw new Exception("Issue with dimension.");
                    if (listy[0] != 10 || listy[1] != 100)
                        throw new Exception("Issue with values.");
                    if (listx.Count != 4)
                        throw new Exception("Issue with dimension.");
                    if (listx[0] != -4)
                        throw new Exception("Issue with values.");
                    if (listx[1] != -6)
                        throw new Exception("Issue with values.");
                    if (listx[2] != -4)
                        throw new Exception("Issue with values.");
                    if (listx[3] != -6)
                        throw new Exception("Issue with values.");
                    if (inob.Count != 2)
                        throw new Exception("Issue with dimension.");
                    if (inob.Values[0] != -5)
                        throw new Exception("Values were overwritten.");
                    if (inob.Values[0] != -5)
                        throw new Exception("Values were overwritten.");
                }
            }
        }


        [TestMethod]
        public void TestTransform2ValueMapperMultiThreadOut()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment();
            {
                var host = env.Register("unittest");

                var inputs = new[] {
                    new InputOutput { X = new float[] { 0, 1 }, Y=10 },
                    new InputOutput { X = new float[] { 2, 3 }, Y=100 }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var trv = LambdaTransform.CreateMap(host, data,
                                            (InputOutput src, InputOutputOut dst, EnvHelper.EmptyState state) =>
                                            {
                                                dst.Xout = new float[] { src.X[0] + 1f, src.X[1] - 1f };
                                            }, (EnvHelper.EmptyState state) => { });

                var ino = new InputOutput { X = new float[] { -5, -5 }, Y = 3 };
                var inob = new VBuffer<float>(2, ino.X);
                var ans = new VBuffer<float>();

                using (var valueMapper = new ValueMapperFromTransformFloat<VBuffer<float>>(host, trv, "X", "Xout", ignoreOtherColumn: true))
                {
                    var mapper = valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();

                    var listy = new List<int>();
                    var listx = new List<float>();
                    int y = 0;
                    for (int i = 0; i < inputs.Length; ++i)
                    {
                        mapper(in inob, ref ans);
                        y = inputs[i].Y;
                        if (ans.Count != 2)
                            throw new Exception("Issue with dimension.");
                        listx.AddRange(ans.GetValues().ToArray());
                        listy.Add((int)y);
                    }
                    if (listy.Count != 2)
                        throw new Exception("Issue with dimension.");
                    if (listy[0] != 10 || listy[1] != 100)
                        throw new Exception("Issue with values.");
                    if (listx.Count != 4)
                        throw new Exception("Issue with dimension.");
                    if (listx[0] != -4)
                        throw new Exception("Issue with values.");
                    if (listx[1] != -6)
                        throw new Exception("Issue with values.");
                    if (listx[2] != -4)
                        throw new Exception("Issue with values.");
                    if (listx[3] != -6)
                        throw new Exception("Issue with values.");
                    if (inob.Count != 2)
                        throw new Exception("Issue with dimension.");
                    if (inob.Values[0] != -5)
                        throw new Exception("Values were overwritten.");
                    if (inob.Values[0] != -5)
                        throw new Exception("Values were overwritten.");
                }
            }
        }

        [TestMethod]
        public void TestTransform2ValueMapperSingleThread()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var host = env.Register("unittest");

                var inputs = new[] {
                    new InputOutput { X = new float[] { 0, 1 }, Y=10 },
                    new InputOutput { X = new float[] { 2, 3 }, Y=100 }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);

                var trv = LambdaTransform.CreateMap(host, data,
                                            (InputOutput src, InputOutput2 dst, EnvHelper.EmptyState state) =>
                                            {
                                                dst.X2 = new float[] { src.X[0] + 1f, src.X[1] - 1f };
                                            }, (state) => { });

                var inos = new InputOutput[] {new InputOutput { X = new float[] { -5, -5 }, Y = 3 },
                new InputOutput { X = new float[] { -6, -6 }, Y = 30 } };
                var ans = new VBuffer<float>();

                using (var valueMapper = new ValueMapperFromTransformFloat<VBuffer<float>>(host, trv, "X", "X2", ignoreOtherColumn: true))
                {
                    var mapper = valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();

                    var listy = new List<int>();
                    var listx = new List<float>();
                    int y = 0;
                    int tour = 0;
                    for (int i = 0; i < inputs.Length; ++i)
                    {
                        var temp = new VBuffer<float>(2, inos[tour++].X);
                        mapper(in temp, ref ans);
                        y = inputs[i].Y;
                        if (ans.Count != 2)
                            throw new Exception("Issue with dimension.");
                        listx.AddRange(ans.GetValues().ToArray());
                        listy.Add((int)y);
                    }
                    if (listy.Count != 2)
                        throw new Exception("Issue with dimension.");
                    if (listy[0] != 10 || listy[1] != 100)
                        throw new Exception("Issue with values.");
                    if (listx.Count != 4)
                        throw new Exception("Issue with dimension.");
                    if (listx[0] != -4)
                        throw new Exception("Issue with values.");
                    if (listx[1] != -6)
                        throw new Exception("Issue with values.");
                    if (listx[2] != -5)
                        throw new Exception("Issue with values.");
                    if (listx[3] != -7)
                        throw new Exception("Issue with values.");
                }
            }
        }

        public class ValueMapperExample : IDisposable
        {
            public class Input
            {
                [VectorType(9)]
                public float[] Features;
            }

            IHostEnvironment _env;
            IDataView _transforms;
            IPredictor _predictor;
            ValueMapper<VBuffer<float>, float> _mapper;
            ValueMapperFromTransformFloat<VBuffer<float>> _valueMapper;

            public ValueMapperExample(string modelName, string features)
            {
                _env = EnvHelper.NewTestEnvironment();
                _predictor = ModelFileUtils.LoadPredictorOrNull(_env, File.OpenRead(modelName));
                var inputs = new Input[0];

                var view = DataViewConstructionUtils.CreateFromEnumerable(_env, inputs);
                _transforms = ModelFileUtils.LoadTransforms(_env, view, File.OpenRead(modelName));
                var data = _env.CreateExamples(_transforms, features);
                var scorer = _env.CreateDefaultScorer(data, _predictor);

                _valueMapper = new ValueMapperFromTransformFloat<VBuffer<float>>(_env, scorer, "Features", "Probability");
                _mapper = _valueMapper.GetMapper<VBuffer<float>, float>();
            }

            public float Predict(float[] features)
            {
                float res = 0f;
                var buf = new VBuffer<float>(features.Length, features);
                _mapper(in buf, ref res);
                return res;
            }

            public void Dispose()
            {
                //(_env as ConsoleEnvironment).Dispose();
                _valueMapper.Dispose();
                _env = null;
                _valueMapper = null;
            }
        }

        public class FloatOutput
        {
            public float Score;
            public float Probability;
        }

        public class PredictionEngineExample : IDisposable
        {
            IHostEnvironment _env;
            PredictionEngine<FloatVectorInput, FloatOutput> _predictor;

            public PredictionEngineExample(string modelName)
            {
                _env = EnvHelper.NewTestEnvironment();

                var view = DataViewConstructionUtils.CreateFromEnumerable(_env, new FloatVectorInput[] { });
                var pipe = DataViewConstructionUtils.LoadPipeWithPredictor(_env, File.OpenRead(modelName),
                                                        new EmptyDataView(_env, view.Schema));
                var transformer = new TransformWrapper(_env, pipe);
                var model = new ModelOperationsCatalog(_env);
                _predictor = model.CreatePredictionEngine<FloatVectorInput, FloatOutput>(transformer);
            }

            public Tuple<float, float> Predict(float[] features)
            {
                var res = _predictor.Predict(new FloatVectorInput() { Features = features });
                return new Tuple<float, float>(res.Score, res.Probability);
            }

            public void Dispose()
            {
                //(_env as ConsoleEnvironment).Dispose();
                _env = null;
            }
        }

        [TestMethod]
        public void TestTransform2ValueMapperSingleThreadSimple()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            using (var example = new ValueMapperExample(name, "Features"))
            {
                var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
                var ans = example.Predict(feat);
                var ans2 = example.Predict(feat);
                if (ans != ans2)
                    throw new Exception(string.Format("Issue {0} != {1}", ans, ans2));

                using (var engine = new PredictionEngineExample(name))
                {
                    var ans6 = engine.Predict(feat);
                    if (ans6.Item2 != ans2)
                        throw new Exception(string.Format("Issue {0} != {1}", ans, ans6.Item2));

                    feat = new float[] { 5, 1, 1, 1, 2, -1, 3, 1, 1 };
                    ans = example.Predict(feat);
                    ans2 = example.Predict(feat);
                    if (ans != ans2)
                        throw new Exception(string.Format("Issue {0} != {1}", ans, ans2));
                    ans6 = engine.Predict(feat);
                    if (ans6.Item2 != ans)
                        throw new Exception(string.Format("Issue {0} != {1}", ans, ans6.Item2));
                }

                using (var exampleNo = new ValueMapperExample(name, "Features"))
                {
                    var ans3 = example.Predict(feat);
                    if (ans != ans3)
                        throw new Exception(string.Format("Issue {0} != {1}", ans, ans3));
                }
            }
        }

        static void RunValueMapperExample(ValueMapperExample run, int n)
        {
            var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
            for (int i = 0; i < n; ++i)
            {
                feat[0] = i;
                run.Predict(feat);
            }
        }

        static void RunPredictionEngineExample(PredictionEngineExample run, int n)
        {
            var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
            for (int i = 0; i < n; ++i)
            {
                feat[0] = i;
                run.Predict(feat);
            }
        }

        [TestMethod]
        public void TestBcLrSameModel()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var output = FileHelper.GetOutputFile("bc-lr.zip", methodName);
            var name = FileHelper.GetOutputFile("bc.txt", methodName);
            var df = DataFrameIO.ReadStr("Label,X1,X2,X3,X4,X5,X6,X7,X8,X9\n" +
                                "0,0.1,1.1,2.1,3.1,4.1,5.1,6.2,7.4,-5\n" +
                                "1,1.1,1.1,2.1,3.1,4.1,5.1,6.2,7.4,-5\n" +
                                "0,2.1,1.1,3.1,3.1,-4.1,5.1,6.2,7.4,-5\n" +
                                "1,3.1,1.1,4.1,3.1,4.1,-5.1,6.2,7.4,-5\n" +
                                "0,4.1,1.1,2.1,3.1,4.1,5.1,6.2,-7.4,-5");
            df.ToCsv(name);
            var cmd = string.Format("Train tr=lr data={0} out={1} loader=text{{col=Label:R4:0 col=Features:R4:1-* sep=, header=+}}",
                                    name, output);

            var stdout = new StringBuilder();
            ILogWriter logout = new LogWriter((string s) => { stdout.Append(s); });
            ILogWriter logerr = new LogWriter((string s) => { stdout.Append(s); });
            /*using (*/
            var env = new DelegateEnvironment(verbose: 2, outWriter: logout, errWriter: logerr);
            MamlHelper.MamlScript(cmd, false, env);
            var stout = stdout.ToString();
            if (string.IsNullOrEmpty(stout))
                throw new Exception(stout);
        }

        [TestMethod]
        public void TestTransform2ValueMapperMeasuringTime()
        {
            // It should be run with the profiler.
#if(DEBUG)
            int n = 10;
#else
            int n = 10000;
#endif
            var name = FileHelper.GetTestFile("bc-lr.zip");
            using (var example = new ValueMapperExample(name, "Features"))
                RunValueMapperExample(example, n);
            using (var engine = new PredictionEngineExample(name))
                RunPredictionEngineExample(engine, n);
        }

        [TestMethod]
        public void TestValueMapperPredictionEngine()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            /*using (*/
            var env = EnvHelper.NewTestEnvironment();
            {
                using (var engine = new ValueMapperPredictionEngineFloat(env, name))
                {
                    var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
                    for (int i = 0; i < 1000; ++i)
                    {
                        feat[0] = i;
                        var res = engine.Predict(feat);
                        Assert.IsFalse(float.IsNaN(res));
                        Assert.IsFalse(float.IsInfinity(res));
                    }
                }
            }
        }

        public class ValueMapperPredictionEngineExample : IDisposable
        {
            ValueMapperPredictionEngineFloat engine;

            public ValueMapperPredictionEngineExample()
            {
            }

            public void Init(string modelName)
            {
                try
                {
                    /*using (*/
                    var env = EnvHelper.NewTestEnvironment();
                    engine = new ValueMapperPredictionEngineFloat(env, modelName, "Probability");
                }
                catch (Exception e)
                {
                    throw new Exception("erreur", e);
                }
            }

            public void Dispose()
            {
                engine.Dispose();
                engine = null;
            }

            public float Predict(float[] features)
            {
                return engine.Predict(features);
            }
        }

        [TestMethod]
        public void TestValueMapperPredictionEngineNotebook()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            using (var engine = new ValueMapperPredictionEngineExample())
            {
                engine.Init(name);
                var res = engine.Predict(new float[] { 8, 10, 10, 8, 7, 10, 9, 7, 1 });
                if (res < 0)
                    throw new Exception("unexpected");
                Assert.IsFalse(float.IsNaN(res));
                Assert.IsFalse(float.IsInfinity(res));
            }
        }

        [TestMethod]
        public void TestLambdaColumnPassThroughTransform()
        {
            /*using (*/
            var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new InputOutputU[] {
                    new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                    new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                    new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
                };

                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                var lambdaView = LambdaColumnHelper.Create<VBuffer<float>, VBuffer<float>>(host,
                                "Lambda", data, "X", "XX", new VectorDataViewType(NumberDataViewType.Single, 2),
                                new VectorDataViewType(NumberDataViewType.Single, 2),
                                (in VBuffer<float> src, ref VBuffer<float> dst) =>
                                {
                                    dst = new VBuffer<float>(2, new float[2]);
                                    dst.Values[0] = src.Values[0] + 1f;
                                    dst.Values[1] = src.Values[1] + 1f;
                                });

                using (var cursor = lambdaView.GetRowCursor(lambdaView.Schema))
                {
                    var labelGetter = cursor.GetGetter<uint>(SchemaHelper._dc(1, cursor));
                    var floatGetter = cursor.GetGetter<VBuffer<float>>(SchemaHelper._dc(2, cursor));
                    var array = new VBuffer<float>();
                    var cont = new List<Tuple<float, float>>();
                    while (cursor.MoveNext())
                    {
                        uint got = 0;
                        labelGetter(ref got);
                        floatGetter(ref array);
                        cont.Add(new Tuple<float, float>(array.Values[0], array.Values[1]));
                    }

                    if (cont.Count != 3)
                        throw new Exception("Should be 3");
                    for (int i = 0; i < cont.Count; ++i)
                    {
                        if (cont[i].Item1 < 1)
                            throw new Exception("Values should be > 1");
                    }
                }
            }
        }

        [TestMethod]
        public void TestValueMapperFromTransform()
        {
            foreach (var each in new[] { false, true })
            {
                /*using (*/
                var host = EnvHelper.NewTestEnvironment();
                {
                    var inputs = new InputOutputU[] {
                    new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                    new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                    new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
                };
                    var inputs_unsued = new InputOutputU[] {
                    new InputOutputU() { X = new float[] { -0.1f, -1.1f }, Y = 1000 },
                };
                    var outputs = new InputOutputU[inputs.Length];
                    for (int i = 0; i < outputs.Length; ++i)
                        outputs[i] = new InputOutputU();

                    var data_unused = DataViewConstructionUtils.CreateFromEnumerable(host, inputs_unsued);
                    var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                    /*using (*/
                    var env = EnvHelper.NewTestEnvironment();
                    {
                        var tr = new PassThroughTransform(env, new PassThroughTransform.Arguments() { }, data_unused);
                        var mapperClass = new ValueMapperFromTransform<InputOutputU, InputOutputU>(env, tr);
                        var mapper = mapperClass.GetMapper<InputOutputU, InputOutputU>();
                        using (var cur = data.GetRowCursor(data.Schema))
                        {
                            for (int i = 0; i < inputs.Length; ++i)
                                mapper(in inputs[i], ref outputs[i]);
                        }
                    }

                    for (int i = 0; i < inputs.Length; ++i)
                    {
                        Assert.AreEqual(inputs[i].Y, outputs[i].Y);
                        Assert.AreEqual(inputs[i].X.Length, outputs[i].X.Length);
                        Assert.AreEqual(inputs[i].X[0], outputs[i].X[0]);
                        Assert.AreEqual(inputs[i].X[1], outputs[i].X[1]);
                    }
                }
            }
        }

        #endregion

        #region DataFrame

        [TestMethod]
        public void TestInfiniteLoopViewCursorRowDataFrame()
        {
            var df = DataFrameIO.ReadStr("Label,X1,X2,X3,X4,X5,X6,X7,X8,X9\n" +
                                "0,0.1,1.1,2.1,3.1,4.1,5.1,6.2,7.4,-5\n" +
                                "1,1.1,1.1,2.1,3.1,4.1,5.1,6.2,7.5,-5\n" +
                                "0,2.1,1.1,3.1,3.1,-4.1,5.1,6.2,7.6,-5\n" +
                                "1,3.1,1.1,4.1,3.1,4.1,-5.1,6.2,7.7,-5\n" +
                                "0,4.1,1.1,2.1,3.1,4.1,5.1,6.2,7.8,-5");
            var df2 = DataFrameIO.ReadStr("Label,X1,X2,X3,X4,X5,X6,X7,X8,X9\n" +
                                "0,0.1,1.1,2.1,3.1,4.1,5.1,6.2,8.4,-5\n" +
                                "1,1.1,1.1,2.1,3.1,4.1,5.1,6.2,8.5,-5");

            var view = new InfiniteLoopViewCursorDataFrame(schema: df.Schema);
            var values = new List<float>();
            using (var cur = view.GetRowCursor(view.Schema))
            {
                var getter = cur.GetGetter<float>(SchemaHelper._dc(8, cur));
                float val = 0f;
                view.Set(df);
                for (int i = 0; i < df.Length; ++i)
                {
                    cur.MoveNext();
                    getter(ref val);
                    values.Add(val);
                }
                view.Set(df2);
                for (int i = 0; i < df2.Length; ++i)
                {
                    cur.MoveNext();
                    getter(ref val);
                    values.Add(val);
                }
            }
            var got = values.ToArray();
            var expected = new[] { 7.4f, 7.5f, 7.6f, 7.7f, 7.8f, 8.4f, 8.5f };
            Assert.IsTrue(expected.SequenceEqual(got));
        }

        [TestMethod]
        public void TestValueMapperDataFrameFromTransform()
        {
            var df = DataFrameHelperTest.CreateDataFrameWithAllTypes();
            var cols = SchemaHelper.EnumerateColumns(df.Schema).Select(c => new Column1x1() { Source = c, Name = c + "_" }).ToArray();
            /*using (*/
            var env = EnvHelper.NewTestEnvironment();
            {
                var tr = new AddRandomTransform(env, new AddRandomTransform.Arguments() { seed = 0, columns = cols }, df);
                var mapperTr = new ValueMapperDataFrameFromTransform(env, tr);
                var mapper = mapperTr.GetMapper<DataFrame, DataFrame>();
                DataFrame df2 = null;
                mapper(in df, ref df2);
                var tr2 = new AddRandomTransform(env, new AddRandomTransform.Arguments() { seed = 0, columns = cols }, df);
                var df3 = DataFrameHelperTest.CreateDataFrameWithAllTypes();
                DataFrame df4 = null;
                mapper(in df3, ref df4);
                var res = df2.AssertAlmostEqual(df4);
                if (res != 0)
                    throw new Exception($"Test failed.");
            }
        }

        #endregion
    }
}
