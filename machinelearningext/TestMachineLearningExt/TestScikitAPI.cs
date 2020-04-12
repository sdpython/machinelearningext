// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Runtime.InteropServices;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Scikit.ML.TestHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestScikitAPI
    {
        [TestMethod]
        public void TestScikitAPI_SimpleTransform()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            /*using (*/
            var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    Assert.IsTrue(predictor != null);
                    var data2 = DataViewConstructionUtils.CreateFromEnumerable(host, inputs2);
                    var predictions = pipe.Transform(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(dfs2, "X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8;-1,-10,-100,1,10,100,100,1000,10000;-2,-3,-5,4,6,10,9,15,25");
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_SimpleTransform_Load()
        {
            ScikitAPI_SimpleTransform_Load(false);
        }

        [TestMethod]
        public void TestScikitAPI_SimpleTransform_LoadNoPass()
        {
            ScikitAPI_SimpleTransform_Load(true);
        }

        private void ScikitAPI_SimpleTransform_Load(bool removeFirstTransform)
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var output = FileHelper.GetOutputFile($"model{removeFirstTransform}.zip", methodName);
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            string expected = null;
            /*using (*/
            var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    Assert.IsTrue(predictor != null);
                    var data2 = DataViewConstructionUtils.CreateFromEnumerable(host, inputs2);
                    var predictions = pipe.Transform(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    expected = dfs2;
                    Assert.AreEqual(dfs2, "X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8;-1,-10,-100,1,10,100,100,1000,10000;-2,-3,-5,4,6,10,9,15,25");
                    pipe.Save(output, removeFirstTransform);
                }
            }
            /*using (*/
            host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data2 = DataViewConstructionUtils.CreateFromEnumerable(host, inputs2);
                using (var pipe2 = new ScikitPipeline(output, host))
                {
                    var predictions = pipe2.Transform(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(expected, dfs2);
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_SimplePredictor()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } },
                new ExampleA() { X = new float[] { 3, 4, 5 } },
                new ExampleA() { X = new float[] { 3, 4, 7 } },
            };

            /*using (*/
            var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    Assert.IsTrue(predictor != null);
                    var data2 = new StreamingDataFrame(DataViewConstructionUtils.CreateFromEnumerable(host, inputs2));
                    var predictions = pipe.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.IsTrue(dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"));
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_SimplePredictor_Load()
        {
            ScikitAPI_SimplePredictor_Load(false);
        }

        [TestMethod]
        public void TestScikitAPI_SimplePredictor_LoadNoPass()
        {
            ScikitAPI_SimplePredictor_Load(true);
        }

        private void ScikitAPI_SimplePredictor_Load(bool removeFirstTransform)
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var output = FileHelper.GetOutputFile($"model{removeFirstTransform}.zip", methodName);
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } },
                new ExampleA() { X = new float[] { 3, 4, 5 } },
                new ExampleA() { X = new float[] { 3, 4, 7 } },
            };

            string expected = null;
            /*using (*/
            var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    Assert.IsTrue(predictor != null);
                    var data2 = DataViewConstructionUtils.CreateFromEnumerable(host, inputs2);
                    var predictions = pipe.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    if (!dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"))
                        throw new Exception($"Wrong starts\n{dfs2}");
                    expected = dfs2;
                    pipe.Save(output, removeFirstTransform);
                }
            }
            /*using (*/
            host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data2 = DataViewConstructionUtils.CreateFromEnumerable(host, inputs2);
                using (var pipe2 = new ScikitPipeline(output, host))
                {
                    var predictions = pipe2.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(expected, dfs2);
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_DelegateEnvironment()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            var stdout = new List<string>();
            var stderr = new List<string>();
            ILogWriter logout = new LogWriter((string s) =>
            {
                stdout.Add(s);
            });
            ILogWriter logerr = new LogWriter((string s) =>
            {
                stderr.Add(s);
            });

            /*using (*/
            var host = new DelegateEnvironment(seed: 0, outWriter: logout, errWriter: logerr, verbose: 3);
            using (var ch = host.Start("Train Pipeline"))
            {
                ComponentHelper.AddStandardComponents(host);
                ch.Info(MessageSensitivity.All, "Polynomial");
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    if (predictor == null)
                        throw new Exception("Predictor is null");
                }
            }
            if (stdout.Count == 0)
                throw new Exception("stdout is empty.");
            if (stderr.Count != 0)
                throw new Exception($"stderr not empty\n{string.Join("\n", stderr)}");
        }

        [TestMethod]
        public void TestScikitAPI_DelegateEnvironmentVerbose0()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            var stdout = new List<string>();
            var stderr = new List<string>();
            ILogWriter logout = new LogWriter(s => stdout.Add(s));
            ILogWriter logerr = new LogWriter(s => stderr.Add(s));

            /*using (*/
            var host = new DelegateEnvironment(seed: 0, outWriter: logout, errWriter: logerr, verbose: 0);
            {
                ComponentHelper.AddStandardComponents(host);
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host: host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    Assert.IsTrue(predictor != null);
                }
            }
            Assert.AreEqual(stdout.Count, 0);
            Assert.AreEqual(stderr.Count, 0);
        }

        [TestMethod]
        public void TestScikitAPI_SimplePredictor_FastValueMapper()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } },
                new ExampleA() { X = new float[] { 3, 4, 5 } },
                new ExampleA() { X = new float[] { 3, 4, 7 } },
            };
            DataFrame df1, df2, df3;
            /*using (*/
            var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                var data2 = DataViewConstructionUtils.CreateFromEnumerable(host, inputs2);
                df1 = DataFrameIO.ReadView(data, env: host, keepVectors: true);
                df2 = DataFrameIO.ReadView(data2, env: host, keepVectors: true);
                df3 = DataFrameIO.ReadView(data2, env: host, keepVectors: true);
            }

            /*using (*/
            host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host))
                {
                    DataFrame pred = null, pred2 = null;
                    var predictor = pipe.Train(df1, feature: "X");
                    Assert.IsTrue(predictor != null);

                    pipe.Predict(df2, ref pred);
                    Assert.AreEqual(pred.Shape, new Tuple<int, int>(4, 3));
                    var dfs = pred.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    if (!dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"))
                        throw new Exception($"Wrong starts\n{dfs2}");

                    pipe.Predict(df3, ref pred2);
                    pred.AssertAlmostEqual(pred2);
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_TrainingWithIris()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrameIO.ReadCsv(iris, sep: '\t');
            df.AddColumn("LabelI", df["Label"].AsType(NumberDataViewType.Single));
            var pipe = new ScikitPipeline(new[] { $"Concat{{col=Features:{df.Columns[1]},{df.Columns[2]}}}" }, "mlr");
            pipe.Train(df, "Features", "LabelI");
            DataFrame pred = null;
            pipe.Predict(df, ref pred);
            Assert.AreEqual(pred.Shape, new ShapeType(150, 9));
        }

        internal static class Mkl
        {
            private const string DllName = "MklImports";

            public enum Layout
            {
                RowMajor = 101,
                ColMajor = 102
            }

            public enum UpLo : byte
            {
                Up = (byte)'U',
                Lo = (byte)'L'
            }

            [DllImport(DllName, EntryPoint = "LAPACKE_dpptrf")]
            public static extern int PptrfInternal(Layout layout, UpLo uplo, int n, Double[] ap);
        }

        [TestMethod]
        public void TestScikitAPI_MKL()
        {
            try
            {
                Mkl.PptrfInternal(Mkl.Layout.ColMajor, Mkl.UpLo.Lo, 2, new double[] { 0.1, 0.3 });
            }
            catch (DllNotFoundException e)
            {
                var os = Environment.OSVersion;
                if (os.Platform == PlatformID.Unix)
                    Console.WriteLine("FAIL: TestScikitAPI_MKL due to {0}", e.ToString());
                else
                {
                    Console.WriteLine("FAIL: TestScikitAPI_MKL, OS={0}", os.ToString());
                    throw e;
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_MKL_TrainingDiabete()
        {
            try
            {
                var diab = FileHelper.GetTestFile("diabete.csv");
                var cols = Enumerable.Range(0, 10).Select(c => NumberDataViewType.Single).ToArray();
                var colsName = string.Join(',', Enumerable.Range(0, 10).Select(c => $"F{c}"));
                var df = DataFrameIO.ReadCsv(diab, sep: ',', dtypes: cols);
                var pipe = new ScikitPipeline(new string[] { $"Concat{{col=Features:{colsName}}}" }, "ols");
                pipe.Train(df, "Features", "Label");
                DataFrame pred = null;
                pipe.Predict(df, ref pred);
                Assert.AreEqual(pred.Shape, new ShapeType(83, 13));
            }
            catch (DllNotFoundException e)
            {
                var os = Environment.OSVersion;
                if (os.Platform == PlatformID.Unix)
                {
                    Console.WriteLine("FAIL(1): TestScikitAPI_MKL due to {0}", e.ToString());
                    return;
                }
                else
                {
                    Console.WriteLine("FAIL(1): TestScikitAPI_MKL, OS={0}", os.ToString());
                    throw e;
                }
            }
            catch (NotSupportedException e)
            {
                var os = Environment.OSVersion;
                if (os.Platform == PlatformID.Unix)
                {
                    Console.WriteLine("FAIL(2): TestScikitAPI_MKL due to {0}", e.ToString());
                    return;
                }
                else
                {
                    Console.WriteLine("FAIL(2): TestScikitAPI_MKL, OS={0}", os.ToString());
                    throw e;
                }
            }
        }
    }
}
