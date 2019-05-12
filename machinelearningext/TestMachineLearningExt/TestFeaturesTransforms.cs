// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineHelper;
using Scikit.ML.FeaturesTransforms;
using Scikit.ML.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestFeaturesTransforms
    {
        #region PolynomTransform

        // The function is create in a separate DLL to allow a call from an executable program.
        // The test is passing. But we need to check that the transform can be serialized.
        // We create a small pipeline, we save, we load, we save, we store data...
        [TestMethod]
        public void TestI_PolynomialTransformDense()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };
            /*using (*/
            var host = EnvHelper.NewTestEnvironment();
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                List<float[]> values;
                CommonTestPolynomialTransform(host, data, 3, out values);
            }
        }

        [TestMethod]
        public void TestI_PolynomialTransformSparse()
        {
            var inputs = new[] {
                new ExampleASparse() { X = new VBuffer<float> (5, 3, new float[] { 1, 10, 100 }, new int[] { 0, 2, 4 }) },
                new ExampleASparse() { X = new VBuffer<float> (5, 3, new float[] { 2, 3, 5 }, new int[] { 1, 2, 3 }) }
            };
            /*using (*/
            var host = EnvHelper.NewTestEnvironment();
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                List<float[]> values;
                CommonTestPolynomialTransform(host, data, 5, out values);

                List<float[]> valuesDense;
                data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                CommonTestPolynomialTransform(host, data, 5, out valuesDense);

                if (values.Count != valuesDense.Count)
                    throw new Exception("Mismath in number of observations.");
                for (int i = 0; i < values.Count; ++i)
                {
                    if (values[i].Length != valuesDense[i].Length)
                        throw new Exception("Mismath in dimensions.");
                    for (int j = 0; j < values[i].Length; ++j)
                        if (values[i][j] != valuesDense[i][j])
                            throw new Exception("Mismath in value.");
                }
            }
        }

        static void CommonTestPolynomialTransform(IHostEnvironment host, IDataView data, int dimension, out List<float[]> values)
        {
            values = null;
            for (int degree = 1; degree <= 3; ++degree)
            {
                var args = new PolynomialTransform.Arguments
                {
                    columns = new[] { new Column1x1() { Source = "X", Name = "poly" } },
                    degree = degree,
                };

                if (degree == 1)
                {
                    try
                    {
                        new PolynomialTransform(host, args, data);
                    }
                    catch (Exception)
                    {
                        continue;  // expected
                    }
                }
                var poly = new PolynomialTransform(host, args, data);

                Func<int, int>[] total = new Func<int, int>[] {
                                k => 0,
                                k => k,
                                k => k * (k + 1) / 2,
                                k => k * (k * k + 3 * k + 2) / 6
                };
                Func<int, int>[] totalCumulated = new Func<int, int>[] {
                                k => 0,
                                k => k,
                                k => total[1](k) + total[2](k),
                                k => total[1](k) + total[2](k) + total[3](k)
                };

                using (var cursor = poly.GetRowCursor(poly.Schema))
                {
                    var outValues = new List<float[]>();
                    var colGetter = cursor.GetGetter<VBuffer<float>>(SchemaHelper._dc(1, cursor));
                    while (cursor.MoveNext())
                    {
                        VBuffer<float> got = new VBuffer<float>();
                        colGetter(ref got);
                        outValues.Add(got.DenseValues().ToArray());
                    }
                    values = outValues;
                    if (outValues.Count != 2)
                        throw new Exception("expected 2");
                    var dist = outValues.Select(c => c.Length).Distinct().ToArray();
                    if (dist.Length != 1)
                        throw new Exception("Not the same number of polynomial features.");
                    if (dist.First() != totalCumulated[degree](dimension))
                        throw new Exception(string.Format("Mismatch in dimensions {0} != {1} degree={2} nbF={3}",
                            dist.First(), totalCumulated[degree](dimension), degree, dimension));
                }
            }
        }

        [TestMethod]
        public void TestI_PolynomialTransformSerialize()
        {
            /*using (*/
            var host = EnvHelper.NewTestEnvironment();
            {
                var inputs = new[] {
                    new ExampleA() { X = new float[] { 1, 10, 100 } },
                    new ExampleA() { X = new float[] { 2, 3, 5 } }
                };

                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                var data = host.CreateTransform("poly{col=poly:X d=3}", loader);

                // We create a specific folder in build/UnitTest which will contain the output.
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
                var outData = FileHelper.GetOutputFile("outData.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

                // This function serializes the output data twice, once before saving the pipeline, once after loading the pipeline.
                // It checks it gives the same result.
                TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
            }
        }

        [TestMethod]
        public void TestI_PolynomialTransformNumericValues()
        {
            /*using (*/
            var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var raw = DataFrameIO.ReadStr("A,B\n1.0,2.0\n2.0,3.0\n10.0,11.0");
                raw.SetShuffle(false);
                var loader = host.CreateTransform("concat{col=X:A,B}", raw);
                var data = host.CreateTransform("Poly{col=X}", loader);
                var res = DataFrameIO.ReadView(data);
                var txt = res.ToString();
                Assert.IsFalse(string.IsNullOrEmpty(txt));
                var exp = "A,B,X.0,X.1,X.2,X.3,X.4\n1.0,2.0,1.0,2.0,1.0,2.0,4.0\n2.0,3.0,2.0,3.0,4.0,6.0,9.0\n10.0,11.0,10.0,11.0,100.0,110.0,121.0";
                var dfexp = DataFrameIO.ReadStr(exp);
                Assert.AreEqual(0, dfexp.AlmostEquals(res, exc: true, printDf: true));
            }
        }

        #endregion

        #region ScalerTransform

        [TestMethod]
        public void TestI_ScalerTransformDenseMeanVar()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };
            /*using (*/
            var host = EnvHelper.NewTestEnvironment();
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                List<float[]> values;
                CommonTestScalerTransform(host, data, 3, ScalerTransform.ScalerStrategy.meanVar, out values);
            }
        }

        [TestMethod]
        public void TestI_ScalerTransformDenseMeanVarNoVector()
        {
            var inputs = new[] {
                new ExampleA0() { X = 1f },
                new ExampleA0() { X = 2f }
            };
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                List<float[]> values;
                CommonTestScalerTransform(host, data, 3, ScalerTransform.ScalerStrategy.meanVar, out values);
            }
        }

        [TestMethod]
        public void TestI_ScalerTransformDenseMinMax()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                List<float[]> values;
                CommonTestScalerTransform(host, data, 3, ScalerTransform.ScalerStrategy.minMax, out values);
            }
        }

        static void CommonTestScalerTransform(IHostEnvironment host, IDataView data,
                                              int dimension, ScalerTransform.ScalerStrategy strategy,
                                              out List<float[]> values)
        {
            var args = new ScalerTransform.Arguments
            {
                columns = new[] { new Column1x1() { Name = "X", Source = "X" } },
                scaling = strategy
            };

            var scaled = new ScalerTransform(host, args, data);

            using (var cursor = scaled.GetRowCursor(scaled.Schema))
            {
                var outValues = new List<float[]>();
                var colGetter = cursor.GetGetter<VBuffer<float>>(SchemaHelper._dc(0, cursor));
                VBuffer<float> got = new VBuffer<float>();
                while (cursor.MoveNext())
                {
                    colGetter(ref got);
                    outValues.Add(got.DenseValues().ToArray());
                }
                values = outValues;
                if (outValues.Count != 2)
                    throw new Exception("expected 2");
                var dist = outValues.Select(c => c.Length).Distinct().ToArray();
                if (dist.Length != 1)
                    throw new Exception("Not the same number of features.");
                var x0 = outValues.Select(c => c[0]).Distinct().ToArray();
                if (x0.Length != 2)
                    throw new Exception("It did not work.");
            }
        }

        [TestMethod]
        public void TestI_ScalerTransformSparse()
        {
            var inputs = new[] {
                new ExampleASparse() { X = new VBuffer<float> (5, 3, new float[] { 1, 10, 100 }, new int[] { 0, 2, 4 }) },
                new ExampleASparse() { X = new VBuffer<float> (5, 3, new float[] { 2, 3, 5 }, new int[] { 0, 1, 3 }) }
            };
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                List<float[]> values;
                CommonTestScalerTransform(host, data, 5, ScalerTransform.ScalerStrategy.meanVar, out values);

                List<float[]> valuesDense;
                data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                CommonTestScalerTransform(host, data, 5, ScalerTransform.ScalerStrategy.meanVar, out valuesDense);

                if (values.Count != valuesDense.Count)
                    throw new Exception("Mismath in number of observations.");
                for (int i = 0; i < values.Count; ++i)
                {
                    if (values[i].Length != valuesDense[i].Length)
                        throw new Exception("Mismath in dimensions.");
                    for (int j = 0; j < values[i].Length; ++j)
                        if (values[i][j] != valuesDense[i][j])
                            throw new Exception("Mismath in value.");
                }
            }
        }

        [TestMethod]
        public void TestI_ScalerTransformSerialize()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment();
            {

                var inputs = new[] {
                    new ExampleA() { X = new float[] { 1, 10, 100 } },
                    new ExampleA() { X = new float[] { 2, 3, 5 } }
                };

                IDataView loader = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                var data = host.CreateTransform("Scaler{col=X}", loader);
                (data as ITrainableTransform).Estimate();

                // We create a specific folder in build/UnitTest which will contain the output.
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
                var outData = FileHelper.GetOutputFile("outData.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                var nb = DataViewUtils.ComputeRowCount(data);
                if (nb < 1)
                    throw new Exception("empty view");

                // This function serializes the output data twice, once before saving the pipeline, once after loading the pipeline.
                // It checks it gives the same result.
                TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
            }
        }

        [TestMethod]
        public void TestI_ScalerTransformNumericValuesMeanVar()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var raw = DataFrameIO.ReadStr("A,B\n1.0,2.0\n2.0,3.0\n10.0,11.0");
                raw.SetShuffle(false);
                var loader = host.CreateTransform("concat{col=X:A,B}", raw);
                var data = host.CreateTransform("Scaler{col=X}", loader);
                (data as ITrainableTransform).Estimate();
                var res = DataFrameIO.ReadView(data);
                var txt = res.ToString();
                Assert.IsNotNull(txt);
                var exp = "A,B,X.0,X.1\n1.0,2.0,-0.827605963,-0.827605963\n2.0,3.0,-0.5793242,-0.5793242\n10.0,11.0,1.40693,1.40693";
                var dfexp = DataFrameIO.ReadStr(exp);
                Assert.AreEqual(0, dfexp.AlmostEquals(res, exc: true, printDf: true));
            }
        }

        [TestMethod]
        public void TestI_ScalerTransformNumericValuesMinMax()
        {
            /*using (*/var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var raw = DataFrameIO.ReadStr("A,B\n1.0,2.0\n2.0,3.0\n10.0,11.0");
                raw.SetShuffle(false);
                var loader = host.CreateTransform("concat{col=X:A,B}", raw);
                var data = host.CreateTransform("Scaler{col=X scale=minMax}", loader);
                (data as ITrainableTransform).Estimate();
                var res = DataFrameIO.ReadView(data);
                var txt = res.ToString();
                var exp = "A,B,X.0,X.1\n1.0,2.0,0.0,0.0\n2.0,3.0,0.11111111,0.11111111\n10.0,11.0,1.0,1.0";
                var dfexp = DataFrameIO.ReadStr(exp);
                Assert.AreEqual(0, dfexp.AlmostEquals(res, exc: true, printDf: true));
            }
        }

        #endregion
    }
}
