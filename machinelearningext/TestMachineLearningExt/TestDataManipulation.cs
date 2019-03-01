// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Scikit.ML.DataManipulation;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineHelper;
using Scikit.ML.ScikitAPI;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestDataManipulation
    {
        #region automated generation



        #endregion

        #region DataFrame IO

        [TestMethod]
        public void TestStreamingDataFrame()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var sdf = StreamingDataFrame.ReadCsv(iris, sep: '\t');
            var df = sdf.ToDataFrame();
            Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 5));
            var sch = df.Schema;
            Assert.AreEqual(sch[0].Name, "Label");
            Assert.AreEqual(sch[1].Name, "Sepal_length");
            Assert.AreEqual(sch[0].Type, NumberDataViewType.Int32);
            Assert.AreEqual(sch[1].Type, NumberDataViewType.Single);
        }

        [TestMethod]
        public void TestReadCsvSimple()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrameIO.ReadCsv(iris, sep: '\t');
            Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 5));
            var sch = df.Schema;
            Assert.AreEqual(sch[0].Name, "Label");
            Assert.AreEqual(sch[1].Name, "Sepal_length");
            Assert.AreEqual(sch[0].Type, NumberDataViewType.Int32);
            Assert.AreEqual(sch[1].Type, NumberDataViewType.Single);
            Assert.AreEqual(df.iloc[0, 0], (int)0);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
            Assert.AreEqual(df.iloc[140, 0], (int)2);
            df.iloc[1, 0] = (int)10;
            Assert.AreEqual(df.iloc[1, 0], (int)10);
            df.loc[1, "Label"] = (int)11;
            Assert.AreEqual(df.loc[1, "Label"], (int)11);
            var d = df[1];
            Assert.AreEqual(d.Count, 5);
            Assert.AreEqual(d["Label"], (int)11);
            var col = df["Label"];
            Assert.AreEqual(col.Length, 150);
            df["Label2"] = df["Label"];
            col = df["Label2"];
            Assert.AreEqual(col.Length, 150);
            Assert.AreEqual(df.loc[1, "Label2"], (int)11);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 6));
        }

        [TestMethod]
        public void TestReadView()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var iris = FileHelper.GetTestFile("iris.txt");
                var loader = DataFrameIO.ReadCsvToTextLoader(iris, sep: '\t', host: env.Register("TextLoader"));
                var df = DataFrameIO.ReadView(loader);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 5));
                var sch = df.Schema;
                Assert.AreEqual(sch[0].Name, "Label");
                Assert.AreEqual(sch[1].Name, "Sepal_length");
                Assert.AreEqual(sch[0].Type, NumberDataViewType.Int32);
                Assert.AreEqual(sch[1].Type, NumberDataViewType.Single);
                Assert.AreEqual(df.iloc[0, 0], (int)0);
                Assert.AreEqual(df.iloc[1, 0], (int)0);
                Assert.AreEqual(df.iloc[140, 0], (int)2);
            }
        }

        [TestMethod]
        public void TestReadViewEqual()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrameIO.ReadCsvToTextLoader(iris, sep: '\t');
            var df1 = DataFrameIO.ReadCsv(iris, sep: '\t');
            var df2 = DataFrameIO.ReadView(loader);
            Assert.IsTrue(df1 == df2);
            df2.iloc[1, 0] = (int)10;
            Assert.IsTrue(df1 != df2);
        }

        [TestMethod]
        public void TestReadTextLoaderSimple()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrameIO.ReadCsvToTextLoader(iris, sep: '\t');
            var sch = loader.Schema;
            Assert.AreEqual(sch.Count, 5);
            Assert.AreEqual(sch[0].Name, "Label");
            Assert.AreEqual(sch[1].Name, "Sepal_length");
        }

        [TestMethod]
        public void TestReadToCsv()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrameIO.ReadCsv(iris, sep: '\t');
            var outfile = FileHelper.GetOutputFile("iris_copy.txt", methodName);
            df.ToCsv(outfile);
            Assert.IsTrue(File.Exists(outfile));
        }

        [TestMethod]
        public void TestReadStr()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var df1 = DataFrameIO.ReadCsv(iris, sep: '\t');
            var content = File.ReadAllText(iris);
            var df2 = DataFrameIO.ReadStr(content, sep: '\t');
            Assert.IsTrue(df1 == df2);
        }

        [TestMethod]
        public void TestReadStrIndex()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text, index: true);
            var tos = df.ToString();
            var exp = "index,AA,BB,CC\n0,0,1,text\n1,1,1.1,text2";
            Assert.AreEqual(exp, tos);
        }

        #endregion

        #region DataFrame ML

        [TestMethod]
        public void TestDataFrameScoringMulti()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var iris = FileHelper.GetTestFile("iris.txt");
                var df = DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new DataViewType[] { NumberDataViewType.Single });
                var conc = env.CreateTransform("Concat{col=Features:Sepal_length,Sepal_width}", df);
                var trainingData = env.CreateExamples(conc, "Features", label: "Label");
                var trainer = env.CreateTrainer("ova{p=ap}");
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, trainingData);
                    var scorer = ScoreUtils.GetScorer(pred, trainingData, env, null);
                    var predictions = DataFrameIO.ReadView(scorer);
                    var v = predictions.iloc[0, 7];
                    Assert.AreEqual(v, (uint)1);
                    Assert.AreEqual(predictions.Schema[5].Name, "Features.0");
                    Assert.AreEqual(predictions.Schema[6].Name, "Features.1");
                    Assert.AreEqual(predictions.Schema[7].Name, "PredictedLabel");
                    Assert.AreEqual(predictions.Shape, new Tuple<int, int>(150, 11));
                }
            }
        }

        [TestMethod]
        public void TestDataFrameScoringBinary()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var iris = FileHelper.GetTestFile("iris.txt");
                var df = DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new DataViewType[] { NumberDataViewType.Single });
                var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
                var trainingData = env.CreateExamples(conc, "Feature", label: "Label");
                var trainer = env.CreateTrainer("lr");
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, trainingData);
                    var scorer = PredictorHelper.Predict(env, pred, trainingData);
                    var predictions = DataFrameIO.ReadView(scorer);
                    var v = predictions.iloc[0, 7];
                    Assert.AreEqual(v, false);
                    Assert.AreEqual(predictions.Schema[5].Name, "Feature.0");
                    Assert.AreEqual(predictions.Schema[6].Name, "Feature.1");
                    Assert.AreEqual(predictions.Schema[7].Name, "PredictedLabel");
                    Assert.AreEqual(predictions.Shape, new Tuple<int, int>(150, 10));
                }
            }
        }

        [TestMethod]
        public void TestDataFrameExtendedAPI()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var iris = FileHelper.GetTestFile("iris.txt");
                var df = DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new DataViewType[] { NumberDataViewType.Single });
                var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
                var trainingData = env.CreateExamples(conc, "Feature", label: "Label");
                ITrainerExtended trainer = env.CreateTrainer("lr");
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, trainingData);
                    var viewpred = PredictorHelper.Predict(env, pred, trainingData);
                    var predictions = DataFrameIO.ReadView(viewpred);
                    var v = predictions.iloc[0, 7];
                    Assert.AreEqual(v, false);
                    Assert.AreEqual(predictions.Schema[5].Name, "Feature.0");
                    Assert.AreEqual(predictions.Schema[6].Name, "Feature.1");
                    Assert.AreEqual(predictions.Schema[7].Name, "PredictedLabel");
                    Assert.AreEqual(predictions.Shape, new Tuple<int, int>(150, 10));
                }
            }
        }

        #endregion

        #region DataFrame Operators

        [TestMethod]
        public void TestDataFrameOperation()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BBxBB"] = df["AA"] + df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 1f);
            Assert.AreEqual(df.iloc[1, 3], 2.1f);

            df["BBxBB2"] = df["BB"] + df["AA"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], 1f);
            Assert.AreEqual(df.iloc[1, 4], 2.1f);

            df["AA2"] = df["AA"] + 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 6));
            Assert.AreEqual(df.iloc[0, 5], (int)10);
            Assert.AreEqual(df.iloc[1, 5], (int)11);

            df["CC2"] = df["CC"] + "10";
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 7));
            Assert.AreEqual(df.iloc[0, 6].ToString(), "text10");
            Assert.AreEqual(df.iloc[1, 6].ToString(), "text210");
        }

        [TestMethod]
        public void TestDataFrameOpMult()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] * df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 0f);
            Assert.AreEqual(df.iloc[1, 3], 1.1f);

            df["AA2"] = df["AA"] * 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], (int)0);
            Assert.AreEqual(df.iloc[1, 4], (int)10);
        }

        [TestMethod]
        public void TestDataFrameOpMinus()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] - df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], -1f);
            Assert.AreEqual(df.iloc[1, 3], 1 - 1.1f);

            df["AA2"] = df["AA"] - 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], (int)(-10));
            Assert.AreEqual(df.iloc[1, 4], (int)(-9));
        }

        [TestMethod]
        public void TestDataFrameOpEqual()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] == df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], false);
            Assert.AreEqual(df.iloc[1, 3], false);

            df["AA2"] = df["AA"] == 0;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], true);
            Assert.AreEqual(df.iloc[1, 4], false);

            var view = df[df["AA"] == 0];
            Assert.AreEqual(view.Shape, new Tuple<int, int>(1, 5));

            df["CC2"] = df["CC"] == "text";
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 6));
            Assert.AreEqual(df.iloc[0, 5], true);
            Assert.AreEqual(df.iloc[1, 5], false);
        }

        [TestMethod]
        public void TestDataFrameOpDiv()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] / df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 0f);
            Assert.AreEqual(df.iloc[1, 3], 1 / 1.1f);

            df["AA2"] = df["AA"] / 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], (int)(0));
            Assert.AreEqual(df.iloc[1, 4], (int)(0));
        }

        [TestMethod]
        public void TestDataFrameOpSup()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] > df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], false);
            Assert.AreEqual(df.iloc[1, 3], false);
        }

        [TestMethod]
        public void TestDataFrameOpSupEqual()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] >= df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], false);
            Assert.AreEqual(df.iloc[1, 3], false);
        }

        [TestMethod]
        public void TestDataFrameOpInf()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] < df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], true);
            Assert.AreEqual(df.iloc[1, 3], true);
        }

        [TestMethod]
        public void TestDataFrameOpInfEqual()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] <= df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], true);
            Assert.AreEqual(df.iloc[1, 3], true);
        }

        [TestMethod]
        public void TestDataFrameOpMinusUni()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["min"] = -df["AA"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], (int)0);
            Assert.AreEqual(df.iloc[1, 3], (int)(-1));
        }

        [TestMethod]
        public void TestDataFrameOpNotUni()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["min"] = !(df["AA"] == 1);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], true);
            Assert.AreEqual(df.iloc[1, 3], false);
        }

        [TestMethod]
        public void TestDataFrameOpPlusEqual()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB"] += df["AA"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            Assert.AreEqual(df.iloc[0, 1], 1f);
            Assert.AreEqual(df.iloc[1, 1], 2.1f);
        }

        [TestMethod]
        public void TestDataFrameOpAnd()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["and"] = (df["AA"] == 0) & (df["BB"] == 1f);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], true);
            Assert.AreEqual(df.iloc[1, 3], false);
        }

        [TestMethod]
        public void TestDataFrameOpOr()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["or"] = (df["AA"] == 1) | (df["BB"] == 1f);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], true);
            Assert.AreEqual(df.iloc[1, 3], true);
        }

        #endregion

        #region DataFrame Copy

        [TestMethod]
        public void TestDataFrameOperationSet()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            df.loc["CC"] = "changed";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed");
        }

        [TestMethod]
        public void TestDataFrameOperationIEnumerable()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            df.iloc[df["AA"].Filter<int>(c => (int)c == 1), 2] = "changed";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed");
            df.loc[df["AA"].Filter<int>(c => (int)c == 1), "CC"] = "changed2";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed2");
        }

        [TestMethod]
        public void TestDataFrameOperationCopy()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            var tos = df.ToString();
            var copy = df.Copy();
            var tos2 = copy.ToString();
            Assert.AreEqual(tos, tos2);
            copy.iloc[copy["AA"].Filter<int>(c => (int)c == 1), 2] = "changed";
            tos2 = copy.ToString();
            Assert.AreNotEqual(tos, tos2);
        }

        [TestMethod]
        public void TestDataViewFrame()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,2.1,text3";
            var df = DataFrameIO.ReadStr(text);
            var view = df[new int[] { 0, 2 }, new int[] { 0, 2 }];
            var tv = view.ToString();
            Assert.AreEqual("AA,CC\n0,text\n2,text3", tv);
            var dfview = view.Copy();
            var tv2 = dfview.ToString();
            Assert.AreEqual("AA,CC\n0,text\n2,text3", tv2);
            dfview["AA1"] = view["AA"] + 1;
            var tv3 = dfview.ToString();
            Assert.AreEqual("AA,CC,AA1\n0,text,1\n2,text3,3", tv3);
            var view2 = df[df.ALL, new[] { "AA", "CC" }];
            var tv4 = view2.ToString();
            Assert.AreEqual("AA,CC\n0,text\n1,text2\n2,text3", tv4);
            var view3 = df[new[] { 0 }, df.ALL];
            var tv5 = view3.ToString();
            Assert.AreEqual("AA,BB,CC\n0,1,text", tv5);
        }

        [TestMethod]
        public void TestCreateFromArrays()
        {
            var df = new DataFrame();
            df.AddColumn("i", new int[] { 0, 1 });
            df.AddColumn("x", new float[] { 0.5f, 1.5f });
            var tx = df.ToString();
            Assert.AreEqual(tx, "i,x\n0,0.5\n1,1.5");
        }

        #endregion

        #region dataframe function

        [TestMethod]
        public void TestDataFrameColumnApply()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["fAA"] = df["AA"].Apply((in int vin, ref float vout) => { vout = (float)vin; });
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 0f);
            Assert.AreEqual(df.iloc[1, 3], 1f);
        }

        [TestMethod]
        public void TestDataFrameColumnDrop()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            var view = df.Drop(new[] { "AA" });
            Assert.AreEqual(view.Shape, new Tuple<int, int>(2, 2));
            Assert.AreEqual(view.iloc[0, 0], 1f);
            Assert.AreEqual(view.iloc[1, 0], 1.1f);
        }

        [TestMethod]
        public void TestDataFrameSortColumn()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            var order = df["AA"].Sort(false, false);
            Assert.AreEqual(order.Length, 2);
            Assert.AreEqual(order[0], 1);
            Assert.AreEqual(order[1], 0);

            df["AA"].Sort(false, true);
            Assert.AreEqual(order.Length, 2);
            Assert.AreEqual(df.iloc[0, 0], (int)1);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
        }

        [TestMethod]
        public void TestDataFrameEnumerateItems()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            var el1 = df.EnumerateItems<int>(new[] { "AA" }).Select(c => c.ToTuple()).ToArray();
            Assert.AreEqual(el1.Length, 2);
            Assert.AreEqual(el1[0].Item1, 0);
            Assert.AreEqual(el1[1].Item1, 1);

            var el2 = df.EnumerateItems<int, float>(new[] { "AA", "BB" }).Select(c => c.ToTuple()).ToArray();
            Assert.AreEqual(el2.Length, 2);
            Assert.AreEqual(el2[0].Item1, 0);
            Assert.AreEqual(el2[1].Item1, 1);
            Assert.AreEqual(el2[0].Item2, 1f);
            Assert.AreEqual(el2[1].Item2, 1.1f);

            var el3 = df.EnumerateItems<int, float, DvText>(new[] { "AA", "BB", "CC" }).Select(c => c.ToTuple()).ToArray();
            Assert.AreEqual(el3.Length, 2);
            Assert.AreEqual(el3[0].Item1, 0);
            Assert.AreEqual(el3[1].Item1, 1);
            Assert.AreEqual(el3[0].Item2, 1f);
            Assert.AreEqual(el3[1].Item2, 1.1f);
            Assert.AreEqual(el3[0].Item3.ToString(), "text");
            Assert.AreEqual(el3[1].Item3.ToString(), "text2");
        }

        #endregion

        #region SQL function

        [TestMethod]
        public void TestDataFrameSort()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2\n0,-1.1,text3";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(3, 3));

            df.TSort<int, float>(new[] { 0, 1 });
            Assert.AreEqual(df.iloc[0, 0], (int)0);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
            Assert.AreEqual(df.iloc[2, 0], (int)1);
            Assert.AreEqual(df.iloc[0, 1], -1.1f);
            Assert.AreEqual(df.iloc[1, 1], 1f);
            Assert.AreEqual(df.iloc[2, 1], 1.1f);

            df.TSort<int, float>(new[] { 0, 1 }, false);
            Assert.AreEqual(df.iloc[2, 0], (int)0);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
            Assert.AreEqual(df.iloc[0, 0], (int)1);
            Assert.AreEqual(df.iloc[2, 1], -1.1f);
            Assert.AreEqual(df.iloc[1, 1], 1f);
            Assert.AreEqual(df.iloc[0, 1], 1.1f);
        }

        [TestMethod]
        public void TestDataFrameSortUnTyped()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2\n0,-1.1,text3";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(3, 3));

            df.Sort(new[] { "AA" });
            Assert.AreEqual(df.iloc[0, 0], (int)0);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
            Assert.AreEqual(df.iloc[2, 0], (int)1);
        }

        [TestMethod]
        public void TestDataFrameSort2Untyped()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2\n0,-1.1,text3";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(3, 3));

            df.Sort(new[] { "AA", "BB" });
            Assert.AreEqual(df.iloc[0, 0], (int)0);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
            Assert.AreEqual(df.iloc[2, 0], (int)1);
            Assert.AreEqual(df.iloc[0, 1], -1.1f);
            Assert.AreEqual(df.iloc[1, 1], 1f);
            Assert.AreEqual(df.iloc[2, 1], 1.1f);

            df.Sort(new[] { "AA", "BB" }, false);
            Assert.AreEqual(df.iloc[2, 0], (int)0);
            Assert.AreEqual(df.iloc[1, 0], (int)0);
            Assert.AreEqual(df.iloc[0, 0], (int)1);
            Assert.AreEqual(df.iloc[2, 1], -1.1f);
            Assert.AreEqual(df.iloc[1, 1], 1f);
            Assert.AreEqual(df.iloc[0, 1], 1.1f);
        }

        [TestMethod]
        public void TestDataFrameSortSimpleTypes()
        {
            var df = DataFrameHelperTest.CreateDataFrameWithAllTypes();
            df.Sort();
            var st = df.ToString();
            Assert.IsTrue(!string.IsNullOrEmpty(st));
            Assert.IsTrue(st.StartsWith("cbool,cint,cuint,cint64,cfloat,cdouble,ctext"));
        }

        [TestMethod]
        public void TestDataFrameSortVectorTypes()
        {
            var df = DataFrameHelperTest.CreateDataFrameWithAllTypes();
            df.Sort(new[] { df.ColumnCount - 1, df.ColumnCount - 2 });
            var st = df.ToString();
            Assert.IsTrue(!string.IsNullOrEmpty(st));
            Assert.IsTrue(st.StartsWith("cbool,cint,cuint,cint64,cfloat,cdouble,ctext"));
        }

        [TestMethod]
        public void TestDataFrameSortView()
        {
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2\n0,-1.1,text3";
            var df = DataFrameIO.ReadStr(text);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(3, 3));
            var view = df[new int[] { 1, 2 }];
            Assert.AreEqual(view.Length, 2);

            view.TSort<int, float>(new[] { 0, 1 });
            Assert.AreEqual(view.iloc[0, 0], (int)0);
            Assert.AreEqual(view.iloc[1, 0], (int)1);
            Assert.AreEqual(view.iloc[0, 1], -1.1f);
            Assert.AreEqual(view.iloc[1, 1], 1.1f);

            view.TSort<int, float>(new[] { 0, 1 }, false);
            Assert.AreEqual(view.iloc[1, 0], (int)0);
            Assert.AreEqual(view.iloc[0, 0], (int)1);
            Assert.AreEqual(view.iloc[1, 1], -1.1f);
            Assert.AreEqual(view.iloc[0, 1], 1.1f);
        }

        [TestMethod]
        public void TestDataFrameDict()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            Assert.AreEqual(text, df.ToString());

            var rows2 = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", (int)0 }, {"BB", 1f }, {"CC", new DvText("text") } },
                new Dictionary<string, object>() { {"AA", (int)1 }, {"BB", 1.1f }, {"CC", new DvText("text2") } },
            };
            df = new DataFrame(rows2);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            Assert.AreEqual(text, df.ToString());
        }

        [TestMethod]
        public void TestDataFrameAggregated()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            var su = df.Sum();
            Assert.AreEqual(su.Shape, new Tuple<int, int>(1, 3));
            var text = "AA,BB,CC\n1,2.1,texttext2";
            Assert.AreEqual(text, su.ToString());

            su = df.Min();
            Assert.AreEqual(su.Shape, new Tuple<int, int>(1, 3));
            text = "AA,BB,CC\n0,1,text";
            Assert.AreEqual(text, su.ToString());

            su = df.Max();
            Assert.AreEqual(su.Shape, new Tuple<int, int>(1, 3));
            text = "AA,BB,CC\n1,1.1,text2";
            Assert.AreEqual(text, su.ToString());

            su = df.Mean();
            Assert.AreEqual(su.Shape, new Tuple<int, int>(1, 3));
            text = "AA,BB,CC\n0,1.05,\"\"";
            Assert.AreEqual(text, su.ToString());

            su = df.Count();
            Assert.AreEqual(su.Shape, new Tuple<int, int>(1, 3));
            text = "AA,BB,CC\n2,2,\"\"";
            Assert.AreEqual(text, su.ToString());

            su = df[new[] { 0 }].Count();
            Assert.AreEqual(su.Shape, new Tuple<int, int>(1, 3));
            text = "AA,BB,CC\n1,1,\"\"";
            Assert.AreEqual(text, su.ToString());
        }

        [TestMethod]
        public void TestDataFrameConcatenate()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            var conc = DataFrame.Concat(new[] { df, df });
            Assert.AreEqual(conc.Shape, new Tuple<int, int>(4, 3));

            rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f } },
            };
            var df2 = new DataFrame(rows);
            conc = DataFrame.Concat(new[] { df, df2 });
            Assert.AreEqual(conc.Shape, new Tuple<int, int>(4, 3));
            Assert.AreEqual(conc.iloc[0, 0], (int)0);
            Assert.AreEqual(conc.iloc[3, 2].ToString(), "");
            Assert.AreEqual(conc.iloc[2, 0], (int)0);

            rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"BB", 1f }, {"CC", new DvText("text") } },
                new Dictionary<string, object>() { {"AA", (int)1 }, {"BB", 1.1f } },
            };
            df2 = new DataFrame(rows);
            conc = DataFrame.Concat(new[] { df, df2 });
            Assert.AreEqual(conc.Shape, new Tuple<int, int>(4, 3));
            Assert.AreEqual(conc.iloc[2, 0], 0);
            Assert.AreEqual(conc.iloc[3, 2], DvText.NA);
        }

        [TestMethod]
        public void TestDataFrameGroupBy()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1.1f }, {"CC", "text3" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text5" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.TGroupBy<int>(new int[] { 0 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            var text = gr.ToString();
            var exp = "AA,BB,CC\n0,2,\"\"\n1,2,\"\"\n2,1,\"\"";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int>(new int[] { 0 }).Sum();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,2.1,texttext3\n1,2.2,text2text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int>(new int[] { 0 }).Min();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int>(new int[] { 0 }).Max();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1.1,text3\n1,1.1,text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int>(new int[] { 0 }).Mean();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1.05,\"\"\n1,1.1,\"\"\n2,1.1,\"\"";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameGroupBy2()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text3" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text5" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.TGroupBy<int, float>(new int[] { 0, 1 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            var text = gr.ToString();
            var exp = "AA,BB,CC\n0,1,\"\"\n1,1.1,\"\"\n2,1.1,\"\"";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float>(new int[] { 0, 1 }).Sum();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,texttext3\n1,1.1,text2text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float>(new int[] { 0, 1 }).Min();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float>(new int[] { 0, 1 }).Max();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text3\n1,1.1,text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float>(new int[] { 0, 1 }).Mean();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,\"\"\n1,1.1,\"\"\n2,1.1,\"\"";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameGroupBy3()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.TGroupBy<int, float, DvText>(new int[] { 0, 1, 2 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            var text = gr.ToString();
            var exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float, DvText>(new int[] { 0, 1, 2 }).Sum();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float, DvText>(new int[] { 0, 1, 2 }).Min();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float, DvText>(new int[] { 0, 1, 2 }).Max();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.TGroupBy<int, float, DvText>(new int[] { 0, 1, 2 }).Mean();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameGroupByAgnostic()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1.1f }, {"CC", "text3" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text5" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.GroupBy(new int[] { 0 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            var text = gr.ToString();
            var exp = "AA,BB,CC\n0,2,\"\"\n1,2,\"\"\n2,1,\"\"";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0 }).Sum();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,2.1,texttext3\n1,2.2,text2text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0 }).Min();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0 }).Max();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1.1,text3\n1,1.1,text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0 }).Mean();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1.05,\"\"\n1,1.1,\"\"\n2,1.1,\"\"";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameGroupByAgnostic2()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text3" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text5" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.GroupBy(new int[] { 0, 1 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            var text = gr.ToString();
            var exp = "AA,BB,CC\n0,1,\"\"\n1,1.1,\"\"\n2,1.1,\"\"";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1 }).Sum();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,texttext3\n1,1.1,text2text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1 }).Min();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1 }).Max();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text3\n1,1.1,text5\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1 }).Mean();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,\"\"\n1,1.1,\"\"\n2,1.1,\"\"";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameGroupByAgnostic3()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.GroupBy(new int[] { 0, 1, 2 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            var text = gr.ToString();
            var exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1, 2 }).Sum();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1, 2 }).Min();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1, 2 }).Max();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);

            gr = df.GroupBy(new int[] { 0, 1, 2 }).Mean();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(3, 3));
            text = gr.ToString();
            exp = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,1.1,text4";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameGroupByOrder()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));

            var gr = df.GroupBy(new int[] { 1 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(2, 3));
            var text = gr.ToString();
            var exp = "BB,AA,CC\n1,2,\"\"\n1.1,3,\"\"";
            Assert.AreEqual(exp, text);

            var view = df[new[] { "AA", "BB" }];
            gr = view.GroupBy(new int[] { 1 }).Count();
            Assert.AreEqual(gr.Shape, new Tuple<int, int>(2, 2));
            text = gr.ToString();
            exp = "BB,AA\n1,2\n1.1,3";
            Assert.AreEqual(exp, text);
        }

        [TestMethod]
        public void TestDataFrameJoin()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text" } },
                new Dictionary<string, object>() { {"AA", 2 }, {"BB", 1.1f }, {"CC", "text4" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df = new DataFrame(rows);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(5, 3));
            var res = df.Join(df, new[] { 0 }, new[] { 0 });
            Assert.AreEqual(res.Shape, new Tuple<int, int>(9, 6));
            var exp = string.Join("\n", new string[] {
                    "AA,BB,CC,AA_y,BB_y,CC_y",
                    "0,1,text,0,1,text",
                    "0,1,text,0,1,text",
                    "0,1,text,0,1,text",
                    "0,1,text,0,1,text",
                    "1,1.1,text2,1,1.1,text2",
                    "1,1.1,text2,1,1.1,text2",
                    "1,1.1,text2,1,1.1,text2",
                    "1,1.1,text2,1,1.1,text2",
                    "2,1.1,text4,2,1.1,text4" });
            var tos = res.ToString();
            Assert.AreEqual(exp, tos);
        }

        [TestMethod]
        public void TestDataFrameJoinType()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }, {"CC", "text0" } },
                new Dictionary<string, object>() { {"AA", 1 }, {"BB", 1.1f }, {"CC", "text2" } },
            };
            var df1 = new DataFrame(rows);
            rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA2", 2 }, {"BB2", 3f }, {"CC", "TEXT2" }, { "DD", 1 }, },
                new Dictionary<string, object>() { {"AA2", 1 }, {"BB2", 4.1f }, {"CC", "TEXT1" }, { "DD", 0 }, },
            };
            var df2 = new DataFrame(rows);

            var res = df1.Join(df2, new[] { 0 }, new[] { 0 });
            var tos = res.ToString();
            Assert.AreEqual(res.Shape, new Tuple<int, int>(1, 7));
            var exp = "AA,BB,CC,AA2,BB2,CC_y,DD\n1,1.1,text2,1,4.1,TEXT1,0";
            Assert.AreEqual(exp, tos);

            res = df1.Join(df2, new[] { 0 }, new[] { 0 }, joinType: JoinStrategy.Left);
            tos = res.ToString();
            Assert.AreEqual(res.Shape, new Tuple<int, int>(2, 7));
            exp = "AA,BB,CC,AA2,BB2,CC_y,DD\n0,1,text0,-2147483648,?,\"\",-2147483648\n1,1.1,text2,1,4.1,TEXT1,0";
            Assert.AreEqual(exp, tos);

            res = df1.Join(df2, new[] { 0 }, new[] { 0 }, joinType: JoinStrategy.Right);
            tos = res.ToString();
            Assert.AreEqual(res.Shape, new Tuple<int, int>(2, 7));
            exp = "AA,BB,CC,AA2,BB2,CC_y,DD\n1,1.1,text2,1,4.1,TEXT1,0\n-2147483648,?,\"\",2,3,TEXT2,1";
            Assert.AreEqual(exp, tos);

            res = df1.Join(df2, new[] { 0 }, new[] { 0 }, joinType: JoinStrategy.Outer);
            tos = res.ToString();
            Assert.AreEqual(res.Shape, new Tuple<int, int>(3, 7));
            exp = "AA,BB,CC,AA2,BB2,CC_y,DD\n0,1,text0,-2147483648,?,\"\",-2147483648\n1,1.1,text2,1,4.1,TEXT1,0\n-2147483648,?,\"\",2,3,TEXT2,1";
            Assert.AreEqual(exp, tos);
        }

        [TestMethod]
        public void TestDataFrameJoinMultiplication()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }},
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 2f }},
            };
            var df1 = new DataFrame(rows);
            rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA2", 0 }, {"BB2", 3f } },
                new Dictionary<string, object>() { {"AA2", 0 }, {"BB2", 4f } },
                new Dictionary<string, object>() { {"AA2", 0 }, {"BB2", 5f } },
            };

            var df2 = new DataFrame(rows);

            var res = df1.Join(df2, new[] { 0 }, new[] { 0 });
            var tos = res.ToString();
            Assert.AreEqual(res.Shape, new Tuple<int, int>(6, 4));
            var exp = "AA,BB,AA2,BB2\n0,1,0,3\n0,2,0,3\n0,1,0,4\n0,2,0,4\n0,1,0,5\n0,2,0,5";
            Assert.AreEqual(exp, tos);
        }

        [TestMethod]
        public void TestDataFrameJoinHeadTail()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }},
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 2f }},
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 3f }},
            };
            var df = new DataFrame(rows);
            var head = df.Head(1);
            var tos = head.ToString();
            var exp = "AA,BB\n0,1";
            Assert.AreEqual(exp, tos);
            head = df.Head(2).Head(1);
            tos = head.ToString();
            Assert.AreEqual(exp, tos);

            var tail = df.Tail(1);
            tos = tail.ToString();
            exp = "AA,BB\n0,3";
            Assert.AreEqual(exp, tos);
            tail = df.Tail(2).Tail(1);
            tos = tail.ToString();
            Assert.AreEqual(exp, tos);
        }


        [TestMethod]
        public void TestDataFrameJoinSample()
        {
            var rows = new Dictionary<string, object>[]
            {
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 1f }},
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 2f }},
                new Dictionary<string, object>() { {"AA", 0 }, {"BB", 3f }},
            };
            var df = new DataFrame(rows);
            var sample = df.Sample(1);
            Assert.AreEqual(sample.Shape, new Tuple<int, int>(1, 2));
            sample = df.Sample(3, true);
            Assert.AreEqual(sample.Shape, new Tuple<int, int>(3, 2));
            sample.Sort(new[] { "BB" });
            var exp = "AA,BB\n0,1\n0,2\n0,3";
            var tos = sample.ToString();
            Assert.AreEqual(exp, tos);
        }

        #endregion

        #region Vector Column

        [TestMethod]
        public void TestDataFrameVectorColumn()
        {
            var inputs = new[] {
                new ExampleXY() { X = 1f, Y=2f },
                new ExampleXY() { X = 3f, Y=4f },
            };

            /*using (*/var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                using (var pipe = new ScikitPipeline(new[] { "Concat{col=Z:X,Y}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    var predictions = pipe.Transform(data);
                    var df = DataFrameIO.ReadView(predictions, keepVectors: true);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(dfs2, "X,Y,Z.0,Z.1;1,2,1,2;3,4,3,4");
                }
            }
        }

        [TestMethod]
        public void TestDataFrame_Flatten()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 7 } },
            };
            DataFrame df1;
            /*using (*/var host = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var data = DataViewConstructionUtils.CreateFromEnumerable(host, inputs);
                df1 = DataFrameIO.ReadView(data, env: host, keepVectors: true);
            }
            var flat = df1.Flatten();
            Assert.AreEqual(flat.Shape, new ShapeType(4, 3));
            var st = flat.ToString();
            Assert.AreEqual(st, "X.0,X.1,X.2\n1,10,100\n2,3,5\n2,4,5\n2,4,7");
        }

        #endregion
    }
}

