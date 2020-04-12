// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Text;
using System.IO;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;
using Scikit.ML.DataManipulation;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DocHelperMlExt;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestMLNet
    {
        [TestMethod]
        public void TestTreePathInnerAPI()
        {
            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var iris = FileHelper.GetTestFile("iris.txt");
                var df = DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new DataViewType[] { NumberDataViewType.Single });
                using (var pipe = new ScikitPipeline(new[] { "Concat{col=Feature:Sepal_length,Sepal_width}",
                                                             "TreeFeat{tr=ft{iter=2} lab=Label feat=Feature}"}))
                {
                    pipe.Train(df);
                    var scorer = pipe.Predict(df);
                    var dfout = DataFrameIO.ReadView(scorer);
                    Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 31));
                    var outfile = FileHelper.GetOutputFile("iris_path.txt", methodName);
                    dfout.ToCsv(outfile);
                    Assert.IsTrue(File.Exists(outfile));
                }
            }
        }

        [TestMethod]
        public void TestLoadModelFromNimbusML()
        {
            var iris = FileHelper.GetTestFile("model_iris.zip");
            /*using (*/
            var env = EnvHelper.NewTestEnvironment();
            {
                try
                {
                    using (var pipe2 = new ScikitPipeline(iris, env))
                    {
                    }
                }
                catch (Exception e)
                {
                    Assert.IsTrue(e.ToString().Contains("because the model is too old"));
                }
            }
        }

        /*
        [TestMethod]
        public void TestCommandLine()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var data = FileHelper.GetTestFile("data_train_test.csv");
            var output = FileHelper.GetOutputFile("model.zip", methodName);
            var bout = new StringBuilder();
            var berr = new StringBuilder();
            ILogWriter stout = new LogWriter(s => bout.Append(s));
            ILogWriter sderr = new LogWriter(s => berr.Append(s));
            var cmd = "chain cmd=train{\n" +
                        "data = __INPUT__\n" +
                        "loader = text{col=ItemID:I8:0 col=Sentiment:I8:1 col=SentimentSource:TX:2 \n" +
                        "              col=SentimentText:TX:3 col=RowNum:I8:4 \n" +
                        "              col=Label:BL:5 col=Train:BL:6 col=Small:BL:7 header=+ sep=,}\n" +
                        "xf = Text {col=transformed1:SentimentText wordExtractor=NGramExtractorTransform{ngram=2}}\n" +
                        "xf = Categorical {col=SentimentSource}\n" +
                        "xf = concat {col=Features:transformed1,SentimentSource}\n" +
                        "tr = FastTreeBinaryClassification\n" +
                        "out = __OUTPUT__} \n" +
                        "cmd = saveonnx{in = __OUTPUT__ \n" +
                        "onnx = ft_sentiment_cs.onnx\n" +
                        "domain = ai.onnx.ml idrop = Label,Sentiment,RowNum,Train,Small }";
            cmd = cmd.Replace("__INPUT__", data);
            cmd = cmd.Replace("__OUTPUT__", output);

            using (var env = new DelegateEnvironment(outWriter: stout, errWriter: sderr, verbose: 3))
            {
                MamlHelper.MamlScript(cmd, false, env);
                var sout = bout.ToString();
                Assert.IsTrue(sout.Length > 0);
                Assert.IsTrue(!sout.Contains("Unknown"));
            }
        }
        */

        [TestMethod]
        public void TestMamlCodeGen()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var data = FileHelper.GetTestFile("data_train_test.csv");
            var output = FileHelper.GetOutputFile("model.zip", methodName);
            var bout = new StringBuilder();
            var berr = new StringBuilder();
            ILogWriter stout = new LogWriter(s => bout.Append(s));
            ILogWriter sderr = new LogWriter(s => berr.Append(s));
            var cmd = "chain cmd=train{\n" +
                        "data = __INPUT__\n" +
                        "loader = text{col=ItemID:I8:0 col=Sentiment:R4:1 col=SentimentSource:TX:2 \n" +
                        "              col=SentimentText:TX:3 col=RowNum:R4:4 \n" +
                        "              col=Label:BL:5 col=Train:BL:6 col=Small:BL:7 header=+ sep=,}\n" +
                        "xf = concat {col=Features:RowNum,Sentiment}\n" +
                        "tr = FastTreeBinaryClassification{iter=2}\n" +
                        "out = __OUTPUT__} \n" +
                        "cmd = saveas{in=__OUTPUT__ code=ft_sentiment_cs}";
            cmd = cmd.Replace("__INPUT__", data);
            cmd = cmd.Replace("__OUTPUT__", output);

            /*using (*/
            var env = new DelegateEnvironment(seed: 0, outWriter: stout, errWriter: sderr, verbose: 3);
            {
                MamlHelper.MamlScript(cmd, false, env);
                var sout = bout.ToString();
                Assert.IsTrue(sout.Length > 0);
                Assert.IsTrue(!sout.Contains("Unknown"));
            }
        }
    }
}

