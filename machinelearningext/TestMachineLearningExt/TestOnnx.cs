// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Scikit.ML.TestHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestOnnx
    {
        [TestMethod]
        public void TestOnnx_TrainingWithIris()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;

            // direct call
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrameIO.ReadCsv(iris, sep: '\t');
            df.AddColumn("LabelI", df["Label"].AsType(NumberDataViewType.Single));
            var pipe = new ScikitPipeline(new[] { $"Concat{{col=Features:{df.Columns[1]},{df.Columns[2]}}}" }, "mlr");
            pipe.Train(df, "Features", "LabelI");
            DataFrame pred = null;
            pipe.Predict(df, ref pred);

            // Onnx Save
            var output = FileHelper.GetOutputFile("model.onnx", methodName);
            var model = pipe.ToOnnx();
            model.Save(output);
            Assert.IsTrue(File.Exists(output));

            // Onnx save no concat.
            output = FileHelper.GetOutputFile("model_vector.onnx", methodName);
            model = pipe.ToOnnx(1);
            model.Save(output);
            Assert.IsTrue(File.Exists(output));

            // Onnx Load Not implemented yet.
            /*
            var restored = new ScikitPipeline(output);
            DataFrame pred2 = null;
            restored.Predict(df, ref pred2);
            pred.AssertAlmostEqual(pred2);
            */
        }
    }
}
