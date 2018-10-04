﻿// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Text;
using System.IO;
using Microsoft.ML.Runtime.Tools;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineHelper;
using Scikit.ML.DocHelperMlExt;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestApiEntryPoint
    {
        [TestMethod]
        public void TestCSGeneratorHelp()
        {
            var cmd = "? CSGenerator";
            using (var std = new Scikit.ML.DocHelperMlExt.StdCapture())
            {
                Maml.MainAll(cmd);
                if (std.StdOut.Length == 0)
                    Assert.Inconclusive("Not accurate on a remote machine.");
            }
        }

        [TestMethod]
        public void TestHelpScorer()
        {
            var bout = new StringBuilder();
            var berr = new StringBuilder();
            ILogWriter stout = new LogWriter(s => bout.Append(s));
            ILogWriter sderr = new LogWriter(s => berr.Append(s));
            using (var env = new DelegateEnvironment(outWriter: stout, errWriter: sderr, verbose: 3))
            {
                var cmd = "? MultiClassClassifierScorer";
                MamlHelper.MamlScript(cmd, false, env);
                var sout = bout.ToString();
                Assert.IsTrue(sout.Length > 0);
                Assert.IsTrue(!sout.Contains("Unknown"));
            }
        }

        [TestMethod]
        public void TestHelpModels()
        {
            foreach (var name in new[] { "Resample" })
            {
                var bout = new StringBuilder();
                var berr = new StringBuilder();
                ILogWriter stout = new LogWriter(s => bout.Append(s));
                ILogWriter sderr = new LogWriter(s => berr.Append(s));
                using (var env = new DelegateEnvironment(outWriter: stout, errWriter: sderr, verbose: 3))
                {
                    var cmd = $"? {name}";
                    MamlHelper.MamlScript(cmd, false, env: env);
                    var sout = bout.ToString();
                    var serr = berr.ToString();
                    Assert.IsTrue(!serr.Contains("Can't instantiate"));
                    Assert.IsTrue(sout.Length > 0);
                    Assert.IsTrue(!sout.Contains("Unknown"));
                }
            }
        }

        [TestMethod]
        public void TestCSGenerator()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var basePath = FileHelper.GetOutputFile("CSharpApiExt.cs", methodName);
            var cmd = $"? generator=cs{{csFilename={basePath} exclude=System.CodeDom.dll}}";
            var bout = new StringBuilder();
            var berr = new StringBuilder();
            ILogWriter stout = new LogWriter(s => bout.Append(s));
            ILogWriter sderr = new LogWriter(s => berr.Append(s));
            using (var env = new DelegateEnvironment(outWriter: stout, errWriter: sderr, verbose: 3))
            {
                MamlHelper.MamlScript(cmd, false, env: env);
                var sout = bout.ToString();
                var serr = berr.ToString();
                Assert.IsTrue(sout.Length > 0);
                Assert.IsTrue(serr.Length == 0);
                Assert.IsFalse(sout.ToLower().Contains("usage"));
            }
            var text = File.ReadAllText(basePath);
            Assert.IsTrue(text.ToLower().Contains("nearest"));
        }
    }
}
