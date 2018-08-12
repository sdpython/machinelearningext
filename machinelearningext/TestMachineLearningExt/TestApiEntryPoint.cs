﻿// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using Microsoft.ML.Runtime.Tools;
using Scikit.ML.TestHelper;

namespace TestMachineLearningExt
{
    [TestClass]
    public class TestApiEntryPoint
    {
        [TestMethod]
        public void TestCSGeneratorHelp()
        {
            var cmd = "? CSGenerator";
            using (var std = new StdCapture())
            {
                Maml.MainAll(cmd);
                Assert.IsTrue(std.StdOut.Length > 0);
            }
        }

        [TestMethod]
        public void TestHelpScorer()
        {
            var cmd = "? MultiClassClassifierScorer";
            using (var std = new StdCapture())
            {
                Maml.MainAll(cmd);
                var sout = std.StdOut.ToString();
                Assert.IsTrue(sout.Length > 0);
                Assert.IsTrue(!sout.Contains("Unknown"));
            }
        }

        [TestMethod]
        public void TestHelpModels()
        {
            foreach (var name in new[] { "Resample" })
            {
                var cmd = $"? {name}";
                using (var std = new StdCapture())
                {
                    Maml.MainAll(cmd);
                    var sout = std.StdOut.ToString();
                    var serr = std.StdErr.ToString();
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
            using (var std = new StdCapture())
            {
                Maml.Main(new[] { cmd });
                Assert.IsTrue(std.StdOut.Length > 0);
                Assert.IsTrue(std.StdErr.Length == 0);
                Assert.IsFalse(std.StdOut.ToLower().Contains("usage"));
            }
            var text = File.ReadAllText(basePath);
            Assert.IsTrue(text.ToLower().Contains("nearest"));
        }
    }
}
