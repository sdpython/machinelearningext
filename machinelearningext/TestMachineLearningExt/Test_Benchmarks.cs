// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Trainers;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.TestHelper;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class Test_Benchmarks
    {
        [TestMethod]
        public void TestValueMapperPredictionEngineMultiThread()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");

            using (var env = EnvHelper.NewTestEnvironment())
            using (var engine0 = new ValueMapperPredictionEngineFloat(env, name, conc: 1))
            {
                var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
                var exp = new float[100];
                for (int i = 0; i < exp.Length; ++i)
                {
                    feat[0] = i;
                    exp[i] = engine0.Predict(feat);
                    Assert.IsFalse(float.IsNaN(exp[i]));
                    Assert.IsFalse(float.IsInfinity(exp[i]));
                }

                var dico = new Dictionary<Tuple<int, bool, int>, double>();

                foreach (var each in new[] { false, true })
                {
                    foreach (int th in new int[] { 2, 0, 1, 3 })
                    {
                        var engine = new ValueMapperPredictionEngineFloat(env, name, conc: th);
                        var sw = new Stopwatch();
                        sw.Start();
                        for (int i = 0; i < exp.Length; ++i)
                        {
                            feat[0] = i;
                            var res = engine.Predict(feat);
                            Assert.AreEqual(exp[i], res);
                        }
                        sw.Stop();
                        dico[new Tuple<int, bool, int>(exp.Length, each, th)] = sw.Elapsed.TotalSeconds;
                    }
                }
                Assert.AreEqual(dico.Count, 8);
                var df = DataFrameIO.Convert(dico, "N", "number of threads", "time(s)");
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var filename = FileHelper.GetOutputFile("benchmark_ValueMapperPredictionEngineMultiThread.txt", methodName);
                df.ToCsv(filename);
            }
        }

        private IDataScorerTransform _TrainSentiment()
        {
            bool normalize = true;

            var args = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[] {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };

            var args2 = new TextFeaturizingEstimator.Arguments()
            {
                Column = new TextFeaturizingEstimator.Column
                {
                    Name = "Features",
                    Source = new[] { "SentimentText" }
                },
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizingEstimator.CaseNormalizationMode.Lower,
                OutputTokens = true,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = normalize ? TextFeaturizingEstimator.TextNormKind.L2 : TextFeaturizingEstimator.TextNormKind.None,
                CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2, AllLengths = true },
            };

            var trainFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-data.tsv");

            using (var env = EnvHelper.NewTestEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env, args).Read(new MultiFileSource(trainFilename));
                var trans = TextFeaturizingEstimator.Create(env, args2, loader);

                // Train
                var trainer = new SdcaBinaryTrainer(env, new SdcaBinaryTrainer.Options
                {
                    NumThreads = 1
                });

                var cached = new CacheDataView(env, trans, prefetch: null);
                var predictor = trainer.Fit(cached);

                var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                return ScoreUtils.GetScorer(predictor.Model, scoreRoles, env, trainRoles.Schema);
            }
        }

        private ITransformer _TrainSentiment2Transformer()
        {
            var args = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[] {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };
            var ml = new MLContext(seed: 1, conc: 1);
            //var reader = ml.Data.TextReader(args);
            var trainFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-data.tsv");

            var data = ml.Data.ReadFromTextFile(trainFilename, args);
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"));
            var model = pipeline.Fit(data);
            return model;
        }

        private List<Tuple<int, TimeSpan, int, float[]>> _MeasureTime(int conc,
            string strategy, string engine, IDataScorerTransform scorer, ITransformer trscorer, int ncall)
        {
            var testFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-test.tsv");
            var times = new List<Tuple<int, TimeSpan, int, float[]>>();

            using (var env = EnvHelper.NewTestEnvironment(seed: 1, conc: conc))
            {
                if (engine == "mlnet")
                {
                    var args = new TextLoader.Arguments()
                    {
                        Separator = "tab",
                        HasHeader = true,
                        Column = new[] {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("SentimentText", DataKind.Text, 1)
                        }
                    };

                    // Take a couple examples out of the test data and run predictions on top.
                    var testLoader = new TextLoader(env, args).Read(new MultiFileSource(testFilename));
                    IDataView cache;
                    if (strategy.Contains("extcache"))
                        cache = new ExtendedCacheTransform(env, new ExtendedCacheTransform.Arguments(), testLoader);
                    else
                        cache = new CacheDataView(env, testLoader, new[] { 0, 1 });
                    //var testData = cache.AsEnumerable<SentimentDataBoolFloat>(env, false);
                    //var testDataArray = cache.AsEnumerable<SentimentDataBoolFloat>(env, false).ToArray();
                    int N = 1;

                    var model = ComponentCreation.CreatePredictionEngine<SentimentDataBoolFloat, SentimentPrediction>(env, trscorer);
                    var sw = new Stopwatch();
                    for (int call = 1; call <= ncall; ++call)
                    {
                        sw.Reset();
                        var pred = new List<float>();
                        sw.Start();
                        for (int i = 0; i < N; ++i)
                        {
                            if (strategy.Contains("array"))
                            {
                                /*
                                foreach (var input in testDataArray)
                                    pred.Add(model.Predict(input).Score);
                                    */
                            }
                            else
                            {
                                /*
                                foreach (var input in testData)
                                    pred.Add(model.Predict(input).Score);
                                    */
                            }
                        }
                        sw.Stop();
                        times.Add(new Tuple<int, TimeSpan, int, float[]>(N, sw.Elapsed, call, pred.ToArray()));
                    }
                }
                else if (engine == "scikit")
                {
                    var args = new TextLoader.Arguments()
                    {
                        Separator = "tab",
                        HasHeader = true,
                        Column = new[] {
                            new TextLoader.Column("Label", DataKind.Bool, 0),
                            new TextLoader.Column("SentimentText", DataKind.Text, 1)
                        }
                    };

                    // Take a couple examples out of the test data and run predictions on top.
                    var testLoader = new TextLoader(env, args).Read(new MultiFileSource(testFilename));
                    IDataView cache;
                    if (strategy.Contains("extcache"))
                        cache = new ExtendedCacheTransform(env, new ExtendedCacheTransform.Arguments(), testLoader);
                    else
                        cache = new CacheDataView(env, testLoader, new[] { 0, 1 });
                    //var testData = cache.AsEnumerable<SentimentDataBool>(env, false);
                    //var testDataArray = cache.AsEnumerable<SentimentDataBool>(env, false).ToArray();
                    int N = 1;

                    string allSchema = SchemaHelper.ToString(scorer.Schema);
                    Assert.IsTrue(allSchema.Contains("PredictedLabel:Bool:4; Score:R4:5; Probability:R4:6"));
                    var model = new ValueMapperPredictionEngine<SentimentDataBool>(env, scorer, conc: conc);
                    var output = new ValueMapperPredictionEngine<SentimentDataBool>.PredictionTypeForBinaryClassification();
                    var sw = new Stopwatch();
                    for (int call = 1; call <= ncall; ++call)
                    {
                        var pred = new List<float>();
                        sw.Reset();
                        sw.Start();
                        for (int i = 0; i < N; ++i)
                        {
                            if (strategy.Contains("array"))
                            {
                                /*
                                foreach (var input in testDataArray)
                                {
                                    model.Predict(input, ref output);
                                    pred.Add(output.Score);
                                }
                                */
                            }
                            else
                            {
                                /*
                                foreach (var input in testData)
                                {
                                    model.Predict(input, ref output);
                                    pred.Add(output.Score);
                                }
                                */
                            }
                        }
                        sw.Stop();
                        times.Add(new Tuple<int, TimeSpan, int, float[]>(N, sw.Elapsed, call, pred.ToArray()));
                    }
                }
                else
                    throw new NotImplementedException($"Unknown engine '{engine}'.");
            }
            return times;
        }

        [TestMethod]
        public void TestScikitAPI_EngineSimpleTrainAndPredict()
        {
            var dico = new Dictionary<Tuple<int, string, string, int, int>, double>();
            var scorer = _TrainSentiment();
            var trscorer = _TrainSentiment2Transformer();
            foreach (var cache in new[] { false, true })
            {
                for (int th = 1; th <= 3; ++th)
                {
                    var memo = new Dictionary<string, float[]>();
                    foreach (var engine in new[] { "mlnet", "scikit" })
                    {
                        foreach (var kind in new[] { "array", "stream" })
                        {
                            var strat_ = new[] {
                                        cache ? "extcache" : "viewcache",
                                        kind,
                                        };
                            var strat = string.Join("+", strat_);
                            foreach (var res in _MeasureTime(th, strat, engine, scorer, trscorer, 2))
                            {
                                dico[new Tuple<int, string, string, int, int>(res.Item1, engine, strat, th, res.Item3)] = res.Item2.TotalSeconds;
                                if (res.Item3 == 1)
                                    memo[engine] = res.Item4;
                            }
                        }
                    }
                    var p1 = memo["mlnet"];
                    var p2 = memo["scikit"];
                    Assert.AreEqual(p1.Length, p2.Length);
                    var abs = 0.0;
                    for (int ii = 0; ii < p1.Length; ++ii)
                        abs += Math.Abs(p1[ii] - p2[ii]);
                    abs /= p1.Length;
                    Assert.IsTrue(abs <= 2);
                }
            }
            var df = DataFrameIO.Convert(dico, "N", "engine", "strategy", "number of threads", "call", "time(s)");
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var filename = FileHelper.GetOutputFile("benchmark_ValueMapperPredictionEngineMultiThread.txt", methodName);
            df.ToCsv(filename);
            Assert.AreEqual(dico.Count, 48);
        }
    }
}
