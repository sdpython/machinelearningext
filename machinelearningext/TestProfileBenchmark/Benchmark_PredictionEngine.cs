// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Core.Data;
using Scikit.ML.TestHelper;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineTransforms;


namespace TestProfileBenchmark
{
    public static class Benchmark_PredictionEngine
    {
        private static IDataScorerTransform _TrainSentiment()
        {
            bool normalize = true;

            var args = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
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
                    NumThreads = 1,
                    LabelColumn = "Label",
                    FeatureColumn = "Features"
                });

                var cached = new Microsoft.ML.Data.CacheDataView(env, trans, prefetch: null);
                var predictor = trainer.Fit(cached);

                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                return ScoreUtils.GetScorer(predictor.Model, scoreRoles, env, trainRoles.Schema);
            }
        }

        private static ITransformer _TrainSentiment2()
        {
            var args = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                           {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };
            var ml = new MLContext(seed: 1, conc: 1);
            //var reader = ml.Data.ReadFromTextFile(args);
            var trainFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-data.tsv");

            var data = ml.Data.ReadFromTextFile(trainFilename, args);
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"));
            var model = pipeline.Fit(data);
            return model;
        }

        private static List<Tuple<int, TimeSpan, int>> _MeasureTime(int conc,
                                string engine, IDataScorerTransform scorer, ITransformer transformer,
                                int N, int ncall, bool cacheScikit)
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

            var testFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-test.tsv");
            var times = new List<Tuple<int, TimeSpan, int>>();

            using (var env = EnvHelper.NewTestEnvironment(seed: 1, conc: conc))
            {

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = new TextLoader(env, args).Read(new MultiFileSource(testFilename));
                IDataView cache;
                if (cacheScikit)
                    cache = new ExtendedCacheTransform(env, new ExtendedCacheTransform.Arguments(), testLoader);
                else
                    cache = new CacheDataView(env, testLoader, new[] { 0, 1 });
                //var testData = cache.AsEnumerable<SentimentDataBoolFloat>(env, false);

                if (engine == "mlnet")
                {
                    Console.WriteLine("engine={0} N={1} ncall={2} cacheScikit={3}", engine, N, ncall, cacheScikit);
                    var fct = ComponentCreation.CreatePredictionEngine<SentimentDataBoolFloat, SentimentPrediction>(env, transformer);
                    var sw = new Stopwatch();
                    for (int call = 1; call <= ncall; ++call)
                    {
                        sw.Reset();
                        sw.Start();
                        for (int i = 0; i < N; ++i)
                            /*
                            foreach (var input in testData)
                                fct.Predict(input);
                                */
                            ;
                        sw.Stop();
                        times.Add(new Tuple<int, TimeSpan, int>(N, sw.Elapsed, call));
                    }
                }
                else if (engine == "scikit")
                {
                    Console.WriteLine("engine={0} N={1} ncall={2} cacheScikit={3}", engine, N, ncall, cacheScikit);
                    var model = new ValueMapperPredictionEngine<SentimentDataBoolFloat>(env, scorer, conc: conc);
                    var output = new ValueMapperPredictionEngine<SentimentDataBoolFloat>.PredictionTypeForBinaryClassification();
                    var sw = new Stopwatch();
                    for (int call = 1; call <= ncall; ++call)
                    {
                        sw.Reset();
                        sw.Start();
                        for (int i = 0; i < N; ++i)
                            /*
                            foreach (var input in testData)
                                model.Predict(input, ref output);
                                */
                            ;
                        sw.Stop();
                        times.Add(new Tuple<int, TimeSpan, int>(N, sw.Elapsed, call));
                    }
                }
                else
                    throw new NotImplementedException($"Unknown engine '{engine}'.");
            }
            return times;
        }

        public static DataFrame TestScikitAPI_EngineSimpleTrainAndPredict(string engine, int th, int N, int ncall, bool cacheScikit)
        {
            var dico = new Dictionary<Tuple<int, string, int, int>, double>();
            var scorer = _TrainSentiment();
            var trscorer = _TrainSentiment2();
            foreach (var res in _MeasureTime(th, engine, scorer, trscorer, N, ncall, cacheScikit))
                dico[new Tuple<int, string, int, int>(res.Item1, engine, th, res.Item3)] = res.Item2.TotalSeconds;
            var df = DataFrameIO.Convert(dico, "N", "engine", "number of threads", "call", "time(s)");
            return df;
        }
    }
}
