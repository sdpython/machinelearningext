// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Scikit.ML.Clustering;
using Scikit.ML.NearestNeighbors;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineHelper;

namespace TestMachineLearningExt
{
    [TestClass]
    public class TestClustering
    {
        private static readonly Random rnd = new Random();

        #region DbScan

        [TestMethod()]
        public void StartTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);

            var dbscan = new DBScan(points);
            var result = dbscan.Cluster(0.5f, 2);
            Assert.IsTrue(result.Count == points.Count);
            Assert.IsTrue(result.Values.Max() <= points.Count);
        }

        [TestMethod()]
        public void RegionQueryPointsListTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);
            var p = new PointIdFloat(new List<float>() { 1.5f, 1.5f, 1.5f });
            Func<IPointIdFloat, Tuple<float, float, float>> keySelector = t => Tuple.Create(
                KdTree.KdTreeNode.KeyByDepth(t, 0), 0.0f, 0.0f);

            var result = DBScan.RegionQuery(points, p, 0.16f);

            Assert.IsTrue(result.Count() == 0);

            result = DBScan.RegionQuery(points, p, 1);
            var expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));

            p = new PointIdFloat(new List<float>() { 0.75f, 0.75f, 0.75f });
            result = DBScan.RegionQuery(points, p, (float)Math.Sqrt(2f / 3f));
            expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));
        }

        [TestMethod()]
        public void RegionQueryKdTreeTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);

            var p = new PointIdFloat(new List<float>() { 1.5f, 1.5f, 1.5f });
            KdTree kdt = new KdTree(points);

            var result = kdt.PointsWithinDistance(p, 0.5f);

            Assert.IsTrue(result.Count == 0);

            result = DBScan.RegionQuery(kdt, p, 1);
            var expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));

            p = new PointIdFloat(new List<float>() { 0.75f, 0.75f, 0.75f });
            result = kdt.PointsWithinDistance(p, (float)Math.Sqrt(2));
            expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));
        }

        [TestMethod()]
        public void RegionQueryTest()
        {
            //Checks that both RegionQuery overloaded methods always return the same result
            for (int d = 2; d <= 5; d++)
            {
                var points = new List<IPointIdFloat>();
                for (int i = 0; i < 200; i++)
                    points.Add(RandomPoint(d));
                KdTree kdt = new KdTree(points);

                for (int i = 0; i < 50; i++)
                {
                    var p = RandomPoint(d);
                    float epsilon = (float)rnd.NextDouble() * int.MaxValue;
                    var l1 = DBScan.RegionQuery(points, p, epsilon).ToList();
                    var l2 = DBScan.RegionQuery(kdt, p, epsilon).ToList();
                    var b = SequenceEquivalent(l1, l2, PointIdFloat.PointsComparison);
                    if (!b)
                    {
                        var l3 = DBScan.RegionQuery(kdt, p, epsilon).ToList();
                        Assert.IsTrue(SequenceEquivalent(l3, l2, PointIdFloat.PointsComparison));
                        Assert.IsTrue(false);
                    }
                }
            }
        }

        [TestMethod()]
        public void ExpandRegionTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1.5f, 1.5f, 1.5f }),
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);

            var clusters = new Dictionary<long, int>();
            var visited = new HashSet<long>();
            var p = points[0];
            float epsilon = 1f;

            var neighbours = DBScan.RegionQuery(points, p, epsilon);
            Assert.IsTrue(neighbours.Contains(p));

            DBScan.ExpandCluster(clusters, visited, points, p, neighbours.ToList(), 0, epsilon, 3);
            Assert.IsTrue(clusters.ContainsKey(p.id));
            foreach (var q in neighbours)
            {
                Assert.IsTrue(clusters.ContainsKey(q.id));
                Assert.IsTrue(clusters[q.id] == clusters[p.id]);
            }

            Assert.IsFalse(clusters.ContainsKey(points.Count() + 1));
        }

        private bool SequenceEquivalent<T>(List<T> list1, List<T> list2, Comparison<T> comparer)
        {
            list1.Sort(comparer);
            list2.Sort(comparer);
            return list1.SequenceEqual(list2);
        }

        private IPointIdFloat RandomPoint(int dimension = 2)
        {
            var coordinates = new List<float>();
            for (int i = 0; i < dimension; i++)
                coordinates.Add((float)(1000 * rnd.NextDouble()));
            return new PointIdFloat(coordinates);
        }

        [TestMethod]
        public void TestDBScanTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("three_classes_2d.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);

            /*using (*/
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            {
                //var loader = env.CreateLoader("text{col=DataViewRowId:I4:0 col=Features:R4:1-2 header=+}", new MultiFileSource(dataFilePath));
                var loader = new TextLoader(env, new TextLoader.Options()
                {
                    HasHeader = true,
                    Columns = new[] { TextLoader.Column.Parse("DataViewRowId:R4:0"),
                                     TextLoader.Column.Parse("Features:R4:1-2")}
                }).Load(new MultiFileSource(dataFilePath));
                var xf = env.CreateTransform("DBScan{col=Features}", loader);

                string schema = SchemaHelper.ToString(xf.Schema);
                if (string.IsNullOrEmpty(schema))
                    throw new Exception("DataViewSchema is null.");
                if (!schema.Contains("Cluster"))
                    throw new Exception("DataViewSchema does not contain Cluster.");
                if (!schema.Contains("Score"))
                    throw new Exception("DataViewSchema does not contain Score.");

                StreamHelper.SaveModel(env, xf, outModelFilePath);

                var saver = env.CreateSaver("Text{header=- schema=-}");
                using (var fs2 = File.Create(outputDataFilePath))
                    saver.SaveData(fs2, TestTransformHelper.AddFlatteningTransform(env, xf),
                                            StreamHelper.GetColumnsIndex(xf.Schema, new[] { "Features", "ClusterId", "Score" }));

                // Checking the values.
                var lines = File.ReadAllLines(outputDataFilePath).Select(c => c.Split('\t')).Where(c => c.Length == 4);
                if (!lines.Any())
                    throw new Exception(string.Format("The output file is empty or not containing three columns '{0}'", outputDataFilePath));
                var clusters = lines.Select(c => c[1]).Distinct();
                if (clusters.Count() <= 1)
                    throw new Exception("Only one cluster, this is unexpected.");

                // Serialization.
                var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, xf, loader, outData, outData2);
            }
        }

        #endregion

        #region Optics

        [TestMethod()]
        public void OrderingTest()
        {
            IReadOnlyDictionary<long, long> orderingMapping;
            IReadOnlyCollection<IPointIdFloat> ordering;

            var points3D = new List<IPointIdFloat>() {
                new PointIdFloat(new List<float>() { 1, 2, 55 }),
                new PointIdFloat(new List<float>() { 0, 1, 0.1f }),
                new PointIdFloat(new List<float>() { 3, 3, 0.01f }),
                new PointIdFloat(new List<float>() { 1.5f, 2, -0.03f }),
                new PointIdFloat(new List<float>() { 5, -1, 44 }),
                new PointIdFloat(new List<float>() { 15, -51, 73 }),
                new PointIdFloat(new List<float>() { 0.5f, -21, 144 })
            };
            PointIdFloat.SetIds(points3D);
            var epsilon = 10;
            var minPoints = 3;

            Optics algo = new Optics(points3D);
            var seeds = new PriorityQueue<float, IPointIdFloat>();

            OpticsOrdering oo = algo.Ordering(epsilon, minPoints);

            ordering = oo.ordering;
            orderingMapping = oo.orderingMapping;

            Assert.AreEqual(points3D.Count, ordering.Count);
            Assert.IsTrue(SequenceEquivalent(points3D, ordering.ToList(), PointIdFloat.PointsComparison));
            Assert.AreEqual(points3D.Count, orderingMapping.Count);

            for (long i = 0; i < orderingMapping.Count; i++)
                Assert.IsTrue(orderingMapping.Values.Contains(i));
            foreach (var p in points3D)
                Assert.IsTrue(orderingMapping.ContainsKey(p.id));
            Assert.IsTrue(oo.reachabilityDistances.Count > 0);
            foreach (var p in points3D)
                Assert.IsTrue(oo.coreDistancesCache.ContainsKey(p.id));
        }

        [TestMethod()]
        public void ClusterTest()
        {
            var points3D = new List<IPointIdFloat>() {
                new PointIdFloat(new List<float>() { 1, 2, 55 }),
                new PointIdFloat(new List<float>() { 0, 1, 0.1f }),
                new PointIdFloat(new List<float>() { 3, 3, 0.01f }),
                new PointIdFloat(new List<float>() { 1.5f, 2, -0.03f }),
                new PointIdFloat(new List<float>() { 5, -1, 44 }),
                new PointIdFloat(new List<float>() { 15, -51, 73 }),
                new PointIdFloat(new List<float>() { 0.5f, -21, 144 })
            };
            PointIdFloat.SetIds(points3D);
            var ids = points3D.Select(p => p.id).ToList();
            ids.Sort();

            var epsilon = 100;
            var epsilonPrime = 10;
            var minPoints = 3;

            Optics algo = new Optics(points3D);
            var seeds = new PriorityQueue<float, IPointIdFloat>();

            OpticsOrdering oo = algo.Ordering(epsilon, minPoints);
            var results = oo.Cluster(epsilonPrime);
            Assert.AreEqual(points3D.Count, results.Count);
            var keys = results.Keys.ToList();
            keys.Sort();
            Assert.IsTrue(keys.SequenceEqual(ids));
        }

        [TestMethod()]
        public void UpdateTest()
        {
            Dictionary<long, float?> reachabilityDistances = new Dictionary<long, float?>();
            Dictionary<long, float?> coreDistancesCache = new Dictionary<long, float?>();
            HashSet<long> processed = new HashSet<long>();

            var points3D = new List<IPointIdFloat>() {
                new PointIdFloat(new List<float>() { 1, 2, 55 }),
                new PointIdFloat(new List<float>() { 0, 1, 0.1f }),
                new PointIdFloat(new List<float>() { 3, 3, 0.01f }),
                new PointIdFloat(new List<float>() { 1.5f, 2, -0.03f }),
                new PointIdFloat(new List<float>() { 5, -1, 44 }),
                new PointIdFloat(new List<float>() { 15, -51, 73 }),
                new PointIdFloat(new List<float>() { 0.5f, -21, 144 })
            };
            PointIdFloat.SetIds(points3D);
            var p = points3D[2];
            var epsilon = 10;
            var minPoints = 3;

            Optics algo = new Optics(points3D);
            var seeds = new PriorityQueue<float, IPointIdFloat>();

            processed.Add(p.id);
            var coreDistance = algo.CoreDistance(algo.kdt, coreDistancesCache, p, epsilon, minPoints);

            algo.Update(ref seeds, processed, reachabilityDistances, coreDistancesCache, p, epsilon, minPoints);

            Assert.AreEqual(2, seeds.Count);
            KeyValuePair<float, IPointIdFloat> top;
            seeds.TryPeek(out top);
            Assert.AreEqual(top.Key, coreDistance);
            Assert.AreEqual(points3D[3], top.Value);
        }

        [TestMethod()]
        public void CoreDistanceTest()
        {
            var points3D = new List<IPointIdFloat>() {
                new PointIdFloat(new List<float>() { 1, 2, 55 }),
                new PointIdFloat(new List<float>() { 0, 1, 0.1f }),
                new PointIdFloat(new List<float>() { 3, 3, 0.01f }),
                new PointIdFloat(new List<float>() { 1.5f, 2, -0.03f }),
                new PointIdFloat(new List<float>() { 5, -1, 44 }),
                new PointIdFloat(new List<float>() { 15, -51, 73 }),
                new PointIdFloat(new List<float>() { 0.5f, -21, 144 })
            };
            PointIdFloat.SetIds(points3D);

            var p = new PointIdFloat(new List<float>() { 1, 2, 3 });

            Optics algo = new Optics(points3D);
            Dictionary<long, float?> coreDist = new Dictionary<long, float?>();

            // - Tests with no cache - 

            //Not enough points in the tree
            Assert.IsNull(algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 1000, 40));
            Assert.IsNull(algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 1000, points3D.Count + 1));

            //Not enough points in the epsilon-neighbourhood
            Assert.IsNull(algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 1, 2));
            Assert.IsNull(algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 10, 6));

            var result = algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 1000, points3D.Count);
            Assert.AreEqual(p.DistanceTo(points3D[6]), result);

            points3D.ForEach(point =>
            {
                result = algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), point, 0.00001f, 1);
                Assert.AreEqual(0, result);
            });

            Assert.IsNull(algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 1, 1));
            Assert.AreEqual(p.DistanceTo(points3D[3]), algo.CoreDistance(algo.kdt, new Dictionary<long, float?>(), p, 5, 1));

            //Tests - using cache -
            Assert.AreEqual(0, coreDist.Count);
            float? tmp;

            Assert.IsNull(algo.CoreDistance(algo.kdt, coreDist, p, 1000, 40));
            Assert.IsTrue(coreDist.ContainsKey(p.id));
            coreDist.TryGetValue(p.id, out tmp);
            Assert.IsNull(tmp);

            coreDist.Clear();

            result = algo.CoreDistance(algo.kdt, coreDist, p, 1000, points3D.Count);
            Assert.AreEqual(p.DistanceTo(points3D[6]), result);

            Assert.IsTrue(coreDist.ContainsKey(p.id));
            coreDist.TryGetValue(p.id, out tmp);
            Assert.AreEqual(result, tmp);
        }

        [TestMethod]
        public void TestOpticsTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("three_classes_2d.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);

            /*using (*/var env = EnvHelper.NewTestEnvironment();
            {
                var loader = env.CreateLoader("text{col=DataViewRowId:I4:0 col=Features:R4:1-2 header=+}",
                                              new MultiFileSource(dataFilePath));
                var xf = env.CreateTransform("Optics{col=Features epsilons=0.3 minPoints=6}", loader);

                string schema = SchemaHelper.ToString(xf.Schema);
                if (string.IsNullOrEmpty(schema))
                    throw new Exception("DataViewSchema is null.");
                if (!schema.Contains("ClusterId"))
                    throw new Exception("DataViewSchema does not contain Cluster.");
                if (!schema.Contains("Score"))
                    throw new Exception("DataViewSchema does not contain Score.");

                StreamHelper.SaveModel(env, xf, outModelFilePath);

                var saver = env.CreateSaver("Text{header=- schema=-}");
                using (var fs2 = File.Create(outputDataFilePath))
                    saver.SaveData(fs2, TestTransformHelper.AddFlatteningTransform(env, xf),
                                    StreamHelper.GetColumnsIndex(xf.Schema, new[] { "Features", "ClusterId", "Score" }));

                // Checking the values.
                var lines = File.ReadAllLines(outputDataFilePath).Select(c => c.Split('\t')).Where(c => c.Length == 4);
                if (!lines.Any())
                    throw new Exception(string.Format("The output file is empty or not containing three columns '{0}'", outputDataFilePath));
                var clusters = lines.Select(c => c[1]).Distinct();
                if (clusters.Count() <= 1)
                    throw new Exception("Only one cluster, this is unexpected.");

                // Serialization.
                var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, xf, loader, outData, outData2);
            }
        }

        [TestMethod]
        public void TestOpticsOrderingTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("three_classes_2d.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);

            /*using (*/var env = EnvHelper.NewTestEnvironment();
            {
                var loader = env.CreateLoader("text{col=DataViewRowId:I4:0 col=Features:R4:1-2 header=+}",
                                              new MultiFileSource(dataFilePath));
                var xf = env.CreateTransform("OpticsOrd{col=Features epsilon=0.3 minPoints=6}", loader);

                string schema = SchemaHelper.ToString(xf.Schema);
                if (string.IsNullOrEmpty(schema))
                    throw new Exception("DataViewSchema is null.");
                if (!schema.Contains("Ordering"))
                    throw new Exception("DataViewSchema does not contain Ordering.");
                if (!schema.Contains("Reachability"))
                    throw new Exception("DataViewSchema does not contain Reachability.");
                StreamHelper.SaveModel(env, xf, outModelFilePath);

                var saver = env.CreateSaver("Text{header=- schema=-}");
                using (var fs2 = File.Create(outputDataFilePath))
                {
                    saver.SaveData(fs2, TestTransformHelper.AddFlatteningTransform(env, xf),
                                    StreamHelper.GetColumnsIndex(xf.Schema, new[] { "Features", "Ordering", "Reachability" }));
                }
                // Checking the values.
                var lines = File.ReadAllLines(outputDataFilePath).Select(c => c.Split('\t')).Where(c => c.Length == 4);
                if (!lines.Any())
                    throw new Exception(string.Format("The output file is empty or not containing three columns '{0}'", outputDataFilePath));
                var clusters = lines.Select(c => c[1]).Distinct();
                if (clusters.Count() <= 1)
                    throw new Exception("Only one cluster, this is unexpected.");

                // Serialization.
                var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
                var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, xf, loader, outData, outData2);
            }
        }

        #endregion
    }
}
