// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

using NearestNeighborsBinaryClassifierPredictor = Scikit.ML.NearestNeighbors.NearestNeighborsBinaryClassifierPredictor;
using NearestNeighborsMulticlassClassifierPredictor = Scikit.ML.NearestNeighbors.NearestNeighborsMulticlassClassifierPredictor;


[assembly: LoadableClass(typeof(NearestNeighborsBinaryClassifierPredictor), null, typeof(SignatureLoadModel),
    NearestNeighborsBinaryClassifierPredictor.LongName, NearestNeighborsBinaryClassifierPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(NearestNeighborsMulticlassClassifierPredictor), null, typeof(SignatureLoadModel),
    NearestNeighborsMulticlassClassifierPredictor.LongName, NearestNeighborsMulticlassClassifierPredictor.LoaderSignature)]


namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsBinaryClassifierPredictor :
        NearestNeighborsPredictor, INearestNeighborsPredictor, IValueMapper, ICanSaveModel
    {
        public const string LoaderSignature = "kNNBinaryClassifier";
        public const string RegistrationName = LoaderSignature;
        public const string LongName = "Nearest Neighbors for Binary Classification";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KNNBINCL",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NearestNeighborsBinaryClassifierPredictor).Assembly.FullName);
        }

        public PredictionKind PredictionKind { get { return PredictionKind.BinaryClassification; } }

        public DataViewType OutputType { get { return NumberDataViewType.Single; } }

        internal static NearestNeighborsBinaryClassifierPredictor Create<TLabel>(IHost host,
                                KdTree[] kdtrees, Dictionary<long, Tuple<TLabel, float>> labelWeights,
                                int k, NearestNeighborsAlgorithm algo, NearestNeighborsWeights weights)
            where TLabel : IComparable<TLabel>
        {
            Contracts.CheckValue(host, "host");
            host.CheckValue(kdtrees, "kdtrees");
            host.Check(!kdtrees.Where(c => c == null).Any(), "kdtrees");
            NearestNeighborsBinaryClassifierPredictor res;
            using (var ch = host.Start("Creating kNN predictor"))
            {
                var trees = new NearestNeighborsTrees(host, kdtrees);
                var pred = new NearestNeighborsValueMapper<TLabel>(host, labelWeights);
                res = new NearestNeighborsBinaryClassifierPredictor(host, trees, pred, k, algo, weights);
            }
            return res;
        }

        internal NearestNeighborsBinaryClassifierPredictor(IHost host, NearestNeighborsTrees trees, INearestNeighborsValueMapper predictor,
                                int k, NearestNeighborsAlgorithm algo, NearestNeighborsWeights weights)
        {
            _host = host;
            _k = k;
            _algo = algo;
            _weights = weights;
            _nearestPredictor = predictor;
            _nearestTrees = trees;
        }

        private NearestNeighborsBinaryClassifierPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckNonWhiteSpace(RegistrationName, "name");
            _host = env.Register(RegistrationName);
            base.ReadCore(_host, ctx);
        }

        public static NearestNeighborsBinaryClassifierPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new NearestNeighborsBinaryClassifierPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            base.SaveCore(ctx);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            _host.Check(typeof(TIn) == typeof(VBuffer<float>));
            _host.Check(typeof(TOut) == typeof(float));
            var res = _nearestPredictor.GetMapper<TIn, TOut>(_nearestTrees, _k, _algo, _weights, PredictionKind.BinaryClassification);
            if (res == null)
                throw _host.Except("Incompatible types {0}, {1}", typeof(TIn), typeof(TOut));
            return res;
        }
    }

    public class NearestNeighborsMulticlassClassifierPredictor :
        NearestNeighborsPredictor, INearestNeighborsPredictor, IValueMapper, ICanSaveModel
#if IMPLIValueMapperDist
        , IValueMapperDist
#endif
    {
        public const string LoaderSignature = "kNNMulticlassClassifier";
        public const string RegistrationName = LoaderSignature;
        public const string LongName = "Nearest Neighbors for Multi Class Classification";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KNNMCLCL",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NearestNeighborsMulticlassClassifierPredictor).Assembly.FullName);
        }

        public PredictionKind PredictionKind { get { return PredictionKind.MulticlassClassification; } }

        int _nbClass;

        public DataViewType OutputType { get { return new VectorType(NumberDataViewType.Single, _nbClass); } }

#if IMPLIValueMapperDist
        public DataViewType DistType { get { return OutputType; } }
#endif

        internal static NearestNeighborsMulticlassClassifierPredictor Create<TLabel>(IHost host,
                                KdTree[] kdtrees, Dictionary<long, Tuple<TLabel, float>> labelWeights,
                                int k, NearestNeighborsAlgorithm algo, NearestNeighborsWeights weights)
            where TLabel : IComparable<TLabel>
        {
            Contracts.CheckValue(host, "host");
            host.CheckValue(kdtrees, "kdtrees");
            host.Check(!kdtrees.Where(c => c == null).Any(), "kdtrees");
            NearestNeighborsMulticlassClassifierPredictor res;
            using (var ch = host.Start("Creating kNN predictor"))
            {
                var trees = new NearestNeighborsTrees(host, kdtrees);
                var pred = new NearestNeighborsValueMapper<TLabel>(host, labelWeights);
                res = new NearestNeighborsMulticlassClassifierPredictor(host, trees, pred, k, algo, weights);
            }
            return res;
        }

        internal NearestNeighborsMulticlassClassifierPredictor(IHost host, NearestNeighborsTrees trees, INearestNeighborsValueMapper predictor,
                                int k, NearestNeighborsAlgorithm algo, NearestNeighborsWeights weights)
        {
            _host = host;
            _k = k;
            _algo = algo;
            _weights = weights;
            _nearestPredictor = predictor;
            _nearestTrees = trees;
            ComputeNbClass();
        }

        private NearestNeighborsMulticlassClassifierPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckNonWhiteSpace(RegistrationName, "name");
            _host = env.Register(RegistrationName);
            base.ReadCore(_host, ctx);
            ComputeNbClass();
        }

        public static NearestNeighborsMulticlassClassifierPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new NearestNeighborsMulticlassClassifierPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            base.SaveCore(ctx);
        }

        void ComputeNbClass()
        {
            Contracts.AssertValue(_nearestPredictor);
            _nbClass = _nearestPredictor.ComputeNbClass(PredictionKind);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            _host.Check(typeof(TIn) == typeof(VBuffer<float>));
            var res = _nearestPredictor.GetMapper<TIn, TOut>(_nearestTrees, _k, _algo, _weights, PredictionKind.MulticlassClassification);
            if (res == null)
                throw _host.Except("Incompatible types {0}, {1}", typeof(TIn), typeof(TOut));
            return res;
        }

#if IMPLIValueMapperDist
        public ValueMapper<TIn, TDst, TDist> GetMapper<TIn, TDst, TDist>()
        {
            _host.Check(typeof(TIn) == typeof(VBuffer<float>));
            var res = _nearestPredictor.GetMapper<TIn, TDst>(_nearestTrees, _k, _algo, _weights, PredictionKind.MulticlassClassification);
            if (res == null)
                throw _host.Except("Incompatible types {0}, {1}", typeof(TIn), typeof(TDst));
            ValueMapper<TIn, TDst, TDst> resDist = (ref TIn input, ref TDst scores, ref TDst probs) =>
            {
                res(ref input, ref scores);
                probs = scores;
            };
            return resDist as ValueMapper<TIn, TDst, TDist>;
        }
#endif
    }
}
