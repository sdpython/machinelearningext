// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;

using NearestNeighborsBinaryClassifierPredictor = Scikit.ML.NearestNeighbors.NearestNeighborsBinaryClassifierPredictor;
using NearestNeighborsMultiClassClassifierPredictor = Scikit.ML.NearestNeighbors.NearestNeighborsMultiClassClassifierPredictor;


[assembly: LoadableClass(typeof(NearestNeighborsBinaryClassifierPredictor), null, typeof(SignatureLoadModel),
    NearestNeighborsBinaryClassifierPredictor.LongName, NearestNeighborsBinaryClassifierPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(NearestNeighborsMultiClassClassifierPredictor), null, typeof(SignatureLoadModel),
    NearestNeighborsMultiClassClassifierPredictor.LongName, NearestNeighborsMultiClassClassifierPredictor.LoaderSignature)]


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

    public class NearestNeighborsMultiClassClassifierPredictor :
        NearestNeighborsPredictor, INearestNeighborsPredictor, IValueMapper, ICanSaveModel
#if IMPLIValueMapperDist
        , IValueMapperDist
#endif
    {
        public const string LoaderSignature = "kNNMultiClassClassifier";
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
                loaderAssemblyName: typeof(NearestNeighborsMultiClassClassifierPredictor).Assembly.FullName);
        }

        public PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        int _nbClass;

        public DataViewType OutputType { get { return new VectorType(NumberDataViewType.Single, _nbClass); } }

#if IMPLIValueMapperDist
        public DataViewType DistType { get { return OutputType; } }
#endif

        internal static NearestNeighborsMultiClassClassifierPredictor Create<TLabel>(IHost host,
                                KdTree[] kdtrees, Dictionary<long, Tuple<TLabel, float>> labelWeights,
                                int k, NearestNeighborsAlgorithm algo, NearestNeighborsWeights weights)
            where TLabel : IComparable<TLabel>
        {
            Contracts.CheckValue(host, "host");
            host.CheckValue(kdtrees, "kdtrees");
            host.Check(!kdtrees.Where(c => c == null).Any(), "kdtrees");
            NearestNeighborsMultiClassClassifierPredictor res;
            using (var ch = host.Start("Creating kNN predictor"))
            {
                var trees = new NearestNeighborsTrees(host, kdtrees);
                var pred = new NearestNeighborsValueMapper<TLabel>(host, labelWeights);
                res = new NearestNeighborsMultiClassClassifierPredictor(host, trees, pred, k, algo, weights);
            }
            return res;
        }

        internal NearestNeighborsMultiClassClassifierPredictor(IHost host, NearestNeighborsTrees trees, INearestNeighborsValueMapper predictor,
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

        private NearestNeighborsMultiClassClassifierPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckNonWhiteSpace(RegistrationName, "name");
            _host = env.Register(RegistrationName);
            base.ReadCore(_host, ctx);
            ComputeNbClass();
        }

        public static NearestNeighborsMultiClassClassifierPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new NearestNeighborsMultiClassClassifierPredictor(env, ctx);
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
            var res = _nearestPredictor.GetMapper<TIn, TOut>(_nearestTrees, _k, _algo, _weights, PredictionKind.MultiClassClassification);
            if (res == null)
                throw _host.Except("Incompatible types {0}, {1}", typeof(TIn), typeof(TOut));
            return res;
        }

#if IMPLIValueMapperDist
        public ValueMapper<TIn, TDst, TDist> GetMapper<TIn, TDst, TDist>()
        {
            _host.Check(typeof(TIn) == typeof(VBuffer<float>));
            var res = _nearestPredictor.GetMapper<TIn, TDst>(_nearestTrees, _k, _algo, _weights, PredictionKind.MultiClassClassification);
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
