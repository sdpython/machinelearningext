// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Runtime;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using NearestNeighborsTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsTrainer;
using NearestNeighborsBinaryClassificationTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsBinaryClassificationTrainer;
using NearestNeighborsMulticlassClassificationTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsMulticlassClassificationTrainer;

[assembly: LoadableClass(NearestNeighborsBinaryClassificationTrainer.Summary,
    typeof(NearestNeighborsBinaryClassificationTrainer),
    typeof(NearestNeighborsTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    NearestNeighborsBinaryClassificationTrainer.LoaderSignature,
    NearestNeighborsBinaryClassificationTrainer.LongName,
    NearestNeighborsBinaryClassificationTrainer.ShortName)]

[assembly: LoadableClass(NearestNeighborsMulticlassClassificationTrainer.Summary,
    typeof(NearestNeighborsMulticlassClassificationTrainer),
    typeof(NearestNeighborsTrainer.Arguments),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    NearestNeighborsMulticlassClassificationTrainer.LoaderSignature,
    NearestNeighborsMulticlassClassificationTrainer.LongName,
    NearestNeighborsMulticlassClassificationTrainer.ShortName)]

namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsBinaryClassificationTrainer : NearestNeighborsTrainer
    {
        public const string LoaderSignature = "NearestNeighborsBC";  // Not more than 24 letters.
        public const string Summary = "k-Nearest Neighbors trainer for Binary Classification";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "kNN";
        public const string LongName = "kNNbc";

        public NearestNeighborsBinaryClassificationTrainer(IHostEnvironment env, Arguments args) : base(env, args, LoaderSignature)
        {
        }

        protected override INearestNeighborsPredictor Train(RoleMappedData data)
        {
            data.CheckBinaryLabel();
            return base.Train(data);
        }

        protected override INearestNeighborsPredictor CreateTrainedPredictor<TLabel>(KdTree[] kdtrees,
            Dictionary<long, Tuple<TLabel, float>> labelsWeights)
        {
            return NearestNeighborsBinaryClassifierPredictor.Create<TLabel>(Host, kdtrees, labelsWeights,
                                _args.k, _args.algo, _args.weighting);
        }
    }

    public class NearestNeighborsMulticlassClassificationTrainer : NearestNeighborsTrainer
    {
        public const string LoaderSignature = "NearestNeighborsMCC";  // Not more than 24 letters.
        public const string Summary = "k-Nearest Neighbors trainer for Multi-Class Classification";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "kNNmc";
        public const string LongName = "kNNmcl";

        public NearestNeighborsMulticlassClassificationTrainer(IHostEnvironment env, Arguments args) : base(env, args, LoaderSignature)
        {
        }

        protected override INearestNeighborsPredictor Train(RoleMappedData data)
        {
            int count;
            data.CheckMulticlassLabel(out count);
            return base.Train(data);
        }

        protected override INearestNeighborsPredictor CreateTrainedPredictor<TLabel>(KdTree[] kdtrees,
            Dictionary<long, Tuple<TLabel, float>> labelsWeights)
        {
            return NearestNeighborsMulticlassClassifierPredictor.Create<TLabel>(Host, kdtrees, labelsWeights,
                                _args.k, _args.algo, _args.weighting);
        }
    }
}
