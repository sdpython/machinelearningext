﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.EntryPoints;
using Scikit.ML.PipelineHelper;

// The following files makes the object visible to maml.
// This way, it can be added to any pipeline.
using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using NearestNeighborsTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsTrainer;
using NearestNeighborsBinaryClassificationTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsBinaryClassificationTrainer;
using NearestNeighborsMultiClassClassificationTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsMultiClassClassificationTrainer;
using EntryPointNearestNeighborsBc = Scikit.ML.NearestNeighbors.EntryPointNearestNeighborsBc;
using EntryPointNearestNeighborsMc = Scikit.ML.NearestNeighbors.EntryPointNearestNeighborsMc;

[assembly: LoadableClass(NearestNeighborsBinaryClassificationTrainer.Summary,
    typeof(NearestNeighborsBinaryClassificationTrainer),
    typeof(NearestNeighborsTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    NearestNeighborsBinaryClassificationTrainer.LoaderSignature,
    NearestNeighborsBinaryClassificationTrainer.LongName,
    NearestNeighborsBinaryClassificationTrainer.ShortName)]

[assembly: LoadableClass(NearestNeighborsMultiClassClassificationTrainer.Summary,
    typeof(NearestNeighborsMultiClassClassificationTrainer),
    typeof(NearestNeighborsTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    NearestNeighborsMultiClassClassificationTrainer.LoaderSignature,
    NearestNeighborsMultiClassClassificationTrainer.LongName,
    NearestNeighborsMultiClassClassificationTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(EntryPointNearestNeighborsBc), null,
    typeof(SignatureEntryPointModule), NearestNeighborsBinaryClassificationTrainer.EntryPointName)]

[assembly: LoadableClass(typeof(void), typeof(EntryPointNearestNeighborsMc), null,
    typeof(SignatureEntryPointModule), NearestNeighborsMultiClassClassificationTrainer.EntryPointName)]


namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsBinaryClassificationTrainer : NearestNeighborsTrainer
    {
        public const string LoaderSignature = "NearestNeighborsBC";  // Not more than 24 letters.
        public const string Summary = "k-Nearest Neighbors trainer for Binary Classification";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "kNN";
        public const string LongName = "kNNbc";
        public const string EntryPointName = "NearestNeighborsBc";

        public NearestNeighborsBinaryClassificationTrainer(IHostEnvironment env, Arguments args) : base(env, args, LoaderSignature)
        {
        }

        public override void Train(RoleMappedData data)
        {
            data.CheckBinaryLabel();
            base.Train(data);
        }

        protected override INearestNeighborsPredictor CreateTrainedPredictor<TLabel>(KdTree[] kdtrees,
            Dictionary<long, Tuple<TLabel, float>> labelsWeights)
        {
            return NearestNeighborsBinaryClassifierPredictor.Create<TLabel>(Host, kdtrees, labelsWeights,
                                _args.k, _args.algo, _args.weight);
        }
    }

    public static partial class EntryPointNearestNeighborsBc
    {
        [TlcModule.EntryPoint(
            Name = "ExtNearestNeighbors." + NearestNeighborsBinaryClassificationTrainer.EntryPointName,
            Desc = NearestNeighborsBinaryClassificationTrainer.Summary,
            UserName = NearestNeighborsBinaryClassificationTrainer.EntryPointName,
            ShortName = NearestNeighborsBinaryClassificationTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, NearestNeighborsBinaryClassificationTrainer.ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return EntryPointsHelper.Train<NearestNeighborsBinaryClassificationTrainer.ArgumentsEntryPoint,
                                           CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new NearestNeighborsBinaryClassificationTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }

    public class NearestNeighborsMultiClassClassificationTrainer : NearestNeighborsTrainer
    {
        public const string LoaderSignature = "NearestNeighborsMCC";  // Not more than 24 letters.
        public const string Summary = "k-Nearest Neighbors trainer for Multi-Class Classification";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "kNNmc";
        public const string LongName = "kNNmcl";
        public const string EntryPointName = "NearestNeighborsMc";

        public NearestNeighborsMultiClassClassificationTrainer(IHostEnvironment env, Arguments args) : base(env, args, LoaderSignature)
        {
        }

        public override void Train(RoleMappedData data)
        {
            int count;
            data.CheckMultiClassLabel(out count);
            base.Train(data);
        }

        protected override INearestNeighborsPredictor CreateTrainedPredictor<TLabel>(KdTree[] kdtrees,
            Dictionary<long, Tuple<TLabel, float>> labelsWeights)
        {
            return NearestNeighborsMultiClassClassifierPredictor.Create<TLabel>(Host, kdtrees, labelsWeights,
                                _args.k, _args.algo, _args.weight);
        }

        public static partial class EntryPointNearestNeighborsBc
        {
            [TlcModule.EntryPoint(
                Name = "ExtNearestNeighbors." + NearestNeighborsBinaryClassificationTrainer.EntryPointName,
                Desc = NearestNeighborsBinaryClassificationTrainer.Summary,
                UserName = NearestNeighborsBinaryClassificationTrainer.EntryPointName,
                ShortName = NearestNeighborsBinaryClassificationTrainer.ShortName)]
            public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, NearestNeighborsBinaryClassificationTrainer.ArgumentsEntryPoint input)
            {
                Contracts.CheckValue(env, nameof(env));
                var host = env.Register("TrainNearestNeighbors");
                host.CheckValue(input, nameof(input));
                EntryPointUtils.CheckInputArgs(host, input);

                return EntryPointsHelper.Train<NearestNeighborsBinaryClassificationTrainer.ArgumentsEntryPoint,
                                               CommonOutputs.BinaryClassificationOutput>(host, input,
                    () => new NearestNeighborsBinaryClassificationTrainer(host, input),
                    getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
            }
        }
    }

    public static partial class EntryPointNearestNeighborsMc
    {
        [TlcModule.EntryPoint(
            Name = "ExtNearestNeighbors." + NearestNeighborsMultiClassClassificationTrainer.EntryPointName,
            Desc = NearestNeighborsMultiClassClassificationTrainer.Summary,
            UserName = NearestNeighborsMultiClassClassificationTrainer.EntryPointName,
            ShortName = NearestNeighborsMultiClassClassificationTrainer.ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, NearestNeighborsMultiClassClassificationTrainer.ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainNearestNeighbors");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return EntryPointsHelper.Train<NearestNeighborsMultiClassClassificationTrainer.ArgumentsEntryPoint,
                                           CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new NearestNeighborsMultiClassClassificationTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }
}
