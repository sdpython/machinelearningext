// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Runtime;


namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsArguments
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of neighbors to consider.")]
        public int k = 5;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Weighting strategy for neighbors", ShortName = "a")]
        public NearestNeighborsAlgorithm algo = NearestNeighborsAlgorithm.kdtree;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Weighting strategy for neighbors", ShortName = "w")]
        public NearestNeighborsWeights weighting = NearestNeighborsWeights.uniform;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Distnace to use", ShortName = "d")]
        public NearestNeighborsDistance distance = NearestNeighborsDistance.L2;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads and number of KD-Tree built to sppeed up the search.", ShortName = "nt")]
        public int? numThreads = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Seed to distribute example over trees.", ShortName = "s")]
        public int? seed = 42;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Column which contains a unique identifier for each observation (optional). " +
                                                      "Type must long.", ShortName = "id")]
        public string colId = null;

        public virtual void Write(ModelSaveContext ctx, IHost host)
        {
            ctx.Writer.Write(k);
            ctx.Writer.Write((int)algo);
            ctx.Writer.Write((int)weighting);
            ctx.Writer.Write((int)distance);
            ctx.Writer.Write(numThreads ?? -1);
            ctx.Writer.Write(seed ?? -1);
            ctx.Writer.Write(string.IsNullOrEmpty(colId) ? "" : colId);
        }

        public virtual void Read(ModelLoadContext ctx, IHost host)
        {
            k = ctx.Reader.ReadInt32();
            algo = (NearestNeighborsAlgorithm)ctx.Reader.ReadInt32();
            weighting = (NearestNeighborsWeights)ctx.Reader.ReadInt32();
            distance = (NearestNeighborsDistance)ctx.Reader.ReadInt32();
            numThreads = ctx.Reader.ReadInt32();
            if (numThreads == -1)
                numThreads = null;
            seed = ctx.Reader.ReadInt32();
            if (seed == -1)
                seed = null;
            colId = ctx.Reader.ReadString();
            if (string.IsNullOrEmpty(colId))
                colId = null;
        }

        public virtual void PostProcess()
        {
        }
    }
}

