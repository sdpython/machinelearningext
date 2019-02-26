// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.NearestNeighbors
{
    public interface INearestNeighborsValueMapper
    {
        DataKind Kind { get; }
        ValueMapper<TIn, TOut> GetMapper<TIn, TOut>(NearestNeighborsTrees trees, int k, NearestNeighborsAlgorithm algo, NearestNeighborsWeights weight, PredictionKind kind);
        void SaveCore(ModelSaveContext ctx);
        int ComputeNbClass(PredictionKind kind);   // Only available for classifier.
    }

    internal class NearestNeighborsValueMapper<TLabel> : INearestNeighborsValueMapper
        where TLabel : IComparable<TLabel>
    {
        readonly IHost _host;
        protected Dictionary<long, Tuple<TLabel, float>> _labelWeights;

        public DataKind Kind { get { return SchemaHelper.GetColumnType<TLabel>().RawKind(); } }

        public NearestNeighborsValueMapper(IHost host, Dictionary<long, Tuple<TLabel, float>> labelWeights)
        {
            _host = host;
            _labelWeights = labelWeights;
        }

        public NearestNeighborsValueMapper(IHost host, ModelLoadContext ctx)
        {
            _host = host;
            _host.CheckValue(ctx, "ctx");
            ReadCore(ctx);
        }

        public void SaveCore(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            // We save the labels. The rest is already saved.
            ctx.Writer.Write(_labelWeights.Count);

            var conv = new TypedConverters<TLabel>();

            foreach (var pair in _labelWeights)
            {
                ctx.Writer.Write(pair.Key);
                conv.Save(ctx, pair.Value.Item1);
                ctx.Writer.Write(pair.Value.Item2);
            }
        }

        protected void ReadCore(ModelLoadContext ctx)
        {
            // We load the labels. The rest is already saved.
            var conv = new TypedConverters<TLabel>();
            int nb = ctx.Reader.ReadInt32();
            _labelWeights = new Dictionary<long, Tuple<TLabel, float>>();

            long key;
            TLabel lab = default(TLabel);
            float weight;

            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadInt64();
                conv.Read(ctx, ref lab);
                weight = ctx.Reader.ReadFloat();
                _labelWeights[key] = new Tuple<TLabel, float>(lab, weight);
            }
        }

        public int ComputeNbClass(PredictionKind kind)
        {
            _host.AssertValue(_labelWeights);
            KeyValuePair<long, Tuple<TLabel, float>> first = new KeyValuePair<long, Tuple<TLabel, float>>(0, new Tuple<TLabel, float>(default(TLabel), 0));
            foreach (var pair in _labelWeights)
            {
                first = pair;
                break;
            }

            TLabel mini = first.Value.Item1, maxi = first.Value.Item1;
            foreach (var pair in _labelWeights)
            {
                if (mini.CompareTo(pair.Value.Item1) > 0)
                    mini = pair.Value.Item1;
                if (maxi.CompareTo(pair.Value.Item1) < 0)
                    maxi = pair.Value.Item1;
            }

            var conv = new TypedConverters<TLabel>(DataKind.Int32);

            int imini = 0, imaxi = 0;
            var convMapper = conv.GetMapper<int>();
            if (convMapper == null)
                throw _host.Except("No conversion from {0} to {1}", typeof(TLabel), typeof(int));
            switch (kind)
            {
                case PredictionKind.BinaryClassification:
                    if (mini.CompareTo(maxi) == 0)
                        throw _host.Except("Only one class, two are expected.");
                    convMapper(in mini, ref imini);
                    convMapper(in maxi, ref imaxi);
                    if (imaxi - imini != 1)
                        throw _host.Except("More than two classes: min(labels)={0} max(labels)={1}", imini, imaxi);
                    return 2;
                case PredictionKind.MultiClassClassification:
                    if (mini.CompareTo(maxi) == 0)
                        throw _host.Except("Only one class, more are expected.");
                    convMapper(in mini, ref imini);
                    convMapper(in maxi, ref imaxi);
                    return imini == 0 ? imaxi + 1 : imini;
                default:
                    throw _host.ExceptNotSupp("Not suported for predictor {0}", kind);
            }
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>(NearestNeighborsTrees trees, int k,
                                NearestNeighborsAlgorithm algo, NearestNeighborsWeights weight, PredictionKind kind)
        {
            _host.Check(typeof(TIn) == typeof(VBuffer<float>));
            _host.CheckValue(_labelWeights, "_labelWeights");
            _host.Check(algo == NearestNeighborsAlgorithm.kdtree, "algo");

            if (weight == NearestNeighborsWeights.uniform)
            {
                switch (kind)
                {
                    case PredictionKind.BinaryClassification:
                        return GetMapperBinaryPrediction<TIn, TOut>(trees, k, algo, weight);
                    case PredictionKind.MultiClassClassification:
                        return GetMapperMultiClassPrediction<TIn, TOut>(trees, k, algo, weight);
                    default:
                        throw _host.ExceptNotImpl("Not implemented yet for kind={0}", kind);
                }
            }
            else
                throw _host.ExceptNotImpl("Not implemented yet for wieght={0}", weight);
        }

        #region binary classification

        ValueMapper<TIn, TOut> GetMapperBinaryPrediction<TIn, TOut>(NearestNeighborsTrees trees, int k,
                        NearestNeighborsAlgorithm algo, NearestNeighborsWeights weight)
        {
            var conv = new TypedConverters<TLabel>();
            TLabel positiveClass = default(TLabel);
            if (typeof(TLabel) == typeof(bool))
            {
                var convMap = conv.GetMapperFrom<bool>();
                var b = true;
                convMap(in b, ref positiveClass);
            }
            else if (typeof(TLabel) == typeof(float))
            {
                var convMap = conv.GetMapperFrom<float>();
                var b = 1f;
                convMap(in b, ref positiveClass);
            }
            else if (typeof(TLabel) == typeof(uint))
            {
                var convMap = conv.GetMapperFrom<uint>();
                uint b = 1;
                convMap(in b, ref positiveClass);
            }
            else
                _host.ExceptNotImpl("Not implemented for type {0}", typeof(TLabel));

            Dictionary<TLabel, float> hist = null;
            if (weight == NearestNeighborsWeights.uniform)
            {
                ValueMapper<VBuffer<float>, float> mapper = (in VBuffer<float> input, ref float output) =>
                {
                    GetMapperUniformBinaryPrediction(trees, k, in input, ref output, positiveClass, ref hist);
                };
                return mapper as ValueMapper<TIn, TOut>;
            }
            else
                throw _host.ExceptNotImpl("Not implemented for {0}", weight);
        }

        void GetMapperUniformBinaryPrediction(NearestNeighborsTrees trees, int k, in VBuffer<float> input,
            ref float output, TLabel positiveClass, ref Dictionary<TLabel, float> hist)
        {
            if (hist == null || hist.Count != 2)
                hist = new Dictionary<TLabel, float>();
            else
            {
                // I did not find a way to set all keys to 0 without creating an intermdiate array.
                // C# should allow such thing.
                foreach (var key in hist.Keys.ToArray())
                    hist[key] = 0;
            }

            var neighbors = trees.NearestNNeighbors(input, k);
            TLabel lab;
            foreach (var pair in neighbors)
            {
                lab = _labelWeights[pair.Value].Item1;
                if (!hist.ContainsKey(lab))
                    hist[lab] = 0;
                ++hist[lab];
            }
            if (hist.Count == 0)
                output = 0f;
            else
            {
                float nb;
                if (hist.TryGetValue(positiveClass, out nb))
                    output = nb / hist.Count;
                else
                    output = 0f;
            }
        }

        #endregion

        #region multiclass classification

        ValueMapper<TIn, TOut> GetMapperMultiClassPrediction<TIn, TOut>(NearestNeighborsTrees trees, int k,
                        NearestNeighborsAlgorithm algo, NearestNeighborsWeights weight)
        {
            Dictionary<TLabel, float> hist = null;
            if (weight == NearestNeighborsWeights.uniform)
            {
                var conv = new TypedConverters<TLabel>(DataKind.UInt32);
                var mapperU4 = conv.GetMapper<uint>();
                var nbClass = ComputeNbClass(PredictionKind.MultiClassClassification);
                ValueMapper<VBuffer<float>, VBuffer<float>> mapper = (in VBuffer<float> input, ref VBuffer<float> output) =>
                {
                    GetMapperMultiClassPrediction(trees, k, in input, ref output, ref mapperU4, nbClass, ref hist);
                };
                return mapper as ValueMapper<TIn, TOut>;
            }
            else
                throw _host.ExceptNotImpl("Not implemented for {0}", weight);
        }

        void GetMapperMultiClassPrediction(NearestNeighborsTrees trees, int k, in VBuffer<float> input,
                ref VBuffer<float> output, ref ValueMapper<TLabel, uint> conv, int nbClass,
                ref Dictionary<TLabel, float> hist)
        {
            if (hist == null || hist.Count != nbClass)
                hist = new Dictionary<TLabel, float>();
            else
            {
                // I did not find a way to set all keys to 0 without creating an intermdiate array.
                // C# should allow such thing.
                foreach (var key in hist.Keys.ToArray())
                    hist[key] = 0;
            }

            int dec = typeof(TLabel) == typeof(float) ? 0 : 1;
            var neighbors = trees.NearestNNeighbors(input, k);
            TLabel lab;
            foreach (var pair in neighbors)
            {
                lab = _labelWeights[pair.Value].Item1;
                if (!hist.ContainsKey(lab))
                    hist[lab] = 0;
                ++hist[lab];
            }

            var foutput = output.Values == null || output.Values.Length < nbClass ? new float[nbClass] : output.Values;
            output = new VBuffer<float>(nbClass, foutput, output.Indices);

            if (hist.Count == 0)
            {
                for (int i = 0; i < output.Length; ++i)
                    foutput[i] = 0f;
            }
            else
            {
                for (int i = 0; i < output.Length; ++i)
                    foutput[i] = 0f;
                float nb = hist.Values.Sum();
                if (nb <= 0)
                    return;

                uint classLabel = 0;
                TLabel temp;
                foreach (var pair in hist)
                {
                    temp = pair.Key;
                    conv(in temp, ref classLabel);
                    foutput[classLabel - dec] = pair.Value / nb;
                }
            }
        }

        #endregion
    }
}
