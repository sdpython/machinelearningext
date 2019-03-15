// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;

using OptimizedOVAPredictor = Scikit.ML.Multiclass.OptimizedOVAPredictor;


[assembly: LoadableClass(typeof(OptimizedOVAPredictor), null, typeof(SignatureLoadModel),
    "Optimized OVA Executor", OptimizedOVAPredictor.LoaderSignature)]


namespace Scikit.ML.Multiclass
{
    using TScalarPredictor = IPredictor; //IPredictorProducing<float>;

    public sealed class OptimizedOVAPredictor :
        PredictorBase<VBuffer<float>>,
        IValueMapper,
#if IMPLIValueMapperDist
        IValueMapperDist,
#endif
        ICanSaveModel
    /*,
    ICanSaveInSourceCode,
    ICanSaveInTextFormat*/
    {
        public const string LoaderSignature = "OptimizedOVAExec";
        public const string RegistrationName = "OptimizedOVAPredictor";

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ORNDOVA ",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OptimizedOVAPredictor).Assembly.FullName);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";

        private readonly ImplBase _impl;

        private readonly DataViewType _outputType;

        public override PredictionKind PredictionKind { get { return PredictionKind.MulticlassClassification; } }

        public DataViewType InputType { get { return _impl.InputType; } }
        public DataViewType OutputType { get { return _outputType; } }
#if IMPLIValueMapperDist
        public DataViewType DistType { get { return _outputType; } }
#endif

        internal static OptimizedOVAPredictor Create(IHost host, bool useProb, TScalarPredictor[] predictors)
        {
            ImplBase impl;

            using (var ch = host.Start("Creating OVA predictor"))
            {
                IValueMapperDist ivmd = null;
                if (useProb &&
                    ((ivmd = predictors[0] as IValueMapperDist) == null ||
                        ivmd.OutputType != NumberDataViewType.Single ||
                        ivmd.DistType != NumberDataViewType.Single))
                {
                    ch.Warning("useProbabilities specified with basePredictor that can't produce probabilities.");
                    ivmd = null;
                }

                if (ivmd != null)
                {
                    var dists = new IValueMapperDist[predictors.Length];
                    for (int i = 0; i < predictors.Length; ++i)
                        dists[i] = (IValueMapperDist)predictors[i];
                    impl = new ImplDist(dists);
                }
                else
                    impl = new ImplRaw(predictors);
            }

            return new OptimizedOVAPredictor(host, impl);
        }

        private OptimizedOVAPredictor(IHostEnvironment env, ImplBase impl) : base(env, RegistrationName)
        {
            Host.AssertValue(impl, "impl");
            Host.Assert(Utils.Size(impl.Predictors) > 0);

            _impl = impl;
            _outputType = new VectorType(NumberDataViewType.Single, _impl.Predictors.Length);
        }

        private OptimizedOVAPredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // bool: useDist
            // int: predictor count
            bool useDist = ctx.Reader.ReadBoolByte();
            int len = ctx.Reader.ReadInt32();
            Host.CheckDecode(len > 0);

            if (useDist)
            {
                var predictors = new IValueMapperDist[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplDist(predictors);
            }
            else
            {
                var predictors = new TScalarPredictor[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplRaw(predictors);
            }

            _outputType = new VectorType(NumberDataViewType.Single, _impl.Predictors.Length);
        }

        public static OptimizedOVAPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            return new OptimizedOVAPredictor(env, ctx);
        }

        private static void LoadPredictors<TPredictor>(IHostEnvironment env, TPredictor[] predictors, ModelLoadContext ctx)
            where TPredictor : class
        {
            for (int i = 0; i < predictors.Length; i++)
                ctx.LoadModel<TPredictor, SignatureLoadModel>(env, out predictors[i], string.Format(SubPredictorFmt, i));
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            var preds = _impl.Predictors;
            ctx.Writer.WriteBoolByte(_impl is ImplDist);
            ctx.Writer.Write(preds.Length);

            for (int i = 0; i < preds.Length; i++)
                ctx.SaveModel(preds[i], string.Format(SubPredictorFmt, i));
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));
            return (ValueMapper<TIn, TOut>)(Delegate)_impl.GetMapper();
        }

        public void SaveAsCode(TextWriter writer, RoleMappedSchema names)
        {
            var preds = _impl.Predictors;
            writer.WriteLine("double[] outputs = new double[{0}];", preds.Length);

            for (int i = 0; i < preds.Length; i++)
            {
                var saveInSourceCode = preds[i] as ICanSaveInSourceCode;
                Host.Check(saveInSourceCode != null, "Saving in code is not supported.");

                writer.WriteLine("{");
                saveInSourceCode.SaveAsCode(writer, names);
                writer.WriteLine("outputs[{0}] = output;", i);
                writer.WriteLine("}");
            }
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema names)
        {
            var preds = _impl.Predictors;

            for (int i = 0; i < preds.Length; i++)
            {
                var saveInText = preds[i] as ICanSaveInTextFormat;
                Host.Check(saveInText != null, "Saving in text is not supported.");

                writer.WriteLine("#region: class-{0} classifier", i);
                saveInText.SaveAsText(writer, names);

                writer.WriteLine("#endregion: class-{0} classifier", i);
                writer.WriteLine();
            }
        }

#if IMPLIValueMapperDist
        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
        {
            _host.Check(typeof(TIn) == typeof(VBuffer<float>));
            _host.Check(typeof(TOut) == typeof(VBuffer<float>));
            _host.Check(typeof(TDist) == typeof(VBuffer<float>));

            return (ValueMapper<TIn, TOut, TDist>)(Delegate)_impl.GetMapperDist();
        }
#endif

        private abstract class ImplBase
        {
            public abstract DataViewType InputType { get; }
            public abstract IValueMapper[] Predictors { get; }
            public abstract ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper();

#if IMPLIValueMapperDist
            public ValueMapper<VBuffer<float>, VBuffer<float>, VBuffer<float>> GetMapperDist()
            {
                var mapper = GetMapper();
                var tempOut = new VBuffer<float>();
                return (ref VBuffer<float> src, ref VBuffer<float> dst, ref VBuffer<float> prob) =>
                {
                    mapper(ref src, ref tempOut);
                    Normalize(ref tempOut, ref prob);
                };
            }
#endif

            protected void Normalize(ref VBuffer<float> src, ref VBuffer<float> dst)
            {
                float sum = 0;
                for (int i = 0; i < src.Count; i++)
                    sum += src.Values[i];

                if (sum != 0)
                {
                    float[] values;
                    int[] indices;
                    int length;
                    if (dst.Values != null && dst.Values.Length < src.Count)
                    {
                        values = dst.Values;
                        length = dst.Length;
                        indices = src.IsDense ? null : (dst.Indices == null || dst.Indices.Length < src.Count ? new int[length] : dst.Indices);
                    }
                    else
                    {
                        values = new float[src.Count];
                        indices = src.IsDense ? null : new int[src.Count];
                        length = values.Length;
                    }
                    dst = new VBuffer<float>(src.Count, length, values, indices);
                    if (src.IsDense)
                    {
                        for (int i = 0; i < src.Count; i++)
                            dst.Values[i] = src.Values[i] / sum;
                    }
                    else
                    {
                        for (int i = 0; i < src.Count; i++)
                        {
                            dst.Values[i] = src.Values[i] / sum;
                            dst.Indices[i] = src.Indices[i];
                        }
                    }
                }
                else
                    src.CopyTo(ref dst);
            }

            protected bool IsValid(IValueMapper mapper, ref DataViewType inputType)
            {
                if (mapper == null)
                    return false;
                if (mapper.OutputType != NumberDataViewType.Single)
                    return false;
                if (!mapper.InputType.IsVector() || mapper.InputType.ItemType() != NumberDataViewType.Single)
                    return false;
                if (inputType == null)
                    inputType = mapper.InputType;
                else if (inputType.VectorSize() != mapper.InputType.VectorSize())
                {
                    if (inputType.VectorSize() == 0)
                        inputType = mapper.InputType;
                    else if (mapper.InputType.VectorSize() != 0)
                        return false;
                }
                return true;
            }
        }

        private sealed class ImplRaw : ImplBase
        {
            private readonly DataViewType _inputType;
            private readonly IValueMapper[] _mappers;

            public override DataViewType InputType { get { return _inputType; } }
            public override IValueMapper[] Predictors { get { return _mappers; } }

            internal ImplRaw(TScalarPredictor[] predictors)
            {
                Contracts.Check(Utils.Size(predictors) > 0);

                _mappers = new IValueMapper[predictors.Length];
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i] as IValueMapper;
                    Contracts.Check(IsValid(vm, ref _inputType), "Predictor doesn't implement the expected interface");
                    _mappers[i] = vm;
                }
            }

            public override ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<float>, float>[_mappers.Length];
                for (int i = 0; i < _mappers.Length; i++)
                    maps[i] = _mappers[i].GetMapper<VBuffer<float>, float>();

                return
                    (in VBuffer<float> src, ref VBuffer<float> dst) =>
                    {
                        if (_inputType.VectorSize() > 0)
                            Contracts.Check(src.Length == _inputType.VectorSize());

                        var values = dst.Values;
                        if (Utils.Size(values) < maps.Length)
                            values = new float[maps.Length];

                        var tmp = src;
                        Parallel.For(0, maps.Length, i => maps[i](in tmp, ref values[i]));
                        dst = new VBuffer<float>(maps.Length, values, dst.Indices);
                    };
            }
        }

        private sealed class ImplDist : ImplBase
        {
            private readonly DataViewType _inputType;
            private readonly IValueMapperDist[] _mappers;

            public override DataViewType InputType { get { return _inputType; } }
            public override IValueMapper[] Predictors { get { return _mappers; } }

            internal ImplDist(IValueMapperDist[] predictors)
            {
                Contracts.Check(Utils.Size(predictors) > 0);

                _mappers = new IValueMapperDist[predictors.Length];
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i];
                    Contracts.Check(IsValid(vm, ref _inputType), "Predictor doesn't implement the expected interface");
                    _mappers[i] = vm;
                }
            }

            private bool IsValid(IValueMapperDist mapper, ref DataViewType inputType)
            {
                if (!base.IsValid(mapper, ref inputType))
                    return false;
                if (mapper.DistType != NumberDataViewType.Single)
                    return false;
                return true;
            }

            public override ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<float>, float, float>[_mappers.Length];
                for (int i = 0; i < _mappers.Length; i++)
                    maps[i] = _mappers[i].GetMapper<VBuffer<float>, float, float>();

                return
                    (in VBuffer<float> src, ref VBuffer<float> dst) =>
                    {
                        if (_inputType.VectorSize() > 0)
                            Contracts.Check(src.Length == _inputType.VectorSize());

                        var values = dst.Values;
                        if (Utils.Size(values) < maps.Length)
                            values = new float[maps.Length];

                        var tmp = src;
                        Parallel.For(0, maps.Length,
                            i =>
                            {
                                float score = 0;
                                maps[i](in tmp, ref score, ref values[i]);
                            });
                        Normalize(values, maps.Length);
                        dst = new VBuffer<float>(maps.Length, values, dst.Indices);
                    };
            }

            private void Normalize(float[] output, int count)
            {
                // Clamp to zero and normalize.
                Double sum = 0;
                for (int i = 0; i < count; i++)
                {
                    var value = output[i];
                    if (value >= 0)
                        sum += value;
                    else
                        output[i] = 0;
                }

                if (sum > 0)
                {
                    for (int i = 0; i < count; i++)
                        output[i] = (float)(output[i] / sum);
                }
            }
        }
    }
}
