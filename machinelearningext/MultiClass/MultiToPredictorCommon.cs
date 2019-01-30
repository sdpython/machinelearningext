// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.MultiClass
{
    using TScalarPredictor = IPredictorProducing<float>;

    /// <summary>
    /// Defines a predictor which does what OVA does but produces only one model.
    /// It adds the label to the features and produces a binary answer which tells whether
    /// or not the input vecor belongs the class label which was added to the features.
    /// </summary>
    public abstract class MultiToPredictorCommon :
            PredictorBase<VBuffer<float>>, IValueMapper, ICanSaveModel, ICanSaveInSourceCode, ICanSaveInTextFormat
#if IMPLIValueMapperDist
                    , IValueMapperDist
#endif
    {
        protected IImplBase _impl;
        public IPredictor[] Predictors { get { return _impl.Predictors; } }
        public IPredictor ReclassificationPredictor { get { return _impl.ReclassificationPredictor; } }

        public override PredictionKind PredictionKind { get; }

        public ColumnType InputType
        {
            get
            {
                var insideInput = _impl.InputType;
                Contracts.Assert(insideInput.IsVector());
                int vs = insideInput.VectorSize();
                Contracts.Assert(vs > 0);
                if (_impl.SingleColumn)
                    return new VectorType(insideInput.AsVector().ItemType(), vs - 1);
                else
                {
                    int nb = _impl.MaxClassIndex() + 1;
                    return new VectorType(insideInput.AsVector().ItemType(), vs - nb);
                }
            }
        }

        public ColumnType OutputType { get { return _impl.OutputType; } }

#if IMPLIValueMapperDist
        public ColumnType DistType { get { return _impl.DistType; } }
#endif
        public ColumnType LabelType { get { return _impl.LabelType; } }

        protected MultiToPredictorCommon(IHostEnvironment env, IImplBase impl, string registrationName)
            : base(env, registrationName)
        {
            Host.AssertValue(impl, "impl");
            Host.Assert(Utils.Size(impl.Predictors) > 0);
            _impl = impl;
        }

        protected MultiToPredictorCommon(IHostEnvironment env, ModelLoadContext ctx, string registrationName)
            : base(env, registrationName, ctx)
        {
        }

        public int GetNbClasses() { return _impl.GetNbClasses(); }
        public string GetStringClasses(string sep = ",") { return _impl.GetStringClasses(sep); }
        public TLabel[] GetClasses<TLabel>()
        {
            var dataKind = LabelType.RawKind();
            switch (dataKind)
            {
                case DataKind.R4:
                    return _impl.GetClasses<float>() as TLabel[];
                case DataKind.U1:
                    return _impl.GetClasses<byte>() as TLabel[];
                case DataKind.U2:
                    return _impl.GetClasses<ushort>() as TLabel[];
                case DataKind.U4:
                    return _impl.GetClasses<uint>() as TLabel[];
                default:
                    throw Host.ExceptNotSupp("Not supported label type.");
            }
        }

        private static void LoadPredictors<TPredictor>(IHostEnvironment env,
            TPredictor[] predictors, out IPredictor reclassPredictor, ModelLoadContext ctx)
            where TPredictor : class
        {
            for (int i = 0; i < predictors.Length; i++)
                ctx.LoadModel<TPredictor, SignatureLoadModel>(env, out predictors[i], string.Format("M2B{0}", i));
            bool doesReclass = ctx.Reader.ReadByte() == 1;
            if (doesReclass)
                ctx.LoadModel<IPredictor, SignatureLoadModel>(env, out reclassPredictor, "Reclassification");
            else
                reclassPredictor = null;
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            return (ValueMapper<TIn, TOut>)(Delegate)_impl.GetMapper();
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

        public void SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            var preds = _impl.Predictors;
            writer.WriteLine("double[] outputs = new double[{0}];", preds.Length);

            for (int i = 0; i < preds.Length; i++)
            {
                var saveInSourceCode = preds[i] as ICanSaveInSourceCode;
                Host.Check(saveInSourceCode != null, "Saving in code is not supported.");

                writer.WriteLine("{");
                saveInSourceCode.SaveAsCode(writer, schema);
                writer.WriteLine("outputs[{0}] = output;", i);
                writer.WriteLine("}");
            }
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            var preds = _impl.Predictors;

            for (int i = 0; i < preds.Length; i++)
            {
                var saveInText = preds[i] as ICanSaveInTextFormat;
                Host.Check(saveInText != null, "Saving in text is not supported.");

                writer.WriteLine("#region: class-{0} classifier", i);
                saveInText.SaveAsText(writer, schema);

                writer.WriteLine("#endregion: class-{0} classifier", i);
                writer.WriteLine();
            }
        }

        protected interface IImplBase
        {
            ColumnType InputType { get; }
            ColumnType LabelType { get; }
            ColumnType OutputType { get; }
#if IMPLIValueMapperDist
            ColumnType DistType { get; }
#endif
            IValueMapper[] ValueMappers { get; }
            IPredictor[] Predictors { get; }
            IPredictor ReclassificationPredictor { get; }
            ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper();
#if IMPLIValueMapperDist
            ValueMapper<VBuffer<float>, VBuffer<float>, VBuffer<float>> GetMapperDist();
#endif
            void SaveCore(ModelSaveContext ctx, IHost host, VersionInfo versionInfo);
            ITLabel[] GetClasses<ITLabel>();
            int GetNbClasses();
            string GetStringClasses(string sep = ",");
            bool SingleColumn { get; }
            int MaxClassIndex();
        }

        protected class ImplRaw<TLabel> : IImplBase
        {
            ColumnType _inputType;
            ColumnType _labelType;
            ColumnType _outputType;
            VBuffer<TLabel> _classes;
            IValueMapper[] _mappers;
            IPredictor[] _predictors;
            IPredictor _reclassificationPredictor;
            Func<TLabel, float> _labelConverter;
            int[] _dstIndices;
            bool _singleColumn;
            bool _labelKey;

            public ColumnType InputType { get { return _inputType; } }
            public ColumnType LabelType { get { return _labelType; } }
            public ColumnType OutputType { get { return _outputType; } }
            public IValueMapper[] ValueMappers { get { return _mappers; } }
            public IPredictor[] Predictors { get { return _predictors; } }
            public IPredictor ReclassificationPredictor { get { return _reclassificationPredictor; } }
            public VBuffer<TLabel> Classes { get { return _classes; } }
            public bool SingleColumn { get { return _singleColumn; } }
            public int MaxClassIndex() { return _dstIndices == null ? _classes.Count - 1 : _dstIndices.Max(); }

#if IMPLIValueMapperDist
            ColumnType _distType;
            public ColumnType DistType { get { return _distType; } }
#endif

            public int GetNbClasses() { return _classes.Length; }
            public string GetStringClasses(string sep = ",") { return string.Join(",", _classes.DenseValues()); }

            public ITLabel[] GetClasses<ITLabel>()
            {
                var res = Classes as ITLabel[];
                Contracts.Assert(res != null);
                return res;
            }

            bool IsValid(IValueMapper mapper, ref ColumnType inputType)
            {
                // inputType
                if (mapper == null)
                    return false;
                if (mapper.OutputType != NumberType.Float)
                    return false;
                if (!mapper.InputType.IsVector() || mapper.InputType.ItemType() != NumberType.Float)
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

            public void SaveCore(ModelSaveContext ctx, IHost host, VersionInfo versionInfo)
            {
                host.Check(Classes.Count > 0, "The model cannot be saved, it was never trained.");
                host.Check(Classes.Count == Classes.Length, "The model cannot be saved, it was never trained.");
                ctx.SetVersionInfo(versionInfo);
                ctx.Writer.WriteIntArray(Classes.Indices);
                if (LabelType == NumberType.R4)
                    ctx.Writer.WriteSingleArray(Classes.Values as float[]);
                else if (LabelType == NumberType.U1)
                    ctx.Writer.WriteByteArray(Classes.Values as byte[]);
                else if (LabelType == NumberType.U2)
                    ctx.Writer.WriteUIntArray((Classes.Values as ushort[]).Select(c => (uint)c).ToArray());
                else if (LabelType == NumberType.U4)
                    ctx.Writer.WriteUIntArray(Classes.Values as uint[]);
                else
                    throw host.Except("Unexpected type for LabelType.");

                ctx.Writer.Write(_singleColumn ? 1 : 0);
                ctx.Writer.Write(_labelKey ? 1 : 0);
                var preds = Predictors;
                ctx.Writer.Write(preds.Length);
                for (int i = 0; i < preds.Length; i++)
                    ctx.SaveModel(preds[i], string.Format("M2B{0}", i));
                ctx.Writer.Write(_reclassificationPredictor != null ? (byte)1 : (byte)0);
                if (_reclassificationPredictor != null)
                    ctx.SaveModel(_reclassificationPredictor, "Reclassification");
                ctx.Writer.Write((byte)213);
            }

            internal ImplRaw(ModelLoadContext ctx, IHostEnvironment env)
            {
                // labelType
                GuessLabelType();
                int[] indices = ctx.Reader.ReadIntArray();

                TLabel[] classes;
                if (LabelType == NumberType.R4)
                {
                    classes = ctx.Reader.ReadFloatArray() as TLabel[];
                    env.CheckValue(classes, "classes");
                }
                else if (LabelType == NumberType.U1)
                {
                    classes = ctx.Reader.ReadByteArray() as TLabel[];
                    env.CheckValue(classes, "classes");
                }
                else if (LabelType == NumberType.U2)
                {
                    var val = ctx.Reader.ReadUIntArray();
                    env.CheckValue(val, "classes");
                    classes = val.Select(c => (ushort)c).ToArray() as TLabel[];
                }
                else if (LabelType == NumberType.U4)
                {
                    var val = ctx.Reader.ReadUIntArray();
                    env.CheckValue(val, "classes");
                    classes = val as TLabel[];
                }
                else
                    throw env.Except("Unexpected type for LabelType.");

                _classes = new VBuffer<TLabel>(classes.Length, classes, indices);
                _singleColumn = ctx.Reader.ReadInt32() == 1;
                _labelKey = ctx.Reader.ReadInt32() == 1;
                FinalizeOutputType();

                int len = ctx.Reader.ReadInt32();
                env.CheckDecode(len > 0);
                var predictors = new TScalarPredictor[len];
                IPredictor reclassPredictor;
                LoadPredictors(env, predictors, out reclassPredictor, ctx);
                Preparation(predictors, reclassPredictor);
                var checkCode = ctx.Reader.ReadByte();
                if (checkCode != 213)
                    throw Contracts.Except("CheckCode is wrong. Serialization failed.");
            }

            internal Func<float, float> GetFuncFloat() { return x => x; }
            internal Func<byte, float> GetFuncByte() { return x => (float)x; }
            internal Func<ushort, float> GetFuncUShort() { return x => (float)x; }
            internal Func<uint, float> GetFuncUInt() { return x => (float)x; }
            internal Func<float, int> GetFuncFloat2Int() { return x => (int)x; }

            internal ImplRaw(VBuffer<TLabel> classes, TScalarPredictor[] predictors,
                                IPredictor reclassPredictor, bool singleColumn, bool labelKey)
            {
                Contracts.Check(Utils.Size(predictors) > 0);
                _singleColumn = singleColumn;
                _labelKey = labelKey;
                _classes = classes;
                Preparation(predictors, reclassPredictor);
                FinalizeOutputType();
            }

            void GuessLabelType()
            {
                var tlabels = new TLabel[0];
                if ((tlabels as float[]) != null)
                {
                    _labelType = NumberType.R4;
                    _labelConverter = GetFuncFloat() as Func<TLabel, float>;
                    var func = GetFuncFloat2Int() as Func<TLabel, int>;
                }
                else if ((tlabels as byte[]) != null)
                {
                    _labelType = NumberType.U1;
                    _labelConverter = GetFuncByte() as Func<TLabel, float>;
                }
                else if ((tlabels as ushort[]) != null)
                {
                    _labelType = NumberType.U2;
                    _labelConverter = GetFuncUShort() as Func<TLabel, float>;
                }
                else if ((tlabels as uint[]) != null)
                {
                    _labelType = NumberType.U4;
                    _labelConverter = GetFuncUInt() as Func<TLabel, float>;
                }
                else
                    Contracts.Assert(false, "Type not supported.");
            }

            void FinalizeOutputType()
            {
                var tlabels = new TLabel[0];
                if ((tlabels as float[]) != null)
                {
                    var func = GetFuncFloat2Int() as Func<TLabel, int>;
                    _dstIndices = _classes.Values.Select(c => func(c)).ToArray();
                    // outputType
                    Contracts.Assert(_classes.Count > 0);
                    _outputType = _outputType = new VectorType(NumberType.Float, _dstIndices.Max() + 1);
#if IMPLIValueMapperDist
                    _distType = _outputType;
#endif
                }
                else
                {
                    _dstIndices = null;
                    // outputType
                    Contracts.Assert(_classes.Count > 0);
                    _outputType = _outputType = new VectorType(NumberType.Float, _classes.Length);
#if IMPLIValueMapperDist
                    _distType = _outputType;
#endif
                }
            }

            void Preparation(TScalarPredictor[] predictors, IPredictor reclassPredictor)
            {
                // labelType
                GuessLabelType();

                Contracts.Assert(predictors != null);
                Contracts.Assert(predictors.Length == 1);
                _mappers = new IValueMapper[predictors.Length];
                _predictors = new IPredictor[predictors.Length];
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i] as IValueMapper;
                    bool r = IsValid(vm, ref _inputType);
                    Contracts.Check(r, "Predictor doesn't implement the expected interface");
                    _mappers[i] = vm;
                    _predictors[i] = predictors[i];
                }
                _reclassificationPredictor = reclassPredictor;
            }

            /// <summary>
            /// This function must be used if the function returns probabilities.
            /// </summary>
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

            #region IValueMapper

            public virtual ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper()
            {
                Contracts.Assert(_mappers != null);
                Contracts.Assert(_mappers.Length == 1);
                var mapper = _mappers[0].GetMapper<VBuffer<float>, float>();
                VBuffer<float> inputs = new VBuffer<float>();
                float[] labelClasses = _classes.Values.Select(c => _labelConverter(c)).ToArray();
                int maxClass = _dstIndices == null ? _classes.Length : _dstIndices.Max() + 1;
                bool labelKey = _labelKey;
                IValueMapper imapperFinal = _reclassificationPredictor == null ? null : _reclassificationPredictor as IValueMapper;
                var mapperFinal = _reclassificationPredictor == null ? null : imapperFinal.GetMapper<VBuffer<float>, VBuffer<float>>();
                var tempOut = new VBuffer<float>();

                if (_singleColumn)
                {
                    if (labelKey)
                    {
                        for (int i = 0; i < labelClasses.Length; ++i)
                            --labelClasses[i];
                    }

                    // Only one new column added which contain the label.
                    return
                        (in VBuffer<float> src, ref VBuffer<float> dst) =>
                        {
                            if (dst.Count != maxClass)
                                dst = new VBuffer<float>(maxClass, _classes.Length, new float[_classes.Length], _dstIndices);

                            var values = inputs.Values;
                            var indices = inputs.Indices;

                            if (src.IsDense)
                            {
                                if (values == null || values.Length <= src.Count)
                                    values = new float[src.Length + 1];
                                Array.Copy(src.Values, values, src.Length);
                                Contracts.Assert(src.Indices == null);
                                inputs = new VBuffer<float>(src.Length + 1, values);
                            }
                            else
                            {
                                if (src.Indices == null)
                                {
                                    if (src.Values == null)
                                    {
                                        // All missing.
                                        for (int i = 0; i < _classes.Length; ++i)
                                            dst.Values[i] = 0;
                                        return;
                                    }
                                    else
                                        throw Contracts.Except("Inconsistency in input vector. Sparse vector with null indices.");
                                }
                                int nb = src.Count + 1;
                                if (values == null || values.Length < nb)
                                    values = new float[nb];
                                if (indices == null || indices.Length < nb)
                                    indices = new int[nb];
                                Array.Copy(src.Values, values, src.Count);
                                Array.Copy(src.Indices, indices, src.Count);
                                indices[src.Count] = src.Length;
                                inputs = new VBuffer<float>(src.Length + 1, src.Count + 1, values, indices);
                            }

                            for (int i = 0; i < _classes.Length; ++i)
                            {
                                inputs.Values[src.Count] = labelClasses[i];
                                mapper(in inputs, ref dst.Values[i]);
                            }

                            // Only if probabilities
                            // Normalize(dst.Values, _classes.Length);
                            #region debug
#if (DEBUG)
                            var dense = dst.DenseValues().ToArray();
                            if (dense.Length != maxClass)
                                throw Contracts.Except("Different dimension {0} != {1}-{2}", maxClass, dst.Length, dst.Count);
#endif
                            #endregion

                            if (mapperFinal != null)
                            {
                                mapperFinal(in dst, ref tempOut);
                                dst = tempOut;
                            }
                        };
                }
                else
                {
                    if (labelKey)
                    {
                        for (int i = 0; i < labelClasses.Length; ++i)
                            --labelClasses[i];
                    }

                    // One column was added for each class.
                    return
                        (in VBuffer<float> src, ref VBuffer<float> dst) =>
                        {
                            if (dst.Count != maxClass)
                                dst = new VBuffer<float>(maxClass, _classes.Length, new float[_classes.Length], _dstIndices);

                            var values = inputs.Values;
                            var indices = inputs.Indices;

                            if (src.IsDense)
                            {
                                if (values == null || values.Length < src.Count + maxClass)
                                    values = new float[src.Length + maxClass];
                                Array.Copy(src.Values, values, src.Length);
                                Contracts.Assert(src.Indices == null);
                                for (int i = src.Length; i < inputs.Length; ++i)
                                    values[i] = 0;
                                inputs = new VBuffer<float>(src.Length + maxClass, values);

                                int k;
                                for (int i = 0; i < _classes.Length; ++i)
                                {
                                    k = (int)labelClasses[i] + src.Count;
                                    inputs.Values[k] = 1;
                                    mapper(in inputs, ref dst.Values[i]);
                                    inputs.Values[k] = 0;
                                }
                            }
                            else
                            {
                                if (src.Indices == null)
                                {
                                    if (src.Values == null)
                                    {
                                        // All missing.
                                        for (int i = 0; i < _classes.Length; ++i)
                                            dst.Values[i] = 0;
                                        return;
                                    }
                                    else
                                        throw Contracts.Except("Inconsistency in input vector. Sparse vector with null indices.");
                                }
                                int nb = src.Count + 1;
                                if (values == null || values.Length < nb)
                                    values = new float[nb];
                                if (indices == null || indices.Length < nb)
                                    indices = new int[nb];
                                Array.Copy(src.Values, values, src.Count);
                                Array.Copy(src.Indices, indices, src.Count);
                                indices[src.Count] = src.Length;
                                inputs = new VBuffer<float>(src.Length + maxClass, nb, values, indices);

                                inputs.Values[src.Count] = 1;
                                for (int i = 0; i < _classes.Length; ++i)
                                {
                                    Contracts.Assert(inputs.Count == nb, "inputs.Count");
                                    inputs.Indices[src.Count] = (int)labelClasses[i] + src.Length;
                                    mapper(in inputs, ref dst.Values[i]);
                                    // If the predictor is called within a LambdaColumnTransform,
                                    // the context is not very well preserved (bug?).
                                    // This second test ensures it passes.
                                    Contracts.Assert(inputs.Count == nb, "inputs.Count");
                                }
                            }

                            // Only if probabiliies
                            // Normalize(dst.Values, _classes.Length);

#if (DEBUG)
                            var dense = dst.DenseValues().ToArray();
                            if (dense.Length != maxClass)
                                throw Contracts.Except("Different dimension {0} != {1}-{2}", maxClass, dst.Length, dst.Count);
#endif

                            if (mapperFinal != null)
                            {
                                mapperFinal(in dst, ref tempOut);
                                dst = tempOut;
                            }
                        };
                }
            }

            #endregion

            #region IValueMapperDist
#if IMPLIValueMapperDist

            public virtual ValueMapper<VBuffer<float>, VBuffer<float>, VBuffer<float>> GetMapperDist()
            {
                if (_reclassificationPredictor == null)
                    return GetMapperDistNoReclass();
                else
                {
                    var mapper = GetMapper();
                    IValueMapperDist imapperFinal = _reclassificationPredictor == null ? null : _reclassificationPredictor as IValueMapperDist;
                    var mapperFinal = _reclassificationPredictor == null ? null : imapperFinal.GetMapper<VBuffer<float>, VBuffer<float>, VBuffer<float>>();
                    var tempOut = new VBuffer<float>();
                    return (ref VBuffer<float> src, ref VBuffer<float> dst, ref VBuffer<float> prob) =>
                    {
                        mapper(ref src, ref dst);
                        mapperFinal(ref dst, ref tempOut, ref prob);
                    };
                }
            }

            ValueMapper<VBuffer<float>, VBuffer<float>, VBuffer<float>> GetMapperDistNoReclass()
            {
                Contracts.Assert(_mappers != null);
                Contracts.Assert(_mappers.Length == 1);
                var mapperDist = _mappers[0] as IValueMapperDist;
                if (mapperDist == null)
                    throw Contracts.Except("The predictor does not output probabilities.");
                var mapper = mapperDist.GetMapper<VBuffer<float>, float, float>();
                VBuffer<float> inputs = new VBuffer<float>();
                float[] labelClasses = _classes.Values.Select(c => _labelConverter(c)).ToArray();
                int maxClass = _dstIndices == null ? _classes.Length : _dstIndices.Max() + 1;
                bool labelKey = _labelKey;

                if (_singleColumn)
                {
                    if (labelKey)
                    {
                        for (int i = 0; i < labelClasses.Length; ++i)
                            --labelClasses[i];
                    }

                    // Only one new column added which contain the label.
                    return
                        (ref VBuffer<float> src, ref VBuffer<float> dst, ref VBuffer<float> prob) =>
                        {
                            if (dst.Count != maxClass)
                                dst = new VBuffer<float>(maxClass, _classes.Length, new float[_classes.Length], _dstIndices);
                            if (prob.Count != maxClass)
                                prob = new VBuffer<float>(maxClass, _classes.Length, new float[_classes.Length], _dstIndices);

                            var values = inputs.Values;
                            var indices = inputs.Indices;

                            if (src.IsDense)
                            {
                                if (values == null || values.Length <= src.Count)
                                    values = new float[src.Length + 1];
                                Array.Copy(src.Values, values, src.Length);
                                Contracts.Assert(src.Indices == null);
                                inputs = new VBuffer<float>(src.Length + 1, values);
                            }
                            else
                            {
                                if (src.Indices == null)
                                {
                                    if (src.Values == null)
                                    {
                                        // All missing.
                                        for (int i = 0; i < _classes.Length; ++i)
                                        {
                                            dst.Values[i] = 0;
                                            prob.Values[i] = 0;
                                        }
                                        return;
                                    }
                                    else
                                        throw Contracts.Except("Inconsistency in input vector. Sparse vector with null indices.");
                                }
                                int nb = src.Count + 1;
                                if (values == null || values.Length < nb)
                                    values = new float[nb];
                                if (indices == null || indices.Length < nb)
                                    indices = new int[nb];
                                Array.Copy(src.Values, values, src.Count);
                                Array.Copy(src.Indices, indices, src.Count);
                                indices[src.Count] = src.Length;
                                inputs = new VBuffer<float>(src.Length + 1, src.Count + 1, values, indices);
                            }

                            for (int i = 0; i < _classes.Length; ++i)
                            {
                                inputs.Values[src.Count] = labelClasses[i];
                                mapper(ref inputs, ref dst.Values[i], ref prob.Values[i]);
                            }

                            Normalize(prob.Values, _classes.Length);

            #region debug
#if (DEBUG)
                            var dense = dst.DenseValues().ToArray();
                            if (dense.Length != maxClass)
                                throw Contracts.Except("Different dimension {0} != {1}-{2}", maxClass, dst.Length, dst.Count);
#endif
            #endregion
                        };
                }
                else
                {
                    if (labelKey)
                    {
                        for (int i = 0; i < labelClasses.Length; ++i)
                            --labelClasses[i];
                    }

                    // One column was added for each class.
                    return
                        (ref VBuffer<float> src, ref VBuffer<float> dst, ref VBuffer<float> prob) =>
                        {
                            if (dst.Count != maxClass)
                                dst = new VBuffer<float>(maxClass, _classes.Length, new float[_classes.Length], _dstIndices);
                            if (prob.Count != maxClass)
                                prob = new VBuffer<float>(maxClass, _classes.Length, new float[_classes.Length], _dstIndices);

                            var values = inputs.Values;
                            var indices = inputs.Indices;

                            if (src.IsDense)
                            {
                                if (values == null || values.Length < src.Count + maxClass)
                                    values = new float[src.Length + maxClass];
                                Array.Copy(src.Values, values, src.Length);
                                Contracts.Assert(src.Indices == null);
                                for (int i = src.Length; i < inputs.Length; ++i)
                                    values[i] = 0;
                                inputs = new VBuffer<float>(src.Length + maxClass, values);

                                int k;
                                for (int i = 0; i < _classes.Length; ++i)
                                {
                                    k = (int)labelClasses[i] + src.Count;
                                    inputs.Values[k] = 1;
                                    mapper(ref inputs, ref dst.Values[i], ref prob.Values[i]);
                                    inputs.Values[k] = 0;
                                }
                            }
                            else
                            {
                                if (src.Indices == null)
                                {
                                    if (src.Values == null)
                                    {
                                        // All missing.
                                        for (int i = 0; i < _classes.Length; ++i)
                                        {
                                            dst.Values[i] = 0;
                                            prob.Values[i] = 0;
                                        }
                                        return;
                                    }
                                    else
                                        throw Contracts.Except("Inconsistency in input vector. Sparse vector with null indices.");
                                }
                                int nb = src.Count + 1;
                                if (values == null || values.Length < nb)
                                    values = new float[nb];
                                if (indices == null || indices.Length < nb)
                                    indices = new int[nb];
                                Array.Copy(src.Values, values, src.Count);
                                Array.Copy(src.Indices, indices, src.Count);
                                indices[src.Count] = src.Length;
                                inputs = new VBuffer<float>(src.Length + maxClass, src.Count + 1, values, indices);

                                inputs.Values[src.Count] = 1;
                                for (int i = 0; i < _classes.Length; ++i)
                                {
                                    inputs.Indices[src.Count] = (int)labelClasses[i] + src.Length;
                                    mapper(ref inputs, ref dst.Values[i], ref prob.Values[i]);
                                }
                            }

                            Normalize(prob.Values, _classes.Length);

#if (DEBUG)
                            var dense = dst.DenseValues().ToArray();
                            if (dense.Length != maxClass)
                                throw Contracts.Except("Different dimension {0} != {1}-{2}", maxClass, dst.Length, dst.Count);
#endif
                        };
                }
            }
#endif
            #endregion
        }
    }
}
