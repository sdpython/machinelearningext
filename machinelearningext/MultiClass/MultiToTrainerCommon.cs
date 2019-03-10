// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.PipelineLambdaTransforms;


namespace Scikit.ML.MultiClass
{
    using TScalarTrainer = ITrainer<IPredictor>; // ITrainer<IPredictorProducing<float>>;
    using TVectorPredictor = IPredictor; //IPredictorProducing<VBuffer<float>>;

    /// <summary>
    /// Train a MultiToBinary predictor. It multiplies the rows by the number of classes to predict.
    /// (multi class problem).
    /// </summary>
    public abstract class MultiToTrainerCommon : TrainerBase<TVectorPredictor>
    {
        #region parameters / command line

        public class Arguments
        {
            #region parameters for m2b

            [Argument(ArgumentType.AtMostOnce, HelpText = "Algorithm to to duplicate rows.", ShortName = "al")]
            public MultiToBinaryTransform.MultiplicationAlgorithm algo = MultiToBinaryTransform.MultiplicationAlgorithm.Default;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weight column", ShortName = "w")]
            public string weight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of time an example can be multipled", ShortName = "m")]
            public float maxMulti = 5f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed to multiply randomly the label.", ShortName = "s")]
            public int seed = 42;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads used to estimate how much a class should resample.", ShortName = "nt")]
            public int? numThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Add one column for the label or one column per class.", ShortName = "sc")]
            public bool singleColumn = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Drop missing labels.", ShortName = "na")]
            public bool dropNALabel = true;

            [Argument(ArgumentType.Multiple, HelpText = "Reclassification using output from the first tree", ShortName = "rp",
                SortOrder = 1, SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<ITrainer> reclassicationPredictor = null;

            #endregion
        }

        #endregion

        #region internal members / accessors

        protected TVectorPredictor[] _predictors;
        protected IPredictor _reclassPredictor;
        protected TScalarTrainer _trainer;

        public override TrainerInfo Info
        {
            get
            {
                return new TrainerInfo(normalization: false, calibration: false, caching: false,
                                       supportValid: false, supportIncrementalTrain: false);
            }
        }

        #endregion

        #region public constructor / serialization / load / save

        /// <summary>
        /// Create a MultiToTrainerCommon transform.
        /// </summary>
        public MultiToTrainerCommon(IHostEnvironment env, Arguments args, string loaderSignature)
            : base(env, loaderSignature)
        {
            Host.CheckValue(args, "args");
        }

        private DataViewSchema.Column _dc(int i)
        {
            return new DataViewSchema.Column(null, i, false, null, null);
        }

        protected Tuple<TLabel, TLabel> MinMaxLabel<TLabel>(MultiToBinaryTransform tr, int index)
            where TLabel : IComparable<TLabel>
        {
            using (var cursor = tr.GetRowCursor(tr.Schema.Where(c => c.Index == index).ToArray()))
            {
                var getter = cursor.GetGetter<TLabel>(_dc(index));
                TLabel cl = default(TLabel), max = default(TLabel), min = default(TLabel);
                bool first = true;
                while (cursor.MoveNext())
                {
                    getter(ref cl);
                    if (first || cl.CompareTo(max) == 1)
                        max = cl;
                    if (first || cl.CompareTo(max) == -1)
                        min = cl;
                    first = false;
                }
                return new Tuple<TLabel, TLabel>(min, max);
            }
        }

        /// <summary>
        /// The function walk through the data to compute the highest label.
        /// </summary>
        protected Tuple<int, int> MinMaxLabelOverDataSet(MultiToBinaryTransform tr, string label, out int nb)
        {
            int index = SchemaHelper.GetColumnIndex(tr.Schema, label);
            var ty = tr.Schema[index].Type;
            switch (ty.RawKind())
            {
                case DataKind.Single:
                    // float is 0 based
                    var tf = MinMaxLabel<float>(tr, index);
                    nb = (int)tf.Item2 + 1;
                    return new Tuple<int, int>((int)tf.Item1, (int)tf.Item2);
                case DataKind.SByte:
                    // key is 1 based
                    var tb = MinMaxLabel<byte>(tr, index);
                    nb = tb.Item2;
                    return new Tuple<int, int>(1, (int)tb.Item2);
                case DataKind.UInt16:
                    // key is 1 based
                    var ts = MinMaxLabel<ushort>(tr, index);
                    nb = ts.Item2;
                    return new Tuple<int, int>(1, (int)ts.Item2);
                case DataKind.UInt32:
                    // key is 1 based
                    var tu = MinMaxLabel<uint>(tr, index);
                    nb = (int)tu.Item2;
                    return new Tuple<int, int>(1, (int)tu.Item2);
                default:
                    throw Contracts.ExceptNotImpl("Type '{0}' not implemented", ty.RawKind());
            }
        }

        /// <summary>
        /// Train the embedded predictor.
        /// </summary>
        protected abstract TVectorPredictor TrainPredictor(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int count);

        /// <summary>
        /// Returns the trained predictor.
        /// </summary>
        protected abstract TVectorPredictor CreatePredictor();

        #region debugging
#if (DEBUG)

        protected void DebugChecking0Vfloat(IDataView viewI, string labName, ulong count)
        {
            var index = SchemaHelper.GetColumnIndexDC(viewI.Schema, labName);
            var ty = viewI.Schema[index.Index].Type;
            Contracts.Assert(ty.IsKey() || ty.IsVector() || ty.RawKind() == DataKind.Single);
            using (var cursor = viewI.GetRowCursor(viewI.Schema.Where(i => i.Index == index.Index).ToArray()))
            {
                var getter = cursor.GetGetter<VBuffer<float>>(index);
                var value = new VBuffer<float>();
                int nb = 0;
                while (cursor.MoveNext())
                {
                    getter(ref value);
                    if (value.Length == 0 || value.Count == 0)
                        throw Host.Except("Issue.");
                    if ((ulong)value.Length > count || (ulong)value.Count > count)
                        throw Host.Except("Issue.");
                    if ((ulong)value.Length != count || value.Count != 1)
                    {
                        getter(ref value);
                        throw Host.Except("Issue.");
                    }
                    if (value.Values[0] != 1)
                        throw Host.Except("Issue.");
                    if ((ulong)value.Indices[0] >= count)
                    {
                        getter(ref value);
                        throw Host.Except("Issue.");
                    }
                    if (value.Indices[0] < 0)
                        throw Host.Except("Issue.");
                    ++nb;
                }
                if (nb < 10)
                    throw Host.Except("Issue.");
            }
        }

        protected void DebugChecking0(IDataView viewI, string labName, bool oneO)
        {
            var index = SchemaHelper.GetColumnIndexDC(viewI.Schema, labName);
            int nbRows = 0;
            using (var cursor = viewI.GetRowCursor(viewI.Schema.Where(c => c.Index == index.Index).ToArray()))
            {
                if (oneO)
                {
                    var gfu = cursor.GetGetter<float>(index);
                    var gff = cursor.GetGetter<uint>(index);
                    Contracts.Assert(gfu != null || gff != null);
                }

                var ty = viewI.Schema[index.Index].Type;
                if (ty.IsVector() && ty.AsVector().ItemType().RawKind() == DataKind.Single)
                {
                    var getter = cursor.GetGetter<VBuffer<float>>(index);
                    var value = new VBuffer<float>();
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        if (value.Length == 0 || value.Count == 0)
                            throw Host.Except("Issue.");
                        ++nbRows;
                    }
                }
                else if (!ty.IsVector() && ty.RawKind() == DataKind.Single)
                {
                    var getter = cursor.GetGetter<float>(index);
                    var sch = SchemaHelper.ToString(cursor.Schema);
                    var value = 0f;
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        ++nbRows;
                    }
                }
                else if (ty.IsKey() && ty.RawKind() == DataKind.UInt32)
                {
                    var getter = cursor.GetGetter<uint>(index);
                    uint value = 0;
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        ++nbRows;
                    }
                }
                else
                    throw Host.ExceptNotSupp();
            }
            if (nbRows == 0)
                throw Contracts.Except("View is empty.");
        }

#if DISABLED
        protected void DebugChecking1(IDataView viewI, IDataView choose, RoleMappedData data,
                                      RoleMappedData td, int count, bool singleColumn)
        {
            var cache1 = new MemoryDataView(Host, viewI, numThreads: 1);
            var cache2 = new MemoryDataView(Host, choose, numThreads: 1);
            var t1 = data.Schema.Feature.Type.AsVector();
            var t2 = td.Schema.Feature.Type.AsVector();
            if (t1.DimCount()() != 1)
                throw Host.Except("Expect only 1 dimension.");
            if (t2.DimCount()() != 1)
                throw Host.Except("Expect only 1 dimension.");
            if (singleColumn && t1.GetDim(0) != t2.GetDim(0) - 1)
                throw Host.Except("Different dimension {0} != {1}-1", t1.GetDim(0), t2.GetDim(0));
            if (!singleColumn && t1.GetDim(0) >= t2.GetDim(0) - 1)
                throw Host.Except("Different dimension {0} != {1}-1", t1.GetDim(0), t2.GetDim(0));
            var nb1 = DataViewUtils.ComputeRowCount(cache1);
            if (nb1 == 0)
                throw Host.Except("empty view");
            var nb2 = DataViewUtils.ComputeRowCount(cache2);
            if (nb2 == 0)
                throw Host.Except("empty view");
            if (!singleColumn)
            {
                using (var cursor = cache2.GetRowCursor(i => true))
                {
                    string sch_ = SchemaHelper.ToString(cursor.Schema);
                    int index;
                    if (!cursor.Schema.TryGetColumnIndex(data.Schema.Label.Name, out index))
                        throw Host.Except("Unable to find '{0}' in\n{1}", data.Schema.Label.Name, sch_);
                    var getter = cursor.GetGetter<VBuffer<float>>(index);
                    var buf = new VBuffer<float>();
                    while (cursor.MoveNext())
                    {
                        getter(ref buf);
                        if (buf.Count > count || buf.Length > count)
                            throw Contracts.Except("Mismath");
                    }
                }
            }
        }

        protected void DebugChecking2(RoleMappedData td, ITrainer trainer)
        {
            var scorer = PredictorHelper.CreateDefaultScorer(Host, td, trainer.CreatePredictor());
            if (trainer.PredictionKind == PredictionKind.Ranking)
            {
                string schemas = SchemaHelper.ToString(scorer.Schema);
                if (!schemas.Contains("Score"))
                    throw Host.Except("Issue with the schema: {0}", schemas);
            }

            using (var cursor = scorer.GetRowCursor(i => true))
            {
                int ilab, ipred, ifeat;
                cursor.Schema.TryGetColumnIndex(td.Schema.Label.Name, out ilab);
                if (trainer.PredictionKind == PredictionKind.Ranking)
                {
                    cursor.Schema.TryGetColumnIndex("Score", out ipred);
                    cursor.Schema.TryGetColumnIndex(td.Schema.Feature.Name, out ifeat);
                    var getter = cursor.GetGetter<uint>(ilab);
                    var fgetter = cursor.GetGetter<VBuffer<float>>(ifeat);
                    var pgetter = cursor.GetGetter<float>(ipred);
                    if (pgetter == null)
                        throw Host.Except("Issue with the schema: {0}", SchemaHelper.ToString(cursor.Schema));
                    uint lab = 0;
                    var counts = new Dictionary<uint, int>();
                    var counts_pred = new Dictionary<float, int>();
                    float pre = 0;
                    VBuffer<float> features = default(VBuffer<float>);
                    int nbrows = 0;
                    int err = 0;
                    while (cursor.MoveNext())
                    {
                        getter(ref lab);
                        pgetter(ref pre);
                        fgetter(ref features);
                        counts[lab] = counts.ContainsKey(lab) ? counts[lab] + 1 : 0;
                        counts_pred[pre] = counts_pred.ContainsKey(pre) ? counts_pred[pre] + 1 : 0;
                        if (trainer.PredictionKind == PredictionKind.Ranking)
                        {
                            var elab = features.Values[features.Count - 1];
                            if (pre > 0 && lab < 0)
                                ++err;
                        }
                        else if (!lab.Equals(pre))
                            ++err;
                        ++nbrows;
                    }
                    if (nbrows == 0)
                        throw Host.Except("No results.");
                    if (err * 2 > nbrows)
                        throw Host.Except("No training.");
                }
                else
                {
                    cursor.Schema.TryGetColumnIndex("PredictedLabel", out ipred);
                    var getter = cursor.GetGetter<bool>(ilab);
                    var pgetter = cursor.GetGetter<bool>(ipred);
                    var counts = new Dictionary<bool, int>();
                    var counts_pred = new Dictionary<bool, int>();
                    bool lab = false;
                    bool pre = false;
                    int nbrows = 0;
                    int err = 0;
                    while (cursor.MoveNext())
                    {
                        getter(ref lab);
                        pgetter(ref pre);
                        counts[lab] = counts.ContainsKey(lab) ? counts[lab] + 1 : 0;
                        counts_pred[pre] = counts_pred.ContainsKey(pre) ? counts_pred[pre] + 1 : 0;
                        if (!lab.Equals(pre))
                            ++err;
                        ++nbrows;
                    }
                    if (nbrows == 0)
                        throw Host.Except("No results.");
                    if (err * 2 > nbrows)
                        throw Host.Except("No training.");
                }
            }
        }

#endif

#endif
        #endregion

        protected IDataView FilterNA(IDataView view, string label, bool dropNALabel)
        {
            if (dropNALabel)
            {
                var args = new NAFilter.Arguments { Columns = new[] { label } };
                return new NAFilter(Host, args, view);
            }
            else
                return view;
        }

        /// <summary>
        /// We insert a MultiToBinaryTransform in the pipeline.
        /// </summary>
        /// <param name="data">input data</param>
        /// <param name="dstName">name of the new label</param>
        /// <param name="ch">channel</param>
        /// <param name="labName">label name as a single column</param>
        /// <param name="count">indication of the number of expected classes</param>
        /// <param name="train">pipeline for training or not</param>
        /// <param name="args">arguments send to the trainer</param>
        /// <returns>created view</returns>
        protected MultiToBinaryTransform MapLabelsAndInsertTransform(IChannel ch, RoleMappedData data,
                                out string dstName, out string labName, int count, bool train, Arguments args)
        {
            var lab = data.Schema.Label.Value;
            Host.Assert(!data.Schema.Schema[lab.Index].IsHidden);
            Host.Assert(lab.Type.GetKeyCount() > 0 || lab.Type == NumberDataViewType.Single);

            IDataView source = data.Data;
            if (train)
            {
                source = FilterNA(source, lab.Name, args.dropNALabel);
                if (lab.Type.IsKey())
                {
                    var uargs = new ULabelToR4LabelTransform.Arguments
                    {
                        columns = new Column1x1[] { new Column1x1() { Source = lab.Name, Name = lab.Name } }
                    };
                    source = new ULabelToR4LabelTransform(Host, uargs, source);
                }
            }

            // Get the destination label column name.
            ch.Info("Multiplying rows label '{0}'", lab.Name);
            switch (args.algo)
            {
                case MultiToBinaryTransform.MultiplicationAlgorithm.Default:
                case MultiToBinaryTransform.MultiplicationAlgorithm.Reweight:
                    dstName = source.Schema.GetTempColumnName() + "BL";
                    break;
                case MultiToBinaryTransform.MultiplicationAlgorithm.Ranking:
                    dstName = source.Schema.GetTempColumnName() + "U4";
                    break;
                default:
                    throw Host.ExceptNotSupp("Not supported algorithm {0}", args.algo);
            }
            var args2 = new MultiToBinaryTransform.Arguments
            {
                label = lab.Name,
                newColumn = dstName,
                algo = args.algo,
                weight = args.weight,
                maxMulti = args.maxMulti,
                seed = args.seed,
                numThreads = args.numThreads,
            };

            labName = lab.Name;

            var tr = new MultiToBinaryTransform(Host, args2, source);
            return tr;
        }

        #endregion

        protected void TrainReclassificationPredictor(RoleMappedData trainer_input, IPredictor predictor,
                            ISubComponent<ITrainer> argsPred)
        {
            if (argsPred == null)
            {
                _reclassPredictor = null;
                return;
            }

            using (var ch = Host.Start("Training reclassification predictor"))
            {
                ch.Info("Training reclassification learner");
                Contracts.CheckUserArg(argsPred.IsGood(), "reclassificationPredictor", "Must specify a base learner type");
                var trainer = argsPred.CreateInstance(Host);

                var args = new PredictTransform.Arguments
                {
                    featureColumn = trainer_input.Schema.Feature.Value.Name,
                    outputColumn = DataViewUtils.GetTempColumnName(trainer_input.Data.Schema)
                };

                var data = new PredictTransform(Host, args, trainer_input.Data, predictor);
                var roles = new[] { RoleMappedSchema.ColumnRole.Feature.Bind(args.outputColumn),
                                     RoleMappedSchema.ColumnRole.Label.Bind(trainer_input.Schema.Label.Value.Name)};
                var data_roles = new RoleMappedData(data, roles);
                var trainerRoles = trainer as ITrainer<TVectorPredictor>;
                if (trainerRoles == null)
                    throw Host.Except("Trainer reclassification cannot be trained.");
                _reclassPredictor = trainerRoles.Train(data_roles);
            }
        }
    }
}
