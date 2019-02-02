// See the LICENSE file in the project root for more information.

using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using TagTrainOrScoreTransform = Scikit.ML.PipelineGraphTransforms.TagTrainOrScoreTransform;


[assembly: LoadableClass(TagTrainOrScoreTransform.Summary, typeof(TagTrainOrScoreTransform),
    typeof(TagTrainOrScoreTransform.Arguments), typeof(SignatureDataTransform),
    "Train and Tag and Score a Predictor", TagTrainOrScoreTransform.LoaderSignature, "TagTrainScore")]

[assembly: LoadableClass(TagTrainOrScoreTransform.Summary, typeof(TagTrainOrScoreTransform),
    null, typeof(SignatureLoadDataTransform),
    "Train and Tag and Score a Predictor", TagTrainOrScoreTransform.LoaderSignature, "TagTrainScore")]


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Train or / and score a predictor hosted by a tagged view.
    /// The transform can also saves the predictor.
    /// </summary>
    public class TagTrainOrScoreTransform : AbstractSimpleTransformTemplate
    {
        public const string LoaderSignature = "TagTrainScoreTransform";
        internal const string Summary = "Trains a predictor, or gets it from a tagged view, and runs it on the data.";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TAGSTRRE",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TagTrainOrScoreTransform).Assembly.FullName);
        }

        public new class Arguments : TrainAndScoreTransformer.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Trainer", ShortName = "tr", SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<ITrainer> trainer = new ScikitSubComponent<ITrainer, SignatureTrainer>("PlattCalibration");

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>",
                SignatureType = typeof(SignatureCalibrator))]
            public IComponentFactory<ICalibratorTrainer> calibrator = new ScikitSubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int maxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>",
                SignatureType = typeof(SignatureDataScorer))]
            public IComponentFactory<IDataScorerTransform> scorer;

            [Argument(ArgumentType.Required, HelpText = "To tag the predictor if it is trained.", ShortName = "tag")]
            public string tag = "taggedPredictor";

            [Argument(ArgumentType.AtMostOnce, HelpText = "To save the trained model.", ShortName = "out")]
            public string outputModel = null;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                IOHelper.Write(ctx, CustomColumns);
                IOHelper.Write(ctx, FeatureColumn);
                IOHelper.Write(ctx, GroupColumn);
                IOHelper.Write(ctx, LabelColumn);
                IOHelper.Write(ctx, NameColumn);
                IOHelper.Write(ctx, WeightColumn);

                ctx.Writer.Write(maxCalibrationExamples);
                ctx.Writer.Write(tag);
                ctx.Writer.Write(string.IsNullOrEmpty(outputModel) ? string.Empty : outputModel);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                CustomColumns = IOHelper.ReadArrayKeyValuePairStringString(ctx);
                FeatureColumn = IOHelper.ReadString(ctx);
                GroupColumn = IOHelper.ReadString(ctx);
                LabelColumn = IOHelper.ReadString(ctx);
                NameColumn = IOHelper.ReadString(ctx);
                WeightColumn = IOHelper.ReadString(ctx);

                maxCalibrationExamples = ctx.Reader.ReadInt32();
                tag = ctx.Reader.ReadString();
                outputModel = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(outputModel))
                    outputModel = null;
            }
        }

        readonly Arguments _args;
        IDataScorerTransform _scorer;
        ICalibratorTrainer _cali;
        IPredictor _predictor;

        public TagTrainOrScoreTransform(IHostEnvironment env, Arguments args, IDataView input) :
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            _args = args;
            _cali = null;
            _scorer = null;
            _predictor = null;
            _sourcePipe = Create(_host, args, input, out _sourceCtx);
        }

        public override void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.Writer.Write(_predictor != null ? (byte)1 : (byte)0);
            ctx.Writer.Write(_cali != null ? (byte)1 : (byte)0);
            ctx.Writer.Write(_scorer != null ? (byte)1 : (byte)0);
            if (_predictor != null)
                ctx.SaveModel(_predictor, "predictor");
            if (_cali != null)
                ctx.SaveModel(_cali, "calibrator");
            if (_scorer != null)
                ctx.SaveModel(_scorer, "scorer");
        }

        public static TagTrainOrScoreTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new TagTrainOrScoreTransform(h, ctx, input));
        }

        public TagTrainOrScoreTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, ctx, input, LoaderSignature)
        {
            _args = new Arguments();
            _args.Read(ctx, _host);

            bool hasPredictor = ctx.Reader.ReadByte() == 1;
            bool hasCali = ctx.Reader.ReadByte() == 1;
            bool hasScorer = ctx.Reader.ReadByte() == 1;

            if (hasPredictor)
                ctx.LoadModel<IPredictor, SignatureLoadModel>(host, out _predictor, "predictor");
            else _predictor = null;

            using (var ch = _host.Start("TagTrainOrScoreTransform loading"))
            {
                var views = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == _args.tag);
                if (views.Any())
                    throw _host.Except("Tag '{0}' is already used.", _args.tag);

                var customCols = TrainUtils.CheckAndGenerateCustomColumns(_host, _args.CustomColumns);
                string feat;
                string group;
                var data = CreateDataFromArgs(_host, ch, new OpaqueDataView(input), _args, out feat, out group);

                if (hasCali)
                    ctx.LoadModel<ICalibratorTrainer, SignatureLoadModel>(host, out _cali, "calibrator", _predictor);
                else
                    _cali = null;

                if (_cali != null)
                    throw ch.ExceptNotImpl("Calibrator is not implemented yet.");

                if (hasScorer)
                    ctx.LoadModel<IDataScorerTransform, SignatureLoadDataTransform>(host, out _scorer, "scorer", data.Data);
                else
                    _scorer = null;

                ch.Info("Tagging with tag '{0}'.", _args.tag);
                var ar = new TagViewTransform.Arguments { tag = _args.tag };
                var res = new TagViewTransform(_host, ar, _scorer, _predictor);
                _sourcePipe = res;
            }
        }

        #region training (not delayed)

        IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input, out IDataView sourceCtx)
        {
            sourceCtx = input;
            Contracts.CheckValue(env, "env");
            env.CheckValue(args, "args");
            env.CheckValue(input, "input");
            env.CheckValue(args.tag, "tag is empty");
            env.CheckValue(args.trainer, "trainer",
                "Trainer cannot be null. If your model is already trained, please use ScoreTransform instead.");

            var views = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.tag);
            if (views.Any())
                throw env.Except("Tag '{0}' is already used.", args.tag);

            var host = env.Register("TagTrainOrScoreTransform");

            using (var ch = host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
                var trainerSett = ScikitSubComponent<ITrainer, SignatureTrainer>.AsSubComponent(args.trainer);
                ITrainer trainer = trainerSett.CreateInstance(host);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.CustomColumns);

                string feat;
                string group;
                var data = CreateDataFromArgs(_host, ch, new OpaqueDataView(input), args, out feat, out group);
                ICalibratorTrainer calibrator = args.calibrator == null
                                    ? null
                                    : ScikitSubComponent<ICalibratorTrainer, SignatureCalibrator>.AsSubComponent(args.calibrator).CreateInstance(host);
                var nameTrainer = args.trainer.ToString().Replace("{", "").Replace("}", "").Replace(" ", "").Replace("=", "").Replace("+", "Y").Replace("-", "N");
                var extTrainer = new ExtendedTrainer(trainer, nameTrainer);
                _predictor = extTrainer.Train(host, ch, data, null, calibrator, args.maxCalibrationExamples);

                if (!string.IsNullOrEmpty(args.outputModel))
                {
                    ch.Info("Saving model into '{0}'", args.outputModel);
                    using (var fs = File.Create(args.outputModel))
                        TrainUtils.SaveModel(env, ch, fs, _predictor, data);
                    ch.Info("Done.");
                }

                if (_cali != null)
                    throw ch.ExceptNotImpl("Calibrator is not implemented yet.");

                ch.Trace("Scoring");
                if (_args.scorer != null)
                {
                    var mapper = new SchemaBindablePredictorWrapper(_predictor);
                    var roles = new RoleMappedSchema(input.Schema, null, feat, group: group);
                    var bound = (mapper as ISchemaBindableMapper).Bind(_host, roles);
                    var scorPars = ScikitSubComponent<IDataScorerTransform, SignatureDataScorer>.AsSubComponent(_args.scorer);
                    _scorer = scorPars.CreateInstance(_host, input, bound, roles);
                }
                else
                    _scorer = PredictorHelper.CreateDefaultScorer(_host, input, feat, group, _predictor);

                ch.Info("Tagging with tag '{0}'.", args.tag);

                var ar = new TagViewTransform.Arguments { tag = args.tag };
                var res = new TagViewTransform(env, ar, _scorer, _predictor);
                return res;
            }
        }

        private static RoleMappedData CreateDataFromArgs(IHostEnvironment env, IExceptionContext ectx, IDataView input,
            TrainAndScoreTransformer.ArgumentsBase args, out string feat, out string group)
        {
            var schema = input.Schema;
            feat = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "FeatureColumn", args.FeatureColumn, DefaultColumnNames.Features);
            var label = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "LabelColumn", args.LabelColumn, DefaultColumnNames.Label);
            group = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "GroupColumn", args.GroupColumn, DefaultColumnNames.GroupId);
            var weight = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "WeightColumn", args.WeightColumn, DefaultColumnNames.Weight);
            var name = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "NameColumn", args.NameColumn, DefaultColumnNames.Name);
            var customCols_ = TrainUtils.CheckAndGenerateCustomColumns(ectx, args.CustomColumns);
            var customCols = customCols_ == null ? new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>() : customCols_.ToList();
            if (!string.IsNullOrEmpty(name))
                customCols.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Name, name));
            return env.CreateExamples(input, feat, label: label, group: group, weight: weight, custom: customCols);
        }

        #endregion
    }
}

