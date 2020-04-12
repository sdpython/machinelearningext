// See the LICENSE file in the project root for more information.

using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.PipelineGraphTransforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using TaggedScoreTransform = Scikit.ML.PipelineGraphTraining.TaggedScoreTransform;

[assembly: LoadableClass(TaggedScoreTransform.Summary, typeof(TaggedScoreTransform),
    typeof(TaggedScoreTransform.Arguments), typeof(SignatureDataTransform),
    "Score a Tagged Predictor", TaggedScoreTransform.LoaderSignature, "TagScore")]

[assembly: LoadableClass(TaggedScoreTransform.Summary, typeof(TaggedScoreTransform),
    null, typeof(SignatureLoadDataTransform),
    "Score a Tagged Predictor", TaggedScoreTransform.LoaderSignature, "TagScore")]


namespace Scikit.ML.PipelineGraphTraining
{
    /// <summary>
    /// Scores a predictor hosted by a tagged view.
    /// </summary>
    public class TaggedScoreTransform : AbstractSimpleTransformTemplate
    {
        public const string LoaderSignature = "TaggedScoreTransform";  // Not more than 24 letters.
        internal const string Summary = "Runs a previously trained predictor on the data. This predictor is hosted by a tagged view.";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TAGSCORE",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TaggedScoreTransform).Assembly.FullName);
        }

        public new class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Tag of the view which holds the predictor.", ShortName = "in")]
            public string taggedPredictor;

            [Argument(ArgumentType.LastOccurrenceWins, HelpText = "Column to use for features when scorer is not defined",
                ShortName = "feat")]
            public string featureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Group column name", ShortName = "group")]
            public string groupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.Multiple,
                HelpText = "Input columns: Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                ShortName = "col")]
            public string customColumn;

            public KeyValuePair<string, string>[] customColumnPair;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>",
                SignatureType = typeof(SignatureDataScorer))]
            public IComponentFactory<IDataScorerTransform> scorer;

            public void PostProcess()
            {
                if (customColumnPair == null && !string.IsNullOrEmpty(customColumn))
                    customColumnPair = customColumn.Split(';').Select(c => c.Split(','))
                                                   .Select(c => new KeyValuePair<string, string>(c[0], c[1]))
                                                   .ToArray();
            }

            public void Write(ModelSaveContext ctx, IHost host)
            {
                IOHelper.Write(ctx, taggedPredictor);
                IOHelper.Write(ctx, featureColumn);
                IOHelper.Write(ctx, groupColumn);
                IOHelper.Write(ctx, customColumn);
                IOHelper.Write(ctx, customColumnPair);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                taggedPredictor = IOHelper.ReadString(ctx);
                featureColumn = IOHelper.ReadString(ctx);
                groupColumn = IOHelper.ReadString(ctx);
                customColumn = IOHelper.ReadString(ctx);
                customColumnPair = IOHelper.ReadArrayKeyValuePairStringString(ctx);
            }
        }

        readonly Arguments _args;
        IDataScorerTransform _scorer;

        public TaggedScoreTransform(IHostEnvironment env, Arguments args, IDataView input) :
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            args.PostProcess();
            _args = args;
            _host.CheckValue(args.taggedPredictor, "taggedPredictor");
            _sourcePipe = Create(_host, args, input, out _input, null);
            if (_scorer == null)
                throw _host.Except("_scorer cannot be null.");
            if (_sourcePipe == null)
                throw _host.Except("_sourcePipe cannot be null.");
        }

        protected override void SaveModel(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.SaveModel(_scorer, "scorer");
        }

        public TaggedScoreTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, ctx, input, LoaderSignature)
        {
            _args = new Arguments();
            _args.Read(ctx, _host);
            ctx.LoadModel<IDataScorerTransform, SignatureLoadDataTransform>(_host, out _scorer, "scorer", input);
            _sourcePipe = Create(_host, _args, input, out _input, _scorer);
        }

        public static TaggedScoreTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new TaggedScoreTransform(h, ctx, input));
        }

        IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input, out IDataView sourceCtx, IDataScorerTransform scorer)
        {
            sourceCtx = input;
            Contracts.CheckValue(env, "env");
            env.CheckValue(args, "args");
            env.CheckValue(input, "input");
            env.CheckUserArg(!string.IsNullOrWhiteSpace(args.taggedPredictor), "taggedPredictor",
                "The input tag is required.");

            if (scorer != null)
            {
                _scorer = scorer;
                return scorer;
            }
            else
            {
                var predictor = TagHelper.GetTaggedPredictor(env, input, args.taggedPredictor);
                string feat = TrainUtils.MatchNameOrDefaultOrNull(env, input.Schema,
                        "featureColumn", args.featureColumn, DefaultColumnNames.Features);
                string group = TrainUtils.MatchNameOrDefaultOrNull(env, input.Schema,
                    "groupColumn", args.groupColumn, DefaultColumnNames.GroupId);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.customColumnPair);

                _scorer = PredictorHelper.CreateDefaultScorer(_host, input, feat, group, predictor);
                return _scorer;
            }
        }
    }
}
