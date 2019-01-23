// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineLambdaTransforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using DeTrendTransform = Scikit.ML.TimeSeries.DeTrendTransform;


[assembly: LoadableClass(DeTrendTransform.Summary, typeof(DeTrendTransform),
    typeof(DeTrendTransform.Arguments), typeof(SignatureDataTransform),
    DeTrendTransform.LoaderSignature, "DeTrend")]

[assembly: LoadableClass(DeTrendTransform.Summary, typeof(DeTrendTransform),
    null, typeof(SignatureLoadDataTransform), "DeTrend Transform", "DeTrend", DeTrendTransform.LoaderSignature)]


namespace Scikit.ML.TimeSeries
{
    public delegate void SignatureMetaLinearSubParameter();


    /// <summary>
    /// Remove the trends of a timeseries.
    /// </summary>
    public class DeTrendTransform : TransformBase
    {
        #region identification

        public const string LoaderSignature = "DeTrendTransform";
        public const string Summary = "Removes the trend of a timeseries.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DETRNDTS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DeTrendTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.MultipleUnique, HelpText = "TimeSeries column", ShortName = "col")]
            public Column1x1[] columns;

            [Argument(ArgumentType.Required, HelpText = "Column which contains the time", ShortName = "time")]
            public string timeColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Parameters to use for the linear optimizer.",
                SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<ITrainer> optim =
                new ScikitSubComponent<ITrainer, SignatureRegressorTrainer>("sasdcar");

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(Column1x1.ArrayToLine(columns));
                ctx.Writer.Write(timeColumn);
                ctx.Writer.Write(optim.ToString());
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                string sr = ctx.Reader.ReadString();
                columns = Column1x1.ParseMulti(sr);
                timeColumn = ctx.Reader.ReadString();
                var opt = ctx.Reader.ReadString();
                optim = new ScikitSubComponent<ITrainer, SignatureRegressorTrainer>(opt);
            }
        }

        #endregion

        #region internal members / accessors

        IDataTransform _transform;      // templated transform (not the serialized version)
        IPredictor _trend;              // Trend predictor
        Arguments _args;                // parameters
        Schema _schema;                 // We need the schema the transform outputs.
        object _lock;

        public override Schema OutputSchema { get { return _schema; } }

        #endregion

        #region public constructor / serialization / load / save

        public DeTrendTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");
            _args = args;
            if (_args.columns == null || _args.columns.Length != 1)
                Host.ExceptUserArg(nameof(_args.columns), "One column must be specified.");
            SchemaHelper.GetColumnIndex(input.Schema, args.timeColumn);
            SchemaHelper.GetColumnIndex(input.Schema, args.columns[0].Source);
            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema,
                                                       new[] { _args.columns[0].Name },
                                                       new[] { NumberType.R4 /*input.Schema.GetColumnType(index)*/ }));
            _trend = null;
            _transform = null;
            _lock = new object();
        }

        public static DeTrendTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new DeTrendTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(_trend, "No trend predictor was ever trained. The model cannot be saved.");
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, Host);
            ctx.SaveModel(_trend, "trend");
        }

        private DeTrendTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, Host);

            ctx.LoadModel<IPredictor, SignatureLoadModel>(host, out _trend, "trend");

            if (_args.columns == null || _args.columns.Length != 1)
                Host.ExceptUserArg(nameof(_args.columns), "One column must be specified.");
            int index = SchemaHelper.GetColumnIndex(input.Schema, _args.columns[0].Source);

            _schema = ExtendedSchema.Create(new ExtendedSchema(input.Schema,
                                            new[] { _args.columns[0].Name },
                                            new[] { NumberType.R4 /*input.Schema.GetColumnType(index)*/ }));
            _lock = new object();
            _transform = BuildTransform(_trend);
        }

        #endregion

        #region IDataTransform API

        public override bool CanShuffle
        {
            get { return Source.CanShuffle; }
        }

        public override long? GetRowCount()
        {
            Host.AssertValue(Source, "Source");
            return Source.GetRowCount();
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return null;
        }

        protected override RowCursor GetRowCursorCore(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            if (_transform == null)
                lock (_lock)
                    if (_transform == null)
                        _transform = CreateTemplatedTransform();
            Host.AssertValue(_transform, "_transform");
            Host.AssertValue(_trend, "_trend");
            return _transform.GetRowCursor(columnsNeeded, rand);
        }

        public override RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            if (_transform == null)
                lock (_lock)
                    if (_transform == null)
                        _transform = CreateTemplatedTransform();
            Host.AssertValue(_transform, "_transform");
            Host.AssertValue(_trend, "_trend");
            return _transform.GetRowCursorSet(columnsNeeded, n, rand);
        }

        #endregion

        #region transform own logic

        private void ValidateInputs(out int indexLabel, out int indexTime, out ColumnType typeLabel, out ColumnType typeTime)
        {
            indexLabel = SchemaHelper.GetColumnIndex(Source.Schema, _args.columns[0].Source);
            typeLabel = Source.Schema[indexLabel].Type;
            if (typeLabel.IsVector())
            {
                if (typeLabel.AsVector().DimCount() != 1 || typeLabel.AsVector().GetDim(0) != 1)
                    throw Host.ExceptNotImpl("Not implemented yet for multiple dimensions.");
            }
            if (typeLabel.RawKind() != DataKind.R4)
                throw Host.ExceptNotImpl("InputColumn must be R4.");
            indexTime = SchemaHelper.GetColumnIndex(Source.Schema, _args.timeColumn);
            typeTime = Source.Schema[indexTime].Type;
            if (typeTime.RawKind() != DataKind.R4)
                throw Host.ExceptNotImpl("Time columne must be R4.");
        }

        private RoleMappedData BuildViewBeforeTraining(out string slotName, out string slotTime, bool train)
        {
            int index, indexTime;
            ColumnType type, typeTime;
            ValidateInputs(out index, out indexTime, out type, out typeTime);
            IDataView input = Source;
            slotName = _args.columns[0].Source;

            if (train && type.IsVector())
            {
                slotName = input.Schema.GetTempColumnName() + "in";
                input = LambdaColumnHelper.Create(Host, "takeslot", input, _args.columns[0].Source, slotName,
                                            new VectorType(NumberType.R4), NumberType.R4,
                                            (in VBuffer<float> src, ref float dst) =>
                                            {
                                                dst = src.GetItemOrDefault(0);
                                            });
            }

            slotTime = _args.timeColumn;
            if (!typeTime.IsVector())
            {
                slotTime = input.Schema.GetTempColumnName() + "time";
                input = LambdaColumnHelper.Create(Host, "makevect", input, _args.timeColumn, slotTime,
                                            NumberType.R4, new VectorType(NumberType.R4, 2),
                                            (in float src, ref VBuffer<float> dst) =>
                                            {
                                                if (dst.Values != null)
                                                    dst = new VBuffer<float>(2, dst.Values);
                                                else
                                                    dst = new VBuffer<float>(2, new float[2]);
                                                dst.Values[0] = src;
                                                dst.Values[1] = 1f;
                                            });
            }
            var roles = new RoleMappedData(input, new[] {
                    new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, slotTime),
                    new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, slotName),
                });
            return roles;
        }

        private IDataTransform CreateTemplatedTransform()
        {
            string slotName, slotTime;
            var roles = BuildViewBeforeTraining(out slotName, out slotTime, true);
            using (var ch = Host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
                var optSett = ScikitSubComponent<ITrainer, SignatureRegressorTrainer>.AsSubComponent(_args.optim);
                ITrainer trainer = optSett.CreateInstance(Host);
                _trend = TrainUtils.Train(Host, ch, roles, trainer, null, null, 0, null);
            }
            return BuildTransform(_trend);
        }

        IDataTransform BuildTransform(IPredictor trend)
        {
            string slotName, slotTime;
            var roles = BuildViewBeforeTraining(out slotName, out slotTime, false);

            var newName = roles.Data.Schema.GetTempColumnName() + "trend";

            // Scoring
            var args = new PredictTransform.Arguments { featureColumn = slotTime, outputColumn = newName, serialize = false };
            var predict = new PredictTransform(Host, args, roles.Data, trend);
            string tempColumn = predict.Schema.GetTempColumnName() + "ConcatDeTrend";
            var cargs = new ColumnConcatenatingTransformer.Arguments()
            {
                Column = new[] {
                    ColumnConcatenatingTransformer.Column.Parse(string.Format("{0}:{1},{2}", tempColumn, slotName, newName)),
               }
            };
            var concat = ColumnConcatenatingTransformer.Create(Host, cargs, predict);

            var lambdaView = LambdaColumnHelper.Create(Host,
                "DeTrendTransform", concat, tempColumn, _args.columns[0].Name, new VectorType(NumberType.R4, 2),
                NumberType.R4,
                (in VBuffer<float> src, ref float dst) =>
                {
                    dst = src.Values[1] - src.Values[0];
                });

            var dropColumns = new List<string>();
            dropColumns.Add(newName);
            if (_args.columns[0].Source != slotName)
                dropColumns.Add(slotName);
            if (_args.timeColumn != slotTime)
                dropColumns.Add(slotTime);
            dropColumns.Add(tempColumn);

            var dropped = ColumnSelectingTransformer.CreateDrop(Host, lambdaView, dropColumns.ToArray());
            return dropped;
        }
    }

    #endregion
}
