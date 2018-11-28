﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using ULabelToR4LabelTransform = Scikit.ML.PipelineTransforms.ULabelToR4LabelTransform;


[assembly: LoadableClass(ULabelToR4LabelTransform.Summary, typeof(ULabelToR4LabelTransform),
    typeof(ULabelToR4LabelTransform.Arguments), typeof(SignatureDataTransform),
    "ULabelToR4Label Transform", ULabelToR4LabelTransform.LoaderSignature, "ULabelToR4Label", "U2R4")]

[assembly: LoadableClass(ULabelToR4LabelTransform.Summary, typeof(ULabelToR4LabelTransform),
    null, typeof(SignatureLoadDataTransform),
    "ULabelToR4Label Transform", ULabelToR4LabelTransform.LoaderSignature, "ULabelToR4Label", "U2R4")]

namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Multiplies features, build polynomial features x1, x1^2, x1x2, x2, x2^2...
    /// </summary>
    public class ULabelToR4LabelTransform : IDataTransform
    {
        #region identification

        public const string LoaderSignature = "ULabelToR4LabelTransform";  // Not more than 24 letters.
        public const string Summary = "Converts a Key label into a Float label (does nothing if the input is a float).";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "U2R4U2R4",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ULabelToR4LabelTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to convert.", ShortName = "col")]
            public Column1x1[] columns;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(Column1x1.ArrayToLine(columns));
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                string sr = ctx.Reader.ReadString();
                columns = Column1x1.ParseMulti(sr);
            }
        }

        #endregion

        #region internal members / accessors

        IDataView _input;
        IDataTransform _transform;          // templated transform (not the serialized version)
        Arguments _args;
        IHost _host;

        public IDataView Source { get { return _input; } }

        #endregion

        #region public constructor / serialization / load / save

        public ULabelToR4LabelTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register("ULabelToR4LabelTransform");
            _host.CheckValue(args, "args");                 // Checks values are valid.
            _host.CheckValue(input, "input");
            _host.CheckValue(args.columns, "columns");

            _input = input;

            int ind;
            var schema = _input.Schema;
            for (int i = 0; i < args.columns.Length; ++i)
                if (!schema.TryGetColumnIndex(args.columns[i].Source, out ind))
                    throw _host.ExceptParam("inputColumn", "Column '{0}' not found in schema.", args.columns[i].Source);
            _args = args;
            _transform = CreateTemplatedTransform();
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        private ULabelToR4LabelTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            _transform = CreateTemplatedTransform();
        }

        public static ULabelToR4LabelTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ULabelToR4LabelTransform(h, ctx, input));
        }

        #endregion

        #region IDataTransform API

        public Schema Schema { get { return _transform.Schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount(); // We do not add or remove any row. Same number of rows as the input.
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            // Fun part we'll see later.
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        #endregion

        #region transform own logic

        /// <summary>
        /// Create the internal transform (not serialized in the zip file).
        /// </summary>
        private IDataTransform CreateTemplatedTransform()
        {
            IDataView view = Source;
            var schema = _input.Schema;
            int index;
            for (int i = 0; i < _args.columns.Length; ++i)
            {
                if (!schema.TryGetColumnIndex(_args.columns[i].Source, out index))
                    throw _host.Except("Unable to find '{0}'", _args.columns[i].Source);
                var typeCol = schema.GetColumnType(index);
                if (typeCol.IsVector())
                    throw _host.Except("Expected a number as input.");

                switch (typeCol.RawKind())
                {
                    case DataKind.R4:
                        view = new PassThroughTransform(_host, new PassThroughTransform.Arguments(),
                                            LambdaColumnMapper.Create(_host, "R42R4", view,
                                            _args.columns[i].Source, _args.columns[i].Name,
                                            NumberType.R4, NumberType.R4,
                                            (in float src, ref float dst) => { dst = src; }));
                        break;
                    case DataKind.U4:
                        view = new PassThroughTransform(_host, new PassThroughTransform.Arguments(),
                                            LambdaColumnMapper.Create(_host, "U42R4", view,
                                            _args.columns[i].Source, _args.columns[i].Name,
                                            NumberType.U4, NumberType.R4,
                                            (in uint src, ref float dst) => { dst = src == 0 ? float.NaN : src - 1; }));
                        break;
                    default:
                        throw Contracts.ExceptNotSupp("Type '{0}' is not handled yet.", typeCol.RawKind());
                }
            }
            return view as IDataTransform;
        }

        #endregion
    }
}
