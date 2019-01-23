// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
//using Microsoft.ML.Api;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using PassThroughTransform = Scikit.ML.PipelineTransforms.PassThroughTransform;

[assembly: LoadableClass(PassThroughTransform.Summary, typeof(PassThroughTransform),
    typeof(PassThroughTransform.Arguments), typeof(SignatureDataTransform),
    "Pass Through Transform", PassThroughTransform.LoaderSignature, "Pass", "PassThrough", "DumpView")]

[assembly: LoadableClass(PassThroughTransform.Summary, typeof(PassThroughTransform),
    null, typeof(SignatureLoadDataTransform),
    "Pass or Dump Transform", PassThroughTransform.LoaderSignature, "Pass", "PassThrough", "DumpView")]


namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Inserts a transform which does nothing just to get a transform pointer.
    /// </summary>
    public class PassThroughTransform : IDataTransform, ISaveAsOnnx
    {
        public const string LoaderSignature = "PassThroughTransform";  // Not more than 24 letters.
        public const string Summary = "Inserts a transform which does nothing just to get a transform pointer. It can be used to dump a view on disk.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PASSTHRO",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PassThroughTransform).Assembly.FullName);
        }

        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Save on disk?", ShortName = "s")]
            public bool saveOnDisk = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Filename if saved.", ShortName = "f")]
            public string filename = null;

            [Argument(ArgumentType.Multiple, HelpText = "Saver settings if data is saved on disk (default is binary).", ShortName = "saver")]
            //public SubComponent<IDataSaver, SignatureDataSaver> saverSettings = new SubComponent<IDataSaver, SignatureDataSaver>("binary");
            public string saverSettings = "binary";

            public string GetSaverSettings()
            {
                var saver = GetSaverComponent();
                var res = string.Format("{0}{{{1}}}", saver.Name, saver.GetSettingsString());
                res = res.Replace("{}", "");
                return res;
            }

            public ISubComponent<IDataSaver> GetSaverComponent()
            {
                return new ScikitSubComponent<IDataSaver, SignatureDataSaver>(saverSettings);
            }

            public void Save(ModelSaveContext ctx)
            {
                ctx.Writer.Write((byte)(saveOnDisk ? 1 : 0));
                ctx.Writer.Write(string.IsNullOrEmpty(filename) ? "" : filename);
                var saver = GetSaverSettings();
                ctx.Writer.Write(saver);
            }

            public void Read(ModelLoadContext ctx)
            {
                saveOnDisk = ctx.Reader.ReadByte() == 1;
                filename = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(filename))
                    filename = null;
                saverSettings = ctx.Reader.ReadString();
            }
        }

        IDataView _input;
        Arguments _args;
        IHost _host;
        object _lock;
        bool _saved;

        public IDataView Source { get { return _input; } }

        /// <summary>
        /// This method changes the sources. It does check the source
        /// has the exact same schema than the previous one.
        /// </summary>
        /// <param name="source">A new data source</param>
        public void SetSource(IDataView source)
        {
            var sch = SchemaHelper.ToString(Source.Schema);
            var sch2 = SchemaHelper.ToString(source.Schema);
            if (sch != sch2)
                throw _host.Except("Schema mismatch. Expected:\n{0}\nGot:\n{1}", sch, sch2);
            _input = source;
        }

        public PassThroughTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            _input = input;
            _args = args;
            _lock = new object();
            _saved = false;
            if (_args.saveOnDisk && string.IsNullOrEmpty(_args.filename))
                throw _host.Except("If saveOnDisk is true, filename cannot be empty.");
        }

        public static PassThroughTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new PassThroughTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Save(ctx);
        }

        private PassThroughTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx);
            _lock = new object();
            _saved = false;
        }

        public Schema Schema { get { return _input.Schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount();
        }

        public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValue(_input, "_input");
            DumpView();
            return Source.GetRowCursor(columnsNeeded, rand);
        }

        public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            _host.AssertValue(_input, "_input");
            DumpView();
            return Source.GetRowCursorSet(columnsNeeded, n, rand);
        }

        public void DumpView()
        {
            if (!_args.saveOnDisk)
                return;
            lock (_lock)
            {
                if (_saved)
                    return;

                using (var ch = _host.Start("Dump View"))
                {
                    ch.Info("Dump view into '{0}'{1}.", _args.filename, File.Exists(_args.filename) ? "(overwriting)" : "");
                    var saver = ComponentCreation.CreateSaver(_host, _args.GetSaverSettings());

                    var columnsList = new List<int>();
                    var schema = _input.Schema;
                    for (int i = 0; i < schema.Count; ++i)
                        columnsList.Add(saver.IsColumnSavable(schema[i].Type) && schema[i].IsHidden ? i : -1);
                    var columns = columnsList.Where(c => c >= 0).ToArray();
                    ch.Info("Save columns: {0}", string.Join(", ", columns.Select(c => c.ToString())));
                    using (var fs2 = File.Create(_args.filename))
                        saver.SaveData(fs2, _input, columns);

                    long length = new FileInfo(_args.filename).Length;
                    ch.Info("Done dump. Size: {0}", length);
                }

                _saved = true;
            }
        }

        public bool CanSaveOnnx(OnnxContext ctx)
        {
            return true;
        }

        public void SaveAsOnnx(OnnxContext ctx)
        {
            // Nothing to do.
        }
    }
}
