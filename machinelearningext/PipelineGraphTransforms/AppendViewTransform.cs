// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using AppendViewTransform = Scikit.ML.PipelineGraphTransforms.AppendViewTransform;

[assembly: LoadableClass(AppendViewTransform.Summary, typeof(AppendViewTransform),
    typeof(AppendViewTransform.Arguments), typeof(SignatureDataTransform),
    "Append View Transform", AppendViewTransform.LoaderSignature, "Append")]

[assembly: LoadableClass(AppendViewTransform.Summary, typeof(AppendViewTransform),
    null, typeof(SignatureLoadDataTransform),
    "Append View Transform", AppendViewTransform.LoaderSignature, "Append")]


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Concatenates two views.
    /// </summary>
    public class AppendViewTransform : IDataTransform
    {
        #region identification

        public const string LoaderSignature = "AppendViewTransform";  // Not more than 24 letters.
        public const string Summary = "Appends tagged views.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "APPETAGS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(AppendViewTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Tags of the view to append with the current view.", ShortName = "t")]
            public string[] tag;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(string.Join(",", tag));
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                string tags = ctx.Reader.ReadString();
                host.CheckValue(tags, "tags");
                tag = tags.Split(',');
            }

            public void PostProcess()
            {
                if (tag != null && tag.Length == 1 && tag[0].Contains(","))
                    tag = tag[0].Split(',');
            }
        }

        #endregion

        #region internal members / accessors

        readonly Arguments _args;
        readonly IHost _host;
        readonly IDataView _source;
        readonly IDataView _mergedView;

        #endregion

        #region API DataTransform

        public IDataView Source { get { return _source; } }

        /// <summary>
        /// Create a AppendViewTransform transform.
        /// </summary>
        public AppendViewTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            args.PostProcess();
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, "args");
            _args = args;
            _host.CheckValue(args.tag, "Tag cannot be empty.");
            _source = input;
            _mergedView = Setup(input);
        }

        private IDataView Setup(IDataView input)
        {
            List<IDataView> concat = new List<IDataView>();
            concat.Add(input);
            foreach (var tag in _args.tag)
            {
                var selected = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == tag);
                if (!selected.Any())
                    throw _host.Except("Unable to find a view to append with tag '{0}'", tag);
                var first = selected.First();
                if (selected.Skip(1).Any())
                    throw _host.Except("Tag '{0}' is ambiguous, {1} views were found.", tag, selected.Count());
                concat.Add(first.Item2);
            }
            return AppendRowsDataView.Create(_host, input.Schema, concat.ToArray());
        }

        public static AppendViewTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new AppendViewTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        private AppendViewTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            _host = host;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            _source = input;
            _mergedView = Setup(input);
        }

        public DataViewSchema Schema { get { return _mergedView.Schema; } }
        public bool CanShuffle { get { return _mergedView.CanShuffle; } }
        public long? GetRowCount()
        {
            _host.AssertValue(_source, "_input");
            return _mergedView.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValue(_source, "_source");
            return _mergedView.GetRowCursor(columnsNeeded, rand);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            _host.AssertValue(_source, "_source");
            return _mergedView.GetRowCursorSet(columnsNeeded, n, rand);
        }

        #endregion
    }
}
