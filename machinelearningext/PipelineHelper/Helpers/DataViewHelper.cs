// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Helpers about DataView.
    /// </summary>
    public static class DataViewHelper
    {
        /// <summary>
        /// Handles specific cases DataViewUtils does not handle.
        /// </summary>
        /// <param name="view">IDataView</param>
        /// <param name="predicate">column selector (null for all)</param>
        /// <returns>number of rows</returns>
        public static long ComputeRowCount(IDataView view, IEnumerable<Schema.Column> columnsNeeded = null)
        {
            var res = view.GetRowCount();
            if (res.HasValue)
                return res.Value;
            long lres = 0;
            using (var cur = view.GetRowCursor(columnsNeeded))
            {
                while (cur.MoveNext())
                    ++lres;
            }
            return lres;
        }

        /// <summary>
        /// Dump a view in csv format
        /// </summary>
        /// <param name="host">IHost</param>
        /// <param name="view">view to dump</param>
        /// <param name="filename">output filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="schema">include the schema</param>
        public static void ToCsv(IHostEnvironment host, IDataView view, string filename, string sep = "\t", bool schema = true)
        {
            var settings = string.Format("Text{{sep={0} header=+ schema={1}}}",
                sep == "\t" ? "tab" : sep, schema ? "+" : "-");
            var saver = ComponentCreation.CreateSaver(host, settings);
            string full_output = Path.GetFullPath(filename);
            using (var ch = host.Start("ToCsv"))
            {
                ch.Info(MessageSensitivity.None, "Saving data into file '{0}' or '{1}'.", filename, full_output);
                using (var fs0 = host.CreateOutputFile(full_output))
                    DataSaverUtils.SaveDataView(ch, saver, view, fs0, true);
            }
        }

        /// <summary>
        /// Dump a view in binary format
        /// </summary>
        /// <param name="host">IHost</param>
        /// <param name="view">view to dump</param>
        /// <param name="filename">output filename</param>
        public static void ToIdv(IHostEnvironment host, IDataView view, string filename)
        {
            var settings = "Binary";
            var saver = ComponentCreation.CreateSaver(host, settings);
            string full_output = Path.GetFullPath(filename);
            using (var ch = host.Start("ToIdv"))
            {
                ch.Info(MessageSensitivity.None, "Saving data into file '{0}' or '{1}'.", filename, full_output);
                using (var fs0 = host.CreateOutputFile(full_output))
                    DataSaverUtils.SaveDataView(ch, saver, view, fs0, true);
            }
        }

        /// <summary>
        /// Retrieves the first view of a pipeline.
        /// </summary>
        /// <param name="view">IDataView</param>
        /// <returns>IDataView</returns>
        public static IDataView GetFirstView(IDataView view)
        {
            var tr = view as IDataTransform;
            while (tr != null)
            {
                view = tr.Source;
                tr = view as IDataTransform;
            }
            return view;
        }
    }
}
