// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements a <see cref="DataFrame"/> which does not necessarily holds
    /// in memory based on a <see cref="IDataView"/> from ML.net.
    /// </summary>
    public class StreamingDataFrame : IDataView
    {
        private IDataView _source;
        IHostEnvironment _env;

        public IDataView Source => _source;
        public DataViewSchema Schema => _source.Schema;

        public StreamingDataFrame(IDataView source, IHostEnvironment env = null)
        {
            _source = source;
            _env = env;
        }

        public StreamingDataFrame(DataFrame source, IHostEnvironment env = null)
        {
            _source = source;
            _env = env;
        }

        public void AddTransform(string transform)
        {
            if (_env == null)
                throw Contracts.ExceptNotSupp("The class must be initialized with an envrionment to enable that functionality.");
            var tr = ComponentCreation.CreateTransform(_env, transform, Source);
            if (tr == null)
                throw Contracts.ExceptNotSupp($"Unable to create transform '{transform}'.");
            AddTransform(tr);
        }

        public void AddTransform(IDataTransform tr)
        {
            if (tr.Source != Source)
                throw Contracts.ExceptNotSupp("Source of the transform must be this StreamingDataFrame.");
            _source = tr;
        }

        public bool CanShuffle => Source.CanShuffle;
        public long? GetRowCount() { return Source.GetRowCount(); }
        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null) { return Source.GetRowCursor(columnsNeeded, rand); }
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return Source.GetRowCursorSet(columnsNeeded, n, rand);
        }

        public static StreamingDataFrame ReadCsv(string filename,
                                        char sep = ',', bool header = true,
                                        string[] names = null, DataViewType[] dtypes = null,
                                        int nrows = -1, int guess_rows = 10,
                                        Encoding encoding = null, bool useThreads = true,
                                        bool index = false, IHost host = null)
        {
            return new StreamingDataFrame(DataFrameIO.ReadCsvToTextLoader(filename, sep, header, names, dtypes, nrows, guess_rows,
                                                                          encoding, useThreads, index, host));
        }

        public static StreamingDataFrame ReadCsv(string[] filenames,
                                        char sep = ',', bool header = true,
                                        string[] names = null, DataViewType[] dtypes = null,
                                        int nrows = -1, int guess_rows = 10,
                                        Encoding encoding = null, bool useThreads = true,
                                        bool index = false, IHost host = null)
        {
            return new StreamingDataFrame(DataFrameIO.ReadCsvToTextLoader(filenames, sep, header, names, dtypes, nrows, guess_rows,
                                                                          encoding, useThreads, index, host));
        }

        /// <summary>
        /// Converts into <see cref="DataFrame"/>.
        /// </summary>
        /// <param name="sep">column separator</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="keepVectors">keep vectors as they are</param>
        /// <param name="numThreads">number of threads to use to fill the dataframe</param>
        /// <returns><see cref="DataFrame"/></returns>
        public DataFrame ToDataFrame(int nrows = -1, bool keepVectors = false, int? numThreads = 1)
        {
            return DataFrameIO.ReadView(Source, nrows, keepVectors, numThreads, _env);
        }
    }
}
