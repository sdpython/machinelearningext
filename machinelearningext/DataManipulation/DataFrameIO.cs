// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;


namespace Scikit.ML.DataManipulation
{
    public static class DataFrameIO
    {
        #region kinds

        static ColumnType GuessKind(int col, List<string[]> read)
        {
            DataKind res = DataKind.TX;
            int nbline = 0;
            foreach (var line in read)
            {
                if (col >= line.Length)
                    throw new FormatException(string.Format("Line {0} has less column than expected.", nbline + 1));
                var val = line[col];

                try
                {
                    bool.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.BL, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.BL, res);
                        continue;
                    }
                }

                try
                {
                    int.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.I4, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.I4, res);
                        continue;
                    }
                }

                try
                {
                    uint.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.U4, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.U4, res);
                        continue;
                    }
                }

                try
                {
                    Int64.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.I8, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.I8, res);
                        continue;
                    }
                }

                try
                {
                    float.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.R4, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.R4, res);
                        continue;
                    }
                }

                try
                {
                    double.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.R8, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.R8, res);
                        continue;
                    }
                }

                res = DetermineDataKind(nbline == 0, DataKind.TX, res);
                ++nbline;
            }
            switch (res)
            {
                case DataKind.BL: return BoolType.Instance;
                case DataKind.I4: return NumberType.I4;
                case DataKind.I8: return NumberType.I8;
                case DataKind.U4: return NumberType.U4;
                case DataKind.U8: return NumberType.U8;
                case DataKind.R4: return NumberType.R4;
                case DataKind.R8: return NumberType.R8;
                case DataKind.TX: return TextType.Instance;
                default:
                    throw Contracts.Except($"Unable to guess ColumnType from '{res}'.");
            }
        }

        /// <summary>
        /// Determines the more generic type with two types.
        /// </summary>
        static DataKind DetermineDataKind(bool first, DataKind suggested, DataKind previous)
        {
            if (first)
                return suggested;
            else
                return MaxKind(suggested, previous);
        }

        /// <summary>
        /// Determines the more generic type with two types.
        /// </summary>
        static DataKind MaxKind(DataKind a, DataKind b)
        {
            if (a == DataKind.TX || b == DataKind.TX)
                return DataKind.TX;
            if (a == DataKind.R8 || b == DataKind.R8)
                return DataKind.R8;
            if (a == DataKind.R4 || b == DataKind.R4)
                return DataKind.R4;
            if (a == DataKind.I8 || b == DataKind.I8)
                return DataKind.I8;
            if (a == DataKind.U4 || b == DataKind.U4)
                return DataKind.U4;
            if (a == DataKind.I4 || b == DataKind.I4)
                return DataKind.I4;
            if (a == DataKind.BL || b == DataKind.BL)
                return DataKind.BL;
            return DataKind.TX;
        }

        #endregion

        #region read/write

        /// <summary>
        /// Saves the dataframe as a file.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="encoding">encoding</param>
        /// <param name="silent">Suppress any info output (not warnings or errors)</param>
        public static void ViewToCsv(IDataView view, string filename, string sep = ",",
                                     bool header = true, Encoding encoding = null, bool silent = false,
                                     IHost host = null)
        {
            using (var fs = new StreamWriter(filename, false, encoding ?? Encoding.ASCII))
                ViewToCsv(view, fs.BaseStream, sep: sep, header: header, silent: silent, host: host);
        }

        /// <summary>
        /// Saves the dataframe in a stream as text format.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="silent">Suppress any info output (not warnings or errors)</param>
        public static void ViewToCsv(IDataView view, Stream st, string sep = ",", bool header = true,
                                     bool silent = false, IHost host = null)
        {
            IHostEnvironment env = host;
            if (env == null)
                env = new ConsoleEnvironment();
            var saver = new TextSaver(env, new TextSaver.Arguments()
            {
                Separator = sep,
                OutputSchema = false,
                OutputHeader = header,
                Silent = silent
            });
            var columns = new int[view.Schema.Count];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = i;
            saver.SaveData(st, view, columns);
        }

        public static IDataView ReadCsvToTextLoader(string filename,
                                        char sep = ',', bool header = true,
                                        string[] names = null, ColumnType[] dtypes = null,
                                        int nrows = -1, int guess_rows = 10,
                                        Encoding encoding = null, bool useThreads = true,
                                        bool index = false, IHost host = null)
        {
            return ReadCsvToTextLoader(new string[] { filename }, sep, header, names, dtypes, nrows, guess_rows, encoding,
                                       useThreads, index, host);
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <param name="useThreads">specific to TextLoader</param>
        /// <param name="host">host</param>
        /// <param name="index">add a column to hold the index</param>
        /// <returns>TextLoader</returns>
        public static IDataView ReadCsvToTextLoader(string[] filenames,
                                        char sep = ',', bool header = true,
                                        string[] names = null, ColumnType[] dtypes = null,
                                        int nrows = -1, int guess_rows = 10,
                                        Encoding encoding = null, bool useThreads = true,
                                        bool index = false, IHost host = null)
        {
            var df = ReadCsv(filenames[0], sep: sep, header: header, names: names, dtypes: dtypes,
                             nrows: guess_rows, guess_rows: guess_rows, encoding: encoding, index: index);
            var sch = df.Schema;
            var cols = new TextLoader.Column[sch.Count];
            for (int i = 0; i < cols.Length; ++i)
                cols[i] = TextLoader.Column.Parse(df.NameType(i));

            var args = new TextLoader.Arguments()
            {
                AllowQuoting = false,
                Separators = new[] { sep },
                Column = cols,
                TrimWhitespace = true,
                UseThreads = useThreads,
                HasHeader = header,
                MaxRows = nrows > 0 ? (int?)nrows : null
            };
            if (host == null)
                host = new ConsoleEnvironment().Register("TextLoader");
            var multiSource = new MultiFileSource(filenames);
            return new TextLoader(host, args, multiSource).Read(multiSource);
        }

        /// <summary>
        /// Reads a string as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="content">data as single string</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="index">add one column with the row index</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadStr(string content,
                                    char sep = ',', bool header = true,
                                    string[] names = null, ColumnType[] dtypes = null,
                                    int nrows = -1, int guess_rows = 10, bool index = false)
        {
            return ReadStream(() => new StreamReader(new MemoryStream(Encoding.UTF8.GetBytes(content))),
                              sep: sep, header: header, names: names, dtypes: dtypes, nrows: nrows,
                              guess_rows: guess_rows, index: index);
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <param name="index">add one column with the row index</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadCsv(string filename,
                                char sep = ',', bool header = true,
                                string[] names = null, ColumnType[] dtypes = null,
                                int nrows = -1, int guess_rows = 10,
                                Encoding encoding = null, bool index = false)
        {
            return ReadStream(() => new StreamReader(filename, encoding ?? Encoding.ASCII),
                              sep: sep, header: header, names: names, dtypes: dtypes, nrows: nrows,
                              guess_rows: guess_rows, index: index);
        }

        public delegate StreamReader FunctionCreateStreamReader();

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="createStream">function which creates a stream</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="index">add one column with the row index</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadStream(FunctionCreateStreamReader createStream,
                                char sep = ',', bool header = true,
                                string[] names = null, ColumnType[] dtypes = null,
                                int nrows = -1, int guess_rows = 10, bool index = false)
        {
            var lines = new List<string[]>();
            int rowline = 0;

            // First pass: schema and number of rows.
            using (var st = createStream())
            {
                string line = st.ReadLine();
                int nbline = 0;
                while (line != null && (nrows == -1 || rowline < nrows))
                {
                    var spl = line.Split(sep);
                    if (header && nbline == 0)
                    {
                        if (names == null)
                            names = spl;
                    }
                    else
                    {
                        ++rowline;
                        if (lines.Count < guess_rows)
                            lines.Add(spl);
                    }
                    ++nbline;
                    line = st.ReadLine();
                }
            }

            if (lines.Count == 0)
                throw new FormatException("File is empty.");
            int numCol = lines.Select(c => c.Length).Max();
            var df = new DataFrame();

            // Guesses types and adds columns.
            for (int i = 0; i < numCol; ++i)
            {
                var kind = GuessKind(i, lines);
                df.AddColumn(names[i],
                            dtypes != null && i < dtypes.Length && dtypes[i] != null
                                        ? dtypes[i]
                                        : kind,
                            rowline);
            }

            // Fills values.
            using (var st = createStream())
            {
                string line = st.ReadLine();
                int nbline = 0;
                rowline = 0;
                while (line != null && (nrows == -1 || rowline < nrows))
                {
                    var spl = line.Split(sep);
                    if (header && nbline == 0)
                    {
                        // Skips.
                    }
                    else
                    {
                        df.FillValues(rowline, spl);
                        ++rowline;
                    }
                    ++nbline;
                    line = st.ReadLine();
                }
            }

            if (index)
            {
                var hashNames = new HashSet<string>(names);
                var nameIndex = "index";
                while (hashNames.Contains(nameIndex))
                    nameIndex += "_";
                var indexValues = Enumerable.Range(0, df.Length).ToArray();
                df.AddColumn(nameIndex, indexValues);
                var newColumns = (new[] { nameIndex }).Concat(names).ToArray();
                df.OrderColumns(newColumns);
            }

            return df;
        }

        /// <summary>
        /// Converts a <see cref="IDataView"/> into a <see cref="DataFrame"/>.
        /// Follows pandas API.
        /// </summary>
        /// <param name="view">IDataView</param>
        /// <param name="sep">column separator</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="keepVectors">keep vectors as they are</param>
        /// <param name="numThreads">number of threads to use to fill the dataframe</param>
        /// <param name="env">mostly for logging</param>
        /// <returns><see cref="DataFrame"/></returns>
        public static DataFrame ReadView(IDataView view, int nrows = -1, bool keepVectors = false, int? numThreads = 1,
                                         IHostEnvironment env = null)
        {
            var df = new DataFrame();
            df.FillValues(view, nrows: nrows, keepVectors: keepVectors, numThreads: numThreads, env: env);
            return df;
        }

        #endregion

        #region data to dataframe

        /// <summary>
        /// Adds one column into a dataframe.
        /// </summary>
        public static void AddColumn<ValueType>(DataFrame df, string name, ValueType[] values)
        {
            if (typeof(ValueType) == typeof(float))
                df.AddColumn(name, values as float[]);
            else if (typeof(ValueType) == typeof(double))
                df.AddColumn(name, values as double[]);
            else if (typeof(ValueType) == typeof(int))
                df.AddColumn(name, values as int[]);
            else if (typeof(ValueType) == typeof(uint))
                df.AddColumn(name, values as uint[]);
            else if (typeof(ValueType) == typeof(string))
                df.AddColumn(name, values as string[]);
            else if (typeof(ValueType) == typeof(bool))
                df.AddColumn(name, values as bool[]);
            else
                throw Contracts.ExceptNotImpl($"DataFrame do not support type {typeof(ValueType)}.");
        }

        /// <summary>
        /// Converts a dictionary into a dataframe with two columns,
        /// one for the keys (needs ToString()), one for the values.
        /// Data is sorted by key.
        /// </summary>
        public static DataFrame Convert<KeyType, ValueType>(Dictionary<KeyType, ValueType> data,
                                                            string columnKey = "key", string columnValue = "value")
        {
            var keys = new List<string>();
            var values = new List<ValueType>();
            foreach (var pair in data.OrderBy(c => c.Key))
            {
                keys.Add(pair.Key.ToString());
                values.Add(pair.Value);
            }

            var df = new DataFrame();
            df.AddColumn(columnKey, keys.ToArray());
            AddColumn(df, columnValue, values.ToArray());
            return df;
        }

        /// <summary>
        /// Converts a dictionary into a dataframe with two columns,
        /// one for the keys (needs ToString()), one for the values.
        /// Data is sorted by key.
        /// </summary>
        public static DataFrame Convert<KeyType1, KeyType2, ValueType>(Dictionary<Tuple<KeyType1, KeyType2>, ValueType> data,
                                                                       string columnKey1 = "key1", string columnKey2 = "key2", string columnValue = "value")
        {
            var keys1 = new List<KeyType1>();
            var keys2 = new List<KeyType2>();
            var values = new List<ValueType>();
            foreach (var pair in data.OrderBy(c => c.Key))
            {
                keys1.Add(pair.Key.Item1);
                keys2.Add(pair.Key.Item2);
                values.Add(pair.Value);
            }

            var df = new DataFrame();
            AddColumn(df, columnKey1, keys1.ToArray());
            AddColumn(df, columnKey2, keys2.ToArray());
            AddColumn(df, columnValue, values.ToArray());
            return df;
        }

        /// <summary>
        /// Converts a dictionary into a dataframe with two columns,
        /// one for the keys (needs ToString()), one for the values.
        /// Data is sorted by key.
        /// </summary>
        public static DataFrame Convert<KeyType1, KeyType2, KeyType3, ValueType>(Dictionary<Tuple<KeyType1, KeyType2, KeyType3>, ValueType> data,
                                                             string columnKey1 = "key1", string columnKey2 = "key2", string columnKey3 = "key3", string columnValue = "value")
        {
            var keys1 = new List<KeyType1>();
            var keys2 = new List<KeyType2>();
            var keys3 = new List<KeyType3>();
            var values = new List<ValueType>();
            foreach (var pair in data.OrderBy(c => c.Key))
            {
                keys1.Add(pair.Key.Item1);
                keys2.Add(pair.Key.Item2);
                keys3.Add(pair.Key.Item3);
                values.Add(pair.Value);
            }

            var df = new DataFrame();
            AddColumn(df, columnKey1, keys1.ToArray());
            AddColumn(df, columnKey2, keys2.ToArray());
            AddColumn(df, columnKey3, keys3.ToArray());
            AddColumn(df, columnValue, values.ToArray());
            return df;
        }

        /// <summary>
        /// Converts a dictionary into a dataframe with two columns,
        /// one for the keys (needs ToString()), one for the values.
        /// Data is sorted by key.
        /// </summary>
        public static DataFrame Convert<KeyType1, KeyType2, KeyType3, KeyType4, ValueType>(Dictionary<Tuple<KeyType1, KeyType2, KeyType3, KeyType4>, ValueType> data,
                                                             string columnKey1 = "key1", string columnKey2 = "key2", string columnKey3 = "key3", string columnKey4 = "key4", string columnValue = "value")
        {
            var keys1 = new List<KeyType1>();
            var keys2 = new List<KeyType2>();
            var keys3 = new List<KeyType3>();
            var keys4 = new List<KeyType4>();
            var values = new List<ValueType>();
            foreach (var pair in data.OrderBy(c => c.Key))
            {
                keys1.Add(pair.Key.Item1);
                keys2.Add(pair.Key.Item2);
                keys3.Add(pair.Key.Item3);
                keys4.Add(pair.Key.Item4);
                values.Add(pair.Value);
            }

            var df = new DataFrame();
            AddColumn(df, columnKey1, keys1.ToArray());
            AddColumn(df, columnKey2, keys2.ToArray());
            AddColumn(df, columnKey3, keys3.ToArray());
            AddColumn(df, columnKey4, keys4.ToArray());
            AddColumn(df, columnValue, values.ToArray());
            return df;
        }

        /// <summary>
        /// Converts a dictionary into a dataframe with two columns,
        /// one for the keys (needs ToString()), one for the values.
        /// Data is sorted by key.
        /// </summary>
        public static DataFrame Convert<KeyType1, KeyType2, KeyType3, KeyType4, KeyType5, ValueType>(Dictionary<Tuple<KeyType1, KeyType2, KeyType3, KeyType4, KeyType5>, ValueType> data,
                                                             string columnKey1 = "key1", string columnKey2 = "key2", string columnKey3 = "key3", string columnKey4 = "key4", string columnKey5 = "key5", string columnValue = "value")
        {
            var keys1 = new List<KeyType1>();
            var keys2 = new List<KeyType2>();
            var keys3 = new List<KeyType3>();
            var keys4 = new List<KeyType4>();
            var keys5 = new List<KeyType5>();
            var values = new List<ValueType>();
            foreach (var pair in data.OrderBy(c => c.Key))
            {
                keys1.Add(pair.Key.Item1);
                keys2.Add(pair.Key.Item2);
                keys3.Add(pair.Key.Item3);
                keys4.Add(pair.Key.Item4);
                keys5.Add(pair.Key.Item5);
                values.Add(pair.Value);
            }

            var df = new DataFrame();
            AddColumn(df, columnKey1, keys1.ToArray());
            AddColumn(df, columnKey2, keys2.ToArray());
            AddColumn(df, columnKey3, keys3.ToArray());
            AddColumn(df, columnKey4, keys4.ToArray());
            AddColumn(df, columnKey5, keys5.ToArray());
            AddColumn(df, columnValue, values.ToArray());
            return df;
        }

        /// <summary>
        /// Converts a dictionary into a dataframe with two columns,
        /// one for the keys (needs ToString()), one for the values.
        /// Data is sorted by key.
        /// </summary>
        public static DataFrame Convert<KeyType1, KeyType2, KeyType3, KeyType4, KeyType5, KeyType6, ValueType>(Dictionary<Tuple<KeyType1, KeyType2, KeyType3, KeyType4, KeyType5, KeyType6>, ValueType> data,
                                                             string columnKey1 = "key1", string columnKey2 = "key2", string columnKey3 = "key3", string columnKey4 = "key4",
                                                             string columnKey5 = "key5", string columnKey6 = "key6", string columnValue = "value")
        {
            var keys1 = new List<KeyType1>();
            var keys2 = new List<KeyType2>();
            var keys3 = new List<KeyType3>();
            var keys4 = new List<KeyType4>();
            var keys5 = new List<KeyType5>();
            var keys6 = new List<KeyType6>();
            var values = new List<ValueType>();
            foreach (var pair in data.OrderBy(c => c.Key))
            {
                keys1.Add(pair.Key.Item1);
                keys2.Add(pair.Key.Item2);
                keys3.Add(pair.Key.Item3);
                keys4.Add(pair.Key.Item4);
                keys5.Add(pair.Key.Item5);
                keys6.Add(pair.Key.Item6);
                values.Add(pair.Value);
            }

            var df = new DataFrame();
            AddColumn(df, columnKey1, keys1.ToArray());
            AddColumn(df, columnKey2, keys2.ToArray());
            AddColumn(df, columnKey3, keys3.ToArray());
            AddColumn(df, columnKey4, keys4.ToArray());
            AddColumn(df, columnKey5, keys5.ToArray());
            AddColumn(df, columnKey6, keys6.ToArray());
            AddColumn(df, columnValue, values.ToArray());
            return df;
        }

        #endregion
    }
}
