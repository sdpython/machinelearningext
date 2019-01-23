// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements a DataFrame based on a IDataView from ML.net.
    /// It replicates some of pandas API for DataFrame except
    /// for the index which can be added as a column but does not
    /// play any particular role (concatenation does not take it
    /// into account).
    /// 
    /// The class follows IDataView API to be nicely integrated in 
    /// machine learning pipelines. However, due to the column type contraints
    /// (<code>IEquatable<DType>, IComparable<DType></code>), two news types
    /// were introduced: <see cref="DvText"/> which a sortable and comparable
    /// <see cref="ReadOnlyMemory{T}"/> and <see cref="VBufferEqSort{T}"/> to
    /// get a sortable and comparable <see cref="VBuffer{T}"/>.
    /// </summary>
    public class DataFrame : IDataFrameView
    {
        #region members

        DataContainer _data;
        bool _shuffle;

        /// <summary>
        /// Can shuffle the data.
        /// </summary>
        public bool CanShuffle => _shuffle;

        public int[] ALL { get { return null; } }

        public IDataFrameView Source => null;
        public int[] ColumnsSet => null;

        #endregion

        #region constructor

        /// <summary>
        /// Initializes an empty dataframe.
        /// </summary>
        /// <param name="shuffle">The dataframe can be shuffled.</param>
        public DataFrame(bool shuffle = true)
        {
            _data = new DataContainer();
            _shuffle = shuffle;
        }

        /// <summary>
        /// Initializes an empty dataframe.
        /// </summary>
        /// <param name="shuffle">The dataframe can be shuffled.</param>
        DataFrame(DataContainer data, bool shuffle)
        {
            _data = data;
            _shuffle = shuffle;
        }

        public void SetShuffle(bool shuffle)
        {
            _shuffle = shuffle;
        }

        /// <summary>
        /// Creates a dataframe from a list of dictionaries.
        /// If *kinds* is null, the function guesses the types from
        /// the first row.
        /// </summary>
        public DataFrame(IEnumerable<Dictionary<string, object>> rows,
                         Dictionary<string, ColumnType> kinds = null)
        {
            _data = new DataContainer(rows, kinds);
        }

        /// <summary>
        /// Creates a dataframe based on a schema.
        /// </summary>
        public DataFrame(Schema schema, int nb = 1)
        {
            _data = new DataContainer(schema, nb);
        }

        public bool CheckSharedSchema(Schema schema)
        {
            return _data.CheckSharedSchema(schema);
        }

        #endregion

        #region IDataView API

        /// <summary>
        /// Returns the number of rows. lazy is unused as the data is stored in memory.
        /// </summary>
        public long? GetRowCount()
        {
            return _data.Length;
        }

        public int Length => _data.Length;
        public int ColumnCount => _data.ColumnCount;
        public string[] Columns => _data.Columns;
        public ColumnType[] Kinds => _data.Kinds;

        public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            return _data.GetRowCursor(columnsNeeded, rand);
        }

        public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            return _data.GetRowCursorSet(columnsNeeded, n, rand);
        }

        public RowCursor GetRowCursor(int[] rows, int[] columns, IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            return _data.GetRowCursor(rows, columns, columnsNeeded, rand);
        }

        public RowCursor[] GetRowCursorSet(int[] rows, int[] columns, IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            return _data.GetRowCursorSet(rows, columns, columnsNeeded, n, rand);
        }

        /// <summary>
        /// Returns the schema of the dataframe, used schema used for IDataView.
        /// </summary>
        public Schema Schema => _data.Schema;
        public ISchema SchemaI => _data.SchemaI;

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        public DataFrame Copy()
        {
            var df = new DataFrame();
            df._data = _data.Copy();
            return df;
        }

        /// <summary>
        /// Resizes the dataframe.
        /// </summary>
        /// <param name="keepData">keeps existing data</param>
        /// <param name="length">new length</param>
        public void Resize(int length, bool keepData = false)
        {
            _data.Resize(length, keepData);
        }

        /// <summary>
        /// Returns the column index.
        /// </summary>
        public int GetColumnIndex(string name)
        {
            return SchemaHelper.GetColumnIndex(Schema, name);
        }

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        public DataFrame Copy(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            var df = new DataFrame();
            df._data = _data.Copy(rows, columns);
            return df;
        }

        #endregion

        #region DataFrame

        /// <summary>
        /// Returns the shape of the dataframe (number of rows, number of columns).
        /// </summary>
        public ShapeType Shape => _data.Shape;

        /// <summary>
        /// Adds a new column. The length must be specified for the first column.
        /// It must be the same for all columns.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="kind">column type</param>
        /// <param name="length">length is needed for the first column to allocated space</param>
        public int AddColumn(string name, ColumnType kind, int? length)
        {
            return _data.AddColumn(name, kind, length);
        }

        /// <summary>
        /// Adds a new column. The length must be specified for the first column.
        /// It must be the same for all columns.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="values">new column</param>
        public int AddColumn(string name, IDataColumn values)
        {
            return _data.AddColumn(name, values.Kind, values.Length, values);
        }

        public int AddColumn<DT>(string name, DT[] values)
        {
            var kind = SchemaHelper.GetColumnType<DT>();
            switch (kind.RawKind())
            {
                case DataKind.BL: return AddColumn(name, values as bool[]);
                case DataKind.I4: return AddColumn(name, values as int[]);
                case DataKind.I8: return AddColumn(name, values as long[]);
                case DataKind.U4: return AddColumn(name, values as uint[]);
                case DataKind.R4: return AddColumn(name, values as float[]);
                case DataKind.R8: return AddColumn(name, values as double[]);
                case DataKind.TX:
                    {
                        if (values as string[] != null)
                            return AddColumn(name, values as string[]);
                        if (values as DvText[] != null)
                            return AddColumn(name, values as DvText[]);
                        if (values as ReadOnlyMemory<char>[] != null)
                            return AddColumn(name, values as ReadOnlyMemory<char>[]);
                        throw Contracts.ExceptNotImpl($"Unable to add a column of type {typeof(DT)}.");
                    }
                default:
                    throw Contracts.ExceptNotImpl($"Unable to add a column of type {typeof(DT)}.");
            }
        }

        public int AddColumn(string name, bool[] values)
        {
            var buf = new bool[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = values[i];
            return AddColumn(name, new DataColumn<bool>(buf));
        }

        public int AddColumn(string name, bool[][] values)
        {
            var buf = new VBufferEqSort<bool>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<bool>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<bool>>(buf));
        }

        public int AddColumn(string name, int[] values)
        {
            var buf = new int[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = values[i];
            return AddColumn(name, new DataColumn<int>(buf));
        }

        public int AddColumn(string name, int[][] values)
        {
            var buf = new VBufferEqSort<int>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<int>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<int>>(buf));
        }

        public int AddColumn(string name, long[] values)
        {
            var buf = new Int64[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = values[i];
            return AddColumn(name, new DataColumn<long>(buf));
        }

        public int AddColumn(string name, uint[] values) { return AddColumn(name, new DataColumn<uint>(values)); }
        public int AddColumn(string name, float[] values) { return AddColumn(name, new DataColumn<float>(values)); }
        public int AddColumn(string name, double[] values) { return AddColumn(name, new DataColumn<double>(values)); }
        public int AddColumn(string name, DvText[] values) { return AddColumn(name, new DataColumn<DvText>(values)); }

        public int AddColumn(string name, uint[][] values)
        {
            var buf = new VBufferEqSort<uint>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<uint>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<uint>>(buf));
        }

        public int AddColumn(string name, float[][] values)
        {
            var buf = new VBufferEqSort<float>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<float>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<float>>(buf));
        }

        public int AddColumn(string name, double[][] values)
        {
            var buf = new VBufferEqSort<double>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<double>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<double>>(buf));
        }

        public int AddColumn(string name, long[][] values)
        {
            var buf = new VBufferEqSort<Int64>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<Int64>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<Int64>>(buf));
        }

        public int AddColumn(string name, DvText[][] values)
        {
            var buf = new VBufferEqSort<DvText>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<DvText>(values[i].Length, values[i]);
            return AddColumn(name, new DataColumn<VBufferEqSort<DvText>>(buf));
        }

        public int AddColumn(string name, string[] values)
        {
            var buf = new DvText[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new DvText(values[i]);
            return AddColumn(name, new DataColumn<DvText>(buf));
        }

        public int AddColumn(string name, string[][] values)
        {
            var buf = new VBufferEqSort<DvText>[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new VBufferEqSort<DvText>(values[i].Length, values[i].Select(c => new DvText(c)).ToArray());
            return AddColumn(name, new DataColumn<VBufferEqSort<DvText>>(buf));
        }

        public int AddColumn(string name, ReadOnlyMemory<char>[] values)
        {
            var buf = new DvText[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new DvText(values[i]);
            return AddColumn(name, new DataColumn<DvText>(buf));
        }

        public MultiGetterAt<MutableTuple<T1>> GetMultiGetterAt<T1>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            if (cols.Length != 1)
                throw new DataValueError($"Dimension mismatch expected 1 not {cols.Length}.");
            var g1 = GetColumn(cols[0]).GetGetterAt<T1>();
            return (int row, ref MutableTuple<T1> value) => { g1(row, ref value.Item1); };
        }

        public MultiGetterAt<MutableTuple<T1, T2>> GetMultiGetterAt<T1, T2>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            if (cols.Length != 2)
                throw new DataValueError($"Dimension mismatch expected 2 not {cols.Length}.");
            var g1 = GetColumn(cols[0]).GetGetterAt<T1>();
            var g2 = GetColumn(cols[1]).GetGetterAt<T2>();
            return (int row, ref MutableTuple<T1, T2> value) =>
            {
                g1(row, ref value.Item1);
                g2(row, ref value.Item2);
            };
        }

        public MultiGetterAt<MutableTuple<T1, T2, T3>> GetMultiGetterAt<T1, T2, T3>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            if (cols.Length != 3)
                throw new DataValueError($"Dimension mismatch expected 3 not {cols.Length}.");
            var g1 = GetColumn(cols[0]).GetGetterAt<T1>();
            var g2 = GetColumn(cols[1]).GetGetterAt<T2>();
            var g3 = GetColumn(cols[2]).GetGetterAt<T3>();
            return (int row, ref MutableTuple<T1, T2, T3> value) =>
            {
                g1(row, ref value.Item1);
                g2(row, ref value.Item2);
                g3(row, ref value.Item3);
            };
        }

        /// <summary>
        /// Raises an exception if two dataframes do not have the same
        /// shape or are two much different.
        /// </summary>
        /// <param name="df">dataframe</param>
        /// <param name="precision">precision</param>
        /// <param name="exc">raises an exception if too different</param>
        /// <returns>max difference</returns>
        public double AssertAlmostEqual(IDataFrameView df, double precision = 1e-5, bool exc = true)
        {
            if (Shape != df.Shape)
                throw new DataValueError(string.Format("Shapes are different ({0}, {1}) != ({2}, {3})",
                            Shape.Item1, Shape.Item2, df.Shape.Item1, df.Shape.Item2));
            double max = 0;
            for (int i = 0; i < df.Shape.Item2; ++i)
            {
                var c1 = GetColumn(i);
                var c2 = GetColumn(i);
                var d = c1.AssertAlmostEqual(c2, precision, exc);
                max = Math.Max(max, d);
            }
            return max;
        }

        #endregion

        #region IO

        /// <summary>
        /// Returns the name and the type of a column such as
        /// <pre>name:type:index</pre>.
        /// </summary>
        public string NameType(int col) { return _data.NameType(col); }

        /// <summary>
        /// Converts the data frame into a string.
        /// Every vector column is skipped.
        /// </summary>
        public override string ToString()
        {
            var df = HasVectorColumn() ? Flatten() : this;
            using (var stream = new MemoryStream())
            {
                DataFrameIO.ViewToCsv(df, stream, silent: true);
                stream.Position = 0;
                using (var reader = new StreamReader(stream))
                    return reader.ReadToEnd().Replace("\r", "").TrimEnd(new char[] { '\n' });
            }
        }

        public bool HasVectorColumn(IEnumerable<int> columns = null)
        {
            return _data.HasVectorColumn(columns);
        }

        /// <summary>
        /// Saves the dataframe as a file.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="encoding">encoding</param>
        /// <param name="silent">Suppress any info output (not warnings or errors)</param>
        public void ToCsv(string filename, string sep = ",", bool header = true,
                          Encoding encoding = null, bool silent = false, IHost host = null)
        {
            DataFrameIO.ViewToCsv(this, filename, sep: sep, header: header, encoding: encoding, silent: silent, host: host);
        }

        public void FillValues(IDataView view, int nrows = -1, bool keepVectors = false, int? numThreads = 1,
                               IHostEnvironment env = null)
        {
            _data.FillValues(view, nrows: nrows, keepVectors: keepVectors, numThreads: numThreads, env: env);
        }

        /// <summary>
        /// Changes the values for an entire row.
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="values">list of values</param>
        public void FillValues(int row, string[] values)
        {
            _data.FillValues(row, values);
        }

        public delegate void RowFillerDelegate(DataFrame df, int row);

        public static RowFillerDelegate GetRowFiller(RowCursor cur)
        {
            var dele = DataContainer.GetRowFiller(cur);
            return (DataFrame df, int row) => { dele(df._data, row); };
        }

        public DataFrame Flatten(IEnumerable<int> rows = null, IEnumerable<int> columns = null)
        {
            return new DataFrame(_data.Flatten(rows, columns), _shuffle);
        }

        #endregion

        #region comparison

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public static bool operator ==(DataFrame df1, DataFrame df2)
        {
            return df1._data == df2._data;
        }

        /// <summary>
        /// Exact difference between two dataframes.
        /// </summary>
        public static bool operator !=(DataFrame df1, DataFrame df2)
        {
            return df1._data != df2._data;
        }

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public bool Equals(DataFrame df)
        {
            return _data.Equals(df._data);
        }

        /// <summary>
        /// Approximated comparison between two dataframes.
        /// It returns 0 if the difference is below the precision
        /// or the difference otherwise, Inf if shapes or schema are different.
        /// </summary>
        public double AlmostEquals(DataFrame df, double precision = 1e-6f, bool exc = false, bool printDf = false)
        {
            if (exc && printDf)
            {
                try
                {
                    return _data.AlmostEquals(df._data, precision, exc);
                }
                catch (Exception e)
                {
                    var addition = $"----\n{ToString()}\n-----\n{df.ToString()}";
                    throw new Exception(addition, e);
                }
            }
            else
                return _data.AlmostEquals(df._data, precision, exc);
        }

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public bool Equals(IDataFrameView dfv)
        {
            return Equals(dfv.Copy());
        }

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public override bool Equals(object o)
        {
            var df = o as DataFrame;
            if (df == null)
                return false;
            return Equals(df);
        }

        /// <summary>
        /// Not implemented.
        /// </summary>
        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

        #endregion

        #region loc / iloc

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public Iloc iloc => new Iloc(this);

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public class Iloc
        {
            readonly DataFrame _parent;

            public Iloc(DataFrame parent)
            {
                _parent = parent;
            }

            DataContainer AsDataContainer()
            {
                var dc = _parent._data;
                if (dc == null)
                    throw new DataTypeError(string.Format("Unexpected container type '{0}'.", _parent._data.GetType()));
                return dc;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, int col]
            {
                get { return _parent._data[row, col]; }
                set { _parent._data[row, col] = value; }
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[IEnumerable<int> rows, int col]
            {
                get { return new DataFrameView(_parent, rows, new[] { col }); }
                set { _parent._data[rows, col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, int col]
            {
                get { return new DataFrameView(_parent, rows.Select((c, i) => c ? -1 : i).Where(c => c >= 0), new[] { col }); }
                set { _parent._data[rows, col] = value; }
            }
        }

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public Loc loc => new Loc(this);

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public class Loc
        {
            DataFrame _parent;

            public Loc(DataFrame parent)
            {
                _parent = parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, string col]
            {
                get { return _parent._data[row, col]; }
                set { _parent._data[row, col] = value; }
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[string col]
            {
                set { _parent._data[col].Set(value); }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, string col]
            {
                set { _parent._data[rows, col] = value; }
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[IEnumerable<int> rows, string col]
            {
                get
                {
                    int icol;
                    _parent.SchemaI.TryGetColumnIndex(col, out icol);
                    return new DataFrameView(_parent, rows, new[] { icol });
                }
                set { _parent._data[rows, col] = value; }
            }
        }

        #endregion

        #region operators []

        /// <summary>
        /// Returns all values in a row as a dictionary.
        /// </summary>
        public Dictionary<string, object> this[int row]
        {
            get { return _data[row]; }
        }

        /// <summary>
        /// Retrieves a column by its name.
        /// </summary>
        public NumericColumn GetColumn(string colname, int[] rows = null)
        {
            return new NumericColumn(_data.GetColumn(colname, rows));
        }

        /// <summary>
        /// Retrieves a column by its position.
        /// </summary>
        public NumericColumn GetColumn(int col, int[] rows = null)
        {
            return new NumericColumn(_data.GetColumn(col, rows));
        }

        /// <summary>
        /// Retrieves a typed column.
        /// </summary>
        public void GetTypedColumn<DType>(int col, out DataColumn<DType> column, int[] rows = null)
            where DType : IEquatable<DType>, IComparable<DType>
        {
            _data.GetTypedColumn(col, out column, rows);
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public NumericColumn this[string colname]
        {
            get { return GetColumn(colname); }
            set { AddColumn(colname, value); }
        }

        /// <summary>
        /// Returns a list of columns.
        /// </summary>
        public DataFrameView this[IEnumerable<string> colNames]
        {
            get { return new DataFrameView(this, null, colNames.Select(c => _data.GetColumnIndex(c))); }
        }

        /// <summary>
        /// Returns a subset of rows.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), null); }
        }

        /// <summary>
        /// Returns a subset of rows.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows]
        {
            get { return new DataFrameView(this, rows, null); }
        }

        /// <summary>
        /// Returns a subset of rows.
        /// </summary>
        public DataFrameView this[NumericColumn boolCol]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(boolCol), null); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, int colname]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), new[] { colname }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, IEnumerable<int> colnames]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), colnames); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, int colname]
        {
            get { return new DataFrameView(this, rows, new[] { colname }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, IEnumerable<int> colnames]
        {
            get { return new DataFrameView(this, rows, colnames); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, string colname]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), new[] { _data.GetColumnIndex(colname) }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, IEnumerable<string> colnames]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), colnames.Select(c => _data.GetColumnIndex(c))); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, string colname]
        {
            get { return new DataFrameView(this, rows, new[] { _data.GetColumnIndex(colname) }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, IEnumerable<string> colnames]
        {
            get { return new DataFrameView(this, rows, colnames.Select(c => _data.GetColumnIndex(c))); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[NumericColumn rows, string colName]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), new[] { _data.GetColumnIndex(colName) }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[NumericColumn rows, IEnumerable<string> colNames]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), colNames.Select(c => _data.GetColumnIndex(c))); }
        }

        /// <summary>
        /// Drops some columns.
        /// Data is not copied.
        /// </summary>
        public DataFrameView Drop(IEnumerable<string> colNames)
        {
            var idrop = new HashSet<int>(colNames.Select(c => _data.GetColumnIndex(c)));
            var ikeep = Enumerable.Range(0, ColumnCount).Where(c => !idrop.Contains(c));
            return new DataFrameView(this, null, ikeep);
        }

        public IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            return EnumerateItems<T1>(columns.Select(c => GetColumnIndex(c)), ascending, rows);
        }

        public IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            return EnumerateItems<T1, T2>(columns.Select(c => GetColumnIndex(c)), ascending, rows);
        }

        public IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            return EnumerateItems<T1, T2, T3>(columns.Select(c => GetColumnIndex(c)), ascending, rows);
        }

        public IEnumerable<TValue> EnumerateItems<TValue>(int[] columns, bool ascending, IEnumerable<int> rows,
                                                          MultiGetterAt<TValue> getter)
            where TValue : ITUple, new()
        {
            var value = new TValue();
            var cols = columns.ToArray();
            if (cols.Length != value.Length)
                throw new DataTypeError($"Dimension mismatch between {cols.Length} and {cols.Length}.");
            for (int i = 0; i < Length; ++i)
            {
                getter(i, ref value);
                yield return value;
            }
        }

        public IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var cols = columns.ToArray();
            return EnumerateItems(cols, ascending, rows, GetMultiGetterAt<T1>(cols));
        }

        public IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var cols = columns.ToArray();
            return EnumerateItems(cols, ascending, rows, GetMultiGetterAt<T1, T2>(cols));
        }

        public IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var cols = columns.ToArray();
            return EnumerateItems(cols, ascending, rows, GetMultiGetterAt<T1, T2, T3>(cols));
        }

        #endregion

        #region SQL functions

        #region head, tail, sample

        /// <summary>
        /// Returns a view on the first rows.
        /// </summary>
        public IDataFrameView Head(int nrows = 5)
        {
            nrows = Math.Min(Length, nrows);
            return new DataFrameView(this, Enumerable.Range(0, nrows).ToArray(), null);
        }

        /// <summary>
        /// Returns a view on the last rows.
        /// </summary>
        public IDataFrameView Tail(int nrows = 5)
        {
            nrows = Math.Min(Length, nrows);
            return new DataFrameView(this, Enumerable.Range(0, nrows).Select(c => c + Length - nrows).ToArray(), null);
        }

        /// <summary>
        /// Returns a sample.
        /// </summary>
        public IDataFrameView Sample(int nrows = 5, bool distinct = false, Random rand = null)
        {
            nrows = Math.Min(Length, nrows);
            return new DataFrameView(this, DataFrameRandom.RandomIntegers(nrows, Length, distinct, rand), null);
        }

        #endregion

        #region sort

        /// <summary>
        /// Order the rows.
        /// </summary>
        public void Order(int[] order)
        {
            _data.Order(order);
        }

        /// <summary>
        /// Reorder the columns. Every view based on it will be impacted.
        /// </summary>
        public void OrderColumns(string[] columns)
        {
            _data.OrderColumns(columns);
        }

        public void RenameColumns(string[] columns)
        {
            _data.RenameColumns(columns);
        }

        /// <summary>
        /// Sorts rows.
        /// </summary>
        public void Sort(IEnumerable<string> columns, bool ascending = true)
        {
            DataFrameSorting.Sort(this, columns.Select(c => GetColumnIndex(c)), ascending);
        }

        /// <summary>
        /// Sorts rows. If <i>columns </i> is null, it is replaced by the first
        /// <see cref="DataFrameSorting.LimitNumberSortingColumns"/> columns.
        /// </summary>
        public void Sort(IEnumerable<int> columns = null, bool ascending = true)
        {
            if (columns == null)
                columns = Enumerable.Range(0, Math.Min(ColumnCount, DataFrameSorting.LimitNumberSortingColumns));
            DataFrameSorting.Sort(this, columns, ascending);
        }

        #endregion

        #region typed sort

        public void TSort<T1>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            int[] order = null;
            DataFrameSorting.TSort<T1>(this, ref order, columns, ascending);
            Order(order);
        }

        public void TSort<T1, T2>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            int[] order = null;
            DataFrameSorting.TSort<T1, T2>(this, ref order, columns, ascending);
            Order(order);
        }

        public void TSort<T1, T2, T3>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            int[] order = null;
            DataFrameSorting.TSort<T1, T2, T3>(this, ref order, columns, ascending);
            Order(order);
        }

        #endregion

        #region aggregate

        /// <summary>
        /// Aggregates over all rows.
        /// </summary>
        public DataFrame Aggregate(AggregatedFunction func, int[] rows = null, int[] columns = null)
        {
            return new DataFrame(_data.Aggregate(func, rows, columns), _shuffle);
        }

        /// <summary>
        /// Sum over all rows.
        /// </summary>
        public DataFrame Sum()
        {
            return Aggregate(AggregatedFunction.Sum);
        }

        /// <summary>
        /// Min over all rows.
        /// </summary>
        public DataFrame Min()
        {
            return Aggregate(AggregatedFunction.Min);
        }

        /// <summary>
        /// Max over all rows.
        /// </summary>
        public DataFrame Max()
        {
            return Aggregate(AggregatedFunction.Max);
        }

        /// <summary>
        /// Average over all rows.
        /// </summary>
        public DataFrame Mean()
        {
            return Aggregate(AggregatedFunction.Mean);
        }

        /// <summary>
        /// Average over all rows.
        /// </summary>
        public DataFrame Count()
        {
            return Aggregate(AggregatedFunction.Count);
        }

        #endregion

        #region concat

        /// <summary>
        /// Concatenates many dataframes.
        /// </summary>
        public static DataFrame Concat(IEnumerable<IDataFrameView> views)
        {
            var arr = views.ToArray();
            var unique = new HashSet<string>();
            var ordered = new List<string>();
            foreach (var df in arr)
            {
                for (int i = 0; i < df.ColumnCount; ++i)
                {
                    var c = df.SchemaI.GetColumnName(i);
                    if (!unique.Contains(c))
                    {
                        unique.Add(c);
                        ordered.Add(c);
                    }
                }
            }

            var res = new DataFrame(arr.All(c => c.CanShuffle));
            int index;
            foreach (var col in ordered)
            {
                var conc = new List<IDataColumn>();
                var first = arr.Where(df => df.SchemaI.TryGetColumnIndex(col, out index))
                               .Select(df => df.GetColumn(col))
                               .First();
                foreach (var df in arr)
                {
                    if (!df.SchemaI.TryGetColumnIndex(col, out index))
                        conc.Add(first.Create(df.Length, true));
                    else
                        conc.Add(df.GetColumn(col));
                }
                var concCol = first.Concat(conc);
                res.AddColumn(col, concCol);
            }
            return res;
        }

        #endregion

        #region groupby

        /// <summary>
        /// Groupby.
        /// </summary>
        public IDataFrameViewGroupResults GroupBy(IEnumerable<string> cols, bool sort = true)
        {
            return new DataFrameView(this, null, null).GroupBy(cols, sort);
        }

        /// <summary>
        /// Groupby.
        /// </summary>
        public IDataFrameViewGroupResults GroupBy(IEnumerable<int> cols, bool sort = true)
        {
            return new DataFrameView(this, null, null).GroupBy(cols, sort);
        }

        public DataFrameViewGroupResults<ImmutableTuple<T1>> TGroupBy<T1>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            return new DataFrameView(this, null, null).TGroupBy<T1>(cols, sort);
        }

        public DataFrameViewGroupResults<ImmutableTuple<T1, T2>> TGroupBy<T1, T2>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            return new DataFrameView(this, null, null).TGroupBy<T1, T2>(cols, sort);
        }

        public DataFrameViewGroupResults<ImmutableTuple<T1, T2, T3>> TGroupBy<T1, T2, T3>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            return new DataFrameView(this, null, null).TGroupBy<T1, T2, T3>(cols, sort);
        }

        #endregion

        #region join

        public IDataFrameView Multiply(int nb, MultiplyStrategy multType = MultiplyStrategy.Block)
        {
            int[] rows = new int[Length * nb];
            switch (multType)
            {
                case MultiplyStrategy.Block:
                    for (int i = 0; i < rows.Length; ++i)
                        rows[i] = i % Length;
                    break;
                case MultiplyStrategy.Row:
                    for (int i = 0; i < rows.Length; ++i)
                        rows[i] = i / nb;
                    break;
                default:
                    throw new DataValueError($"Unkown multiplication strategy '{multType}'.");
            }
            return new DataFrameView(this, rows, null);
        }

        /// <summary>
        /// Join.
        /// </summary>
        public DataFrame Join(IDataFrameView right, IEnumerable<string> colsLeft, IEnumerable<string> colsRight,
                        string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return new DataFrameView(this, null, null).Join(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame Join(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<string> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return new DataFrameView(this, null, null).Join(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame Join(IDataFrameView right, IEnumerable<string> colsLeft, IEnumerable<int> colsRight,
                            string leftSuffix = null, string rightSuffix = null,
                           JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return new DataFrameView(this, null, null).Join(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame Join(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight,
                        string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return new DataFrameView(this, null, null).Join(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame TJoin<T1>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            return new DataFrameView(this, null, null).TJoin<T1>(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame TJoin<T1, T2>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            return new DataFrameView(this, null, null).TJoin<T1, T2>(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame TJoin<T1, T2, T3>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            return new DataFrameView(this, null, null).TJoin<T1, T2, T3>(right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        #endregion

        #endregion
    }
}
