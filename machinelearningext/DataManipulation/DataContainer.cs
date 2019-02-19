// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Contains data.
    /// </summary>
    public class DataContainer : IEquatable<DataContainer>
    {
        #region All possible types to hold data.

        List<string> _names;
        List<DataViewType> _kinds;
        int _length;
        Dictionary<string, int> _naming;
        Dictionary<int, Tuple<DataViewType, int>> _mapping;
        ISchema _schema;

        List<IDataColumn> _colsBL;
        List<IDataColumn> _colsI4;
        List<IDataColumn> _colsU4;
        List<IDataColumn> _colsI8;
        List<IDataColumn> _colsR4;
        List<IDataColumn> _colsR8;
        List<IDataColumn> _colsTX;

        List<IDataColumn> _colsABL;
        List<IDataColumn> _colsAI4;
        List<IDataColumn> _colsAU4;
        List<IDataColumn> _colsAI8;
        List<IDataColumn> _colsAR4;
        List<IDataColumn> _colsAR8;
        List<IDataColumn> _colsATX;

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public DataContainer Copy()
        {
            var dc = new DataContainer();
            dc._names = new List<string>(_names);
            dc._kinds = new List<DataViewType>(_kinds);
            dc._length = _length;
            dc._naming = new Dictionary<string, int>(_naming);
            dc._mapping = new Dictionary<int, Tuple<DataViewType, int>>(_mapping);
            dc._schema = new DataContainerSchema(dc);

            dc._colsBL = _colsBL == null ? null : new List<IDataColumn>(_colsBL.Select(c => c.Copy()));
            dc._colsI4 = _colsI4 == null ? null : new List<IDataColumn>(_colsI4.Select(c => c.Copy()));
            dc._colsU4 = _colsU4 == null ? null : new List<IDataColumn>(_colsU4.Select(c => c.Copy()));
            dc._colsI8 = _colsI8 == null ? null : new List<IDataColumn>(_colsI8.Select(c => c.Copy()));
            dc._colsR4 = _colsR4 == null ? null : new List<IDataColumn>(_colsR4.Select(c => c.Copy()));
            dc._colsR8 = _colsR8 == null ? null : new List<IDataColumn>(_colsR8.Select(c => c.Copy()));
            dc._colsTX = _colsTX == null ? null : new List<IDataColumn>(_colsTX.Select(c => c.Copy()));

            dc._colsABL = _colsABL == null ? null : new List<IDataColumn>(_colsABL.Select(c => c.Copy()));
            dc._colsAI4 = _colsAI4 == null ? null : new List<IDataColumn>(_colsAI4.Select(c => c.Copy()));
            dc._colsAU4 = _colsAU4 == null ? null : new List<IDataColumn>(_colsAU4.Select(c => c.Copy()));
            dc._colsAI8 = _colsAI8 == null ? null : new List<IDataColumn>(_colsAI8.Select(c => c.Copy()));
            dc._colsAR4 = _colsAR4 == null ? null : new List<IDataColumn>(_colsAR4.Select(c => c.Copy()));
            dc._colsAR8 = _colsAR8 == null ? null : new List<IDataColumn>(_colsAR8.Select(c => c.Copy()));
            dc._colsATX = _colsATX == null ? null : new List<IDataColumn>(_colsATX.Select(c => c.Copy()));

            return dc;
        }

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public DataContainer Copy(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            var arows = rows.ToArray();
            var dc = new DataContainer();
            foreach (var c in columns)
                dc.AddColumn(_names[c], _kinds[c], arows.Length, GetColumn(c).Copy(arows));
            return dc;
        }

        /// <summary>
        /// Converts a filter into a list of row indices.
        /// </summary>
        public IEnumerable<int> EnumerateRowsIndex(IEnumerable<bool> filter)
        {
            int row = 0;
            foreach (var b in filter)
            {
                if (b)
                    yield return row;
                ++row;
            }
        }

        /// <summary>
        /// Converts a filter into a list of row indices.
        /// </summary>
        public IEnumerable<int> EnumerateRowsIndex(NumericColumn filter)
        {
            if (filter.Kind.IsVector() || filter.Kind.RawKind() != DataKind.Bool)
                throw Contracts.ExceptNotSupp("Only boolean column are allowed for operator [].");
            var th = filter.Column as DataColumn<bool>;
            if (th == null)
                throw Contracts.ExceptNotSupp("filter is not a boolean column.");
            int row = 0;
            foreach (var b in th.Data)
            {
                if ((bool)b)
                    yield return row;
                ++row;
            }
        }

        /// <summary>
        /// Returns the dimension of the container.
        /// </summary>
        public ShapeType Shape => new ShapeType(Length, _names == null ? 0 : _names.Count);

        /// <summary>
        /// Returns the name and the type of a column such as
        /// <pre>name:type:index</pre>.
        /// </summary>
        public string NameType(int col) { return string.Format("{0}:{1}:{2}", _names[col], _kinds[col], col); }

        /// <summary>
        /// Returns the number of rows.
        /// </summary>
        public int Length => _length;

        /// <summary>
        /// Returns the number of columns.
        /// </summary>
        public int ColumnCount => _names.Count;

        /// <summary>
        /// Returns the list of columns.
        /// </summary>
        public string[] Columns => _names.ToArray();

        /// <summary>
        /// Returns the list of columns.
        /// </summary>
        public DataViewType[] Kinds => _kinds.ToArray();

        /// <summary>
        /// Returns the name of a column.
        /// </summary>
        public string GetColumnName(int col) { return _names[col]; }

        /// <summary>
        /// Returns the position of a column.
        /// </summary>
        public int GetColumnIndex(string name) { return _naming[name]; }

        /// <summary>
        /// Returns the type of a column.
        /// </summary>
        public DataViewType GetDType(int col) { return _kinds[col]; }

        /// <summary>
        /// Returns the number of rows actually allocated.
        /// </summary>
        public int MemoryLength
        {
            get
            {
                if (ColumnCount == 0)
                    return 0;
                else
                    return GetColumn(0).MemoryLength;
            }
        }

        /// <summary>
        /// Resizes all columns.
        /// </summary>
        /// <param name="keepData">keeps existing data</param>
        /// <param name="length">new length</param>
        public void Resize(int length, bool keepData = false)
        {
            if (length <= _length)
                _length = length;
            else if (length <= MemoryLength)
                _length = length;
            else
                for (int i = 0; i < ColumnCount; ++i)
                    GetColumn(i).Resize(length, keepData);
        }

        /// <summary>
        /// Returns a typed container of column col.
        /// </summary>
        public void GetTypedColumn<DType>(int col, out DataColumn<DType> column, int[] rows = null)
            where DType : IEquatable<DType>, IComparable<DType>
        {
            var coor = _mapping[col];
            DataColumn<DType> found = null;
            if (coor.Item1.IsVector())
            {
                switch (coor.Item1.ItemType().RawKind())
                {
                    case DataKind.BL: found = _colsABL[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.I4: found = _colsAI4[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.U4: found = _colsAU4[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.I8: found = _colsAI8[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.R4: found = _colsAR4[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.R8: found = _colsAR8[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.TX: found = _colsATX[coor.Item2] as DataColumn<DType>; break;
                    default:
                        throw new DataTypeError(string.Format("Type '{0}' is not handled.", coor.Item1));
                }
            }
            else
            {
                switch (coor.Item1.RawKind())
                {
                    case DataKind.BL: found = _colsBL[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.I4: found = _colsI4[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.U4: found = _colsU4[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.I8: found = _colsI8[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.R4: found = _colsR4[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.R8: found = _colsR8[coor.Item2] as DataColumn<DType>; break;
                    case DataKind.TX: found = _colsTX[coor.Item2] as DataColumn<DType>; break;
                    default:
                        throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
                }
            }
            if (found == null)
                throw new DataTypeError(string.Format("Column {0} is not of type {1} (kind={2}, c={3})",
                                col, typeof(DType), coor.Item1, coor.Item2));
            if (rows != null)
            {
                // TODO: This is not absolutely efficient as it copies the data.
                found = found.Copy(rows) as DataColumn<DType>;
            }
            column = found;
        }

        /// <summary>
        /// Returns the container of column col as an interface.
        /// </summary>
        public IDataColumn GetColumn(string colname, int[] rows = null)
        {
            return GetColumn(_naming[colname], rows);
        }

        /// <summary>
        /// Returns the container of column col as an interface.
        /// </summary>
        public IDataColumn GetColumn(int col, int[] rows = null)
        {
            var coor = _mapping[col];
            if (coor.Item1.IsVector())
            {
                switch (coor.Item1.ItemType().RawKind())
                {
                    case DataKind.BL: DataColumn<VBufferEqSort<bool>> objbl; GetTypedColumn(col, out objbl, rows); return objbl;
                    case DataKind.I4: DataColumn<VBufferEqSort<int>> obji4; GetTypedColumn(col, out obji4, rows); return obji4;
                    case DataKind.U4: DataColumn<VBufferEqSort<uint>> obju4; GetTypedColumn(col, out obju4, rows); return obju4;
                    case DataKind.I8: DataColumn<VBufferEqSort<long>> obji8; GetTypedColumn(col, out obji8, rows); return obji8;
                    case DataKind.R4: DataColumn<VBufferEqSort<float>> objf; GetTypedColumn(col, out objf, rows); return objf;
                    case DataKind.R8: DataColumn<VBufferEqSort<double>> objd; GetTypedColumn(col, out objd, rows); return objd;
                    case DataKind.TX: DataColumn<VBufferEqSort<DvText>> objs; GetTypedColumn(col, out objs, rows); return objs;
                    default:
                        throw new DataTypeError(string.Format("Type '{0}' is not handled.", coor.Item1));
                }
            }
            else
            {
                switch (coor.Item1.RawKind())
                {
                    case DataKind.BL: DataColumn<bool> objbl; GetTypedColumn(col, out objbl, rows); return objbl;
                    case DataKind.I4: DataColumn<int> obji4; GetTypedColumn(col, out obji4, rows); return obji4;
                    case DataKind.U4: DataColumn<uint> obju4; GetTypedColumn(col, out obju4, rows); return obju4;
                    case DataKind.I8: DataColumn<long> obji8; GetTypedColumn(col, out obji8, rows); return obji8;
                    case DataKind.R4: DataColumn<float> objf; GetTypedColumn(col, out objf, rows); return objf;
                    case DataKind.R8: DataColumn<double> objd; GetTypedColumn(col, out objd, rows); return objd;
                    case DataKind.TX: DataColumn<DvText> objs; GetTypedColumn(col, out objs, rows); return objs;
                    default:
                        throw new DataTypeError(string.Format("Type '{0}' is not handled.", coor.Item1));
                }
            }
        }

        #endregion

        #region constructor

        /// <summary>
        /// Data Container.
        /// </summary>
        public DataContainer()
        {
            _init();
        }

        /// <summary>
        /// Initializes the class as empty.
        /// </summary>
        void _init()
        {
            _length = 0;
            _names = null;
            _kinds = null;
            _colsBL = null;
            _colsI4 = null;
            _colsU4 = null;
            _colsI8 = null;
            _colsR4 = null;
            _colsR8 = null;
            _colsTX = null;
            _colsABL = null;
            _colsAI4 = null;
            _colsAU4 = null;
            _colsAI8 = null;
            _colsAR4 = null;
            _colsAR8 = null;
            _colsATX = null;
            _mapping = new Dictionary<int, Tuple<DataViewType, int>>();
            _naming = new Dictionary<string, int>();
            _schema = new DataContainerSchema(this);
            _schemaCache = null;
            _lock = new object();
        }

        /// <summary>
        /// Creates a dataframe from a list of dictionaries.
        /// If *kinds* is null, the function guesses the types from
        /// the first row.
        /// </summary>
        public DataContainer(IEnumerable<Dictionary<string, object>> rows,
                             Dictionary<string, DataViewType> kinds = null)
        {
            _init();
            var array = rows.ToArray();
            if (kinds == null)
                kinds = GuessKinds(array);
            foreach (var pair in kinds)
            {
                var values = array.Select(c => c.ContainsKey(pair.Key)
                                                ? c[pair.Key]
                                                : DataFrameMissingValue.GetMissingOrDefaultValue(pair.Value, array.Where(d => d.ContainsKey(pair.Key))
                                                                       .Select(e => e[pair.Key]).First()))
                                                                       .ToArray();
                var data = CreateColumn(pair.Value, values);
                AddColumn(pair.Key, pair.Value, array.Length, data);
            }
        }

        public DataContainer(DataViewSchema schema, int nb = 1)
        {
            _init();
            for (int i = 0; i < schema.Count; ++i)
            {
                var name = schema[i].Name;
                var type = schema[i].Type;
                AddColumn(name, type, nb);
            }
        }

        public bool CheckSharedSchema(DataViewSchema schema)
        {
            if (schema.Count != ColumnCount)
                throw Contracts.Except($"Different number of columns ${ColumnCount} != ${schema.Count}.");
            string name;
            DataViewType colType;
            for (int i = 0; i < ColumnCount; ++i)
            {
                name = schema[i].Name;
                if (name != _names[i])
                    throw Contracts.Except($"Different column name at position ${i}: ${_names[i]} != ${name}.");
                colType = schema[i].Type;
                if (colType != _kinds[i])
                    throw Contracts.Except($"Different column type at position ${i}: ${_kinds[i]} != ${colType}.");
            }
            return true;
        }

        Dictionary<string, DataViewType> GuessKinds(Dictionary<string, object>[] rows)
        {
            var res = new Dictionary<string, DataViewType>();
            foreach (var row in rows.Take(10))
            {
                foreach (var pair in row)
                {
                    if (res.ContainsKey(pair.Key))
                        continue;
                    if (pair.Value == null)
                        continue;
                    if (pair.Value is bool || pair.Value is bool)
                    {
                        res[pair.Key] = BooleanDataViewType.Instance;
                        continue;
                    }
                    if (pair.Value is int || pair.Value is int)
                    {
                        res[pair.Key] = NumberDataViewType.Int32;
                        continue;
                    }
                    if (pair.Value is uint)
                    {
                        res[pair.Key] = NumberDataViewType.UInt32;
                        continue;
                    }
                    if (pair.Value is long || pair.Value is Int64)
                    {
                        res[pair.Key] = NumberDataViewType.Int64;
                        continue;
                    }
                    if (pair.Value is float)
                    {
                        res[pair.Key] = NumberDataViewType.Single;
                        continue;
                    }
                    if (pair.Value is double)
                    {
                        res[pair.Key] = NumberDataViewType.Double;
                        continue;
                    }
                    if (pair.Value is ReadOnlyMemory<char> || pair.Value is string || pair.Value is DvText)
                    {
                        res[pair.Key] = TextDataViewType.Instance;
                        continue;
                    }
                    throw Contracts.ExceptNotImpl($"Type '{pair.Value.GetType()}' is not implemented.");
                }
            }
            return res;
        }

        IDataColumn CreateColumn(DataViewType kind, IEnumerable<object> values)
        {
            if (kind.IsVector())
                throw new NotImplementedException();
            switch (kind.RawKind())
            {
                case DataKind.BL:
                    try
                    {
                        return new DataColumn<bool>(values.Select(c => (bool)c).ToArray());
                    }
                    catch (InvalidCastException)
                    {
                        return new DataColumn<bool>(values.Select(c => (bool)(bool)c).ToArray());
                    }
                case DataKind.I4:
                    try
                    {
                        return new DataColumn<int>(values.Select(c => (int)c).ToArray());
                    }
                    catch (InvalidCastException)
                    {
                        return new DataColumn<int>(values.Select(c => (int)(int)c).ToArray());
                    }
                case DataKind.U4:
                    return new DataColumn<uint>(values.Select(c => (uint)c).ToArray());
                case DataKind.I8:
                    try
                    {
                        return new DataColumn<long>(values.Select(c => (long)c).ToArray());
                    }
                    catch (InvalidCastException)
                    {
                        return new DataColumn<long>(values.Select(c => (long)(Int64)c).ToArray());
                    }
                case DataKind.R4:
                    return new DataColumn<float>(values.Select(c => (float)c).ToArray());
                case DataKind.R8:
                    return new DataColumn<double>(values.Select(c => (double)c).ToArray());
                case DataKind.TX:
                    try
                    {
                        return new DataColumn<DvText>(values.Select(c => (DvText)c).ToArray());
                    }
                    catch (InvalidCastException)
                    {
                        return new DataColumn<DvText>(values.Select(c => new DvText((string)c)).ToArray());
                    }
                default:
                    throw Contracts.ExceptNotImpl($"Kind {kind} is not implemented.");
            }
        }

        #endregion

        #region column

        /// <summary>
        /// Adds a new column.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="kind">column type</param>
        /// <param name="length">changes the length</param>
        /// <param name="values">values (can be null)</param>
        public int AddColumn(string name, DataViewType kind, int? length, IDataColumn values = null)
        {
            if (_naming.ContainsKey(name))
            {
                // Works as replacement.
                var column = GetColumn(name);
                if (column.Kind == kind ||
                    (kind.IsKey() && !kind.IsVector() && !column.Kind.IsVector() && column.Kind.RawKind() == kind.RawKind()))
                {
                    if (values == null)
                        column.SetDefault();
                    else
                        column.Set(values);
                    return _naming[name];
                }
                else if (column.Kind.IsVector() && kind.IsVector() && column.Kind.ItemType().RawKind() == kind.ItemType().RawKind())
                {
                    if (values == null)
                        column.SetDefault();
                    else
                        column.Set(values);
                    return _naming[name];
                }
                else
                    throw new DataNameError(string.Format("Column '{0}' already exists but types are different {1} != {2}", name, column.Kind, kind));
            }
            if (values != null && ((object)(values as NumericColumn) != null))
                values = (values as NumericColumn).Column;
            if (_names == null)
                _names = new List<string>();
            if (_kinds == null)
                _kinds = new List<DataViewType>();
            if (length.HasValue && Length > 0 && length.Value != Length)
                throw new DataTypeError(string.Format("Length mismatch, expected {0} got {1}", Length, length.Value));

            int last = _names.Count;
            _names.Add(name);
            _kinds.Add(kind);
            int pos = -1;
            int nb = length ?? Length;
            _length = nb;
            if (kind.IsVector())
            {
                switch (kind.ItemType().RawKind())
                {
                    case DataKind.BL:
                        if (_colsABL == null)
                            _colsABL = new List<IDataColumn>();
                        pos = _colsABL.Count;
                        _colsABL.Add(values ?? new DataColumn<VBufferEqSort<bool>>(nb));
                        break;
                    case DataKind.I4:
                        if (_colsAI4 == null)
                            _colsAI4 = new List<IDataColumn>();
                        pos = _colsAI4.Count;
                        _colsAI4.Add(values ?? new DataColumn<VBufferEqSort<int>>(nb));
                        break;
                    case DataKind.U4:
                        if (_colsAU4 == null)
                            _colsAU4 = new List<IDataColumn>();
                        pos = _colsAU4.Count;
                        _colsAU4.Add(values ?? new DataColumn<VBufferEqSort<uint>>(nb));
                        break;
                    case DataKind.I8:
                        if (_colsAI8 == null)
                            _colsAI8 = new List<IDataColumn>();
                        pos = _colsAI8.Count;
                        _colsAI8.Add(values ?? new DataColumn<VBufferEqSort<long>>(nb));
                        break;
                    case DataKind.R4:
                        if (_colsAR4 == null)
                            _colsAR4 = new List<IDataColumn>();
                        pos = _colsAR4.Count;
                        _colsAR4.Add(values ?? new DataColumn<VBufferEqSort<float>>(nb));
                        break;
                    case DataKind.R8:
                        if (_colsAR8 == null)
                            _colsAR8 = new List<IDataColumn>();
                        pos = _colsAR8.Count;
                        _colsAR8.Add(values ?? new DataColumn<VBufferEqSort<double>>(nb));
                        break;
                    case DataKind.TX:
                        if (_colsATX == null)
                            _colsATX = new List<IDataColumn>();
                        pos = _colsATX.Count;
                        _colsATX.Add(values ?? new DataColumn<VBufferEqSort<DvText>>(nb));
                        break;
                    default:
                        throw new DataTypeError(string.Format("Type {0} is not handled.", kind.ItemType()));
                }
            }
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.BL:
                        if (_colsBL == null)
                            _colsBL = new List<IDataColumn>();
                        pos = _colsBL.Count;
                        _colsBL.Add(values ?? new DataColumn<bool>(nb));
                        break;
                    case DataKind.I4:
                        if (_colsI4 == null)
                            _colsI4 = new List<IDataColumn>();
                        pos = _colsI4.Count;
                        _colsI4.Add(values ?? new DataColumn<int>(nb));
                        break;
                    case DataKind.U4:
                        if (_colsU4 == null)
                            _colsU4 = new List<IDataColumn>();
                        pos = _colsU4.Count;
                        _colsU4.Add(values ?? new DataColumn<uint>(nb));
                        break;
                    case DataKind.I8:
                        if (_colsI8 == null)
                            _colsI8 = new List<IDataColumn>();
                        pos = _colsI8.Count;
                        _colsI8.Add(values ?? new DataColumn<long>(nb));
                        break;
                    case DataKind.R4:
                        if (_colsR4 == null)
                            _colsR4 = new List<IDataColumn>();
                        pos = _colsR4.Count;
                        _colsR4.Add(values ?? new DataColumn<float>(nb));
                        break;
                    case DataKind.R8:
                        if (_colsR8 == null)
                            _colsR8 = new List<IDataColumn>();
                        pos = _colsR8.Count;
                        _colsR8.Add(values ?? new DataColumn<double>(nb));
                        break;
                    case DataKind.TX:
                        if (_colsTX == null)
                            _colsTX = new List<IDataColumn>();
                        pos = _colsTX.Count;
                        _colsTX.Add(values ?? new DataColumn<DvText>(nb));
                        break;
                    default:
                        throw new DataTypeError(string.Format("Type {0} is not handled.", kind));
                }
            }
            _mapping[last] = new Tuple<DataViewType, int>(kind, pos);
            _naming[name] = last;
            return last;
        }

        /// <summary>
        /// Adds all columns in a container.
        /// </summary>
        public void AddColumn(DataContainer dc)
        {
            for (int i = 0; i < dc.ColumnCount; ++i)
                AddColumn(GetColumnName(i), GetDType(i), dc.Length, GetColumn(i));
        }

        /// <summary>
        /// Adds all columns in a container.
        /// </summary>
        public void AddColumn(IDataFrameView dc)
        {
            var names = dc.Columns;
            var kinds = dc.Kinds;
            for (int i = 0; i < dc.ColumnCount; ++i)
                AddColumn(names[i], kinds[i], dc.Length, dc.GetColumn(i));
        }

        /// <summary>
        /// Sets values for a row.
        /// </summary>
        /// <param name="row">row to fill</param>
        /// <param name="spl">value as strings, they will be converted</param>
        public void FillValues(int row, string[] spl)
        {
            for (int i = 0; i < spl.Length; ++i)
                Set(row, i, spl[i]);
        }

        /// <summary>
        /// Changes a value at a specific location.
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="col">column</param>
        /// <param name="value">value</param>
        public void Set(int row, int col, string value)
        {
            var coor = _mapping[col];
            if (_kinds[col].IsVector())
                throw new NotImplementedException();
            else
            {
                switch (_kinds[col].RawKind())
                {
                    case DataKind.BL: _colsBL[coor.Item2].Set(row, (bool)bool.Parse(value)); break;
                    case DataKind.I4: _colsI4[coor.Item2].Set(row, (int)int.Parse(value)); break;
                    case DataKind.U4: _colsU4[coor.Item2].Set(row, uint.Parse(value)); break;
                    case DataKind.I8: _colsI8[coor.Item2].Set(row, (long)long.Parse(value)); break;
                    case DataKind.R4: _colsR4[coor.Item2].Set(row, float.Parse(value)); break;
                    case DataKind.R8: _colsR8[coor.Item2].Set(row, double.Parse(value)); break;
                    case DataKind.TX: _colsTX[coor.Item2].Set(row, new DvText(value)); break;
                    default:
                        throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
                }
            }
        }

        public DataContainer Flatten(IEnumerable<int> rows = null, IEnumerable<int> columns = null)
        {
            if (columns == null)
                columns = Enumerable.Range(0, ColumnCount);
            var srows = rows == null ? Enumerable.Range(0, Length).ToArray() : rows.ToArray();
            var dc = new DataContainer();
            foreach (var col in columns)
            {
                var kind = GetDType(col);
                if (kind.IsVector())
                {
                    var dc2 = GetColumn(col).Flatten(GetColumnName(col), rows);
                    dc.AddColumn(dc2);
                }
                else
                    dc.AddColumn(GetColumnName(col), kind, srows.Length,
                                 rows == null ? GetColumn(col) : GetColumn(col, srows));

            }
            return dc;
        }

        public bool HasVectorColumn(IEnumerable<int> columns = null)
        {
            if (columns == null)
                columns = Enumerable.Range(0, ColumnCount);
            foreach (var i in columns)
            {
                var kind = GetDType(i);
                if (kind.IsVector())
                    return true;
            }
            return false;
        }

        #endregion

        #region IDataView API

        /// <summary>
        /// Implements DataViewSchema interface for this container.
        /// </summary>
        public class DataContainerSchema : ISchema
        {
            DataContainer _cont;
            public DataContainerSchema(DataContainer cont) { _cont = cont; }
            public int ColumnCount => _cont._names == null ? 0 : _cont._names.Count;
            public string GetColumnName(int col) { return _cont._names[col]; }
            public bool TryGetColumnIndex(string name, out int col)
            {
                return _cont._naming.TryGetValue(name, out col);
            }

            public DataViewType GetColumnType(int col) { return _cont._kinds[col]; }

            public DataViewType GetMetadataTypeOrNull(string kind, int col)
            {
                if (kind == _cont._kinds[col].ToString())
                    return GetColumnType(col);
                return null;
            }

            string[] GetSlotNames(int col)
            {
                string name = GetColumnName(col);
                var type = GetColumnType(col);
                if (type.IsVector())
                {
                    var vec = type.AsVector();
                    if (vec.DimCount() != 1)
                        throw Contracts.ExceptNotImpl("Only one dimension is implemented.");
                    var res = new string[vec.GetDim(0)];
                    for (int i = 0; i < res.Length; ++i)
                        res[i] = string.Format("{0}{1}", name, i);
                    return res;
                }
                else
                    return new string[] { name };
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                if (kind == MetadataUtils.Kinds.SlotNames)
                {
                    var res = GetSlotNames(col);
                    DvText[] dres = new DvText[res.Length];
                    for (int i = 0; i < res.Length; ++i)
                        dres[i] = new DvText(res[i]);
                    var vec = new VBuffer<DvText>(res.Length, dres);
                    ValueGetter<VBuffer<DvText>> conv = (ref VBuffer<DvText> val) => { val = vec; };
                    var conv2 = conv as ValueGetter<TValue>;
                    conv2(ref value);
                    return;
                }

                int index;
                if (TryGetColumnIndex(kind, out index))
                {
                    if (typeof(TValue) == typeof(ReadOnlyMemory<char>))
                    {
                        ValueMapper<string, DvText> convs = (in string src, ref DvText dst) =>
                        {
                            dst = new DvText(src);
                        };
                        var convs2 = convs as ValueMapper<string, TValue>;
                        convs2(in kind, ref value);
                    }
                }
                else
                    throw new IndexOutOfRangeException();
            }

            public IEnumerable<KeyValuePair<string, DataViewType>> GetMetadataTypes(int col)
            {
                if (col < 0 || col >= _cont.ColumnCount)
                    throw new IndexOutOfRangeException();
                yield return new KeyValuePair<string, DataViewType>(_cont._names[col], _cont._kinds[col]);
            }
        }

        /// <summary>
        /// Returns the schema. It should not be used unless it is necessary
        /// as it makes a copy of the existing schema.
        /// </summary>
        public DataViewSchema Schema
        {
            get
            {
                lock (_lock)
                {
                    if (_schemaCache == null || _schemaCache.Count != _schema.ColumnCount)
                    {
                        _schemaCache = ExtendedSchema.Create(_schema);
                        return _schemaCache;
                    }
                    if (Enumerable.Range(0, _schema.ColumnCount).Where(i => _schema.GetColumnName(i) != _schemaCache[i].Name).Any() ||
                        Enumerable.Range(0, _schema.ColumnCount).Where(i => _schema.GetColumnType(i) != _schemaCache[i].Type).Any())
                    {
                        _schemaCache = ExtendedSchema.Create(_schema);
                        return _schemaCache;

                    }
                }
                return _schemaCache;
            }
        }

        private DataViewSchema _schemaCache;
        private object _lock;

        public ISchema SchemaI => _schema;

        /// <summary>
        /// Fills the value with values coming from a IDataView.
        /// nrow must be specified for the first column.
        /// The method checks that all column have the same number of elements.
        /// </summary>
        public void FillValues(IDataView view, int nrows = -1, bool keepVectors = false, int? numThreads = 1, IHostEnvironment env = null)
        {
            long? numRows = view.GetRowCount();
            if (!numRows.HasValue)
                numRows = DataViewUtils.ComputeRowCount(view);

            var memory = new Dictionary<int, Tuple<int, int>>();
            var sch = view.Schema;
            int pos = 0;
            for (int i = 0; i < sch.Count; ++i)
            {
                if (sch[i].IsHidden)
                    continue;
                var ty = sch[i].Type;
                if (!keepVectors && ty.IsVector())
                {
                    var tyv = ty.AsVector();
                    if (tyv.DimCount() != 1)
                        throw new NotSupportedException("Only arrays with one dimension are supported.");
                    for (int j = 0; j < tyv.GetDim(0); ++j)
                    {
                        AddColumn(string.Format("{0}.{1}", sch[i].Name, j), tyv.ItemType(), (int)numRows.Value);
                        memory[pos++] = new Tuple<int, int>(i, j);
                    }
                }
                else
                {
                    memory[pos] = new Tuple<int, int>(i, -1);
                    AddColumn(sch[i].Name, ty, (int)numRows.Value);
                    ++pos;
                }
            }

            ILogWriter logout = new LogWriter((string s) => { });
            ILogWriter logerr = new LogWriter((string s) => { });
            int nth;
            bool dispose = env == null;
            IHostEnvironment host = env ?? new DelegateEnvironment(conc: 1, outWriter: logout, errWriter: logerr, verbose: 1);
            var ch = host.Register("Estimate n threads");
            nth = numThreads.HasValue ? numThreads.Value : DataViewUtils.GetThreadCount(ch, 0, true);
            if (dispose)
                (host as DelegateEnvironment).Dispose();

            // Fills values.
            if (nth == 1)
            {
                using (var cursor = view.GetRowCursor(view.Schema))
                    FillValues(cursor, memory);
            }
            else
            {
                var cursors = view.GetRowCursorSet(view.Schema, nth);
                // FillValues(cursors, memory);
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Fills the value with values coming from a DataViewRowCursor.
        /// Called by the previous method.
        /// </summary>
        void FillValues(DataViewRowCursor cursor, Dictionary<int, Tuple<int, int>> memory)
        {
            var getterBL = new ValueGetter<bool>[_colsBL == null ? 0 : _colsBL.Count];
            var getterI4 = new ValueGetter<int>[_colsI4 == null ? 0 : _colsI4.Count];
            var getterU4 = new ValueGetter<uint>[_colsU4 == null ? 0 : _colsU4.Count];
            var getterI8 = new ValueGetter<long>[_colsI8 == null ? 0 : _colsI8.Count];
            var getterR4 = new ValueGetter<float>[_colsR4 == null ? 0 : _colsR4.Count];
            var getterR8 = new ValueGetter<double>[_colsR8 == null ? 0 : _colsR8.Count];
            var getterTX = new ValueGetter<DvText>[_colsTX == null ? 0 : _colsTX.Count];

            var getterABL = new ValueGetter<VBuffer<bool>>[_colsABL == null ? 0 : _colsABL.Count];
            var getterAI4 = new ValueGetter<VBuffer<int>>[_colsAI4 == null ? 0 : _colsAI4.Count];
            var getterAU4 = new ValueGetter<VBuffer<uint>>[_colsAU4 == null ? 0 : _colsAU4.Count];
            var getterAI8 = new ValueGetter<VBuffer<long>>[_colsAI8 == null ? 0 : _colsAI8.Count];
            var getterAR4 = new ValueGetter<VBuffer<float>>[_colsAR4 == null ? 0 : _colsAR4.Count];
            var getterAR8 = new ValueGetter<VBuffer<double>>[_colsAR8 == null ? 0 : _colsAR8.Count];
            var getterATX = new ValueGetter<VBuffer<DvText>>[_colsATX == null ? 0 : _colsATX.Count];

            for (int i = 0; i < _names.Count; ++i)
            {
                var coor = _mapping[i];
                if (coor.Item1.IsVector())
                {
                    switch (coor.Item1.ItemType().RawKind())
                    {
                        case DataKind.BL:
                            getterABL[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, false);
                            break;
                        case DataKind.I4:
                            getterAI4[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, 0);
                            break;
                        case DataKind.U4:
                            getterAU4[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, (uint)0);
                            break;
                        case DataKind.I8:
                            getterAI8[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, (long)0);
                            break;
                        case DataKind.R4:
                            getterAR4[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, float.NaN);
                            break;
                        case DataKind.R8:
                            getterAR8[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, double.NaN);
                            break;
                        case DataKind.TX:
                            getterATX[coor.Item2] = GetGetterCursorVector(cursor, memory[i].Item1, memory[i].Item2, DvText.NA);
                            break;
                        default:
                            throw new NotImplementedException(string.Format("Not implemented for kind {0}", coor.Item1));
                    }
                }
                else
                {
                    switch (coor.Item1.RawKind())
                    {
                        case DataKind.BL:
                            getterBL[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, false);
                            break;
                        case DataKind.I4:
                            getterI4[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, 0);
                            break;
                        case DataKind.U4:
                            getterU4[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, (uint)0);
                            break;
                        case DataKind.I8:
                            getterI8[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, (long)0);
                            break;
                        case DataKind.R4:
                            getterR4[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, float.NaN);
                            break;
                        case DataKind.R8:
                            getterR8[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, double.NaN);
                            break;
                        case DataKind.TX:
                            getterTX[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, DvText.NA);
                            break;
                        default:
                            throw new NotImplementedException(string.Format("Not implemented for kind {0}", coor.Item1));
                    }
                }
            }

            var valueBL = new bool();
            var valueI4 = new int();
            uint valueU4 = 0;
            var valueI8 = new long();
            float valueR4 = 0;
            double valueR8 = 0;
            var valueTX = new DvText();

            var avalueBL = new VBuffer<bool>();
            var avalueI4 = new VBuffer<int>();
            var avalueU4 = new VBuffer<uint>();
            var avalueI8 = new VBuffer<long>();
            var avalueR4 = new VBuffer<float>();
            var avalueR8 = new VBuffer<double>();
            var avalueTX = new VBuffer<DvText>();

            var aqvalueBL = new VBufferEqSort<bool>();
            var aqvalueI4 = new VBufferEqSort<int>();
            var aqvalueU4 = new VBufferEqSort<uint>();
            var aqvalueI8 = new VBufferEqSort<long>();
            var aqvalueR4 = new VBufferEqSort<float>();
            var aqvalueR8 = new VBufferEqSort<double>();
            var aqvalueTX = new VBufferEqSort<DvText>();

            int row = 0;
            while (cursor.MoveNext())
            {
                for (int i = 0; i < _names.Count; ++i)
                {
                    var coor = _mapping[i];
                    if (coor.Item1.IsVector())
                    {
                        switch (coor.Item1.ItemType().RawKind())
                        {
                            case DataKind.BL:
                                getterABL[coor.Item2](ref avalueBL);
                                aqvalueBL = new VBufferEqSort<bool>(avalueBL);
                                _colsABL[coor.Item2].Set(row, aqvalueBL);
                                break;
                            case DataKind.I4:
                                getterAI4[coor.Item2](ref avalueI4);
                                aqvalueI4 = new VBufferEqSort<int>(avalueI4);
                                _colsAI4[coor.Item2].Set(row, aqvalueI4);
                                break;
                            case DataKind.U4:
                                getterAU4[coor.Item2](ref avalueU4);
                                aqvalueU4 = new VBufferEqSort<uint>(avalueU4);
                                _colsAU4[coor.Item2].Set(row, aqvalueU4);
                                break;
                            case DataKind.I8:
                                getterAI8[coor.Item2](ref avalueI8);
                                aqvalueI8 = new VBufferEqSort<long>(avalueI8);
                                _colsAI8[coor.Item2].Set(row, aqvalueI8);
                                break;
                            case DataKind.R4:
                                getterAR4[coor.Item2](ref avalueR4);
                                aqvalueR4 = new VBufferEqSort<float>(avalueR4);
                                _colsAR4[coor.Item2].Set(row, aqvalueR4);
                                break;
                            case DataKind.R8:
                                getterAR8[coor.Item2](ref avalueR8);
                                aqvalueR8 = new VBufferEqSort<double>(avalueR8);
                                _colsAR8[coor.Item2].Set(row, aqvalueR8);
                                break;
                            case DataKind.TX:
                                getterATX[coor.Item2](ref avalueTX);
                                aqvalueTX = new VBufferEqSort<DvText>(avalueTX);
                                _colsATX[coor.Item2].Set(row, aqvalueTX);
                                break;
                            default:
                                throw new NotImplementedException(string.Format("Not implemented for kind {0}", coor.Item1));
                        }
                    }
                    else
                    {
                        switch (coor.Item1.RawKind())
                        {
                            case DataKind.BL:
                                getterBL[coor.Item2](ref valueBL);
                                _colsBL[coor.Item2].Set(row, valueBL);
                                break;
                            case DataKind.I4:
                                getterI4[coor.Item2](ref valueI4);
                                _colsI4[coor.Item2].Set(row, valueI4);
                                break;
                            case DataKind.U4:
                                getterU4[coor.Item2](ref valueU4);
                                _colsU4[coor.Item2].Set(row, valueU4);
                                break;
                            case DataKind.I8:
                                getterI8[coor.Item2](ref valueI8);
                                _colsI8[coor.Item2].Set(row, valueI8);
                                break;
                            case DataKind.R4:
                                getterR4[coor.Item2](ref valueR4);
                                _colsR4[coor.Item2].Set(row, valueR4);
                                break;
                            case DataKind.R8:
                                getterR8[coor.Item2](ref valueR8);
                                _colsR8[coor.Item2].Set(row, valueR8);
                                break;
                            case DataKind.TX:
                                getterTX[coor.Item2](ref valueTX);
                                _colsTX[coor.Item2].Set(row, valueTX);
                                break;
                            default:
                                throw new NotImplementedException(string.Format("Not implemented for kind {0}", coor.Item1));
                        }
                    }
                }
                ++row;
            }
        }

        /// <summary>
        /// Returns a getter of a certain type.
        /// </summary>
        ValueGetter<DType> GetGetterCursor<DType>(DataViewRowCursor cursor, int col, int index, DType defaultValue)
        {
            var dt = cursor.Schema[col].Type;
            if (dt.IsVector())
            {
                ValueGetter<VBuffer<DType>> getter;
                try
                {
                    getter = cursor.GetGetter<VBuffer<DType>>(col);
                }
                catch (Exception e)
                {
                    throw new Exception($"Unable to extract getter for column {col} and type {typeof(DType)}, schema:\n{SchemaHelper.ToString(Schema)}", e);
                }

                var temp = new VBuffer<DType>();
                return (ref DType value) =>
                {
                    getter(ref temp);
                    if (temp.IsDense)
                        value = temp.Values[index];
                    else
                    {
                        value = defaultValue;
                        for (int j = 0; j < temp.Count; ++j)
                        {
                            if (temp.Indices[j] == index)
                            {
                                value = temp.Values[j];
                                break;
                            }
                        }
                    }
                };
            }
            else
            {
                try
                {
                    return cursor.GetGetter<DType>(col);
                }
                catch (InvalidOperationException e)
                {
                    // DvText does not exist in ml.net world.
                    ValueGetter<ReadOnlyMemory<char>> getter;
                    try
                    {
                        getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);
                    }
                    catch (InvalidOperationException)
                    {
                        throw new InvalidOperationException($"Unable to extract getter for column {col} and type {typeof(DType)}, schema:\n{SchemaHelper.ToString(Schema)}", e);
                    }

                    var getter2 = ConvGetter(getter) as ValueGetter<DType>;
                    if (getter2 == null)
                        throw new InvalidOperationException($"Unable to extract getter for column {col} and type {typeof(DType)}, schema:\n{SchemaHelper.ToString(Schema)}");
                    return getter2;
                }
            }
        }

        private static ValueGetter<DvText> ConvGetter(ValueGetter<ReadOnlyMemory<char>> getter)
        {
            return (ref DvText value) =>
            {
                ReadOnlyMemory<char> buffer = null;
                getter(ref buffer);
                value.Set(buffer);
            };
        }

        /// <summary>
        /// Returns a getter of a certain type.
        /// </summary>
        ValueGetter<VBuffer<DType>> GetGetterCursorVector<DType>(DataViewRowCursor cursor, int col, int index, DType defaultValue)
        {
            var dt = cursor.Schema[col].Type;
            if (dt.IsVector())
                return cursor.GetGetter<VBuffer<DType>>(col);
            else
            {
                var getter = cursor.GetGetter<DType>(col);
                var temp = defaultValue;
                return (ref VBuffer<DType> value) =>
                {
                    getter(ref temp);
                    if (value.Length != 1 && value.Count != 1)
                        value = new VBuffer<DType>(1, new[] { temp });
                    else
                        value.Values[0] = temp;
                };
            }
        }

        public delegate void RowFillerDelegate(DataContainer cont, int row);
        public delegate void RowColumnSetterDelegate(DataContainer cont, int row);

        public static RowFillerDelegate GetRowFiller(DataViewRowCursor cur)
        {
            var setters = GetAllSetters(cur);

            return (DataContainer cont, int row) =>
            {
                for (int i = 0; i < setters.Length; ++i)
                    setters[i](cont, row);
            };
        }

        /// <summary>
        /// Builds setters for each column in the DataFrame based on the schema of a cursor.
        /// </summary>
        public static RowColumnSetterDelegate[] GetAllSetters(DataViewRowCursor cur)
        {
            var sch = cur.Schema;
            var res = new List<RowColumnSetterDelegate>();
            var getters = CursorHelper.GetAllGetters(cur);
            int pos = 0;
            for (int i = 0; i < sch.Count; ++i)
            {
                if (sch[i].IsHidden)
                    continue;
                var getter = getters[pos];
                res.Add(GetColumnSetter(cur, getter, pos, sch[i].Type));
                ++pos;
            }
            return res.ToArray();
        }

        /// <summary>
        /// Builds a setter for a specific column, the function then chooses the best
        /// option based on the type. It deals with ambiguities introduced by
        /// DvText and VBufferEqSort.
        /// </summary>
        public static RowColumnSetterDelegate GetColumnSetter(DataViewRowCursor cur, Delegate getter, int col, DataViewType colType)
        {
            if (colType.IsVector())
            {
                switch (colType.ItemType().RawKind())
                {
                    case DataKind.BL: return GetColumnSetterVector<bool>(cur, getter, col);
                    case DataKind.I4: return GetColumnSetterVector<int>(cur, getter, col);
                    case DataKind.U4: return GetColumnSetterVector<uint>(cur, getter, col);
                    case DataKind.I8: return GetColumnSetterVector<Int64>(cur, getter, col);
                    case DataKind.R4: return GetColumnSetterVector<float>(cur, getter, col);
                    case DataKind.R8: return GetColumnSetterVector<double>(cur, getter, col);
                    case DataKind.TX: return GetColumnSetterVectorText(cur, getter, col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
            else
            {
                switch (colType.RawKind())
                {
                    case DataKind.BL: return GetColumnSetter<bool>(cur, getter, col);
                    case DataKind.I4: return GetColumnSetter<int>(cur, getter, col);
                    case DataKind.U4: return GetColumnSetter<uint>(cur, getter, col);
                    case DataKind.I8: return GetColumnSetter<Int64>(cur, getter, col);
                    case DataKind.R4: return GetColumnSetter<float>(cur, getter, col);
                    case DataKind.R8: return GetColumnSetter<double>(cur, getter, col);
                    case DataKind.TX: return GetColumnSetterText(cur, getter, col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
        }

        public static RowColumnSetterDelegate GetColumnSetter<DType>(DataViewRowCursor cur, Delegate getter, int col)
             where DType : IEquatable<DType>, IComparable<DType>
        {
            var typedGetter = getter as ValueGetter<DType>;
            if (typedGetter == null)
                throw new DataTypeError($"Unable to convert a getter {getter.GetType()} for type {typeof(DType)}.");
            return (DataContainer cont, int row) =>
            {
                DataColumn<DType> typedCol;
                cont.GetTypedColumn(col, out typedCol);
                typedCol.Set(row, typedGetter);
            };
        }

        public static RowColumnSetterDelegate GetColumnSetterVector<DType>(DataViewRowCursor cur, Delegate getter, int col)
            where DType : IEquatable<DType>, IComparable<DType>
        {
            var typedGetter2 = getter as ValueGetter<VBufferEqSort<DType>>;
            if (typedGetter2 != null)
                return (DataContainer cont, int row) =>
                {
                    DataColumn<VBufferEqSort<DType>> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedCol.Set(row, typedGetter2);
                };
            var typedGetter = getter as ValueGetter<VBuffer<DType>>;
            if (typedGetter != null)
            {
                var buffer = new VBuffer<DType>();
                return (DataContainer cont, int row) =>
                {
                    DataColumn<VBufferEqSort<DType>> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedGetter(ref buffer);
                    typedCol.Set(row, new VBufferEqSort<DType>(buffer));
                };
            }
            throw new DataTypeError($"Unable to convert a getter {getter.GetType()} for type {typeof(DType)}.");
        }

        public static RowColumnSetterDelegate GetColumnSetterVectorText(DataViewRowCursor cur, Delegate getter, int col)
        {
            var typedGetter3 = getter as ValueGetter<VBufferEqSort<DvText>>;
            if (typedGetter3 != null)
                return (DataContainer cont, int row) =>
                {
                    DataColumn<VBufferEqSort<DvText>> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedCol.Set(row, typedGetter3);
                };
            var typedGetter2 = getter as ValueGetter<VBuffer<DvText>>;
            if (typedGetter2 != null)
            {
                var buffer = new VBuffer<DvText>();
                return (DataContainer cont, int row) =>
                {
                    DataColumn<VBufferEqSort<DvText>> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedGetter2(ref buffer);
                    typedCol.Set(row, new VBufferEqSort<DvText>(buffer));
                };
            }
            var typedGetter = getter as ValueGetter<VBuffer<ReadOnlyMemory<char>>>;
            if (typedGetter != null)
            {
                var buffer = new VBuffer<ReadOnlyMemory<char>>();
                return (DataContainer cont, int row) =>
                {
                    DataColumn<VBufferEqSort<DvText>> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedGetter(ref buffer);
                    typedCol.Set(row, new VBufferEqSort<DvText>(buffer.Length, buffer.Count,
                                                                buffer.Values.Select(c => new DvText(c)).ToArray(),
                                                                buffer.Indices));
                };
            }
            throw new DataTypeError($"Unable to convert a getter {getter.GetType()} for type VBufferEqSort<DvText> or equivalent.");
        }

        public static RowColumnSetterDelegate GetColumnSetterText(DataViewRowCursor cur, Delegate getter, int col)
        {
            var typedGetter2 = getter as ValueGetter<DvText>;
            if (typedGetter2 != null)
                return (DataContainer cont, int row) =>
                {
                    DataColumn<DvText> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedCol.Set(row, typedGetter2);
                };
            var typedGetter = getter as ValueGetter<ReadOnlyMemory<char>>;
            if (typedGetter != null)
            {
                var buffer = new ReadOnlyMemory<char>();
                return (DataContainer cont, int row) =>
                {
                    DataColumn<DvText> typedCol;
                    cont.GetTypedColumn(col, out typedCol);
                    typedGetter(ref buffer);
                    typedCol.Set(row, new DvText(buffer));
                };
            }
            throw new DataTypeError($"Unable to convert a getter {getter.GetType()} for type DvText or ReadOnlyMemory<char>.");
        }

        #endregion

        #region Cursor

        /// <summary>
        /// Returns a cursor on the data.
        /// </summary>
        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return new DataRowCursor(this, columnsNeeded, rand);
        }

        /// <summary>
        /// Returns a cursor on a subset of the data.
        /// </summary>
        public DataRowCursor GetRowCursor(int[] rows, int[] columns, IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return new DataRowCursor(this, columnsNeeded, rand, rows: rows, columns: columns);
        }

        /// <summary>
        /// Returns a set of aliased cursors on the data.
        /// </summary>
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return GetRowCursorSet(null, null, columnsNeeded, n, rand);
        }

        /// <summary>
        /// Returns a set of aliased cursors on the data.
        /// </summary>
        public DataViewRowCursor[] GetRowCursorSet(int[] rows, int[] columns, IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            var host = new ConsoleEnvironment().Register("Estimate n threads");
            n = DataViewUtils.GetThreadCount(host, n);
            if (n > 1 && (long)n > Length)
                n = Length;

            if (n <= 1)
                return new DataViewRowCursor[] { GetRowCursor(rows, columns, columnsNeeded, rand) };
            else
            {
                var cursors = new DataViewRowCursor[n];
                for (int i = 0; i < cursors.Length; ++i)
                    cursors[i] = new DataRowCursor(this, columnsNeeded, rand, n, i, rows: rows, columns: columns);
                return cursors;
            }
        }

        /// <summary>
        /// Implements a cursor for this container.
        /// </summary>
        public class DataRowCursor : DataViewRowCursor
        {
            DataContainer _cont;
            public override long Batch => _first;
            Random _rand;
            IEnumerable<DataViewSchema.Column> _columnsNeeded;
            long _inc;
            readonly long _first;
            long _position;

            int[] _rowsSet;
            int[] _colsSet;
            Dictionary<int, int> _revColsSet;
            DataViewSchema _schema;
            int[] _shuffled;

            public DataRowCursor(DataContainer cont, IEnumerable<DataViewSchema.Column> columnsNeeded,
                             Random rand = null, int inc = 1, int first = 0,
                             int[] rows = null, int[] columns = null)
            {
                _cont = cont;
                _position = -1;
                _inc = inc;
                _first = first;
                _columnsNeeded = columnsNeeded;
                _rand = rand;
                _rowsSet = rows;
                _colsSet = columns;
                if (_colsSet != null)
                {
                    _revColsSet = new Dictionary<int, int>();
                    for (int i = 0; i < columns.Length; ++i)
                        _revColsSet[columns[i]] = i;
                    _schema = ExtendedSchema.Create(new DataFrameViewSchema(_cont.Schema, columns));
                }
                else
                    _schema = null;
                if (_rand != null)
                {
                    _shuffled = new int[LastPosition];
                    for (int i = 0; i < _shuffled.Length; ++i)
                        _shuffled[i] = i;
                    Utils.Shuffle<int>(_rand, _shuffled);
                }
                else
                    _shuffled = null;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return (ref DataViewRowId idrow) => { idrow = new DataViewRowId(Position >= 0 ? (ulong)Position : (ulong)(-Position), Position >= 0 ? (ulong)1 : (ulong)0); };
            }

            protected override void Dispose(bool disposing)
            {
            }

            public override bool IsColumnActive(int col) { return Schema.Where(c => c.Index == col).Any(); }
            public override DataViewSchema Schema => _colsSet == null ? _cont.Schema : _schema;

            public override long Position => _rowsSet == null
                                        ? _position
                                        : (_shuffled == null
                                                ? _rowsSet[_position]
                                                : (_position < _shuffled.Length
                                                    ? _rowsSet[_shuffled[_position]]
                                                    : _position));
            int LastPosition => _rowsSet == null ? _cont.Length : _rowsSet.Length;

            public override bool MoveNext()
            {
                if (_position == -1)
                    _position = _first;
                else
                    _position += _inc;
                return _position < LastPosition;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                col = _colsSet == null ? col : _colsSet[col];
                var coor = _cont._mapping[col];
                if (coor.Item1.IsVector())
                {
                    switch (coor.Item1.ItemType().RawKind())
                    {
                        case DataKind.BL: return _cont._colsABL[coor.Item2].GetGetterVector<bool>(this) as ValueGetter<TValue>;
                        case DataKind.I4: return _cont._colsAI4[coor.Item2].GetGetterVector<int>(this) as ValueGetter<TValue>;
                        case DataKind.U4: return _cont._colsAU4[coor.Item2].GetGetterVector<uint>(this) as ValueGetter<TValue>;
                        case DataKind.I8: return _cont._colsAI8[coor.Item2].GetGetterVector<long>(this) as ValueGetter<TValue>;
                        case DataKind.R4: return _cont._colsAR4[coor.Item2].GetGetterVector<float>(this) as ValueGetter<TValue>;
                        case DataKind.R8: return _cont._colsAR8[coor.Item2].GetGetterVector<double>(this) as ValueGetter<TValue>;
                        case DataKind.TX: return _cont._colsATX[coor.Item2].GetGetterVector<ReadOnlyMemory<char>>(this) as ValueGetter<TValue>;
                        default:
                            throw new NotImplementedException();
                    }
                }
                else
                {
                    switch (coor.Item1.RawKind())
                    {
                        case DataKind.BL: return _cont._colsBL[coor.Item2].GetGetter<TValue>(this);
                        case DataKind.I4: return _cont._colsI4[coor.Item2].GetGetter<TValue>(this);
                        case DataKind.U4: return _cont._colsU4[coor.Item2].GetGetter<TValue>(this);
                        case DataKind.I8: return _cont._colsI8[coor.Item2].GetGetter<TValue>(this);
                        case DataKind.R4: return _cont._colsR4[coor.Item2].GetGetter<TValue>(this);
                        case DataKind.R8: return _cont._colsR8[coor.Item2].GetGetter<TValue>(this);
                        case DataKind.TX: return _cont._colsTX[coor.Item2].GetGetter<TValue>(this);
                        default:
                            throw new NotImplementedException();
                    }
                }
            }
        }

        #endregion

        #region comparison

        /// <summary>
        /// Order the rows.
        /// </summary>
        public void Order(int[] order)
        {
            for (int i = 0; i < ColumnCount; ++i)
                GetColumn(i).Order(order);
        }

        /// <summary>
        /// Order the columns.
        /// </summary>
        public void OrderColumns(string[] columns)
        {
            var colind = columns.Select(c => GetColumnIndex(c)).ToArray();
            var new_names = columns;
            var new_kinds = colind.Select(i => _kinds[i]).ToList();
            var new_length = columns.Length;
            var new_naming = new Dictionary<string, int>();
            var new_mapping = new Dictionary<int, Tuple<DataViewType, int>>();
            for (int i = 0; i < new_length; ++i)
            {
                new_naming[new_names[i]] = i;
                new_mapping[i] = _mapping[colind[i]];
            }

            _names = new_names.ToList();
            _mapping = new_mapping;
            _kinds = new_kinds;
            _naming = new_naming;
            _schema = new DataContainerSchema(this);
        }

        public void RenameColumns(string[] columns)
        {
            if (columns.Length != _names.Count)
                throw new DataNameError("Unexpected number of columns.");
            var new_names = _names.Select((c, i) => string.IsNullOrEmpty(columns[i]) ? c : columns[i]).ToArray();
            var new_naming = new Dictionary<string, int>();
            for (int i = 0; i < _names.Count; ++i)
                new_naming[new_names[i]] = i;

            _names = new_names.ToList();
            _naming = new_naming;
            _schema = new DataContainerSchema(this);
        }

        /// <summary>
        /// Checks that containers are exactly the same.
        /// </summary>
        public static bool operator ==(DataContainer c1, DataContainer c2)
        {
            return c1.Equals(c2);
        }

        /// <summary>
        /// Checks that containers are different.
        /// </summary>
        public static bool operator !=(DataContainer c1, DataContainer c2)
        {
            return !c1.Equals(c2);
        }

        /// <summary>
        /// Checks that containers are exactly the same.
        /// </summary>
        public override bool Equals(object c)
        {
            var cd = c as DataContainer;
            if (cd == null)
                return false;
            return Equals(cd);
        }

        /// <summary>
        /// Not implemented.
        /// </summary>
        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Checks that containers are exactly the same.
        /// </summary>
        public bool Equals(DataContainer c)
        {
            if (Shape.Item1 != c.Shape.Item1 || Shape.Item2 != c.Shape.Item2)
                return false;
            for (int i = 0; i < _names.Count; ++i)
            {
                if (_names[i] != c._names[i])
                    return false;
                if (_kinds[i] != c._kinds[i])
                    return false;
                if (_mapping[i].Item1 != c._mapping[i].Item1 || _mapping[i].Item2 != c._mapping[i].Item2)
                    return false;
            }
            if (_colsBL != null)
            {
                for (int i = 0; i < _colsBL.Count; ++i)
                    if (!_colsBL[i].Equals(c._colsBL[i]))
                        return false;
            }
            if (_colsI4 != null)
            {
                for (int i = 0; i < _colsI4.Count; ++i)
                    if (!_colsI4[i].Equals(c._colsI4[i]))
                        return false;
            }
            if (_colsU4 != null)
            {
                for (int i = 0; i < _colsU4.Count; ++i)
                    if (!_colsU4[i].Equals(c._colsU4[i]))
                        return false;
            }
            if (_colsI8 != null)
            {
                for (int i = 0; i < _colsI8.Count; ++i)
                    if (!_colsI8[i].Equals(c._colsI8[i]))
                        return false;
            }
            if (_colsR4 != null)
            {
                for (int i = 0; i < _colsR4.Count; ++i)
                    if (!_colsR4[i].Equals(c._colsR4[i]))
                        return false;
            }
            if (_colsR8 != null)
            {
                for (int i = 0; i < _colsR8.Count; ++i)
                    if (!_colsR8[i].Equals(c._colsR8[i]))
                        return false;
            }
            if (_colsTX != null)
            {
                for (int i = 0; i < _colsTX.Count; ++i)
                    if (!_colsTX[i].Equals(c._colsTX[i]))
                        return false;
            }
            return true;
        }

        string AlmostEqualsSchemaMessage(DataContainer c)
        {
            var rows = new List<string>();
            for (int i = 0; i < Math.Max(Shape.Item1, c.Shape.Item1); ++i)
            {
                var row = new List<string>();
                if (i < Schema.Count)
                    row.Add($"'{Schema[i].Name}':{Schema[i].Type}");
                else
                    row.Add("###");
                if (i < c.SchemaI.ColumnCount)
                    row.Add($"'{c.SchemaI.GetColumnName(i)}':{c.SchemaI.GetColumnType(i)}");
                else
                    row.Add("###");
                var s = string.Join(" --- ", row);
                rows.Add($"{i}: {s}");
            }
            return string.Join("\n", rows);
        }

        /// <summary>
        /// Checks that containers are almost exactly the same for real values
        /// or exactly the same of other types.
        /// It returns 0 if the difference is below the precision
        /// or the difference otherwise, Inf if shapes or schema are different.
        /// </summary>
        public double AlmostEquals(DataContainer c, double precision = 1e-6f, bool exc = false)
        {
            if (Shape.Item1 != c.Shape.Item1 || Shape.Item2 != c.Shape.Item2)
            {
                if (exc)
                    throw new Exception($"Shapes do not match: ({Shape.Item1},{Shape.Item2}) != ({c.Shape.Item1},{c.Shape.Item2})");
                return double.PositiveInfinity;
            }
            for (int i = 0; i < _names.Count; ++i)
            {
                if (_names[i] != c._names[i])
                {
                    if (exc)
                        throw new Exception($"Name at position {i} do not match: '{_names[i]}' != '{c._names[i]}'\n{AlmostEqualsSchemaMessage(c)}");
                    return double.PositiveInfinity;
                }
                if (_kinds[i] != c._kinds[i])
                {
                    if (exc)
                        throw new Exception($"Type at position {i} do not match: '{_kinds[i]}' != '{c._kinds[i]}'\n{AlmostEqualsSchemaMessage(c)}");
                    return double.PositiveInfinity;
                }
                if (_mapping[i].Item1 != c._mapping[i].Item1 || _mapping[i].Item2 != c._mapping[i].Item2)
                {
                    if (exc)
                        throw new Exception($"Mapping at position {i} do not match: {_mapping[i].Item1}, {_mapping[i].Item2} != {c._mapping[i].Item1}, {c._mapping[i].Item2}\n{AlmostEqualsSchemaMessage(c)}");
                    return double.PositiveInfinity;
                }
            }
            if (_colsBL != null)
            {
                for (int i = 0; i < _colsBL.Count; ++i)
                    if (!_colsBL[i].Equals(c._colsBL[i]))
                    {
                        if (exc)
                            throw new Exception($"Mismatch in BL column {i}");
                        return (double)DataKind.BL;
                    }
            }
            if (_colsI4 != null)
            {
                for (int i = 0; i < _colsI4.Count; ++i)
                    if (!_colsI4[i].Equals(c._colsI4[i]))
                    {
                        if (exc)
                            throw new Exception($"Mismatch in I4 column {i}");
                        return (double)DataKind.I4;
                    }
            }
            if (_colsU4 != null)
            {
                for (int i = 0; i < _colsU4.Count; ++i)
                    if (!_colsU4[i].Equals(c._colsU4[i]))
                    {
                        if (exc)
                            throw new Exception($"Mismatch in U4 column {i}");
                        return (double)DataKind.U4;
                    }
            }
            if (_colsI8 != null)
            {
                for (int i = 0; i < _colsI8.Count; ++i)
                    if (!_colsI8[i].Equals(c._colsI8[i]))
                    {
                        if (exc)
                            throw new Exception($"Mismatch in I8 column {i}");
                        return (double)DataKind.I8;
                    }
            }
            if (_colsR4 != null)
            {
                for (int i = 0; i < _colsR4.Count; ++i)
                {
                    var c1 = (_colsR4[i] as DataColumn<float>).Data;
                    var c2 = (c._colsR4[i] as DataColumn<float>).Data;
                    var d = NumericHelper.AlmostEqual(c1, c2, (float)precision);
                    if (d != 0f)
                    {
                        if (exc)
                            throw new Exception($"Mismatch in R4 column {i} - {d}");
                        return (double)d;
                    }
                }
            }
            if (_colsR8 != null)
            {
                for (int i = 0; i < _colsR8.Count; ++i)
                {
                    var c1 = (_colsR8[i] as DataColumn<double>).Data;
                    var c2 = (c._colsR8[i] as DataColumn<double>).Data;
                    var d = NumericHelper.AlmostEqual(c1, c2, precision);
                    if (d != 0)
                    {
                        if (exc)
                            throw new Exception($"Mismatch in R8 column {i} - {d}");
                        return d;
                    }
                }
            }
            if (_colsTX != null)
            {
                for (int i = 0; i < _colsTX.Count; ++i)
                    if (!_colsTX[i].Equals(c._colsTX[i]))
                    {
                        if (exc)
                            throw new Exception($"Mismatch in TX column {i}");
                        return (double)DataKind.TX;
                    }
            }
            return 0;
        }

        #endregion

        #region operators

        /// <summary>
        /// Usual operator [i,j].
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="col">column</param>
        /// <returns>value</returns>
        public object this[int row, int col]
        {
            get { return GetColumn(col).Get(row); }
            set { GetColumn(col).Set(row, value); }
        }

        /// <summary>
        /// Usual operator [i,colname].
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="col">column</param>
        /// <returns>value</returns>
        public object this[int row, string colname]
        {
            get { return GetColumn(colname).Get(row); }
            set { GetColumn(colname).Set(row, value); }
        }


        /// <summary>
        /// Returns a column.
        /// </summary>
        public NumericColumn this[string colname]
        {
            get { return new NumericColumn(GetColumn(colname)); }
        }

        /// <summary>
        /// Returns all values in a row as a dictionary.
        /// </summary>
        public Dictionary<string, object> this[int row]
        {
            get
            {
                var res = new Dictionary<string, object>();
                for (int i = 0; i < _names.Count; ++i)
                    res[_names[i]] = this[row, i];
                return res;
            }
        }

        /// <summary>
        /// Changes the value of a column and a subset of rows.
        /// </summary>
        public object this[IEnumerable<bool> rows, int col]
        {
            set { GetColumn(col).Set(rows, value); }
        }

        /// <summary>
        /// Changes the value of a column and a subset of rows.
        /// </summary>
        public object this[IEnumerable<int> rows, int col]
        {
            set { GetColumn(col).Set(rows, value); }
        }

        /// <summary>
        /// Changes the value of a column and a subset of rows.
        /// </summary>
        public object this[IEnumerable<bool> rows, string col]
        {
            set { GetColumn(col).Set(rows, value); }
        }

        /// <summary>
        /// Changes the value of a column and a subset of rows.
        /// </summary>
        public object this[IEnumerable<int> rows, string col]
        {
            set { GetColumn(col).Set(rows, value); }
        }

        #endregion

        #region SQL functions

        /// <summary>
        /// Aggregates over all rows.
        /// </summary>
        public DataContainer Aggregate(AggregatedFunction agg, int[] rows = null, int[] columns = null)
        {
            var res = new DataContainer();
            if (columns == null)
            {
                for (int i = 0; i < ColumnCount; ++i)
                    res.AddColumn(_names[i], _kinds[i], 1, GetColumn(i).Aggregate(agg, rows));
            }
            else
            {
                int i;
                for (int c = 0; c < columns.Length; ++c)
                {
                    i = columns[c];
                    res.AddColumn(_names[i], _kinds[i], 1, GetColumn(i).Aggregate(agg, rows));
                }
            }
            return res;
        }

        #endregion
    }
}
