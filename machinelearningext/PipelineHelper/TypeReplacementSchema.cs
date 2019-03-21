// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Change a type for an existing column in an existing schema.
    /// The input view must provide getters (GetGetter) which returns getters for different types.
    /// </summary>
    public class TypeReplacementSchema : ISchema
    {
        readonly ISchema _schemaInput;
        readonly Dictionary<int, DataViewType> _types;
        readonly Dictionary<int, int> _mappedColumns;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="inputSchema">existing schema</param>
        /// <param name="names">new columns</param>
        /// <param name="types">corresponding types</param>
        public TypeReplacementSchema(ISchema inputSchema, string[] names, DataViewType[] types)
        {
            _schemaInput = inputSchema;
            if (names == null || names.Length == 0)
                throw Contracts.ExceptEmpty("The extended schema must contain new names.");
            if (types == null || types.Length != names.Length)
                throw Contracts.Except("names and types must have the same length.");
            _types = new Dictionary<int, DataViewType>();
            _mappedColumns = new Dictionary<int, int>();
            Contracts.Assert(types.Length == names.Length);
            int index;
            for (int i = 0; i < names.Length; ++i)
            {
                if (!inputSchema.TryGetColumnIndex(names[i], out index))
                    throw Contracts.Except("Unable to find column '{0}' in '{1}'", names[i], SchemaHelper.ToString(inputSchema));
                _types[index] = types[i];
                _mappedColumns[inputSchema.ColumnCount + i] = index;
            }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="inputSchema">existing schema</param>
        /// <param name="names">new columns</param>
        /// <param name="types">corresponding types</param>
        public TypeReplacementSchema(DataViewSchema inputSchema, string[] names, DataViewType[] types)
        {
            _schemaInput = new ExtendedSchema(inputSchema);
            if (names == null || names.Length == 0)
                throw Contracts.ExceptEmpty("The extended schema must contain new names.");
            if (types == null || types.Length != names.Length)
                throw Contracts.Except("names and types must have the same length.");
            _types = new Dictionary<int, DataViewType>();
            _mappedColumns = new Dictionary<int, int>();
            Contracts.Assert(types.Length == names.Length);
            int index;
            for (int i = 0; i < names.Length; ++i)
            {
                index = SchemaHelper.GetColumnIndex(inputSchema, names[i]);
                _types[index] = types[i];
                _mappedColumns[inputSchema.Count + i] = index;
            }
        }

        /// <summary>
        /// Returns the extended number of columns.
        /// </summary>
        public int ColumnCount { get { return _schemaInput.ColumnCount; } }

        public int GetColumnIndex(string name)
        {
            int res;
            var r = TryGetColumnIndex(name, out res);
            if (r)
                return res;
            throw new IndexOutOfRangeException(string.Format("Unable to find column '{0}'.", name));
        }

        public bool TryGetColumnIndex(string name, out int res)
        {
            return _schemaInput.TryGetColumnIndex(name, out res);
        }

        public string GetColumnName(int col)
        {
            return _schemaInput.GetColumnName(col);
        }

        public DataViewType GetColumnType(int col)
        {
            if (_types.ContainsKey(col))
                return _types[col];
            else
                return _schemaInput.GetColumnType(col);
        }

        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            int count = _schemaInput.ColumnCount;
            if (col < count)
            {
                _schemaInput.GetMetadata<TValue>(kind, col, ref value);
                return;
            }
            throw new IndexOutOfRangeException();
        }

        public DataViewType GetMetadataTypeOrNull(string kind, int col)
        {
            int count = _schemaInput.ColumnCount;
            if (col < count)
                return _schemaInput.GetMetadataTypeOrNull(kind, col);
            return null;
        }

        public IEnumerable<KeyValuePair<string, DataViewType>> GetMetadataTypes(int col)
        {
            int count = _schemaInput.ColumnCount;
            if (col < count)
            {
                foreach (var s in _schemaInput.GetMetadataTypes(col))
                    yield return s;
            }
            else
                throw new IndexOutOfRangeException();
        }
    }
}