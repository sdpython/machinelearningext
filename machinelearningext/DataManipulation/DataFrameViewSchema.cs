// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements ISchema interface for this container.
    /// </summary>
    public class DataFrameViewSchema : ISchema
    {
        ISchema _schema;
        Dictionary<int, int> _mapping;
        int[] _revmapping;

        public DataFrameViewSchema(ISchema schema, IEnumerable<int> colIndices)
        {
            _schema = schema;
            _mapping = new Dictionary<int, int>();
            var li = new List<int>();
            int c = 0;
            foreach (var col in colIndices)
            {
                _mapping[col] = c;
                li.Add(col);
                ++c;
            }
            _revmapping = li.ToArray();
        }

        public DataFrameViewSchema(DataViewSchema schema, IEnumerable<int> colIndices)
        {
            _schema = new ExtendedSchema(schema);
            _mapping = new Dictionary<int, int>();
            var li = new List<int>();
            int c = 0;
            foreach (var col in colIndices)
            {
                _mapping[col] = c;
                li.Add(col);
                ++c;
            }
            _revmapping = li.ToArray();
        }

        public int ColumnCount => _mapping.Count;
        public string GetColumnName(int col) { return _schema.GetColumnName(_revmapping[col]); }
        public bool TryGetColumnIndex(string name, out int col)
        {
            bool r = _schema.TryGetColumnIndex(name, out col);
            if (!r)
                return r;
            col = _mapping[col];
            return r;
        }
        public DataViewType GetColumnType(int col) { return _schema.GetColumnType(_revmapping[col]); }
        public DataViewType GetMetadataTypeOrNull(string kind, int col) { return _schema.GetMetadataTypeOrNull(kind, _revmapping[col]); }
        public void GetMetadata<TValue>(string kind, int col, ref TValue value) { _schema.GetMetadata(kind, _revmapping[col], ref value); }
        public IEnumerable<KeyValuePair<string, DataViewType>> GetMetadataTypes(int col) { return _schema.GetMetadataTypes(_revmapping[col]); }
    }
}
