// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Similar to OpaqueDataView implementation. It provides a view which changes the type of one column
    /// but does not change the cursor. We assume the cursor is able to return the same column in
    /// a different type (so GetGetter&lt;TYPE&gt;(col) returns two different getters on TYPE).
    /// </summary>
    public sealed class TypeReplacementDataView : IDataView
    {
        private readonly IDataView _source;
        private readonly DataViewSchema _schema;

        public IDataView SourceTags { get { return _source; } }

        public TypeReplacementDataView(IDataView source, TypeReplacementSchema newSchema)
        {
            _source = source;
            _schema = ExtendedSchema.Create(newSchema);
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public DataViewSchema Schema
        {
            get { return _schema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var res = new TypeReplacementCursor(_source.GetRowCursor(columnsNeeded, rand), Schema);
#if(DEBUG)
            if (!SchemaHelper.CompareSchema(_schema, res.Schema))
                SchemaHelper.CompareSchema(_schema, res.Schema, true);
#endif
            return res;
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return _source.GetRowCursorSet(columnsNeeded, n, rand)
                          .Select(c => new TypeReplacementCursor(c, Schema)).ToArray();
        }

        class TypeReplacementCursor : DataViewRowCursor
        {
            DataViewRowCursor _cursor;
            DataViewSchema _schema;

            public TypeReplacementCursor(DataViewRowCursor cursor, ISchema newSchema)
            {
                _cursor = cursor;
                _schema = ExtendedSchema.Create(newSchema);
            }

            public TypeReplacementCursor(DataViewRowCursor cursor, DataViewSchema newSchema)
            {
                _cursor = cursor;
                _schema = newSchema;
            }

            public override DataViewSchema Schema { get { return _schema; } }
            public override bool IsColumnActive(DataViewSchema.Column col) { return _cursor.IsColumnActive(col); }
            public override ValueGetter<DataViewRowId> GetIdGetter() { return _cursor.GetIdGetter(); }
            public override long Batch { get { return _cursor.Batch; } }
            public override long Position { get { return _cursor.Position; } }
            public override bool MoveNext() { return _cursor.MoveNext(); }
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                return _cursor.GetGetter<TValue>(col);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _cursor.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
