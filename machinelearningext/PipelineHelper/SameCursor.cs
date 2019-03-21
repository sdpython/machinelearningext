// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Combines an existing cursor with a schema not necessarily related.
    /// Used in <see cref="ScalerTransform"/>.
    /// </summary>
    public class SameCursor : DataViewRowCursor
    {
        readonly DataViewRowCursor _inputCursor;
        readonly DataViewSchema _schema;
        readonly DataViewSchema _cursorSchema;

        public SameCursor(DataViewRowCursor cursor, DataViewSchema schema)
        {
            _schema = schema;
            _inputCursor = cursor;
            _cursorSchema = _inputCursor.Schema;
        }

        public override bool IsColumnActive(DataViewSchema.Column col)
        {
            if (col.Index < _cursorSchema.Count)
                return _inputCursor.IsColumnActive(col);
            return false;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                _inputCursor.Dispose();
            GC.SuppressFinalize(this);
        }

        public override ValueGetter<DataViewRowId> GetIdGetter() { return _inputCursor.GetIdGetter(); }
        public override long Batch { get { return _inputCursor.Batch; } }
        public override long Position { get { return _inputCursor.Position; } }
        public override DataViewSchema Schema { get { return _schema; } }
        public override bool MoveNext() { return _inputCursor.MoveNext(); }
        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col) { return _inputCursor.GetGetter<TValue>(col); }
    }
}
