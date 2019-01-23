// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Combines an existing cursor with a schema not necessarily related.
    /// Used in <see cref="ScalerTransform"/>.
    /// </summary>
    public class SameCursor : RowCursor
    {
        readonly RowCursor _inputCursor;
        readonly Schema _schema;
        readonly Schema _cursorSchema;

        public SameCursor(RowCursor cursor, Schema schema)
        {
            _schema = schema;
            _inputCursor = cursor;
            _cursorSchema = _inputCursor.Schema;
        }

        public override bool IsColumnActive(int col)
        {
            if (col < _cursorSchema.Count)
                return _inputCursor.IsColumnActive(col);
            return false;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                _inputCursor.Dispose();
            GC.SuppressFinalize(this);
        }

        public override ValueGetter<RowId> GetIdGetter() { return _inputCursor.GetIdGetter(); }
        public override long Batch { get { return _inputCursor.Batch; } }
        public override long Position { get { return _inputCursor.Position; } }
        public override Schema Schema { get { return _schema; } }
        public override bool MoveNext() { return _inputCursor.MoveNext(); }
        public override ValueGetter<TValue> GetGetter<TValue>(int col) { return _inputCursor.GetGetter<TValue>(col); }
    }
}
