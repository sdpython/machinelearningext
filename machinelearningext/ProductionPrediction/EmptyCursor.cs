// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.ProductionPrediction
{
    public class EmptyCursor : DataViewRowCursor
    {
        Func<int, bool> _needCol;
        IDataView _view;

        public EmptyCursor(IDataView view, Func<int, bool> needCol)
        {
            _needCol = needCol;
            _view = view;
        }

        //public override int Count() { return 0; }
        public override long Batch { get { return 0; } }
        public override long Position { get { return 0; } }
        public override DataViewSchema Schema { get { return _view.Schema; } }
        public override ValueGetter<DataViewRowId> GetIdGetter() { return (ref DataViewRowId uid) => { uid = new DataViewRowId(0, 1); }; }

        protected override void Dispose(bool disposing)
        {
            GC.SuppressFinalize(this);
        }

        public override bool MoveNext()
        {
            return false;
        }

        public override bool IsColumnActive(DataViewSchema.Column col)
        {
            return _needCol(col.Index);
        }

        /// <summary>
        /// The getter return the default value. A null getter usually fails the pipeline.
        /// </summary>
        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
        {
            return (ref TValue value) =>
            {
                value = default(TValue);
            };
        }
    }
}
