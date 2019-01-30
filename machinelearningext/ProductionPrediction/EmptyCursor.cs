// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;


namespace Scikit.ML.ProductionPrediction
{
    public class EmptyCursor : RowCursor
    {
        Func<int, bool> _needCol;
        IDataView _view;

        public EmptyCursor(IDataView view, Func<int, bool> needCol)
        {
            _needCol = needCol;
            _view = view;
        }

        public override int Count() { return 0; }
        public override long Batch { get { return 0; } }
        public override long Position { get { return 0; } }
        public override Schema Schema { get { return _view.Schema; } }
        public override ValueGetter<RowId> GetIdGetter() { return (ref RowId uid) => { uid = new RowId(0, 1); }; }

        protected override void Dispose(bool disposing)
        {
            GC.SuppressFinalize(this);
        }

        public override bool MoveNext()
        {
            return false;
        }

        public override bool IsColumnActive(int col)
        {
            return _needCol(col);
        }

        /// <summary>
        /// The getter return the default value. A null getter usually fails the pipeline.
        /// </summary>
        public override ValueGetter<TValue> GetGetter<TValue>(int col)
        {
            return (ref TValue value) =>
            {
                value = default(TValue);
            };
        }
    }
}
