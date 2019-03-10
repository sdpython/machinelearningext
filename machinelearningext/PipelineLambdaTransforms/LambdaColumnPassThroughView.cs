//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.PipelineLambdaTransforms
{
    public static class LambdaColumnHelper
    {
        /// <summary>
        /// Modifies or add another column by applying a lambda column.
        /// </summary>
        /// <typeparam name="TSrc">input type</typeparam>
        /// <typeparam name="TDst">output type</typeparam>
        /// <param name="env">environment</param>
        /// <param name="name">name to register</param>
        /// <param name="input">input view</param>
        /// <param name="src">name of the input column</param>
        /// <param name="dst">name of the output column</param>
        /// <param name="typeSrc">input column type</param>
        /// <param name="typeDst">output column type</param>
        /// <param name="mapper">mapper to apply</param>
        /// <returns>IDataView</returns>
        public static IDataView Create<TSrc, TDst>(IHostEnvironment env, string name, IDataView input,
                        string src, string dst, DataViewType typeSrc, DataViewType typeDst,
                        ValueMapper<TSrc, TDst> mapper)
        {
            return new LambdaColumnPassThroughView<TSrc, TDst>(env, name, input, src, dst, typeSrc, typeDst, mapper);
        }
    }

    /// <summary>
    /// This implement a view which runs a lambda function.
    /// The view is more simple than <see cref="LambdaColumnTransform"/>
    /// and tends to make it as a pass-through transform.
    /// </summary>
    public class LambdaColumnPassThroughView<TSrc, TDst> : IDataView
    {
        readonly IDataView _source;
        readonly string _columnSrc;
        readonly string _columnDst;
        readonly DataViewType _typeSrc;
        readonly DataViewType _typeDst;
        readonly ValueMapper<TSrc, TDst> _mapper;
        readonly DataViewSchema _newSchema;
        readonly IHost _host;
        readonly int _srcIndex;

        public IDataView SourceTags { get { return _source; } }
        protected IHost Host { get { return _host; } }

        public LambdaColumnPassThroughView(IHostEnvironment env, string name, IDataView input,
                        string src, string dst, DataViewType typeSrc, DataViewType typeDst,
                        ValueMapper<TSrc, TDst> mapper)
        {
            _host = env.Register(name);
            _source = input;
            _mapper = mapper;
            _columnDst = dst;
            _columnSrc = src;
            _typeDst = typeDst;
            _typeSrc = typeSrc;
            _newSchema = ExtendedSchema.Create(new ExtendedSchema(_source.Schema, new[] { dst }, new[] { typeDst }));
            _srcIndex = SchemaHelper.GetColumnIndex(_source.Schema, _columnSrc);
            _host.Except("Unable to find column '{0}' in input schema.", _columnSrc);
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public DataViewSchema Schema
        {
            get { return _newSchema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            if (columnsNeeded.Where(c => c.Index == _source.Schema.Count).Any())
            {
                var only = SchemaHelper.SchemaIntersection(_source.Schema, columnsNeeded, _srcIndex);
                var cursor = _source.GetRowCursor(only, rand);
                return new LambdaCursor(this, cursor);
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return new SameCursor(_source.GetRowCursor(columnsNeeded, rand), Schema);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            if (columnsNeeded.Where(c => c.Index == _source.Schema.Count).Any())
            {
                var only = SchemaHelper.SchemaIntersection(_source.Schema, columnsNeeded, _srcIndex);
                var cursors = _source.GetRowCursorSet(only, n, rand);
                return cursors.Select(c => new LambdaCursor(this, c)).ToArray();
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return _source.GetRowCursorSet(columnsNeeded, n, rand)
                              .Select(c => new SameCursor(c, Schema))
                              .ToArray();
        }

        public class LambdaCursor : DataViewRowCursor
        {
            readonly LambdaColumnPassThroughView<TSrc, TDst> _view;
            readonly DataViewRowCursor _inputCursor;

            public LambdaCursor(LambdaColumnPassThroughView<TSrc, TDst> view, DataViewRowCursor cursor)
            {
                _view = view;
                _inputCursor = cursor;
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                if (col.Index < _inputCursor.Schema.Count)
                    return _inputCursor.IsColumnActive(col);
                return true;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref DataViewRowId pos) =>
                {
                    getId(ref pos);
                };
            }

            public override long Batch { get { return _inputCursor.Batch; } }
            public override long Position { get { return _inputCursor.Position; } }
            public override DataViewSchema Schema { get { return _view.Schema; } }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveNext()
            {
                return _inputCursor.MoveNext();
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                if (col.Index < _view.SourceTags.Schema.Count)
                    return _inputCursor.GetGetter<TValue>(col);
                else if (col.Index == _view.SourceTags.Schema.Count)
                    return GetLambdaGetter() as ValueGetter<TValue>;
                else
                    throw _view.Host.Except("Column index {0} does not exist.", col);
            }

            private DataViewSchema.Column _dc(int i)
            {
                return new DataViewSchema.Column(null, i, false, null, null);
            }

            protected ValueGetter<TDst> GetLambdaGetter()
            {
                var getter = _inputCursor.GetGetter<TSrc>(_dc(_view._srcIndex));
                TSrc temp = default(TSrc);
                return (ref TDst dst) =>
                {
                    getter(ref temp);
                    _view._mapper(in temp, ref dst);
                };
            }
        }
    }
}
