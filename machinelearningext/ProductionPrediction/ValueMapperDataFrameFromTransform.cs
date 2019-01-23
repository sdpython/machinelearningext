// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.DataManipulation;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Converts a <see cref="IDataTransform" /> into a <see cref="ValueMapper" />.
    /// This mapper is far from being efficient (execution time).
    /// The mapper creates a new view, a new iterator, serializes and deserializes the transform for each row.
    /// The serialization can be avoided by exposing a setter on _source.
    /// </summary>
    public class ValueMapperDataFrameFromTransform : IValueMapper, IDisposable
    {
        readonly IDataTransform _transform;
        readonly IDataView _sourceToReplace;
        readonly IHostEnvironment _env;
        IHostEnvironment _computeEnv;
        readonly bool _disposeEnv;

        public ColumnType InputType => null;
        public ColumnType OutputType => null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="transform">transform to convert</param>
        /// <param name="sourceToReplace">source to replace</param>
        /// <param name="conc">number of concurrency threads</param>
        public ValueMapperDataFrameFromTransform(IHostEnvironment env, IDataTransform transform,
                                                 IDataView sourceToReplace = null,
                                                 int conc = 1)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(transform);
            _env = env;
            _transform = transform;
            _sourceToReplace = sourceToReplace;
            _disposeEnv = conc > 0;
            _computeEnv = _disposeEnv ? new PassThroughEnvironment(env, conc: conc, verbose: false) : env;
        }

        public void Dispose()
        {
            if (_disposeEnv)
            {
                (_computeEnv as PassThroughEnvironment).Dispose();
                _computeEnv = null;
            }
        }

        delegate void DelegateSetterRow<TDst, TValue>(ref TDst row, ref TValue value);

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            if (typeof(TSrc) != typeof(DataFrame))
                throw _env.Except($"Cannot create a mapper with input type {typeof(TSrc)} != {typeof(DataFrame)} (expected).");

            ValueMapper<TSrc, TDst> res;
            if (typeof(TDst) == typeof(DataFrame))
                res = GetMapperRow() as ValueMapper<TSrc, TDst>;
            else
                res = GetMapperColumn<TDst>() as ValueMapper<TSrc, TDst>;
            if (res == null)
                throw _env.ExceptNotSupp($"Unable to create mapper from {typeof(TSrc)} to {typeof(TDst)}.");
            return res;
        }

        /// <summary>
        /// Returns a getter on particuler column.
        /// </summary>
        ValueMapper<DataFrame, TDst> GetMapperColumn<TDst>()
        {
            throw _env.ExceptNotImpl("Not implemented yet as it is missing the column index.");
        }

        ValueMapper<DataFrame, DataFrame> GetMapperRow()
        {
            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(_transform);
            var schema = firstView.Schema;

            var inputView = new InfiniteLoopViewCursorDataFrame(null, firstView.Schema);

            // This is extremely time consuming as the transform is serialized and deserialized.
            var outputView = _sourceToReplace == _transform.Source
                                ? ApplyTransformUtils.ApplyTransformToData(_computeEnv, _transform, inputView)
                                : ApplyTransformUtils.ApplyAllTransformsToData(_computeEnv, _transform, inputView, _sourceToReplace);

            // We assume all columns are needed, otherwise they should be removed.
            using (var cur = outputView.GetRowCursor(outputView.Schema))
            {
                var getRowFiller = DataFrame.GetRowFiller(cur);

                return (in DataFrame src, ref DataFrame dst) =>
                {
                    if (dst is null)
                        dst = new DataFrame(outputView.Schema, src.Length);
                    else if (!dst.CheckSharedSchema(outputView.Schema))
                        throw _env.Except($"DataFrame does not share the same schema, expected {SchemaHelper.ToString(outputView.Schema)}.");
                    dst.Resize(src.Length);

                    inputView.Set(src);
                    for (int i = 0; i < src.Length; ++i)
                    {
                        cur.MoveNext();
                        getRowFiller(dst, i);
                    }
                };
            }
        }
    }
}
