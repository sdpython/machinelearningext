// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Converts a <see cref="IDataTransform" /> into a <see cref="ValueMapper" />.
    /// This mapper is far from being efficient (execution time).
    /// The mapper creates a new view, a new iterator, serializes and deserializes the transform for each row.
    /// The serialization can be avoided by exposing a setter on _source.
    /// The class is not multithreaded, it is safer to use it from the same thread.
    /// </summary>
    public class ValueMapperFromTransformFloat<TColValue> : IValueMapper, IDisposable
    {
        public DataViewType InputType { get { return _transform.Source.Schema[_inputIndex].Type; } }
        public DataViewType OutputType { get { return _outputType; } }

        readonly IDataTransform _transform;
        readonly IDataView _sourceToReplace;
        readonly IHostEnvironment _env;
        readonly string _outputColumn;
        readonly int _inputIndex;
        readonly DataViewType _outputType;
        IHostEnvironment _computeEnv;
        readonly bool _disposeEnv;
        readonly bool _ignoreOtherColumn;
        List<IDisposable> _toDispose;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="transform">transform to convert</param>
        /// <param name="inputColumn">input column of the mapper</param>
        /// <param name="outputColumn">output column of the mapper</param>
        /// <param name="sourceToReplace">source to replace</param>
        /// <param name="conc">number of concurrency threads</param>
        /// <param name="ignoreOtherColumn">ignore other columns instead of raising an exception if they are requested</param>
        public ValueMapperFromTransformFloat(IHostEnvironment env, IDataTransform transform,
                                             string inputColumn, string outputColumn,
                                             IDataView sourceToReplace = null, int conc = 1,
                                             bool ignoreOtherColumn = false)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(transform);
            _env = env;
            _transform = transform;
            _sourceToReplace = sourceToReplace;
            _outputColumn = outputColumn;
            _ignoreOtherColumn = ignoreOtherColumn;
            _toDispose = new List<IDisposable>();

            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(transform);

            int index = SchemaHelper.GetColumnIndex(firstView.Schema, inputColumn);
            _inputIndex = index;
            index = SchemaHelper.GetColumnIndex(transform.Schema, outputColumn);
            _outputType = _transform.Schema[index].Type;

            _disposeEnv = false;// conc > 0;
            _computeEnv = _disposeEnv ? new PassThroughEnvironment(env, verbose: false) : env;
        }

        public void Dispose()
        {
            if (_disposeEnv)
            {
                //(_computeEnv as PassThroughEnvironment).Dispose();
                _computeEnv = null;
            }
            foreach (var disp in _toDispose)
                disp.Dispose();
            _toDispose.Clear();
        }

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            var mapper = GetMapperDispose<TSrc, TDst>();
            _toDispose.Add(mapper);
            return mapper.GetMapper<TSrc, TDst>();
        }

        ValueMapperDispose<TSrc, TDst> GetMapperDispose<TSrc, TDst>()
        {
            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(_transform);
            var inputView = new InfiniteLoopViewCursorColumn<TSrc>(_inputIndex, firstView.Schema, ignoreOtherColumn: _ignoreOtherColumn);

            // This is extremely time consuming as the transform is serialized and deserialized.
            var outputView = _sourceToReplace == _transform.Source
                                ? ApplyTransformUtils.ApplyTransformToData(_computeEnv, _transform, inputView)
                                : ApplyTransformUtils.ApplyAllTransformsToData(_computeEnv, _transform, inputView,
                                                                               _sourceToReplace);
            var index = SchemaHelper.GetColumnIndexDC(outputView.Schema, _outputColumn);
            var newOutputIndex = index;
            var cur = outputView.GetRowCursor(outputView.Schema.Where(c => c.Index == newOutputIndex.Index).ToArray());
            var getter = cur.GetGetter<TDst>(newOutputIndex);
            if (getter == null)
                throw _env.Except("Unable to get a getter on the transform for type {0}", default(TDst).GetType());
            return new ValueMapperDispose<TSrc, TDst>((in TSrc src, ref TDst dst) =>
            {
                inputView.Set(in src);
                cur.MoveNext();
                getter(ref dst);
            }, new IDisposable[] { cur });
        }
    }
}
