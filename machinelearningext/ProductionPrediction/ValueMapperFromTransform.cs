// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Converts a <see cref="IDataTransform" /> into a <see cref="ValueMapper" />.
    /// This mapper is far from being efficient (execution time).
    /// The mapper creates a new view, a new iterator, serializes and deserializes the transform for each row.
    /// The serialization can be avoided by exposing a setter on _source.
    /// </summary>
    public class ValueMapperFromTransform<TRowInput, TRowOutput> : IValueMapper, IDisposable
        where TRowInput : class, IClassWithGetter<TRowInput>, new()
        where TRowOutput : class, IClassWithSetter<TRowOutput>, new()
    {
        readonly IDataTransform _transform;
        readonly IDataView _sourceToReplace;
        readonly IHostEnvironment _env;
        IHostEnvironment _computeEnv;
        readonly bool _disposeEnv;
        ValueMapperDispose<TRowInput, TRowOutput> _valueMapperDispose;

        public DataViewType InputType => null;
        public DataViewType OutputType => null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="transform">transform to convert</param>
        /// <param name="sourceToReplace">source to replace</param>
        /// <param name="conc">number of concurrency threads</param>
        public ValueMapperFromTransform(IHostEnvironment env, IDataTransform transform,
                                        IDataView sourceToReplace = null, int conc = 1)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(transform);
            _env = env;
            _transform = transform;
            _sourceToReplace = sourceToReplace;
            _disposeEnv = conc > 0;
            _computeEnv = _disposeEnv ? new PassThroughEnvironment(env, conc: conc, verbose: false) : env;
            _valueMapperDispose = GetMapperDispose();
        }

        public void Dispose()
        {
            if (_disposeEnv)
            {
                (_computeEnv as PassThroughEnvironment).Dispose();
                _computeEnv = null;
            }
            if (_valueMapperDispose != null)
            {
                _valueMapperDispose.Dispose();
                _valueMapperDispose = null;
            }
        }

        delegate void DelegateSetterRow<TDst, TValue>(ref TDst row, ref TValue value);

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            var disp = _valueMapperDispose as ValueMapperDispose<TSrc, TDst>;
            if (disp == null)
                throw _env.Except($"Cannot create a mapper, probable issue with requested types.");
            return disp.GetMapper<TSrc, TDst>();
        }

        ValueMapperDispose<TRowInput, TRowOutput> GetMapperDispose()
        {
            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(_transform);
            var schema = SchemaDefinition.Create(typeof(TRowOutput), SchemaDefinition.Direction.Read);
            var inputView = new InfiniteLoopViewCursorRow<TRowInput>(null, firstView.Schema,
                                overwriteRowGetter: GetterSetterHelper.GetGetter<TRowInput>());

            // This is extremely time consuming as the transform is serialized and deserialized.
            var outputView = _sourceToReplace == _transform.Source
                                ? ApplyTransformUtils.ApplyTransformToData(_computeEnv, _transform, inputView)
                                : ApplyTransformUtils.ApplyAllTransformsToData(_computeEnv, _transform, inputView, _sourceToReplace);

            // We assume all columns are needed, otherwise they should be removed.
            using (var cur = outputView.GetRowCursor(outputView.Schema))
            {
                Delegate[] dels;
                try
                {
                    dels = new TRowOutput().GetCursorGetter(cur);
                }
                catch (InvalidOperationException e)
                {
                    throw new InvalidOperationException($"Unable to create getter for the schema\n{SchemaHelper.ToString(cur.Schema)}", e);
                }

                return new ValueMapperDispose<TRowInput, TRowOutput>((in TRowInput src, ref TRowOutput dst) =>
                {
                    inputView.Set(in src);
                    cur.MoveNext();
                    dst.Set(dels);
                }, new IDisposable[] { cur });
            }
        }
    }
}
