// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    public class ValueMapperDispose<TSrc, TDst> : IDisposable, IValueMapper
    {
        public ValueMapper<TSrc, TDst> _mapper;
        public IDisposable[] _toDispose;

        public DataViewType InputType => SchemaHelper.GetColumnType<TSrc>();
        public DataViewType OutputType => SchemaHelper.GetColumnType<TDst>();

        public ValueMapperDispose(ValueMapper<TSrc, TDst> mapper, IDisposable[] toDispose)
        {
            _mapper = mapper;
            _toDispose = toDispose;
        }

        public ValueMapper<TSrc2, TDst2> GetMapper<TSrc2, TDst2>()
        {
            Contracts.Assert(_mapper != null, nameof(_mapper));
            var mapper = GetInnerMapper() as ValueMapper<TSrc2, TDst2>;
            if (mapper == null)
                throw Contracts.Except($"Unable to create a mapper. Probable issue with requested types.");
            return mapper;
        }

        ValueMapper<TSrc, TDst> GetInnerMapper()
        {
            return (in TSrc src, ref TDst dst) =>
            {
                _mapper(in src, ref dst);
            };
        }

        public void Dispose()
        {
            if (_toDispose != null)
            {
                foreach (var disp in _toDispose)
                    disp.Dispose();
                _toDispose = null;
            }
            _mapper = null;
        }
    }
}
