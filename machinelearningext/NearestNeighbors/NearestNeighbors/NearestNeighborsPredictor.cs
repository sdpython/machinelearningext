// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsPredictor
    {
        protected int _k;
        protected NearestNeighborsWeights _weights;
        protected NearestNeighborsAlgorithm _algo;

        protected IHost _host;
        protected NearestNeighborsTrees _nearestTrees;
        protected INearestNeighborsValueMapper _nearestPredictor;

        public DataViewType InputType { get { return _nearestTrees.InputType; } }

        public void SaveCore(ModelSaveContext ctx)
        {
            ctx.Writer.Write(_k);
            ctx.Writer.Write((int)_algo);
            ctx.Writer.Write((int)_weights);
            _nearestTrees.Save(ctx);
            ctx.Writer.Write((int)_nearestPredictor.Kind);
            _nearestPredictor.SaveCore(ctx);
        }

        protected void ReadCore(IHost host, ModelLoadContext ctx)
        {
            _k = ctx.Reader.ReadInt32();
            _algo = (NearestNeighborsAlgorithm)ctx.Reader.ReadInt32();
            _weights = (NearestNeighborsWeights)ctx.Reader.ReadInt32();
            _nearestTrees = new NearestNeighborsTrees(_host, ctx);
            _host.CheckValue(_nearestTrees, "_nearestTrees");
            var kind_ = ctx.Reader.ReadInt32();
            var kind = (DataKind)kind_;
            switch (kind)
            {
                case DataKind.Boolean:
                    _nearestPredictor = new NearestNeighborsValueMapper<bool>(host, ctx);
                    break;
                case DataKind.SByte:
                    _nearestPredictor = new NearestNeighborsValueMapper<byte>(host, ctx);
                    break;
                case DataKind.UInt16:
                    _nearestPredictor = new NearestNeighborsValueMapper<ushort>(host, ctx);
                    break;
                case DataKind.UInt32:
                    _nearestPredictor = new NearestNeighborsValueMapper<uint>(host, ctx);
                    break;
                case DataKind.Single:
                    _nearestPredictor = new NearestNeighborsValueMapper<float>(host, ctx);
                    break;
                default:
                    throw _host.ExceptNotSupp("Not suported kind={0}", kind);
            }
            _host.CheckValue(_nearestPredictor, "_nearestPredictor");
        }
    }
}
