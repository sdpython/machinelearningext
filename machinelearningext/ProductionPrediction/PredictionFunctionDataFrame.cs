// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Scikit.ML.DataManipulation;
using ITransformer = Microsoft.ML.ITransformer;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// A prediction engine class, that takes instances of <see cref="DataFrame"/> through
    /// the transformer pipeline and produces instances of <see cref="DataFrame"/> as outputs.
    /// </summary>
    public class PredictionFunctionDataFrame
    {
        private ValueMapperDataFrameFromTransform _fastValueMapperObject;
        private ValueMapper<DataFrame, DataFrame> _fastValueMapper;

        /// <summary>
        /// Creates an instance of <see cref="PredictionFunctionDataFrame"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="transformer">The model (transformer) to use for prediction.</param>
        /// <param name="inputSchema">Input schema.</param>
        /// <param name="conc">Number of threads.</param>
        public PredictionFunctionDataFrame(IHostEnvironment env, ITransformer transformer, DataViewSchema inputSchema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));
            var df = new DataFrame(transformer.GetOutputSchema(inputSchema), 0);
            var tr = transformer.Transform(df) as IDataTransform;
            _fastValueMapperObject = new ValueMapperDataFrameFromTransform(env, tr);
            _fastValueMapper = _fastValueMapperObject.GetMapper<DataFrame, DataFrame>();
        }

        /// <summary>
        /// Performs one or several predictions using the model.
        /// </summary>
        /// <param name="example">The object that holds values to predict from.</param>
        /// <returns>The object populated with prediction results.</returns>
        public DataFrame Predict(DataFrame src)
        {
            DataFrame res = null;
            Predict(src, ref res);
            return res;
        }

        /// <summary>
        /// Performs one or several predictions using the model.
        /// Reuses the provided prediction object, which is more efficient in high-load scenarios.
        /// </summary>
        /// <param name="example">The object that holds values to predict from.</param>
        /// <param name="prediction">The object to store the predictions in. If it's <c>null</c>, a new object is created,
        /// otherwise the provided object is used.</param>
        public void Predict(DataFrame example, ref DataFrame prediction)
        {
            _fastValueMapper(in example, ref prediction);
        }
    }

    public static class PredictionFunctionDataFrameExtensions
    {
        /// <summary>
        /// Create an instance of the 'prediction function', or 'prediction machine', from a model
        /// denoted by <paramref name="transformer"/>.
        /// It will be accepting instances of <typeparamref name="TSrc"/> as input, and produce
        /// instances of <typeparamref name="TDst"/> as output.
        /// </summary>
        public static PredictionFunctionDataFrame MakePredictionFunctionDataFrame(this ITransformer transformer, IHostEnvironment env, DataViewSchema inputSchema)
            => new PredictionFunctionDataFrame(env, transformer, inputSchema);
    }
}
