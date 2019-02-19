// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;


namespace Scikit.ML.ProductionPrediction
{
    public class FloatVectorInput
    {
        /// Dummy field with a known dimension.
        /// Known or unknown dimension is checked at loading time.
        /// The dimension cannot be checked.
        [VectorType(1)]
        public float[] Features;
    }

    /// <summary>
    /// Creates a prediction engine which does not create getters each time.
    /// It is much faster as it does not recreate getter for every observation.
    /// </summary>
    public class ValueMapperPredictionEngineFloat : IDisposable
    {
        readonly IHostEnvironment _env;
        readonly IDataView _transforms;
        readonly IPredictor _predictor;
        readonly ValueMapper<VBuffer<float>, float> _mapper;
        readonly ValueMapper<VBuffer<float>, VBuffer<float>> _mapperVector;
        ValueMapperFromTransformFloat<VBuffer<float>> _valueMapper;

        public ValueMapperPredictionEngineFloat()
        {
            throw Contracts.Except("Use arguments.");
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelName">filename</param>
        /// <param name="output">name of the output column</param>
        /// <param name="outputIsFloat">output is a gloat (true) or a vector of floats (false)</param>
        /// <param name="conc">number of concurrency threads</param>
        /// <param name="features">features name</param>
        public ValueMapperPredictionEngineFloat(IHostEnvironment env, string modelName,
                string output = "Probability", bool outputIsFloat = true, int conc = 1,
                string features = "Features") :
            this(env, File.OpenRead(modelName), output, outputIsFloat, conc, features)
        {
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelStream">stream</param>
        /// <param name="output">name of the output column</param>
        /// <param name="outputIsFloat">output is a gloat (true) or a vector of floats (false)</param>
        /// <param name="conc">number of concurrency threads</param>
        /// <param name="features">features name</param>
        public ValueMapperPredictionEngineFloat(IHostEnvironment env, Stream modelStream,
                string output = "Probability", bool outputIsFloat = true, int conc = 1,
                string features = "Features")
        {
            _env = env;
            if (_env == null)
                throw Contracts.Except("env must not be null");
            var inputs = new FloatVectorInput[0];
            var view = DataViewConstructionUtils.CreateFromEnumerable<FloatVectorInput>(_env, inputs);

            long modelPosition = modelStream.Position;
            _predictor = ModelFileUtils.LoadPredictorOrNull(_env, modelStream);
            if (_predictor == null)
                throw _env.Except("Unable to load a model.");
            modelStream.Seek(modelPosition, SeekOrigin.Begin);
            _transforms = ModelFileUtils.LoadTransforms(_env, view, modelStream);
            if (_transforms == null)
                throw _env.Except("Unable to load a model.");

            var data = _env.CreateExamples(_transforms, features);
            if (data == null)
                throw _env.Except("Cannot create rows.");
            var scorer = _env.CreateDefaultScorer(data, _predictor);
            if (scorer == null)
                throw _env.Except("Cannot create a scorer.");

            _valueMapper = new ValueMapperFromTransformFloat<VBuffer<float>>(_env,
                                scorer, features, output, conc: conc);
            if (_valueMapper == null)
                throw _env.Except("Cannot create a mapper.");
            if (outputIsFloat)
            {
                _mapper = _valueMapper.GetMapper<VBuffer<float>, float>();
                _mapperVector = null;
            }
            else
            {
                _mapper = null;
                _mapperVector = _valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();
            }
        }

        public void Dispose()
        {
            _valueMapper.Dispose();
            _valueMapper = null;
        }

        /// <summary>
        /// Produces prediction assuming the input accepts a features vectors as inputs.
        /// </summary>
        /// <param name="features">feature vector</param>
        /// <returns>prediction as float</returns>
        public float Predict(float[] features)
        {
            if (_mapper == null)
                throw _env.Except("The mapper is outputting a vector not a float.");
            float res = 0f;
            var buf = new VBuffer<float>(features.Length, features);
            _mapper(in buf, ref res);
            return res;
        }

        /// <summary>
        /// Produces prediction assuming the input accepts a features vectors as inputs.
        /// </summary>
        /// <param name="features">feature vector</param>
        /// <returns>predictions</returns>
        public float[] PredictVector(float[] features)
        {
            if (_mapperVector == null)
                throw _env.Except("The mapper is outputting a float not a vector.");
            VBuffer<float> res = new VBuffer<float>();
            var buf = new VBuffer<float>(features.Length, features);
            _mapperVector(in buf, ref res);
            if (!res.IsDense)
                throw _env.Except("The output of the predictor or transform must be dense.");
            return res.DenseValues().ToArray();
        }
    }
}
