// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Creates a prediction engine which does not create getters each time.
    /// It is much faster as it does not recreate getter for every observation.
    /// </summary>
    public class ValueMapperPredictionEngine<TRowValue> : IDisposable
        where TRowValue : class, IClassWithGetter<TRowValue>, new()
    {
        #region result type

        public class PredictionTypeForBinaryClassification : IClassWithSetter<PredictionTypeForBinaryClassification>
        {
            public bool PredictedLabel;
            public float Score;
            public float Probability;

            public Delegate[] GetCursorGetter(DataViewRowCursor cursor)
            {
                int indexL = SchemaHelper.GetColumnIndex(cursor.Schema, "PredictedLabel");
                int indexS = SchemaHelper.GetColumnIndex(cursor.Schema, "Score");
                int indexP = SchemaHelper.GetColumnIndex(cursor.Schema, "Probability");
                return new Delegate[]
                {
                    cursor.GetGetter<bool>(indexL),
                    cursor.GetGetter<float>(indexS),
                    cursor.GetGetter<float>(indexP),
                };
            }

            public void Set(Delegate[] delegates)
            {
                var del1 = delegates[0] as ValueGetter<bool>;
                del1(ref PredictedLabel);
                var del2 = delegates[1] as ValueGetter<float>;
                del2(ref Score);
                var del3 = delegates[2] as ValueGetter<float>;
                del3(ref Probability);
            }
        }

        #endregion

        readonly IHostEnvironment _env;
        readonly IDataView _transforms;
        readonly IPredictor _predictor;

        ValueMapper<TRowValue, PredictionTypeForBinaryClassification> _mapperBinaryClassification;
        IDisposable _valueMapper;

        public ValueMapperPredictionEngine()
        {
            throw Contracts.Except("Use arguments.");
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelName">filename</param>
        /// <param name="conc">number of concurrency threads</param>
        /// <param name="features">features name</param>
        public ValueMapperPredictionEngine(IHostEnvironment env, string modelName,
                bool outputIsFloat = true, int conc = 1, string features = "Features") :
            this(env, File.OpenRead(modelName), conc, features)
        {
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelStream">stream</param>
        /// <param name="conc">number of concurrency threads</param>
        /// <param name="features">features column</param>
        public ValueMapperPredictionEngine(IHostEnvironment env, Stream modelStream,
                                           int conc = 1, string features = "Features")
        {
            _env = env;
            if (_env == null)
                throw Contracts.Except("env must not be null");
            var inputs = new TRowValue[0];
            var view = DataViewConstructionUtils.CreateFromEnumerable<TRowValue>(_env, inputs);

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
            _CreateMapper(scorer, conc);
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelStream">stream</param>
        /// <param name="output">name of the output column</param>
        /// <param name="outputIsFloat">output is a gloat (true) or a vector of floats (false)</param>
        /// <param name="conc">number of concurrency threads</param>
        public ValueMapperPredictionEngine(IHostEnvironment env, IDataScorerTransform scorer, int conc = 1)
        {
            _env = env;
            if (_env == null)
                throw Contracts.Except("env must not be null");
            _CreateMapper(scorer, conc);
        }

        void _CreateMapper(IDataScorerTransform scorer, int conc)
        {
            _mapperBinaryClassification = null;
            var schema = scorer.Schema;
            int i1, i2, i3;
            i1 = SchemaHelper.GetColumnIndex(schema, "PredictedLabel");
            i2 = SchemaHelper.GetColumnIndex(schema, "Score");
            i3 = SchemaHelper.GetColumnIndex(schema, "Probability");
            var map = new ValueMapperFromTransform<TRowValue, PredictionTypeForBinaryClassification>(_env,
                                scorer, conc: conc);
            _mapperBinaryClassification = map.GetMapper<TRowValue, PredictionTypeForBinaryClassification>();
            _valueMapper = map;
        }

        public void Dispose()
        {
            _valueMapper.Dispose();
            _valueMapper = null;
        }

        /// <summary>
        /// Produces prediction for a binary classification.
        /// </summary>
        /// <param name="features">features</param>
        /// <param name="res">prediction</param>
        public void Predict(TRowValue features, ref PredictionTypeForBinaryClassification res)
        {
            if (_mapperBinaryClassification != null)
                _mapperBinaryClassification(in features, ref res);
            else
                throw _env.Except("Unrecognized machine learn problem.");
        }
    }
}
