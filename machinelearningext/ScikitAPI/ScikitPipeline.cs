// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Model.OnnxConverter;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.OnnxHelper;


namespace Scikit.ML.ScikitAPI
{
    /// <summary>
    /// Creates an interface easy to use and close to what others languages offer.
    /// The user should be careful when Predict or Transform methods are used
    /// in a multi-threaded environment.
    /// Predict and Transform methods are equivalent.
    /// </summary>
    public class ScikitPipeline : IDisposable
    {
        #region members

        public enum SaveFormat
        {
            Zip = 1,
            Onnx = 2,
        };

        public class StepTransform
        {
            public string transformSettings;
            public IDataTransform transform;
        }

        public class StepPredictor
        {
            public string trainerSettings;
            public ITrainerExtended trainer;
            public IPredictor predictor;
            public RoleMappedData roleMapData;
        }

        private IHostEnvironment _env;
        private StepTransform[] _transforms;
        private StepPredictor _predictor;
        private string _loaderSettings;
        private List<KeyValuePair<RoleMappedSchema.ColumnRole, string>> _roles;
        private bool _dispose;
        private ValueMapperDataFrameFromTransform _fastValueMapperObject;
        private ValueMapper<DataFrame, DataFrame> _fastValueMapper;

        #endregion

        #region constructor

        /// <summary>
        /// Initializes the pipeline.
        /// </summary>
        /// <param name="transforms">list of transform, can be empty</param>
        /// <param name="predictor">a predictor, can be empty</param>
        /// <param name="host">can be empty too, a <see cref="ExtendedConsoleEnvironment"/> is then created</param>
        public ScikitPipeline(string[] transforms = null,
                              string predictor = null,
                              IHostEnvironment host = null)
        {
            _dispose = false;
            _env = host ?? ExtendedConsoleEnvironment();
            _transforms = new StepTransform[transforms == null ? 1 : transforms.Length + 1];
            // We add a PassThroughTransform to be able to change the source.
            _transforms[0] = new StepTransform() { transformSettings = "pass", transform = null };
            if (transforms != null)
                for (int i = 0; i < transforms.Length; ++i)
                    _transforms[i + 1] = new StepTransform() { transformSettings = transforms[i], transform = null };
            _predictor = predictor == null ? null : new StepPredictor()
            {
                trainerSettings = predictor,
                predictor = null,
                trainer = null,
                roleMapData = null
            };
            _loaderSettings = null;
            _roles = null;
            _fastValueMapper = null;
            _fastValueMapperObject = null;
        }

        public void Dispose()
        {
            if (_dispose)
            {
                // (_env as ConsoleEnvironment).Dispose();
                _env = null;
            }
        }

        private IHostEnvironment ExtendedConsoleEnvironment()
        {
            _dispose = true;
            var env = new ConsoleEnvironment();
            ComponentHelper.AddStandardComponents(env);
            return env;
        }

        /// <summary>
        /// Initializes a pipeline based on an a filename.
        /// The pipeline can restore a pipeline saved into Onnx format but
        /// it cannot be trained again. You need to save it as a zip format to do so.
        /// </summary>
        /// <param name="filename">zip</param>
        /// <param name="host">can be empty too, a <see cref="ExtendedConsoleEnvironment"/> is then created</param>
        public ScikitPipeline(string filename, IHostEnvironment host = null)
        {
            _dispose = false;
            _env = host ?? ExtendedConsoleEnvironment();
            using (var st = File.OpenRead(filename))
            {
                if (filename.EndsWith(".zip"))
                    Load(st);
                else
                    LoadOnnx(st);
            }
        }

        /// <summary>
        /// Initializes a pipeline based on an a stream.
        /// </summary>
        /// <param name="filename">stream on a zip file</param>
        /// <param name="host">can be empty too, a <see cref="ExtendedConsoleEnvironment"/> is then created</param>
        public ScikitPipeline(Stream st, SaveFormat format = SaveFormat.Zip, IHostEnvironment host = null)
        {
            _dispose = false;
            _env = host ?? ExtendedConsoleEnvironment();
            switch (format)
            {
                case SaveFormat.Zip:
                    Load(st);
                    break;
                case SaveFormat.Onnx:
                    LoadOnnx(st);
                    break;
                default:
                    throw _env.Except($"Unable to save pipeline into format {format}.");
            }
        }

        /// <summary>
        /// Loads a pipeline saved in zip format.
        /// </summary>
        protected void Load(Stream fs)
        {
            var transformPipe = ModelFileUtils.LoadPipeline(_env, fs, new MultiFileSource(null), true);
            var ipred = ModelFileUtils.LoadPredictorOrNull(_env, fs);

            IDataView root;
            for (root = transformPipe; root is IDataTransform && !(root is PassThroughTransform); root = ((IDataTransform)root).Source) ;
            if (!(root is PassThroughTransform))
            {
                var tr = new PassThroughTransform(_env, new PassThroughTransform.Arguments(), root);
                transformPipe = ApplyTransformUtils.ApplyAllTransformsToData(_env, transformPipe, tr, root);
            }

            var stack = new List<IDataView>();
            for (root = transformPipe; root is IDataTransform; root = ((IDataTransform)root).Source)
                stack.Add(root);
            stack.Reverse();

            _transforms = new StepTransform[stack.Count];
            for (int i = 0; i < _transforms.Length; ++i)
                _transforms[i] = new StepTransform() { transform = stack[i] as IDataTransform, transformSettings = null };

            if (ipred == null)
                _predictor = new StepPredictor() { predictor = null, roleMapData = null, trainer = null, trainerSettings = null };
            else
            {
                _roles = ModelFileUtils.LoadRoleMappingsOrNull(_env, fs).ToList();
                var data = new RoleMappedData(transformPipe, _roles);
                _predictor = new StepPredictor() { predictor = ipred, roleMapData = data, trainer = null, trainerSettings = null };
            }
            _fastValueMapper = null;
        }

        /// <summary>
        /// Loads a pipeline saved in onnx format.
        /// </summary>
        protected void LoadOnnx(Stream fs)
        {
            var root = new PassThroughTransform(_env, new PassThroughTransform.Arguments(), null);
            _transforms = new StepTransform[2];
            _transforms[0] = new StepTransform() { transform = root as IDataTransform, transformSettings = null };
            _transforms[1] = new StepTransform() { transform = ConvertFromOnnx.ReadOnnx(fs, root), transformSettings = null };
            _predictor = new StepPredictor() { predictor = null, roleMapData = null, trainer = null, trainerSettings = null };
            _fastValueMapper = null;
        }

        /// <summary>
        /// Saves the pipeline as a filename.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="removeFirstTransform">remove the first transform which is a PassThroughTransform</param>
        public void Save(string filename, bool removeFirstTransform = false)
        {
            using (var fs = File.Create(filename))
                if (filename.EndsWith(".zip"))
                    Save(fs, removeFirstTransform);
                else
                    ToOnnx(removeFirstTransform ? 1 : 0).Save(fs);
        }

        /// <summary>
        /// Saves the pipeline in a stream.
        /// </summary>
        /// <param name="fs">opened stream</param>
        /// <param name="removeFirstTransform">remove the first transform which is a PassThroughTransform</param>
        public void Save(Stream fs, bool removeFirstTransform = false)
        {
            RoleMappedData roleMap = null;
            if (removeFirstTransform)
            {
                var source = _transforms.First().transform;
                if (!(source is PassThroughTransform))
                    throw Contracts.ExceptNotSupp($"The first transform should be of type PassThroughTransform.");
                var replace = (source as PassThroughTransform).Source;
                var last = _transforms.Last().transform;
                var newPipe = ApplyTransformUtils.ApplyAllTransformsToData(_env, last, replace, source);
                var roles = _predictor.roleMapData.Schema.GetColumnRoleNames().ToArray();
                roleMap = new RoleMappedData(newPipe, roles);
            }
            else
                roleMap = _predictor.roleMapData;
            using (var ch = _env.Start("Save Predictor"))
                TrainUtils.SaveModel(_env, ch, fs, _predictor.predictor, roleMap);
        }

        #endregion

        #region train

        /// <summary>
        /// Trains the pipeline with data stored in a filename.
        /// </summary>
        public ScikitPipeline Train(string loaderSettings, string filename,
                                    string feature = "Feature", string label = null,
                                    string weight = null, string groupId = null)
        {
            _loaderSettings = loaderSettings;
            var loader = _env.CreateLoader(loaderSettings, new MultiFileSource(filename));
            return Train(loader);
        }

        /// <summary>
        /// Trains the pipeline with data coming from a <see cref="IDataView"/>.
        /// </summary>
        public ScikitPipeline Train(IDataView data,
                                    string feature = "Feature", string label = null,
                                    string weight = null, string groupId = null)
        {
            IDataView trans = data;
            using (var ch = _env.Start("Create transforms"))
            {
                for (int i = 0; i < _transforms.Length; ++i)
                {
                    try
                    {
                        trans = _env.CreateTransform(_transforms[i].transformSettings, trans);
                    }
                    catch (Exception e)
                    {
                        if (e.ToString().Contains("Unknown loadable class"))
                        {
                            var nn = _env.ComponentCatalog.GetAllClasses().Length;
                            var filt = _env.ComponentCatalog.GetAllClasses()
                                                            .Select(c => c.UserName)
                                                            .OrderBy(c => c)
                                                            .Where(c => c.Trim().Length > 2);
                            var regis = string.Join("\n", filt);
                            throw Contracts.Except(e, $"Unable to create transform '{_transforms[i].transformSettings}', assembly not registered among {nn}\n{regis}");
                        }
                        throw e;
                    }
                    _transforms[i].transform = trans as IDataTransform;
                }
            }

            if (_predictor != null)
            {
                using (var ch = _env.Start("Create Predictor"))
                {
                    _predictor.trainer = TrainerHelper.CreateTrainer(_env, _predictor.trainerSettings);
                    _roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
                    _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, feature));
                    if (!string.IsNullOrEmpty(label))
                        _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, label));
                    if (!string.IsNullOrEmpty(groupId))
                        _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Group, groupId));
                    if (!string.IsNullOrEmpty(weight))
                        _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Weight, weight));
                    var roleMap = new RoleMappedData(trans, label, feature, group: groupId, weight: weight);
                    _predictor.predictor = _predictor.trainer.Train(_env, ch, roleMap);
                    _predictor.roleMapData = roleMap;
                }
            }
            else
            {
                _predictor = new StepPredictor()
                {
                    predictor = null,
                    trainer = null,
                    trainerSettings = null,
                    roleMapData = new RoleMappedData(trans)
                };

                // We predict one to make sure everything works fine.
                using (var ch = _env.Start("Compute one prediction."))
                {
                    var df = DataFrameIO.ReadView(trans, 1, keepVectors: true, env: _env);
                    if (df.Length == 0)
                        throw _env.ExceptEmpty("Something went wrong. The pipeline does not produce any output.");
                }
            }
            return this;
        }

        #endregion

        #region regular predict or transform

        /// <summary>
        /// Computes the predictions for data stored in a filename.
        /// It must follow the same schema as the training data.
        /// This method can be called from any thread as it creates new getters.
        /// </summary>
        public IDataView Transform(string filename)
        {
            return Predict(filename);
        }

        /// <summary>
        /// Computes the predictions for data stored in a filename.
        /// It must follow the same schema as the training data.
        /// This method can be called from any thread as it creates new getters.
        /// </summary>
        public IDataView Predict(string filename)
        {
            if (string.IsNullOrEmpty(_loaderSettings))
                throw _env.Except("A dataframe was used for training, another one must be used for prediction.");
            var loader = _env.CreateLoader(_loaderSettings, new MultiFileSource(filename));
            return Predict(loader);
        }

        /// <summary>
        /// Computes the predictions for data stored in a <see cref="IDataView"/>.
        /// It must follow the same schema as the training data.
        /// This method can be called from any thread as it creates new getters.
        /// </summary>
        public IDataView Transform(IDataView data)
        {
            return Predict(data);
        }

        /// <summary>
        /// Computes the predictions for data stored in a <see cref="StreamingDataFrame"/>.
        /// It must follow the same schema as the training data.
        /// This method can be called from any thread as it creates new getters.
        /// </summary>
        public StreamingDataFrame Transform(StreamingDataFrame data)
        {
            return new StreamingDataFrame(Transform(data.Source), _env);
        }

        /// <summary>
        /// Computes the predictions for data stored in a <see cref="StreamingDataFrame"/>.
        /// It must follow the same schema as the training data.
        /// This method can be called from any thread as it creates new getters.
        /// </summary>
        public StreamingDataFrame Predict(StreamingDataFrame data)
        {
            return new StreamingDataFrame(Predict(data.Source), _env);
        }

        /// <summary>
        /// Computes the predictions for data stored in a <see cref="IDataView"/>.
        /// This method can be called from any thread as it creates new getters.
        /// It must follow the same schema as the training data.
        /// </summary>
        public IDataView Predict(IDataView data)
        {
            IDataView features = null;
            if (_transforms == null)
                features = data;
            else
            {
                var first = _transforms.First().transform;
                var pass = first as PassThroughTransform;
                pass.SetSource(data);
                features = _transforms.Last().transform;
            }

            if (_predictor == null || _predictor.predictor == null)
                return features;
            var roles = new RoleMappedData(features, _roles ?? new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>());
            return PredictorHelper.Predict(_env, _predictor.predictor, roles);
        }

        #endregion

        #region fast predictions or transform

        /// <summary>
        /// Fast predictions, it does not create getter, setter each time the predictions
        /// are done. The first call is slower as it creates getters.
        /// It must not be called from different threads.
        /// The function uses no lock.
        /// </summary>
        /// <param name="df">input data</param>
        /// <param name="view">output, reused</param>
        /// <param name="conc">number of threads used to compute the predictions</param>
        public void Predict(DataFrame df, ref DataFrame view)
        {
            if (_fastValueMapper == null)
                CreateFastValueMapper();
            _fastValueMapper(in df, ref view);
        }

        /// <summary>
        /// Fast predictions, do not create getter, setter each time the predictions
        /// are done. The first call is slower as it creates getters.
        /// It must not be called from different threads.
        /// The function uses no lock.
        /// </summary>
        /// <param name="df">input data</param>
        /// <param name="view">output, reused</param>
        public void Transform(DataFrame df, ref DataFrame view)
        {
            Predict(df, ref view);
        }

        protected void CreateFastValueMapper()
        {
            IDataTransform tr = _transforms.Last().transform;

            if (_predictor != null && _predictor.predictor != null)
            {
                var roles = new RoleMappedData(tr, _roles ?? new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>());
                tr = PredictorHelper.Predict(_env, _predictor.predictor, roles);
            }

            _fastValueMapperObject = new ValueMapperDataFrameFromTransform(_env, tr);
            _fastValueMapper = _fastValueMapperObject.GetMapper<DataFrame, DataFrame>();
        }

        #endregion

        #region onnx

        /// <summary>
        /// Exports to onnx.
        /// </summary>
        /// <param name="firstTransform">select the first transform</param>
        /// <param name="inputs">inputs, modified by the function to contains what is actually used</param>
        /// <param name="outputs">outputs, modified by the function to contains what is actually used</param>
        public ScikitOnnxContext ToOnnx(int firstTransform = 0, string[] inputs = null, string[] outputs = null,
                                        string name = null, string producer = "Scikit.ML",
                                        long version = 0, string domain = "onnx.ai.ml",
                                        OnnxVersion onnxVersion = OnnxVersion.Stable)
        {
            var begin = _transforms == null ? null : _transforms[firstTransform].transform;
            var res = Convert2Onnx.ToOnnx(_predictor == null ? _transforms.Last().transform : GetScorer(),
                                          ref inputs, ref outputs, name, producer, version, domain,
                                          onnxVersion, new IDataView[] { begin }, _env);
            return res;
        }

        IDataScorerTransform GetScorer()
        {
            return PredictorHelper.CreateDefaultScorer(_env, _predictor.roleMapData, _predictor.predictor, null);
        }

        #endregion
    }
}
