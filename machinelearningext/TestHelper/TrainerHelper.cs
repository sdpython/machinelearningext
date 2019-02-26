// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.TestHelper
{
    public static class TestTrainerHelper
    {
        /// <summary>
        /// Finalizes the test on a predictor, calls the predictor with a scorer,
        /// saves the data, saves the models, loads it back, saves the data again,
        /// checks the output is the same.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="outModelFilePath">output filename</param>
        /// <param name="predictor">predictor</param>
        /// <param name="roles">label, feature, ...</param>
        /// <param name="outData">first output data</param>
        /// <param name="outData2">second output data</param>
        /// <param name="kind">prediction kind</param>
        /// <param name="checkError">checks errors</param>
        /// <param name="ratio">check the error is below that threshold (if checkError is true)</param>
        /// <param name="ratioReadSave">check the predictions difference after reloading the model are below this threshold</param>
        public static void FinalizeSerializationTest(IHostEnvironment env,
                            string outModelFilePath, IPredictor predictor,
                            RoleMappedData roles, string outData, string outData2,
                            PredictionKind kind, bool checkError = true,
                            float ratio = 0.8f, float ratioReadSave = 0.06f,
                            bool checkType = true)
        {
            string labelColumn = kind != PredictionKind.Clustering ? roles.Schema.Label.Value.Name : null;

            #region save, reading, running

            // Saves model.
            using (var ch = env.Start("Save"))
            using (var fs = File.Create(outModelFilePath))
                TrainUtils.SaveModel(env, ch, fs, predictor, roles);
            if (!File.Exists(outModelFilePath))
                throw new FileNotFoundException(outModelFilePath);

            // Loads the model back.
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var pred_local = ModelFileUtils.LoadPredictorOrNull(env, fs);
                if (pred_local == null)
                    throw new Exception(string.Format("Unable to load '{0}'", outModelFilePath));
                if (checkType && predictor.GetType() != pred_local.GetType())
                    throw new Exception(string.Format("Type mismatch {0} != {1}", predictor.GetType(), pred_local.GetType()));
            }

            // Checks the outputs.
            var sch1 = SchemaHelper.ToString(roles.Schema.Schema);
            var scorer = PredictorHelper.CreateDefaultScorer(env, roles, predictor);

            var sch2 = SchemaHelper.ToString(scorer.Schema);
            if (string.IsNullOrEmpty(sch1) || string.IsNullOrEmpty(sch2))
                throw new Exception("Empty schemas");

            var saver = env.CreateSaver("Text");
            var columns = new int[scorer.Schema.Count];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = saver.IsColumnSavable(scorer.Schema[i].Type) ? i : -1;
            columns = columns.Where(c => c >= 0).ToArray();
            using (var fs2 = File.Create(outData))
                saver.SaveData(fs2, scorer, columns);

            if (!File.Exists(outModelFilePath))
                throw new FileNotFoundException(outData);

            // Check we have the same output.
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var model = ModelFileUtils.LoadPredictorOrNull(env, fs);
                scorer = PredictorHelper.CreateDefaultScorer(env, roles, model);
                saver = env.CreateSaver("Text");
                using (var fs2 = File.Create(outData2))
                    saver.SaveData(fs2, scorer, columns);
            }

            var t1 = File.ReadAllLines(outData);
            var t2 = File.ReadAllLines(outData2);
            if (t1.Length != t2.Length)
                throw new Exception(string.Format("Not the same number of lines: {0} != {1}", t1.Length, t2.Length));
            var linesN = new List<int>();
            for (int i = 0; i < t1.Length; ++i)
            {
                if (t1[i] != t2[i])
                    linesN.Add(i);
            }
            if (linesN.Count > (int)(t1.Length * ratioReadSave))
            {
                var rows = linesN.Select(i => string.Format("1-Mismatch on line {0}/{3}:\n{1}\n{2}", i, t1[i], t2[i], t1.Length)).ToList();
                rows.Add($"Number of differences: {linesN.Count}/{t1.Length}");
                throw new Exception(string.Join("\n", rows));
            }

            #endregion

            #region clustering 

            if (kind == PredictionKind.Clustering)
                // Nothing to do here.
                return;

            #endregion

            #region supervized

            string expectedOuput = kind == PredictionKind.Regression ? "Score" : "PredictedLabel";

            // Get label and basic checking about performance.
            using (var cursor = scorer.GetRowCursor(scorer.Schema))
            {
                int ilabel, ipred;
                ilabel = SchemaHelper.GetColumnIndex(cursor.Schema, labelColumn);
                ipred = SchemaHelper.GetColumnIndex(cursor.Schema, expectedOuput);
                var ty1 = cursor.Schema[ilabel].Type;
                var ty2 = cursor.Schema[ipred].Type;
                var dist1 = new Dictionary<int, int>();
                var dist2 = new Dictionary<int, int>();
                var conf = new Dictionary<Tuple<int, int>, long>();

                if (kind == PredictionKind.MultiClassClassification)
                {
                    #region multiclass

                    if (!ty2.IsKey())
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));

                    if (ty1.RawKind() == DataKind.Single)
                    {
                        var lgetter = cursor.GetGetter<float>(ilabel);
                        var pgetter = cursor.GetGetter<uint>(ipred);
                        float ans = 0;
                        uint pre = 0;
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            // The scorer +1 to the argmax.
                            ++ans;

                            var key = new Tuple<int, int>((int)pre, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else if (ty1.RawKind() == DataKind.UInt32 && ty1.IsKey())
                    {
                        var lgetter = cursor.GetGetter<uint>(ilabel);
                        var pgetter = cursor.GetGetter<uint>(ipred);
                        uint ans = 0;
                        uint pre = 0;
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            var key = new Tuple<int, int>((int)pre, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else
                        throw new NotImplementedException(string.Format("Not implemented for type {0}", ty1.ToString()));
                    #endregion
                }
                else if (kind == PredictionKind.BinaryClassification)
                {
                    #region binary classification

                    if (ty2.RawKind() != DataKind.Boolean)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));

                    if (ty1.RawKind() == DataKind.Single)
                    {
                        var lgetter = cursor.GetGetter<float>(ilabel);
                        var pgetter = cursor.GetGetter<bool>(ipred);
                        float ans = 0;
                        bool pre = default(bool);
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            if (ans != 0 && ans != 1)
                                throw Contracts.Except("The problem is not binary, expected answer is {0}", ans);

                            var key = new Tuple<int, int>(pre ? 1 : 0, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey(pre ? 1 : 0))
                                dist2[pre ? 1 : 0] = 1;
                            else
                                ++dist2[pre ? 1 : 0];
                        }
                    }
                    else if (ty1.RawKind() == DataKind.UInt32)
                    {
                        var lgetter = cursor.GetGetter<uint>(ilabel);
                        var pgetter = cursor.GetGetter<bool>(ipred);
                        uint ans = 0;
                        bool pre = default(bool);
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);
                            if (ty1.IsKey())
                                --ans;

                            if (ans != 0 && ans != 1)
                                throw Contracts.Except("The problem is not binary, expected answer is {0}", ans);

                            var key = new Tuple<int, int>(pre ? 1 : 0, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey(pre ? 1 : 0))
                                dist2[pre ? 1 : 0] = 1;
                            else
                                ++dist2[pre ? 1 : 0];
                        }
                    }
                    else if (ty1.RawKind() == DataKind.Boolean)
                    {
                        var lgetter = cursor.GetGetter<bool>(ilabel);
                        var pgetter = cursor.GetGetter<bool>(ipred);
                        bool ans = default(bool);
                        bool pre = default(bool);
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            var key = new Tuple<int, int>(pre ? 1 : 0, ans ? 1 : 0);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];

                            if (!dist1.ContainsKey(ans ? 1 : 0))
                                dist1[ans ? 1 : 0] = 1;
                            else
                                ++dist1[ans ? 1 : 0];
                            if (!dist2.ContainsKey(pre ? 1 : 0))
                                dist2[pre ? 1 : 0] = 1;
                            else
                                ++dist2[pre ? 1 : 0];
                        }
                    }
                    else
                        throw new NotImplementedException(string.Format("Not implemented for type {0}", ty1));

                    #endregion
                }
                else if (kind == PredictionKind.Regression)
                {
                    #region regression

                    if (ty1.RawKind() != DataKind.Single)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));
                    if (ty2.RawKind() != DataKind.Single)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));

                    var lgetter = cursor.GetGetter<float>(ilabel);
                    var pgetter = cursor.GetGetter<float>(ipred);
                    float ans = 0;
                    float pre = 0f;
                    float error = 0f;
                    while (cursor.MoveNext())
                    {
                        lgetter(ref ans);
                        pgetter(ref pre);
                        error += (ans - pre) * (ans - pre);
                        if (!dist1.ContainsKey((int)ans))
                            dist1[(int)ans] = 1;
                        else
                            ++dist1[(int)ans];
                        if (!dist2.ContainsKey((int)pre))
                            dist2[(int)pre] = 1;
                        else
                            ++dist2[(int)pre];
                    }

                    if (float.IsNaN(error) || float.IsInfinity(error))
                        throw new Exception("Regression wen wrong. Error is infinite.");

                    #endregion
                }
                else
                    throw new NotImplementedException(string.Format("Not implemented for kind {0}", kind));

                var nbError = conf.Where(c => c.Key.Item1 != c.Key.Item2).Select(c => c.Value).Sum();
                var nbTotal = conf.Select(c => c.Value).Sum();

                if (checkError && (nbError * 1.0 > nbTotal * ratio || dist2.Count <= 1))
                {
                    var sconf = string.Join("\n", conf.OrderBy(c => c.Key)
                                      .Select(c => string.Format("pred={0} exp={1} count={2}", c.Key.Item1, c.Key.Item2, c.Value)));
                    var sdist2 = string.Join("\n", dist1.OrderBy(c => c.Key)
                                       .Select(c => string.Format("label={0} count={1}", c.Key, c.Value)));
                    var sdist1 = string.Join("\n", dist2.OrderBy(c => c.Key).Take(20)
                                       .Select(c => string.Format("label={0} count={1}", c.Key, c.Value)));
                    throw new Exception(string.Format("Too many errors {0}/{1}={7}\n###########\nConfusion:\n{2}\n########\nDIST1\n{3}\n###########\nDIST2\n{4}\nOutput:\n{5}\n...\n{6}",
                                    nbError, nbTotal,
                                    sconf, sdist1, sdist2,
                                    string.Join("\n", t1.Take(Math.Min(30, t1.Length))),
                                    string.Join("\n", t1.Skip(Math.Max(0, t1.Length - 30))),
                                    nbError * 1.0 / nbTotal));
                }
            }

            #endregion
        }
    }
}
