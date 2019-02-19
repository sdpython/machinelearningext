// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.TestHelper
{
    public static class StreamHelper
    {
        public static int[] GetColumnsIndex(DataViewSchema schema, IEnumerable<string> subsetColumns = null)
        {
            if (subsetColumns == null)
                return Enumerable.Range(0, schema.Count).Where(c => !schema[c].IsHidden).ToArray();
            else
                return subsetColumns.Select(c => SchemaHelper.GetColumnIndex(schema, c))
                                    .ToArray();
        }

        /// <summary>
        /// Computes the prediction given a model as a zip file
        /// and some data in a view.
        /// </summary>
        public static void SavePredictions(IHostEnvironment env, string modelPath,
                                           string outFilePath, IDataView data,
                                           IEnumerable<string> subsetColumns = null)
        {
            using (var fs = File.OpenRead(modelPath))
            {
                var deserializedData = ModelFileUtils.LoadTransforms(env, data, fs);
                var saver2 = env.CreateSaver("Text");
                var columns = GetColumnsIndex(data.Schema, subsetColumns);
                using (var fs2 = File.Create(outFilePath))
                    saver2.SaveData(fs2, deserializedData, columns);
            }
        }

        /// <summary>
        /// Computes the prediction given a model as a zip file
        /// and some data in a view.
        /// </summary>
        public static void SavePredictions(IHostEnvironment env, IDataView tr, string outFilePath,
                                           IEnumerable<string> subsetColumns = null)
        {
            var saver2 = env.CreateSaver("Text");
            var columns = GetColumnsIndex(tr.Schema, subsetColumns);
            using (var fs2 = File.Create(outFilePath))
                saver2.SaveData(fs2, tr, columns);
        }

        /// <summary>
        /// Saves a model in a zip file.
        /// </summary>
        public static void SaveModel(IHostEnvironment env, IDataTransform tr, string outModelFilePath)
        {
            using (var ch = env.Start("SaveModel"))
            using (var fs = File.Create(outModelFilePath))
            {
                var trainingExamples = env.CreateExamples(tr, null);
                TrainUtils.SaveModel(env, ch, fs, null, trainingExamples);
            }
        }
    }
}
