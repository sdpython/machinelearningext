// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineGraphTransforms;


namespace Scikit.ML.OnnxHelper
{
    /// <summary>
    /// Helpers for ONNX conversion.
    /// </summary>
    public static class Convert2Onnx
    {
        public class GaphViewNode
        {
            public string variableName;
            public IDataView view;
            public TagHelper.GraphPositionEnum position;
            public DataViewType variableType;
        }

        /// <summary>
        /// Enumerates all variables in all transforms in a pipeline.
        /// </summary>
        /// <param name="trans"><see cref="IDataTransform"/></param>
        /// <returns>iterator of 4-uple: column name, data view, unique column name, column type</returns>
        public static IEnumerable<GaphViewNode> EnumerateVariables(IDataTransform trans, IDataView[] begin = null)
        {
            var unique = new HashSet<string>();
            foreach (var view in TagHelper.EnumerateAllViews(trans, begin))
            {
                var sch = view.Item1.Schema;
                for (int i = 0; i < sch.Count; ++i)
                {
                    var name = sch[i].Name;
                    var prop = name;
                    int k = 1;
                    while (unique.Contains(prop))
                    {
                        prop = $"{name}_{k}";
                        ++k;
                    }
                    unique.Add(prop);

                    yield return new GaphViewNode()
                    {
                        variableName = name,
                        view = view.Item1,
                        position = view.Item2,
                        variableType = sch[i].Type
                    };
                }
            }
        }

        /// <summary>
        /// Guesses all inputs in a pipeline.
        /// If empty, the function looks into every view with no predecessor.
        /// </summary>
        public static void GuessInputs(IDataTransform trans, ref string[] inputs, IDataView[] begin = null)
        {
            var vars = EnumerateVariables(trans, begin).Where(c => c.position == TagHelper.GraphPositionEnum.first).ToArray();
            if (inputs == null)
            {
                inputs = vars.Select(c => c.variableName).ToArray();
                var has = new HashSet<string>(inputs);
                if (has.Count != inputs.Length)
                    throw Contracts.Except($"One column is duplicated.");
            }
            else
            {
                var has = new HashSet<string>(inputs);
                if (has.Count != inputs.Length)
                    throw Contracts.Except($"One column is duplicated.");
                has = new HashSet<string>(vars.Select(c => c.variableName));
                foreach (var inp in inputs)
                {
                    if (!has.Contains(inp))
                        throw Contracts.Except($"Unable to find column '{inp}' in {string.Join(", ", has.OrderBy(c => c))}.");
                }
            }
        }

        /// <summary>
        /// Extracts outputs of a transform is not specified.
        /// </summary>
        /// <param name="view">transform</param>
        /// <param name="outputs">requested outputs or null to get them all</param>
        /// <param name="hidden">include hidden variables</param>
        public static void GuessOutputs(IDataView view, ref string[] outputs, bool hidden = false)
        {
            if (outputs == null)
            {
                var sch = view.Schema;
                outputs = Enumerable.Range(0, sch.Count)
                                    .Where(c => hidden || !sch[c].IsHidden)
                                    .Select(c => sch[c].Name).ToArray();
            }
            else
            {
                var sch = view.Schema;
                foreach (var name in outputs)
                    SchemaHelper.GetColumnIndex(sch, name);
            }
        }

        public static ScikitOnnxContext ToOnnx(IDataTransform trans, ref string[] inputs, ref string[] outputs,
                                               string name = null, string producer = "Scikit.ML",
                                               long version = 0, string domain = "onnx.ai.ml",
                                               OnnxVersion onnxVersion = OnnxVersion.Stable,
                                               IDataView[] begin = null, IHostEnvironment host = null)
        {
            if (host == null)
            {
                /*using (*/
                var env = new DelegateEnvironment();/*)*/
                return ToOnnx(trans, ref inputs, ref outputs, name, producer, version, domain, onnxVersion, begin, env);
            }
            if (name == null)
                name = trans.GetType().Name;

            GuessOutputs(trans, ref outputs);
            GuessInputs(trans, ref inputs, begin);

            if (inputs == null || inputs.Length == 0)
                throw host.Except("Inputs cannot be empty.");
            if (outputs == null || outputs.Length == 0)
                throw host.Except("Outputs cannot be empty.");

            var assembly = System.Reflection.Assembly.GetExecutingAssembly();
            var versionInfo = System.Diagnostics.FileVersionInfo.GetVersionInfo(assembly.Location);
            var ctx = new ScikitOnnxContext(host, name, producer, versionInfo.FileVersion,
                                            version, domain, onnxVersion);

            var hasin = new HashSet<string>(inputs);
            var uniqueVars = EnumerateVariables(trans, begin).ToArray();
            var mapInputType = new Dictionary<string, DataViewType>();
            foreach (var it in uniqueVars.Where(c => c.position == TagHelper.GraphPositionEnum.first))
                mapInputType[it.variableName] = it.variableType;

            foreach (var col in inputs)
                ctx.AddInputVariable(mapInputType[col], col);

            var views = TagHelper.EnumerateAllViews(trans, begin);
            var transforms = views.Where(c => (c.Item1 as IDataTransform) != null)
                                  .Select(c => c.Item1)
                                  .ToArray();

            foreach (var tr in transforms.Reverse())
            {
                var tron = tr as ICanSaveOnnx;
                if (tron == null)
                    throw host.ExceptNotSupp($"Transform {tr.GetType()} cannot be saved in Onnx format.");
                if (!tron.CanSaveOnnx(ctx))
                    throw host.ExceptNotSupp($"Transform {tr.GetType()} cannot be saved in ONNX format.");
                var tron2 = tron as ISaveAsOnnx;
                if (!tron2.CanSaveOnnx(ctx))
                    throw host.ExceptNotSupp($"Transform {tr.GetType()} does not implement SaveAsOnnx.");
                tron2.SaveAsOnnx(ctx);
            }

            var mapOuputType = new Dictionary<string, DataViewType>();
            foreach (var it in uniqueVars.Where(c => c.position == TagHelper.GraphPositionEnum.last))
                mapOuputType[it.variableName] = it.variableType;

            foreach (var col in outputs)
            {
                var variableName = ctx.TryGetVariableName(col);
                var trueVariableName = ctx.AddIntermediateVariable(null, col, true);
                ctx.CreateNode("Identity", variableName, trueVariableName, ctx.GetNodeName("Identity"), "");
                ctx.AddOutputVariable(mapOuputType[col], trueVariableName);
            }

            return ctx;
        }
    }
}
