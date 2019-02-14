// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;


namespace Scikit.ML.OnnxHelper
{
    /// <summary>
    /// Helpers to read ONNX.
    /// </summary>
    public static class ConvertFromOnnx
    {
        /// <summary>
        /// Reads an onnx file.
        /// </summary>
        public static IDataTransform ReadOnnx(Stream fs, IDataView view)
        {
            var args = new OnnxTransformer.Options();
            throw Contracts.ExceptNotImpl("Reading ONNX format is not implemented yet.");
        }
    }
}
