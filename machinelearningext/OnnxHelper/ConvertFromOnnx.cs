﻿// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;


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
            throw Contracts.ExceptNotImpl("Reading ONNX format is not implemented yet.");
        }
    }
}
