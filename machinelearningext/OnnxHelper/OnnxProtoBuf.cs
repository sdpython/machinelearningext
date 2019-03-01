// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Google.Protobuf;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.OnnxHelper
{
    /// <summary>
    /// Contains methods to create ONNX models in protocol buffer.
    /// </summary>
    internal static class OnnxUtils
    {
        private static OnnxCSharpToProtoWrapper.TypeProto MakeType(OnnxCSharpToProtoWrapper.TypeProto typeProto,
                                                                   OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType,
            List<long> dims, List<bool> dimsParam)
        {
            Contracts.CheckValue(typeProto, nameof(typeProto));

            if (typeProto.TensorType == null)
                typeProto.TensorType = new OnnxCSharpToProtoWrapper.TypeProto.Types.Tensor();

            typeProto.TensorType.ElemType = dataType;
            if (dims != null)
            {
                for (int index = 0; index < dims.Count; index++)
                {
                    var d = new OnnxCSharpToProtoWrapper.TensorShapeProto.Types.Dimension();
                    if (typeProto.TensorType.Shape == null)
                        typeProto.TensorType.Shape = new OnnxCSharpToProtoWrapper.TensorShapeProto();

                    if (dimsParam != null && dimsParam.Count > index && dimsParam[index])
                        d.DimParam = "None";
                    else
                        d.DimValue = dims[index];

                    typeProto.TensorType.Shape.Dim.Add(d);
                }
            }

            return typeProto;
        }

        private static OnnxCSharpToProtoWrapper.ValueInfoProto MakeValue(OnnxCSharpToProtoWrapper.ValueInfoProto value, string name,
                                                                         OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType,
            List<long> dims, List<bool> dimsParam)
        {
            Contracts.CheckValue(value, nameof(value));
            Contracts.CheckNonEmpty(name, nameof(name));

            value.Name = name;
            if (value.Type == null)
                value.Type = new OnnxCSharpToProtoWrapper.TypeProto();

            MakeType(value.Type, dataType, dims, dimsParam);
            return value;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key)
        {
            Contracts.CheckNonEmpty(key, nameof(key));

            var attribute = new OnnxCSharpToProtoWrapper.AttributeProto();
            attribute.Name = key;
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, double value)
        {
            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Float;
            attribute.F = (float)value;
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, IEnumerable<double> value)
        {
            Contracts.CheckValue(value, nameof(value));

            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Floats;
            attribute.Floats.Add(value.Select(x => (float)x));
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, IEnumerable<float> value)
        {
            Contracts.CheckValue(value, nameof(value));

            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Floats;
            attribute.Floats.Add(value.Select(x => x));
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, long value)
        {
            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Int;
            attribute.I = value;
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, IEnumerable<long> value)
        {
            Contracts.CheckValue(value, nameof(value));

            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Ints;
            attribute.Ints.Add(value);
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, ByteString value)
        {
            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.String;
            attribute.S = value;
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, IEnumerable<ByteString> value)
        {
            Contracts.CheckValue(value, nameof(value));

            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Strings;
            attribute.Strings.Add(value);
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, OnnxCSharpToProtoWrapper.GraphProto value)
        {
            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Graph;
            attribute.G = value;
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, IEnumerable<OnnxCSharpToProtoWrapper.GraphProto> value)
        {
            Contracts.CheckValue(value, nameof(value));

            OnnxCSharpToProtoWrapper.AttributeProto attribute = MakeAttribute(key);
            attribute.Type = OnnxCSharpToProtoWrapper.AttributeProto.Types.AttributeType.Graphs;
            attribute.Graphs.Add(value);
            return attribute;
        }

        private static OnnxCSharpToProtoWrapper.AttributeProto MakeAttribute(string key, bool value) => MakeAttribute(key, value ? 1 : 0);

        public static OnnxCSharpToProtoWrapper.NodeProto MakeNode(string opType, IEnumerable<string> inputs, IEnumerable<string> outputs, string name, string domain = null)
        {
            Contracts.CheckNonEmpty(opType, nameof(opType));
            Contracts.CheckValue(inputs, nameof(inputs));
            Contracts.CheckValue(outputs, nameof(outputs));
            Contracts.CheckNonEmpty(name, nameof(name));

            var node = new OnnxCSharpToProtoWrapper.NodeProto();
            node.OpType = opType;
            node.Input.Add(inputs);
            node.Output.Add(outputs);
            node.Name = name;
            node.Domain = domain ?? "ai.onnx.ml";
            return node;
        }

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, double value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<double> value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<float> value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<bool> value)
            => node.Attribute.Add(MakeAttribute(argName, value.Select(v => v ? (long)1 : 0)));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, long value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<long> value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, ReadOnlyMemory<char> value)
            => node.Attribute.Add(MakeAttribute(argName, StringToByteString(value)));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, string[] value)
            => node.Attribute.Add(MakeAttribute(argName, StringToByteString(value)));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<ReadOnlyMemory<char>> value)
            => node.Attribute.Add(MakeAttribute(argName, StringToByteString(value)));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<string> value)
            => node.Attribute.Add(MakeAttribute(argName, StringToByteString(value)));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, string value)
            => node.Attribute.Add(MakeAttribute(argName, StringToByteString(value)));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, OnnxCSharpToProtoWrapper.GraphProto value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, IEnumerable<OnnxCSharpToProtoWrapper.GraphProto> value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        public static void NodeAddAttributes(OnnxCSharpToProtoWrapper.NodeProto node, string argName, bool value)
            => node.Attribute.Add(MakeAttribute(argName, value));

        private static ByteString StringToByteString(ReadOnlyMemory<char> str) => ByteString.CopyFrom(Encoding.UTF8.GetBytes(str.ToString()));
        private static IEnumerable<ByteString> StringToByteString(IEnumerable<ReadOnlyMemory<char>> str)
            => str.Select(s => ByteString.CopyFrom(Encoding.UTF8.GetBytes(s.ToString())));

        private static IEnumerable<ByteString> StringToByteString(IEnumerable<string> str)
            => str.Select(s => ByteString.CopyFrom(Encoding.UTF8.GetBytes(s)));

        private static ByteString StringToByteString(string str) => ByteString.CopyFrom(Encoding.UTF8.GetBytes(str));

        public sealed class ModelArgs
        {
            public readonly string Name;
            public readonly OnnxCSharpToProtoWrapper.TensorProto.Types.DataType DataType;
            public readonly List<long> Dims;
            public readonly List<bool> DimParams;

            public ModelArgs(string name, OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType, List<long> dims, List<bool> dimParams)
            {
                Name = name;
                DataType = dataType;
                Dims = dims;
                DimParams = dimParams;
            }
        }

        public static OnnxCSharpToProtoWrapper.ModelProto MakeModel(List<OnnxCSharpToProtoWrapper.NodeProto> nodes, string producerName, string name,
            string domain, string producerVersion, long modelVersion, List<ModelArgs> inputs,
            List<ModelArgs> outputs, List<ModelArgs> intermediateValues, List<OnnxCSharpToProtoWrapper.TensorProto> initializers)
        {
            Contracts.CheckValue(nodes, nameof(nodes));
            Contracts.CheckValue(inputs, nameof(inputs));
            Contracts.CheckValue(outputs, nameof(outputs));
            Contracts.CheckValue(intermediateValues, nameof(intermediateValues));
            Contracts.CheckValue(initializers, nameof(initializers));
            Contracts.CheckNonEmpty(producerName, nameof(producerName));
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckNonEmpty(domain, nameof(domain));
            Contracts.CheckNonEmpty(producerVersion, nameof(producerVersion));

            var model = new OnnxCSharpToProtoWrapper.ModelProto();
            model.Domain = domain;
            model.ProducerName = producerName;
            model.ProducerVersion = producerVersion;
            model.IrVersion = (long)OnnxCSharpToProtoWrapper.Version.IrVersion;
            model.ModelVersion = modelVersion;
            model.OpsetImport.Add(new OnnxCSharpToProtoWrapper.OperatorSetIdProto() { Domain = "ai.onnx.ml", Version = 1 });
            model.OpsetImport.Add(new OnnxCSharpToProtoWrapper.OperatorSetIdProto() { Domain = "", Version = 7 });
            model.Graph = new OnnxCSharpToProtoWrapper.GraphProto();
            var graph = model.Graph;
            graph.Node.Add(nodes);
            graph.Name = name;
            foreach (var arg in inputs)
            {
                var val = new OnnxCSharpToProtoWrapper.ValueInfoProto();
                graph.Input.Add(val);
                MakeValue(val, arg.Name, arg.DataType, arg.Dims, arg.DimParams);
            }

            foreach (var arg in outputs)
            {
                var val = new OnnxCSharpToProtoWrapper.ValueInfoProto();
                graph.Output.Add(val);
                MakeValue(val, arg.Name, arg.DataType, arg.Dims, arg.DimParams);
            }

            foreach (var arg in intermediateValues)
            {
                var val = new OnnxCSharpToProtoWrapper.ValueInfoProto();
                graph.ValueInfo.Add(val);
                MakeValue(val, arg.Name, arg.DataType, arg.Dims, arg.DimParams);
            }

            graph.Initializer.AddRange(initializers);

            return model;
        }

        public static ModelArgs GetModelArgs(DataViewType type, string colName,
            List<long> dims = null, List<bool> dimsParams = null)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckNonEmpty(colName, nameof(colName));

            OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Undefined;
            DataKind rawKind;
            if (type.IsVector())
                rawKind = type.AsVector().ItemType().RawKind();
            else if (type.IsKey())
                rawKind = type.AsKey().RawKind();
            else
                rawKind = type.RawKind();

            switch (rawKind)
            {
                case DataKind.Boolean:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Float;
                    break;
                case DataKind.String:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.String;
                    break;
                case DataKind.SByte:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int8;
                    break;
                case DataKind.Byte:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint8;
                    break;
                case DataKind.Int16:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int16;
                    break;
                case DataKind.UInt16:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint16;
                    break;
                case DataKind.Int32:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int32;
                    break;
                case DataKind.UInt32:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int64;
                    break;
                case DataKind.Int64:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int64;
                    break;
                case DataKind.UInt64:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint64;
                    break;
                case DataKind.Single:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Float;
                    break;
                case DataKind.Double:
                    dataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Double;
                    break;
                default:
                    string msg = "Unsupported type: DataKind " + rawKind.ToString();
                    Contracts.Check(false, msg);
                    break;
            }

            string name = colName;
            List<long> dimsLocal = null;
            List<bool> dimsParamLocal = null;
            if (dims != null)
            {
                dimsLocal = dims;
                dimsParamLocal = dimsParams;
            }
            else
            {
                dimsLocal = new List<long>();
                if (type.ValueCount() == 0) //Unknown size.
                {
                    dimsLocal.Add(1);
                    dimsParamLocal = new List<bool>() { false, true }; //false for batch size, true for dims.
                }
                else if (type.ValueCount() == 1)
                    dimsLocal.Add(1);
                else if (type.ValueCount() > 1)
                {
                    var vec = type.AsVector();
                    for (int i = 0; i < vec.DimCount(); i++)
                        dimsLocal.Add(vec.GetDim(i));
                }
            }
            //batch size.
            dimsLocal?.Insert(0, 1);

            return new ModelArgs(name, dataType, dimsLocal, dimsParamLocal);
        }

        // Make long scalar in ONNX from native C# number
        public static OnnxCSharpToProtoWrapper.TensorProto MakeInt64(string name, long value)
        {
            var tensor = new OnnxCSharpToProtoWrapper.TensorProto();
            tensor.Name = name;
            tensor.DataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int64;
            tensor.Int64Data.Add(value);
            return tensor;
        }

        // Make long vector (i.e., 1-D tensor) with dims=null. Otherwise, dims is used as the shape of the produced tensor.
        public static OnnxCSharpToProtoWrapper.TensorProto MakeInt64s(string name, IEnumerable<long> values, IEnumerable<long> dims = null)
        {
            var tensor = new OnnxCSharpToProtoWrapper.TensorProto();
            tensor.Name = name;
            tensor.DataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int64;
            tensor.Int64Data.AddRange(values);
            if (dims != null)
                tensor.Dims.AddRange(dims);
            else
                tensor.Dims.Add(values.Count());
            return tensor;
        }

        // Make float scalar in ONNX from native C# number
        public static OnnxCSharpToProtoWrapper.TensorProto MakeFloat(string name, float value)
        {
            var tensor = new OnnxCSharpToProtoWrapper.TensorProto();
            tensor.Name = name;
            tensor.DataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Float;
            tensor.FloatData.Add(value);
            return tensor;
        }

        // Make float vector (i.e., 1-D tensor) with dims=null. Otherwise, dims is used as the shape of the produced tensor.
        public static OnnxCSharpToProtoWrapper.TensorProto MakeFloats(string name, IEnumerable<float> values, IEnumerable<long> dims = null)
        {
            var tensor = new OnnxCSharpToProtoWrapper.TensorProto();
            tensor.Name = name;
            tensor.DataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Float;
            tensor.FloatData.AddRange(values);
            if (dims != null)
                tensor.Dims.AddRange(dims);
            else
                tensor.Dims.Add(values.Count());
            return tensor;
        }

        // Make string scalar in ONNX from native C# number
        public static OnnxCSharpToProtoWrapper.TensorProto MakeString(string name, string value)
        {
            var tensor = new OnnxCSharpToProtoWrapper.TensorProto();
            tensor.Name = name;
            tensor.DataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.String;
            tensor.StringData.Add(StringToByteString(value));
            return tensor;
        }

        // Make string vector (i.e., 1-D tensor) with dims=null. Otherwise, dims is used as the shape of the produced tensor.
        public static OnnxCSharpToProtoWrapper.TensorProto MakeStrings(string name, IEnumerable<string> values, IEnumerable<long> dims = null)
        {
            var tensor = new OnnxCSharpToProtoWrapper.TensorProto();
            tensor.Name = name;
            tensor.DataType = OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.String;
            tensor.StringData.AddRange(StringToByteString(values));
            if (dims != null)
                tensor.Dims.AddRange(dims);
            else
                tensor.Dims.Add(values.Count());
            return tensor;
        }
    }
}