// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Helpers about ISchema.
    /// </summary>
    public static class SchemaHelper
    {
        public static DataViewType DataKind2ColumnType(DataKind kind, IChannel ch = null)
        {
            switch (kind)
            {
                case DataKind.Boolean:
                    return BooleanDataViewType.Instance;
                case DataKind.Single:
                    return NumberDataViewType.Single;
                case DataKind.Double:
                    return NumberDataViewType.Double;
                case DataKind.SByte:
                    return NumberDataViewType.SByte;
                case DataKind.Int16:
                    return NumberDataViewType.Int16;
                case DataKind.Int32:
                    return NumberDataViewType.Int32;
                case DataKind.Int64:
                    return NumberDataViewType.Int64;
                case DataKind.Byte:
                    return NumberDataViewType.Byte;
                case DataKind.UInt16:
                    return NumberDataViewType.UInt16;
                case DataKind.UInt32:
                    return NumberDataViewType.UInt32;
                case DataKind.UInt64:
                    return NumberDataViewType.UInt64;
                case DataKind.String:
                    return TextDataViewType.Instance;
                case DataKind.DateTime:
                    return DateTimeDataViewType.Instance;
                case DataKind.DateTimeOffset:
                    return DateTimeOffsetDataViewType.Instance;
                default:
                    throw ch != null ? ch.Except("Unknown kind {0}", kind) : Contracts.Except("Unknown kind {0}", kind);
            }
        }

        public static DataViewType DataKind2ColumnType(InternalDataKind kind, IChannel ch = null)
        {
            switch (kind)
            {
                case InternalDataKind.BL:
                    return BooleanDataViewType.Instance;
                case InternalDataKind.R4:
                    return NumberDataViewType.Single;
                case InternalDataKind.R8:
                    return NumberDataViewType.Double;
                case InternalDataKind.I1:
                    return NumberDataViewType.SByte;
                case InternalDataKind.I2:
                    return NumberDataViewType.Int16;
                case InternalDataKind.I4:
                    return NumberDataViewType.Int32;
                case InternalDataKind.I8:
                    return NumberDataViewType.Int64;
                case InternalDataKind.U1:
                    return NumberDataViewType.Byte;
                case InternalDataKind.U2:
                    return NumberDataViewType.UInt16;
                case InternalDataKind.U4:
                    return NumberDataViewType.UInt32;
                case InternalDataKind.U8:
                    return NumberDataViewType.UInt64;
                case InternalDataKind.TX:
                    return TextDataViewType.Instance;
                case InternalDataKind.DateTime:
                    return DateTimeDataViewType.Instance;
                case InternalDataKind.DateTimeZone:
                    return DateTimeOffsetDataViewType.Instance;
                default:
                    throw ch != null ? ch.Except("Unknown kind {0}", kind) : Contracts.Except("Unknown kind {0}", kind);
            }
        }

        public static DataKind InternalDataKind2DataKind(InternalDataKind kind, IChannel ch = null)
        {
            switch (kind)
            {
                case InternalDataKind.BL: return DataKind.Boolean;
                case InternalDataKind.R4: return DataKind.Single;
                case InternalDataKind.R8: return DataKind.Double;
                case InternalDataKind.I1: return DataKind.SByte;
                case InternalDataKind.I2: return DataKind.Int16;
                case InternalDataKind.I4: return DataKind.Int32;
                case InternalDataKind.I8: return DataKind.Int64;
                case InternalDataKind.U1: return DataKind.Byte;
                case InternalDataKind.U2: return DataKind.UInt16;
                case InternalDataKind.U4: return DataKind.UInt32;
                case InternalDataKind.U8: return DataKind.UInt64;
                case InternalDataKind.TX: return DataKind.String;
                case InternalDataKind.DateTime: return DataKind.DateTime;
                case InternalDataKind.DateTimeZone: return DataKind.DateTimeOffset;
                default:
                    throw ch != null ? ch.Except("Unknown kind {0}", kind) : Contracts.Except("Unknown kind {0}", kind);
            }
        }

        public static InternalDataKind DataKind2InternalDataKind(DataKind kind, IChannel ch = null)
        {
            switch (kind)
            {
                case DataKind.Boolean: return InternalDataKind.BL;
                case DataKind.Single: return InternalDataKind.R4;
                case DataKind.Double: return InternalDataKind.R8;
                case DataKind.SByte: return InternalDataKind.I1;
                case DataKind.Int16: return InternalDataKind.I2;
                case DataKind.Int32: return InternalDataKind.I4;
                case DataKind.Int64: return InternalDataKind.I8;
                case DataKind.Byte: return InternalDataKind.U1;
                case DataKind.UInt16: return InternalDataKind.U2;
                case DataKind.UInt32: return InternalDataKind.U4;
                case DataKind.UInt64: return InternalDataKind.U8;
                case DataKind.String: return InternalDataKind.TX;
                case DataKind.DateTime: return InternalDataKind.DateTime;
                case DataKind.DateTimeOffset: return InternalDataKind.DateTimeZone;
                default:
                    throw ch != null ? ch.Except("Unknown kind {0}", kind) : Contracts.Except("Unknown kind {0}", kind);
            }
        }

        /// <summary>
        /// Convert a type description into another type description.
        /// </summary>

        public static DataViewType Convert(TextLoader.Column col, IChannel ch = null)
        {
            if (col.Source != null && col.Source.Length > 0)
            {
                if (col.Source.Length != 1)
                    throw Contracts.ExceptNotImpl("Convert of TextLoader.Column is not implemented for more than on range.");
                if (col.Source[0].ForceVector)
                {
                    if (!col.Source[0].Max.HasValue)
                        throw ch != null ? ch.Except("A vector column needs a dimension")
                                         : Contracts.Except("A vector column needs a dimension");
                    int delta = col.Source[0].Max.Value - col.Source[0].Min + 1;
                    var colType = DataKind2ColumnType(col.Type, ch);
                    return new VectorType(colType.AsPrimitive(), delta);
                }
            }
            if (col.KeyCount != null)
            {
                var r = col.KeyCount;
                return new KeyType(DataKind2ColumnType(col.Type).RawType, r.Count.HasValue ? r.Count.Value : 0);
            }
            else
                return DataKind2ColumnType(col.Type, ch);
        }

        /// <summary>
        /// To display a schema.
        /// </summary>
        /// <param name="schema">schema</param>
        /// <param name="sep">column separator</param>
        /// <param name="vectorVec">if true, show Vec<R4, 2> and false, shows :R4:5-6 </R4> does the same for keys</param>
        /// <param name="keepHidden">keepHidden columns?</param>
        /// <returns>schema as a string</returns>
        public static string ToString(ISchema schemaInst, string sep = "; ", bool vectorVec = true, bool keepHidden = false)
        {
            return ToString(ExtendedSchema.Create(schemaInst), sep, vectorVec, keepHidden);
        }

        public static string ToString(DataViewSchema schema, string sep = "; ", bool vectorVec = true, bool keepHidden = false)
        {
            var builder = new StringBuilder();
            string name, type;
            string si;
            int lag = 0;
            for (int i = 0; i < schema.Count; ++i)
            {
                if (!keepHidden && schema[i].IsHidden)
                    continue;
                if (builder.Length > 0)
                    builder.Append(sep);
                name = schema[i].Name;
                var t = schema[i].Type;
                if (vectorVec || (!t.IsVector() && !t.IsKey()))
                {
                    type = schema[i].Type.ToString().Replace(" ", "");
                    si = (i + lag).ToString();
                }
                else
                {
                    if (t.IsVector())
                    {
                        if (t.AsVector().DimCount() != 1)
                            throw Contracts.ExceptNotSupp("Only vector with one dimension are supported.");
                        type = t.ItemType().RawKind().ToString();
                        si = string.Format("{0}-{1}", i + lag, i + lag + t.AsVector().GetDim(0) - 1);
                        lag += t.AsVector().GetDim(0) - 1;
                    }
                    else if (t.IsKey())
                    {
                        var k = t.AsKey();
                        type = k.Count > 0
                                    ? string.Format("{0}[{1}]", k.RawKind(), k.Count)
                                    : string.Format("{0}[{1}]", k.RawKind(), "*");
                        si = i.ToString();
                    }
                    else
                        throw Contracts.ExceptNotImpl(string.Format("Unable to process type '{0}'.", t));
                }

                builder.Append(string.Format("{0}:{1}:{2}", name, type, si));
            }
            return builder.ToString();
        }

        public static void CheckSchema(IHostEnvironment host, ISchema sch1, ISchema sch2)
        {
            if (sch1.ColumnCount != sch2.ColumnCount)
                throw host.Except("Mismatch between input schema and cached schema #columns {0} != # cached columns {1}.\nSchema 1:{2}\nSchema 2: {3}",
                    sch1.ColumnCount, sch2.ColumnCount, ToString(sch1), ToString(sch2));
            for (int i = 0; i < sch2.ColumnCount; ++i)
            {
                if (sch1.GetColumnName(i) != sch2.GetColumnName(i))
                    throw host.Except("Name mismatch at column {0}: '{1}' != '{2}'.\nSchema 1:{3}\nSchema 2: {4}", i,
                        sch1.GetColumnName(i), sch2.GetColumnName(i), ToString(sch1), ToString(sch2));
                if (!sch1.GetColumnType(i).Equals(sch2.GetColumnType(i)))
                    throw host.Except("Type mismatch at column {0}: '{1}' != '{2}'.\nSchema 1:{3}\nSchema 2: {4}", i,
                        sch1.GetColumnType(i), sch2.GetColumnType(i), ToString(sch1), ToString(sch2));
            }
        }

        public static void CheckSchema(IHostEnvironment host, DataViewSchema sch1, DataViewSchema sch2)
        {
            if (sch1.Count != sch2.Count)
                throw host.Except("Mismatch between input schema and cached schema #columns {0} != # cached columns {1}.\nSchema 1:{2}\nSchema 2: {3}",
                    sch1.Count, sch2.Count, ToString(sch1), ToString(sch2));
            for (int i = 0; i < sch2.Count; ++i)
            {
                if (sch1[i].Name != sch2[i].Name)
                    throw host.Except("Name mismatch at column {0}: '{1}' != '{2}'.\nSchema 1:{3}\nSchema 2: {4}", i,
                        sch1[i].Name, sch2[i].Name, ToString(sch1), ToString(sch2));
                if (!sch1[i].Type.Equals(sch2[i].Type))
                    throw host.Except("Type mismatch at column {0}: '{1}' != '{2}'.\nSchema 1:{3}\nSchema 2: {4}", i,
                        sch1[i].Type, sch2[i].Type, ToString(sch1), ToString(sch2));
            }
        }

        public static bool CompareSchema(ISchema sch1, ISchema sch2, bool raise = false)
        {
            if (sch1.ColumnCount != sch2.ColumnCount)
            {
                if (raise)
                    throw Contracts.Except("Different number of columns {0} != {1}\nS1: {2}\nS2: {3}",
                        sch1.ColumnCount, sch2.ColumnCount,
                        ToString(sch1), ToString(sch2));
                else
                    return false;
            }
            for (int i = 0; i < sch1.ColumnCount; ++i)
            {
                if (sch1.GetColumnName(i) != sch2.GetColumnName(i))
                {
                    if (raise)
                        throw Contracts.Except("Column name {0} is different {1} != {2}\nS1: {3}\nS2: {4}",
                                i, sch1.GetColumnName(i), sch2.GetColumnName(i),
                                ToString(sch1), ToString(sch2));
                    else
                        return false;
                }
                if (sch1.GetColumnType(i) != sch2.GetColumnType(i))
                {
                    var t1 = sch1.GetColumnType(i);
                    var t2 = sch2.GetColumnType(i);
                    bool r = t1 != t2;
                    if (r && t1.IsVector() && t2.IsVector())
                    {
                        var v1 = t1.AsVector();
                        var v2 = t2.AsVector();
                        r = v1.DimCount() != v2.DimCount() || v1.GetKeyCount() != v2.GetKeyCount();
                        r |= v1.RawKind() != v2.RawKind();
                        r |= v1.ItemType() != v2.ItemType();
                        r |= v1.IsKnownSizeVector() != v2.IsKnownSizeVector();
                    }
                    if (r)
                    {
                        if (raise)
                        {
                            throw Contracts.Except("Column type {0} is different {1} != {2}\nS1: {3}\nS2: {4}",
                                    i, t1, t2, ToString(sch1), ToString(sch2));
                        }
                        else
                            return false;
                    }
                }
            }
            return true;
        }

        public static bool CompareSchema(DataViewSchema sch1, DataViewSchema sch2, bool raise = false)
        {
            if (sch1.Count != sch2.Count)
            {
                if (raise)
                    throw Contracts.Except("Different number of columns {0} != {1}\nS1: {2}\nS2: {3}",
                        sch1.Count, sch2.Count,
                        ToString(sch1), ToString(sch2));
                else
                    return false;
            }
            for (int i = 0; i < sch1.Count; ++i)
            {
                if (sch1[i].Name != sch2[i].Name)
                {
                    if (raise)
                        throw Contracts.Except("Column name {0} is different {1} != {2}\nS1: {3}\nS2: {4}",
                                i, sch1[i].Name, sch2[i].Name,
                                ToString(sch1), ToString(sch2));
                    else
                        return false;
                }
                if (sch1[i].Type != sch2[i].Type)
                {
                    var t1 = sch1[i].Type;
                    var t2 = sch2[i].Type;
                    bool r = t1 != t2;
                    if (r && t1.IsVector() && t2.IsVector())
                    {
                        var v1 = t1.AsVector();
                        var v2 = t2.AsVector();
                        r = v1.DimCount() != v2.DimCount() || v1.GetKeyCount() != v2.GetKeyCount();
                        r |= v1.RawKind() != v2.RawKind();
                        r |= v1.ItemType() != v2.ItemType();
                        r |= v1.IsKnownSizeVector() != v2.IsKnownSizeVector();
                    }
                    if (r)
                    {
                        if (raise)
                        {
                            throw Contracts.Except("Column type {0} is different {1} != {2}\nS1: {3}\nS2: {4}",
                                    i, t1, t2, ToString(sch1), ToString(sch2));
                        }
                        else
                            return false;
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Saves a type into a stream.
        /// </summary>
        public static void WriteType(ModelSaveContext ctx, DataViewType type)
        {
            ctx.Writer.Write(type.IsVector());
            if (type.IsVector())
            {
                ctx.Writer.Write(type.AsVector().DimCount());
                for (int i = 0; i < type.AsVector().DimCount(); ++i)
                    ctx.Writer.Write(type.AsVector().GetDim(i));
                ctx.Writer.Write((byte)type.AsVector().ItemType().RawKind());
            }
            else if (type.IsKey())
                throw Contracts.ExceptNotImpl("Key cannot be serialized yet.");
            else
                ctx.Writer.Write((byte)type.RawKind());
        }

        /// <summary>
        /// From DataKind to DataViewType
        /// </summary>
        public static DataViewType FromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.String:
                    return TextDataViewType.Instance;
                case DataKind.Boolean:
                    return BooleanDataViewType.Instance;
                case DataKind.DateTime:
                    return DateTimeDataViewType.Instance;
                case DataKind.TimeSpan:
                    return TimeSpanDataViewType.Instance;
                default:
                    return ColumnTypeHelper.NumberFromKind(kind);
            }
        }

        public static DataViewType ReadType(ModelLoadContext ctx)
        {
            bool isVector = ctx.Reader.ReadBoolean();
            if (isVector)
            {
                int dimCount = ctx.Reader.ReadInt32();
                if (dimCount != 1)
                    throw Contracts.ExceptNotImpl("Number of dimensions should be 1.");
                var dims = new int[dimCount];
                for (int i = 0; i < dimCount; ++i)
                    dims[i] = ctx.Reader.ReadInt32();
                var kind = (DataKind)ctx.Reader.ReadByte();
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(kind), dims[0]);
            }
            else
            {
                var kind = (DataKind)ctx.Reader.ReadByte();
                return FromKind(kind);
            }
        }

        /// <summary>
        /// To avoid code duplication in many transform which convert a column into another one.
        /// </summary>
        public class OneToOneColumnForArgument : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type")]
            public DataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyCount KeyCount;

            public static OneToOneColumnForArgument Parse(string str)
            {
                var res = new OneToOneColumnForArgument();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:T:S where N is the new column name, T is the new type,
                // and S is source column names.
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                InternalDataKind kind;
                if (!TypeParsingUtils.TryParseDataKind(extra, out kind, out KeyCount))
                    return false;
                ResultType = InternalDataKind2DataKind(kind);
                return true;
            }

            public override string ToString()
            {
                var sb = new StringBuilder();
                if (!TryUnparse(sb))
                    throw Contracts.Except("Unable to convert a column into string.");
                return sb.ToString();
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (ResultType == null && KeyCount == null)
                    return TryUnparseCore(sb);

                if (!TrySanitize())
                    return false;
                if (CmdQuoter.NeedsQuoting(Name) || CmdQuoter.NeedsQuoting(Source))
                    return false;

                int ich = sb.Length;
                sb.Append(Name);
                sb.Append(':');
                if (ResultType != null)
                    sb.Append(ResultType.Value);
                if (KeyCount != null)
                {
                    sb.Append('[');
                    if (!KeyCount.TryUnparse(sb))
                    {
                        sb.Length = ich;
                        return false;
                    }
                    sb.Append(']');
                }
                sb.Append(':');
                sb.Append(Source);
                return true;
            }
        }

        /// <summary>
        /// Suggests a default schema for a predictor
        /// </summary>
        /// <param name="kind">DataKind</param>
        /// <param name="dim">dimension</param>
        /// <returns>ISchema</returns>
        public static ISchema PredictionDefaultSchema(PredictionKind kind, int dim = 0)
        {
            switch (kind)
            {
                case PredictionKind.BinaryClassification:
                    return new ExtendedSchema((ISchema)null, new[] { "Score" }, new[] { NumberDataViewType.Single });
                case PredictionKind.Regression:
                    return new ExtendedSchema((ISchema)null, new[] { "Prediction" }, new[] { NumberDataViewType.Single });
                case PredictionKind.MultiClassClassification:
                    return new ExtendedSchema((ISchema)null, new[] { "Scores" }, new[] { new VectorType(NumberDataViewType.Single, dim) });
                case PredictionKind.MultiOutputRegression:
                    return new ExtendedSchema((ISchema)null, new[] { "Predictions" }, new[] { new VectorType(NumberDataViewType.Single, dim) });
                default:
                    throw Contracts.Except("Unable to build the schema for kind {0}", kind);
            }
        }

        /// <summary>
        /// Returns the data kind based on a type.
        /// </summary>
        public static DataViewType GetColumnType<TLabel>()
        {
            return GetColumnType(typeof(TLabel));
        }

        /// <summary>
        /// Returns the data kind based on a type.
        /// </summary>
        public static DataViewType GetColumnType(Type type)
        {
            if (type == typeof(bool))
                return BooleanDataViewType.Instance;
            if (type == typeof(byte))
                return NumberDataViewType.Byte;
            if (type == typeof(ushort))
                return NumberDataViewType.UInt16;
            if (type == typeof(uint))
                return NumberDataViewType.UInt32;
            if (type == typeof(int))
                return NumberDataViewType.Int32;
            if (type == typeof(Int64))
                return NumberDataViewType.Int64;
            if (type == typeof(float))
                return NumberDataViewType.Single;
            if (type == typeof(double))
                return NumberDataViewType.Double;
            if (type == typeof(ReadOnlyMemory<char>) || type == typeof(string) || type == typeof(DvText))
                return ColumnTypeHelper.PrimitiveFromKind(DataKind.String);

            if (type == typeof(VBuffer<bool>) || type == typeof(VBufferEqSort<bool>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.Boolean));
            if (type == typeof(VBuffer<byte>) || type == typeof(VBufferEqSort<byte>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.SByte));
            if (type == typeof(VBuffer<ushort>) || type == typeof(VBufferEqSort<ushort>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.UInt16));
            if (type == typeof(VBuffer<uint>) || type == typeof(VBufferEqSort<uint>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.UInt32));
            if (type == typeof(VBuffer<int>) || type == typeof(VBufferEqSort<int>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.Int32));
            if (type == typeof(VBuffer<Int64>) || type == typeof(VBufferEqSort<Int64>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.Int64));
            if (type == typeof(VBuffer<float>) || type == typeof(VBufferEqSort<float>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.Single));
            if (type == typeof(VBuffer<double>) || type == typeof(VBufferEqSort<double>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.Double));
            if (type == typeof(VBuffer<ReadOnlyMemory<char>>) || type == typeof(VBuffer<string>) || type == typeof(VBuffer<DvText>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.String));
            if (type == typeof(VBufferEqSort<string>) || type == typeof(VBufferEqSort<DvText>))
                return new VectorType(ColumnTypeHelper.PrimitiveFromKind(DataKind.String));

            throw Contracts.ExceptNotSupp("Unsupported output type {0}.", type);
        }

        public enum ArrayKind
        {
            None = 0,
            Array = 1,
            VBuffer = 2
        }

        public static Tuple<DataKind, ArrayKind> GetKindArray(DataViewType type)
        {
            if (type.IsVector())
            {
                int dc = type.AsVector().DimCount();
                return new Tuple<DataKind, ArrayKind>(type.ItemType().RawKind(), dc == 1 &&
                                                      type.AsVector().GetDim(0) > 0 ? ArrayKind.Array : ArrayKind.VBuffer);
            }
            else
                return new Tuple<DataKind, ArrayKind>(type.RawKind(), ArrayKind.None);
        }

        /// <summary>
        /// Returns the data kind based on a type.
        /// </summary>
        public static Tuple<DataKind, ArrayKind> GetKindArray(Type type)
        {
            if (type == typeof(bool))
                return new Tuple<DataKind, ArrayKind>(DataKind.Boolean, ArrayKind.None);
            if (type == typeof(byte))
                return new Tuple<DataKind, ArrayKind>(DataKind.SByte, ArrayKind.None);
            if (type == typeof(ushort))
                return new Tuple<DataKind, ArrayKind>(DataKind.UInt16, ArrayKind.None);
            if (type == typeof(uint))
                return new Tuple<DataKind, ArrayKind>(DataKind.UInt32, ArrayKind.None);
            if (type == typeof(int))
                return new Tuple<DataKind, ArrayKind>(DataKind.Int32, ArrayKind.None);
            if (type == typeof(Int64))
                return new Tuple<DataKind, ArrayKind>(DataKind.Int64, ArrayKind.None);
            if (type == typeof(float))
                return new Tuple<DataKind, ArrayKind>(DataKind.Single, ArrayKind.None);
            if (type == typeof(double))
                return new Tuple<DataKind, ArrayKind>(DataKind.Double, ArrayKind.None);
            if (type == typeof(ReadOnlyMemory<char>) || type == typeof(string) || type == typeof(DvText))
                return new Tuple<DataKind, ArrayKind>(DataKind.String, ArrayKind.None);

            if (type == typeof(VBuffer<bool>))
                return new Tuple<DataKind, ArrayKind>(DataKind.Boolean, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<byte>))
                return new Tuple<DataKind, ArrayKind>(DataKind.SByte, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<ushort>))
                return new Tuple<DataKind, ArrayKind>(DataKind.UInt16, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<uint>))
                return new Tuple<DataKind, ArrayKind>(DataKind.UInt32, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<int>))
                return new Tuple<DataKind, ArrayKind>(DataKind.Int32, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<Int64>))
                return new Tuple<DataKind, ArrayKind>(DataKind.Int64, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<float>))
                return new Tuple<DataKind, ArrayKind>(DataKind.Single, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<double>))
                return new Tuple<DataKind, ArrayKind>(DataKind.Double, ArrayKind.VBuffer);
            if (type == typeof(VBuffer<ReadOnlyMemory<char>>) || type == typeof(VBuffer<string>) || type == typeof(VBuffer<DvText>))
                return new Tuple<DataKind, ArrayKind>(DataKind.String, ArrayKind.VBuffer);

            if (type == typeof(bool[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.Boolean, ArrayKind.Array);
            if (type == typeof(byte[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.SByte, ArrayKind.Array);
            if (type == typeof(ushort[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.UInt16, ArrayKind.Array);
            if (type == typeof(uint[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.UInt32, ArrayKind.Array);
            if (type == typeof(int[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.Int32, ArrayKind.Array);
            if (type == typeof(Int64[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.Int64, ArrayKind.Array);
            if (type == typeof(float[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.Single, ArrayKind.Array);
            if (type == typeof(double[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.Double, ArrayKind.Array);
            if (type == typeof(ReadOnlyMemory<char>[]) || type == typeof(string[]) || type == typeof(DvText[]))
                return new Tuple<DataKind, ArrayKind>(DataKind.String, ArrayKind.Array);

            throw Contracts.ExceptNotSupp("Unsupported output type {0}.", type);
        }

        public static ValueMapper<TLabel, TDest> GetConverter<TLabel, TDest>(out bool identity)
        {
            var col1 = GetColumnType<TLabel>();
            var col2 = GetColumnType<TDest>();
            if (typeof(TLabel) == typeof(float))
            {
                if (typeof(TDest) == typeof(uint))
                {
                    ValueMapper<float, uint> temp = (in float src, ref uint dst) =>
                    {
                        if (src < 0)
                            throw Contracts.ExceptValue("Unable to converter {0} '{1}' into {2}.", typeof(float).ToString(), src, typeof(uint));
                        dst = (uint)src;
                    };
                    identity = false;
                    return temp as ValueMapper<TLabel, TDest>;
                }
                else if (typeof(TDest) == typeof(int))
                {
                    ValueMapper<float, int> temp = (in float src, ref int dst) =>
                    {
                        dst = (int)src;
                    };
                    identity = false;
                    return temp as ValueMapper<TLabel, TDest>;
                }
            }
            return Conversions.Instance.GetStandardConversion<TLabel, TDest>(col1, col2, out identity);
        }

        public static DataViewType GetColumnType(ISchema schema, string name)
        {
            int index;
            if (!schema.TryGetColumnIndex(name, out index))
                throw Contracts.Except($"Unable to find column '{name}' in schema\n{ToString(schema)}.");
            return schema.GetColumnType(index);
        }

        public static DataViewType GetColumnType(DataViewSchema schema, string name)
        {
            int index = GetColumnIndex(schema, name);
            return schema[index].Type;
        }

        public static int GetColumnIndex(ISchema schema, string name, bool allowNull = false)
        {
            int index;
            if (!schema.TryGetColumnIndex(name, out index))
            {
                if (allowNull)
                    return -1;
                throw Contracts.Except($"Unable to find column '{name}' in schema\n{ToString(schema)}.");
            }
            return index;
        }

        public static int GetColumnIndex(DataViewSchema schema, string name, bool allowNull = false, bool allowHidden = false)
        {
            int index;
            for (index = 0; index < schema.Count; ++index)
                if (schema[index].Name == name && (allowHidden || !schema[index].IsHidden))
                    return index;
            if (allowNull)
                return -1;
            throw Contracts.Except($"Unable to find column '{name}' in schema\n{ToString(schema)}.");
        }

        public static DataViewSchema.Column GetColumnIndexDC(DataViewSchema schema, string name, bool allowNull = false, bool allowHidden = false)
        {
            int index;
            for (index = 0; index < schema.Count; ++index)
                if (schema[index].Name == name && (allowHidden || !schema[index].IsHidden))
                    return schema[index];
            if (allowNull)
                return new DataViewSchema.Column(null, -1, false, null, null);
            throw Contracts.Except($"Unable to find column '{name}' in schema\n{ToString(schema)}.");
        }

        public static int NeedColumn(Dictionary<int, int> mapping, int col)
        {
            int res;
            if (mapping.TryGetValue(col, out res))
                return res;
            else
                return col;
        }

        public static IEnumerable<DataViewSchema.Column> ColumnsNeeded(IEnumerable<DataViewSchema.Column> columnsNeeded,
                                                                       DataViewSchema schema, Column1x1[] columns = null)
        {
            var hash = new HashSet<int>(columnsNeeded.Select(c => c.Index));
            var hashName = new HashSet<string>(columnsNeeded.Select(c => c.Name));
            var di = new Dictionary<string, string>();
            if (columns != null)
                foreach (var c in columns)
                    if (hashName.Contains(c.Name))
                        hashName.Add(c.Source);
            return schema.Where(c => hash.Contains(c.Index) || hashName.Contains(c.Name)).ToArray();
        }

        public static IEnumerable<DataViewSchema.Column> ColumnsNeeded(IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewSchema schema, int[] columns)
        {
            var hash = new HashSet<int>(columnsNeeded.Select(c => c.Index));
            foreach (var c in columns)
                hash.Add(c);
            return schema.Where(c => hash.Contains(c.Index)).ToArray();
        }

        public static IEnumerable<DataViewSchema.Column> ColumnsNeeded(IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewSchema schema, string column)
        {
            var hash = new HashSet<int>(columnsNeeded.Select(c => c.Index));
            var hashName = new HashSet<string>(columnsNeeded.Select(c => c.Name));
            hashName.Add(column);
            return schema.Where(c => hash.Contains(c.Index) || hashName.Contains(c.Name)).ToArray();
        }

        public static IEnumerable<string> EnumerateColumns(ISchema sch)
        {
            for (int i = 0; i < sch.ColumnCount; ++i)
                yield return sch.GetColumnName(i);
        }

        public static IEnumerable<string> EnumerateColumns(DataViewSchema sch)
        {
            for (int i = 0; i < sch.Count; ++i)
                yield return sch[i].Name;
        }

        /// <summary>
        /// When the last column is requested, we also need the column used to compute it.
        /// This function ensures that this column is requested when the last one is.
        /// </summary>
        public static IEnumerable<DataViewSchema.Column> ColumnsNeeded(IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewSchema schema, int newCol, int dependsOn)
        {
            var cols = columnsNeeded.ToList();
            var hash = new HashSet<int>(columnsNeeded.Select(c => c.Index));
            if (hash.Contains(newCol) && !hash.Contains(dependsOn))
            {
                hash.Add(dependsOn);
                return schema.Where(c => hash.Contains(c.Index)).ToArray();
            }
            else
                return columnsNeeded;
        }

        /// <summary>
        /// Returns from superset inside subset.
        /// </summary>
        public static IEnumerable<DataViewSchema.Column> SchemaIntersection(IEnumerable<DataViewSchema.Column> subset, IEnumerable<DataViewSchema.Column> superset, int addition = -1)
        {
            var hash = new HashSet<Tuple<string, int>>(subset.Select(c => new Tuple<string, int>(c.Name, c.Index)).ToArray());
            var auto = superset.Where(c => hash.Contains(new Tuple<string, int>(c.Name, c.Index))).ToList();
            if (addition != -1)
            {
                var any = auto.Where(c => c.Index == addition).Any();
                if (!any)
                {
                    var add = subset.Where(c => c.Index == addition).ToArray();
                    if (add == null || add.Length != 1)
                        throw Contracts.Except($"Missing column {addition}.");
                    auto.Add(add[0]);
                }
            }
            return auto.ToArray();
        }
    }
}
