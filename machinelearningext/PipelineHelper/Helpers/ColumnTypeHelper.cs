// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Implements functions declared as internal.
    /// </summary>
    public static class ColumnTypeHelper
    {
        public static bool IsPrimitive(this DataViewType column)
        {
            return column is PrimitiveDataViewType;
        }

        public static bool IsVector(this DataViewType column)
        {
            return column is VectorType;
        }

        public static bool IsNumber(this DataViewType column)
        {
            return column is NumberDataViewType;
        }

        public static bool IsKey(this DataViewType column)
        {
            return column is KeyType;
        }

        public static bool IsText(this DataViewType column)
        {
            if (!(column is TextDataViewType))
                return false;
            Contracts.Assert(column == TextDataViewType.Instance);
            return true;
        }

        public static KeyType AsKey(this DataViewType column)
        {
            return IsKey(column) ? (KeyType)column : null;
        }

        public static PrimitiveDataViewType AsPrimitive(this DataViewType column)
        {
            return IsPrimitive(column) ? (PrimitiveDataViewType)column : null;
        }

        public static VectorType AsVector(this DataViewType column)
        {
            return IsVector(column) ? (VectorType)column : null;
        }

        public static int DimCount(this DataViewType column)
        {
            return IsVector(column) ? AsVector(column).Dimensions.Length : 0;
        }

        public static int GetDim(this DataViewType column, int dim)
        {
            return IsVector(column) ? AsVector(column).Dimensions[dim] : 0;
        }

        public static int ValueCount(this DataViewType column)
        {
            return IsVector(column) ? AsVector(column).Size : 1;
        }

        public static PrimitiveDataViewType ItemType(this DataViewType column)
        {
            return IsVector(column) ? AsVector(column).ItemType : AsPrimitive(column);
        }

        public static DataKind RawKind(this DataViewType type)
        {
            InternalDataKind kind;
            if (IsVector(type))
                if (InternalDataKindExtensions.TryGetDataKind(ItemType(type).RawType, out kind))
                    return Internal2DataKind(kind);
            if (InternalDataKindExtensions.TryGetDataKind(type.RawType, out kind))
                return Internal2DataKind(kind);
            throw Contracts.ExceptNotSupp($"Unable to guess kind for type {type}.");
        }

        public static DataKind Internal2DataKind(InternalDataKind kind)
        {
            switch (kind)
            {
                case InternalDataKind.BL: return DataKind.Boolean;
                case InternalDataKind.I1: return DataKind.SByte;
                case InternalDataKind.U1: return DataKind.Byte;
                case InternalDataKind.I2: return DataKind.Int16;
                case InternalDataKind.U2: return DataKind.UInt16;
                case InternalDataKind.I4: return DataKind.Int32;
                case InternalDataKind.U4: return DataKind.UInt32;
                case InternalDataKind.I8: return DataKind.Int64;
                case InternalDataKind.U8: return DataKind.UInt64;
                case InternalDataKind.R4: return DataKind.Single;
                case InternalDataKind.R8: return DataKind.Double;
                case InternalDataKind.TX: return DataKind.String;
                default:
                    throw Contracts.ExceptNotImpl($"Datakind not implemented for kind {kind}.");
            }
        }

        public static int VectorSize(this DataViewType column)
        {
            return IsVector(column) ? AsVector(column).Size : 0;
        }

        public static NumberDataViewType NumberFromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.SByte: return NumberDataViewType.SByte;
                case DataKind.Byte: return NumberDataViewType.Byte;
                case DataKind.Int16: return NumberDataViewType.Int16;
                case DataKind.UInt16: return NumberDataViewType.UInt16;
                case DataKind.Int32: return NumberDataViewType.Int32;
                case DataKind.UInt32: return NumberDataViewType.UInt32;
                case DataKind.Int64: return NumberDataViewType.Int64;
                case DataKind.UInt64: return NumberDataViewType.UInt64;
                case DataKind.Single: return NumberDataViewType.Single;
                case DataKind.Double: return NumberDataViewType.Double;
                default:
                    throw Contracts.ExceptNotImpl($"Number from kind not implemented for kind {kind}.");
            }
        }

        public static PrimitiveDataViewType PrimitiveFromKind(DataKind kind)
        {
            if (kind == DataKind.String)
                return TextDataViewType.Instance;
            if (kind == DataKind.Boolean)
                return BooleanDataViewType.Instance;
            if (kind == DataKind.TimeSpan)
                return TimeSpanDataViewType.Instance;
            if (kind == DataKind.DateTime)
                return DateTimeDataViewType.Instance;
            if (kind == DataKind.DateTimeOffset)
                return DateTimeOffsetDataViewType.Instance;
            return NumberFromKind(kind);
        }       

        public static bool IsBool(this DataViewType column)
        {
            if (!(column is BooleanDataViewType))
                return false;
            Contracts.Assert(column == BooleanDataViewType.Instance);
            return true;
        }

        public static bool IsValidDataKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.SByte:
                case DataKind.UInt16:
                case DataKind.UInt32:
                case DataKind.UInt64:
                    return true;
                default:
                    return false;
            }
        }
    }
}
