// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;


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
            DataKind kind;
            if (IsVector(type))
                if (DataKindExtensions.TryGetDataKind(ItemType(type).RawType, out kind))
                    return kind;
            if (DataKindExtensions.TryGetDataKind(type.RawType, out kind))
                return kind;
            throw Contracts.ExceptNotSupp($"Unable to guess kind for type {type}.");
        }

        public static int VectorSize(this DataViewType column)
        {
            return IsVector(column) ? AsVector(column).Size : 0;
        }

        public static NumberDataViewType NumberFromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1: return NumberDataViewType.SByte;
                case DataKind.U1: return NumberDataViewType.Byte;
                case DataKind.I2: return NumberDataViewType.Int16;
                case DataKind.U2: return NumberDataViewType.UInt16;
                case DataKind.I4: return NumberDataViewType.Int32;
                case DataKind.U4: return NumberDataViewType.UInt32;
                case DataKind.I8: return NumberDataViewType.Int64;
                case DataKind.U8: return NumberDataViewType.UInt64;
                case DataKind.R4: return NumberDataViewType.Single;
                case DataKind.R8: return NumberDataViewType.Double;
                case DataKind.UG: return NumberDataViewType.DataViewRowId;
                default:
                    throw Contracts.ExceptNotImpl($"Number from kind not implemented for kind {kind}.");
            }
        }

        public static PrimitiveDataViewType PrimitiveFromKind(DataKind kind)
        {
            if (kind == DataKind.TX)
                return TextDataViewType.Instance;
            if (kind == DataKind.BL)
                return BooleanDataViewType.Instance;
            if (kind == DataKind.TS)
                return TimeSpanDataViewType.Instance;
            if (kind == DataKind.DT)
                return DateTimeDataViewType.Instance;
            if (kind == DataKind.DZ)
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
                case DataKind.I1:
                case DataKind.U2:
                case DataKind.U4:
                case DataKind.U8:
                    return true;
                default:
                    return false;
            }
        }
    }
}
