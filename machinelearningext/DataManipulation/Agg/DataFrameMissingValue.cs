// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    public static class DataFrameMissingValue
    {
        public static object GetMissingValue(ColumnType kind, object subcase = null)
        {
            if (kind.IsVector())
                return null;
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.BL:
                        throw new NotImplementedException("NA is not available for bool");
                    case DataKind.I4:
                        throw new NotImplementedException("NA is not available for int");
                    case DataKind.U4:
                        return 0;
                    case DataKind.I8:
                        throw new NotImplementedException("NA is not available for long");
                    case DataKind.R4:
                        return float.NaN;
                    case DataKind.R8:
                        return double.NaN;
                    case DataKind.TX:
                        return subcase is string ? (object)(string)null : DvText.NA;
                    default:
                        throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
                }
            }
        }

        public static object GetMissingOrDefaultValue(ColumnType kind, object subcase = null)
        {
            if (kind.IsVector())
                return null;
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.BL:
                        return false;
                    case DataKind.I4:
                        return 0;
                    case DataKind.U4:
                        return 0;
                    case DataKind.I8:
                        return 0;
                    case DataKind.R4:
                        return float.NaN;
                    case DataKind.R8:
                        return double.NaN;
                    case DataKind.TX:
                        return subcase is string ? (object)(string)null : DvText.NA;
                    default:
                        throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
                }
            }
        }

        public static object GetMissingOrDefaultMissingValue(ColumnType kind, object subcase = null)
        {
            if (kind.IsVector())
                return null;
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.BL:
                        throw new NotSupportedException("No missing value for boolean. Convert to int.");
                    case DataKind.I4:
                        return int.MinValue;
                    case DataKind.U4:
                        return uint.MaxValue;
                    case DataKind.I8:
                        return long.MinValue;
                    case DataKind.R4:
                        return float.NaN;
                    case DataKind.R8:
                        return double.NaN;
                    case DataKind.TX:
                        return subcase is string ? (object)(string)null : DvText.NA;
                    default:
                        throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
                }
            }
        }
    }
}
