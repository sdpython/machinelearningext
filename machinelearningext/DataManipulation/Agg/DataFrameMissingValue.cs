// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    public static class DataFrameMissingValue
    {
        public static object GetMissingValue(DataViewType kind, object subcase = null)
        {
            if (kind.IsVector())
                return null;
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.Boolean:
                        throw new NotImplementedException("NA is not available for bool");
                    case DataKind.Int32:
                        throw new NotImplementedException("NA is not available for int");
                    case DataKind.UInt32:
                        return 0;
                    case DataKind.Int64:
                        throw new NotImplementedException("NA is not available for long");
                    case DataKind.Single:
                        return float.NaN;
                    case DataKind.Double:
                        return double.NaN;
                    case DataKind.String:
                        return subcase is string ? (object)(string)null : DvText.NA;
                    default:
                        throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
                }
            }
        }

        public static object GetMissingOrDefaultValue(DataViewType kind, object subcase = null)
        {
            if (kind.IsVector())
                return null;
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.Boolean:
                        return false;
                    case DataKind.Int32:
                        return 0;
                    case DataKind.UInt32:
                        return 0;
                    case DataKind.Int64:
                        return 0;
                    case DataKind.Single:
                        return float.NaN;
                    case DataKind.Double:
                        return double.NaN;
                    case DataKind.String:
                        return subcase is string ? (object)(string)null : DvText.NA;
                    default:
                        throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
                }
            }
        }

        public static object GetMissingOrDefaultMissingValue(DataViewType kind, object subcase = null)
        {
            if (kind.IsVector())
                return null;
            else
            {
                switch (kind.RawKind())
                {
                    case DataKind.Boolean:
                        throw new NotSupportedException("No missing value for boolean. Convert to int.");
                    case DataKind.Int32:
                        return int.MinValue;
                    case DataKind.UInt32:
                        return uint.MaxValue;
                    case DataKind.Int64:
                        return long.MinValue;
                    case DataKind.Single:
                        return float.NaN;
                    case DataKind.Double:
                        return double.NaN;
                    case DataKind.String:
                        return subcase is string ? (object)(string)null : DvText.NA;
                    default:
                        throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
                }
            }
        }
    }
}
