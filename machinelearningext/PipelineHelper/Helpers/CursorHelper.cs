// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Helpers about DataView.
    /// </summary>
    public static class CursorHelper
    {
        /// <summary>
        /// Returns the getters for all columns.
        /// </summary>
        public static Delegate[] GetAllGetters(DataViewRowCursor cur)
        {
            var sch = cur.Schema;
            var res = new List<Delegate>();
            for (int i = 0; i < sch.Count; ++i)
            {
                if (sch[i].IsHidden)
                    continue;
                var getter = GetColumnGetter(cur, new DataViewSchema.Column(string.Empty, i, false, null, null), sch);
                if (getter == null)
                    throw Contracts.Except($"Unable to get getter for column {i} from schema\n{SchemaHelper.ToString(sch)}.");
                res.Add(getter);
            }
            return res.ToArray();
        }

        public static Delegate GetGetterChoice<T1, T2>(DataViewRowCursor cur, DataViewSchema.Column col)
        {
            Delegate res = null;
            try
            {
                res = cur.GetGetter<T1>(col);
                if (res != null)
                    return res;
            }
            catch (Exception)
            {
            }
            try
            {
                res = cur.GetGetter<T2>(col);
                if (res != null)
                    return res;
            }
            catch (Exception)
            {
            }
            if (res == null)
                throw Contracts.ExceptNotImpl($"Unable to get a getter for column {col} of type {typeof(T1)} or {typeof(T2)} from schema\n{SchemaHelper.ToString(cur.Schema)}.");
            return res;
        }

        public static Delegate GetGetterChoice<T1, T2, T3>(DataViewRowCursor cur, DataViewSchema.Column col)
        {
            Delegate res = null;
            try
            {
                res = cur.GetGetter<T1>(col);
                if (res != null)
                    return res;
            }
            catch (Exception)
            {
            }
            try
            {
                res = cur.GetGetter<T2>(col);
                if (res != null)
                    return res;
            }
            catch (Exception)
            {
            }
            try
            {
                res = cur.GetGetter<T3>(col);
                if (res != null)
                    return res;
            }
            catch (Exception)
            {
            }
            if (res == null)
                throw Contracts.ExceptNotImpl($"Unable to get a getter for column {col} of type {typeof(T1)} or {typeof(T2)} or {typeof(T3)} from schema\n{SchemaHelper.ToString(cur.Schema)}.");
            return res;
        }

        public static Delegate GetColumnGetter(DataViewRowCursor cur, DataViewSchema.Column col, DataViewSchema sch = null)
        {
            if (sch == null)
                sch = cur.Schema;
            var colType = sch[col.Index].Type;
            if (colType.IsVector())
            {
                switch (colType.ItemType().RawKind())
                {
                    case DataKind.Boolean: return GetGetterChoice<VBufferEqSort<bool>, VBuffer<bool>>(cur, col);
                    case DataKind.Int32: return GetGetterChoice<VBufferEqSort<int>, VBuffer<int>>(cur, col);
                    case DataKind.UInt32: return GetGetterChoice<VBufferEqSort<uint>, VBuffer<uint>>(cur, col);
                    case DataKind.Int64: return GetGetterChoice<VBufferEqSort<long>, VBuffer<long>>(cur, col);
                    case DataKind.Single: return GetGetterChoice<VBufferEqSort<float>, VBuffer<float>>(cur, col);
                    case DataKind.Double: return GetGetterChoice<VBufferEqSort<double>, VBuffer<double>>(cur, col);
                    case DataKind.String: return GetGetterChoice<VBufferEqSort<DvText>, VBuffer<DvText>, VBuffer<ReadOnlyMemory<char>>>(cur, col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
            else
            {
                switch (colType.RawKind())
                {
                    case DataKind.Boolean: return cur.GetGetter<bool>(col);
                    case DataKind.Int32: return cur.GetGetter<int>(col);
                    case DataKind.UInt32: return cur.GetGetter<uint>(col);
                    case DataKind.Int64: return cur.GetGetter<Int64>(col);
                    case DataKind.Single: return cur.GetGetter<float>(col);
                    case DataKind.Double: return cur.GetGetter<double>(col);
                    case DataKind.String: return GetGetterChoice<DvText, ReadOnlyMemory<char>>(cur, col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
        }
    }
}
