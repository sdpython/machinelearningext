// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;


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
        public static Delegate[] GetAllGetters(RowCursor cur)
        {
            var sch = cur.Schema;
            var res = new List<Delegate>();
            for (int i = 0; i < sch.Count; ++i)
            {
                if (sch[i].IsHidden)
                    continue;
                var getter = GetColumnGetter(cur, i, sch);
                if (getter == null)
                    throw Contracts.Except($"Unable to get getter for column {i} from schema\n{SchemaHelper.ToString(sch)}.");
                res.Add(getter);
            }
            return res.ToArray();
        }

        public static Delegate GetGetterChoice<T1, T2>(RowCursor cur, int col)
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

        public static Delegate GetGetterChoice<T1, T2, T3>(RowCursor cur, int col)
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

        public static Delegate GetColumnGetter(RowCursor cur, int col, Schema sch = null)
        {
            if (sch == null)
                sch = cur.Schema;
            var colType = sch[col].Type;
            if (colType.IsVector())
            {
                switch (colType.ItemType().RawKind())
                {
                    case DataKind.BL: return GetGetterChoice<VBufferEqSort<bool>, VBuffer<bool>>(cur, col);
                    case DataKind.I4: return GetGetterChoice<VBufferEqSort<int>, VBuffer<int>>(cur, col);
                    case DataKind.U4: return GetGetterChoice<VBufferEqSort<uint>, VBuffer<uint>>(cur, col);
                    case DataKind.I8: return GetGetterChoice<VBufferEqSort<long>, VBuffer<long>>(cur, col);
                    case DataKind.R4: return GetGetterChoice<VBufferEqSort<float>, VBuffer<float>>(cur, col);
                    case DataKind.R8: return GetGetterChoice<VBufferEqSort<double>, VBuffer<double>>(cur, col);
                    case DataKind.TX: return GetGetterChoice<VBufferEqSort<DvText>, VBuffer<DvText>, VBuffer<ReadOnlyMemory<char>>>(cur, col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
            else
            {
                switch (colType.RawKind())
                {
                    case DataKind.BL: return cur.GetGetter<bool>(col);
                    case DataKind.I4: return cur.GetGetter<int>(col);
                    case DataKind.U4: return cur.GetGetter<uint>(col);
                    case DataKind.I8: return cur.GetGetter<Int64>(col);
                    case DataKind.R4: return cur.GetGetter<float>(col);
                    case DataKind.R8: return cur.GetGetter<double>(col);
                    case DataKind.TX: return GetGetterChoice<DvText, ReadOnlyMemory<char>>(cur, col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
        }
    }
}
