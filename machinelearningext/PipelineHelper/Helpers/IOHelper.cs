// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Model;


namespace Scikit.ML.PipelineHelper
{
    public static class IOHelper
    {
        public static void Write(ModelSaveContext ctx, string value)
        {
            if (string.IsNullOrEmpty(value))
                ctx.Writer.Write("");
            else
                ctx.Writer.Write(value);
        }

        public static string ReadString(ModelLoadContext ctx)
        {
            string res = ctx.Reader.ReadString();
            return string.IsNullOrEmpty(res) ? null : res;
        }

        public static void Write(ModelSaveContext ctx, string[] values)
        {
            ctx.Writer.Write((byte)(values == null ? 1 : 0));
            if (values != null)
            {
                ctx.Writer.Write(values.Length);
                for (int i = 0; i < values.Length; ++i)
                    ctx.Writer.Write(values[i]);
            }
        }

        public static string[] ReadArrayString(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            int nb = ctx.Reader.ReadInt32();
            var array = new string[nb];
            for (int i = 0; i < array.Length; ++i)
                array[i] = ctx.Reader.ReadString();
            return array;
        }

        public static void Write(ModelSaveContext ctx, KeyValuePair<string, string> values)
        {
            IOHelper.Write(ctx, values.Key);
            IOHelper.Write(ctx, values.Value);
        }

        public static KeyValuePair<string, string> ReadKeyValuePairStringString(ModelLoadContext ctx)
        {
            return new KeyValuePair<string, string>(IOHelper.ReadString(ctx), IOHelper.ReadString(ctx));
        }

        public static void Write(ModelSaveContext ctx, Tuple<float, int> values)
        {
            ctx.Writer.Write(values.Item1);
            ctx.Writer.Write(values.Item2);
        }

        public static Tuple<float, int> ReadTupleFloatInt(ModelLoadContext ctx)
        {
            return new Tuple<float, int>(ctx.Reader.ReadSingle(), ctx.Reader.ReadInt32());
        }

        public static void Write(ModelSaveContext ctx, KeyValuePair<string,string>[] values)
        {
            ctx.Writer.Write((byte)(values == null ? 1 : 0));
            if (values != null)
            {
                ctx.Writer.Write(values.Length);
                for (int i = 0; i < values.Length; ++i)
                    IOHelper.Write(ctx, values[i]);
            }
        }

        public static KeyValuePair<string, string>[] ReadArrayKeyValuePairStringString(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            int nb = ctx.Reader.ReadInt32();
            var array = new KeyValuePair<string, string>[nb];
            for (int i = 0; i < array.Length; ++i)
                array[i] = IOHelper.ReadKeyValuePairStringString(ctx);
            return array;
        }

        public static void Write(ModelSaveContext ctx, float[] values)
        {
            ctx.Writer.Write((byte)(values == null ? 1 : 0));
            if (values != null)
            {
                ctx.Writer.Write(values.Length);
                for (int i = 0; i < values.Length; ++i)
                    ctx.Writer.Write(values[i]);
            }
        }

        public static void Write(ModelSaveContext ctx, double[] values)
        {
            ctx.Writer.Write((byte)(values == null ? 1 : 0));
            if (values != null)
            {
                ctx.Writer.Write(values.Length);
                for (int i = 0; i < values.Length; ++i)
                    ctx.Writer.Write(values[i]);
            }
        }

        public static float[] ReadFloatArray(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            int nb = ctx.Reader.ReadInt32();
            var array = new float[nb];
            for (int i = 0; i < array.Length; ++i)
                array[i] = ctx.Reader.ReadSingle();
            return array;
        }

        public static double[] ReadDoubleArray(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            int nb = ctx.Reader.ReadInt32();
            var array = new double[nb];
            for (int i = 0; i < array.Length; ++i)
                array[i] = ctx.Reader.ReadDouble();
            return array;
        }

        public static void Write(ModelSaveContext ctx, int[] values)
        {
            ctx.Writer.Write((byte)(values == null ? 1 : 0));
            if (values != null)
            {
                ctx.Writer.Write(values.Length);
                for (int i = 0; i < values.Length; ++i)
                    ctx.Writer.Write(values[i]);
            }
        }

        public static int[] ReadInt32Array(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            int nb = ctx.Reader.ReadInt32();
            var array = new int[nb];
            for (int i = 0; i < array.Length; ++i)
                array[i] = ctx.Reader.ReadInt32();
            return array;
        }

        public static void Write(ModelSaveContext ctx, VBuffer<float> buf)
        {
            ctx.Writer.Write(buf.Length);
            ctx.Writer.Write(buf.Count);
            IOHelper.Write(ctx, buf.Values);
            IOHelper.Write(ctx, buf.Indices);
        }

        public static void Write(ModelSaveContext ctx, VBuffer<double> buf)
        {
            ctx.Writer.Write(buf.Length);
            ctx.Writer.Write(buf.Count);
            IOHelper.Write(ctx, buf.Values);
            IOHelper.Write(ctx, buf.Indices);
        }

        public static VBuffer<float> ReadVBufferFloat(ModelLoadContext ctx)
        {
            int len = ctx.Reader.ReadInt32();
            int cou = ctx.Reader.ReadInt32();
            var fl = ReadFloatArray(ctx);
            var nd = ReadInt32Array(ctx);
            return new VBuffer<float>(len, cou, fl, nd);
        }

        public static VBuffer<double> ReadVBufferDouble(ModelLoadContext ctx)
        {
            int len = ctx.Reader.ReadInt32();
            int cou = ctx.Reader.ReadInt32();
            var fl = IOHelper.ReadDoubleArray(ctx);
            var nd = IOHelper.ReadInt32Array(ctx);
            return new VBuffer<double>(len, cou, fl, nd);
        }

        public static void Write(ModelSaveContext ctx, Dictionary<string, long> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public static Dictionary<string, long> ReadDictStringLong(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<string, long>();
            int nb = ctx.Reader.ReadInt32();
            string key;
            long value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadString();
                value = ctx.Reader.ReadInt64();
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, Dictionary<float, long> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public static void Write(ModelSaveContext ctx, Dictionary<double, long> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public static Dictionary<float, long> ReadDictFloatLong(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<float, long>();
            int nb = ctx.Reader.ReadInt32();
            float key;
            long value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadSingle();
                value = ctx.Reader.ReadInt64();
                res[key] = value;
            }
            return res;
        }

        public static Dictionary<double, long> ReadDictDoubleLong(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<double, long>();
            int nb = ctx.Reader.ReadInt32();
            double key;
            long value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadDouble();
                value = ctx.Reader.ReadInt64();
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, Dictionary<float, float> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public static Dictionary<float, float> ReadDictFloatFloat(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<float, float>();
            int nb = ctx.Reader.ReadInt32();
            float key;
            float value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadSingle();
                value = ctx.Reader.ReadSingle();
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, Dictionary<Tuple<float, int>, float> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    IOHelper.Write(ctx, pair.Key);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public static Dictionary<Tuple<float, int>, float> ReadDictFloatIntFloat(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<Tuple<float, int>, float>();
            int nb = ctx.Reader.ReadInt32();
            Tuple<float, int> key;
            float value;
            for (int i = 0; i < nb; ++i)
            {
                key = IOHelper.ReadTupleFloatInt(ctx);
                value = ctx.Reader.ReadSingle();
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, Dictionary<int, float> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public static Dictionary<int, float> ReadDictIntFloat(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<int, float>();
            int nb = ctx.Reader.ReadInt32();
            int key;
            float value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadInt32();
                value = ctx.Reader.ReadSingle();
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, Dictionary<long, Dictionary<float, float>> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    Write(ctx, pair.Value);
                }
            }
        }

        public static Dictionary<long, Dictionary<float, float>> ReadDictLongFloatFloat(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<long, Dictionary<float, float>>();
            int nb = ctx.Reader.ReadInt32();
            long key;
            Dictionary<float, float> value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadInt64();
                value = ReadDictFloatFloat(ctx);
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, Dictionary<int, Dictionary<float, float>> di)
        {
            ctx.Writer.Write((byte)(di == null ? 1 : 0));
            if (di != null)
            {
                ctx.Writer.Write(di.Count);
                foreach (var pair in di)
                {
                    ctx.Writer.Write(pair.Key);
                    Write(ctx, pair.Value);
                }
            }
        }

        public static Dictionary<int, Dictionary<float, float>> ReadDictIntFloatFloat(ModelLoadContext ctx)
        {
            bool isNull = ctx.Reader.ReadByte() == 1;
            if (isNull)
                return null;
            var res = new Dictionary<int, Dictionary<float, float>>();
            int nb = ctx.Reader.ReadInt32();
            int key;
            Dictionary<float, float> value;
            for (int i = 0; i < nb; ++i)
            {
                key = ctx.Reader.ReadInt32();
                value = ReadDictFloatFloat(ctx);
                res[key] = value;
            }
            return res;
        }

        public static void Write(ModelSaveContext ctx, RoleMappedSchema schema)
        {
            var array = schema.GetColumnRoleNames().ToArray();
            ctx.Writer.Write(array.Length);
            foreach( var pair in array)
            {
                ctx.Writer.Write(pair.Key.Value);
                ctx.Writer.Write(pair.Value);
            }
        }

        public static RoleMappedSchema ReadRoleMappedSchema(ModelLoadContext ctx, Schema schema)
        {
            int nb = ctx.Reader.ReadInt32();
            var array = new KeyValuePair<RoleMappedSchema.ColumnRole, string>[nb];
            for(int i = 0; i < nb;++i)
            {
                var key = ctx.Reader.ReadString();
                var value = ctx.Reader.ReadString();
                array[i] = new KeyValuePair<RoleMappedSchema.ColumnRole, string>(new RoleMappedSchema.ColumnRole(key), value);
            }
            return new RoleMappedSchema(schema, array);
        }
    }
}
