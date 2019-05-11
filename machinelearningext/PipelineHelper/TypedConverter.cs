// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// The class is not protected against multithreading.
    /// </summary>
    internal class TypedConverters<TLabel>
    {
        bool identity;

        ValueMapper<TLabel, bool> mapperBL;
        ValueMapper<TLabel, byte> mapperU1;
        ValueMapper<TLabel, ushort> mapperU2;
        ValueMapper<TLabel, uint> mapperU4;
        ValueMapper<TLabel, int> mapperI4;
        ValueMapper<TLabel, float> mapperR4;

        ValueMapper<bool, TLabel> mapperFromBL;
        ValueMapper<byte, TLabel> mapperFromU1;
        ValueMapper<ushort, TLabel> mapperFromU2;
        ValueMapper<uint, TLabel> mapperFromU4;
        ValueMapper<int, TLabel> mapperFromI4;
        ValueMapper<float, TLabel> mapperFromR4;

        bool _bl;
        byte _u1;
        ushort _u2;
        uint _u4;
        float _r4;

        DataKind _kind;
        DataKind _destKind;

        public TypedConverters()
        {
            Init(SchemaHelper.GetColumnType<TLabel>().RawKind());
        }

        public TypedConverters(DataKind destKind)
        {
            Init(destKind);
        }

        void Init(DataKind destKind)
        {
            _kind = SchemaHelper.GetColumnType<TLabel>().RawKind();
            _destKind = destKind;

            mapperBL = null;
            mapperU1 = null;
            mapperU2 = null;
            mapperU4 = null;
            mapperI4 = null;
            mapperR4 = null;

            mapperFromBL = null;
            mapperFromU1 = null;
            mapperFromU2 = null;
            mapperFromU4 = null;
            mapperFromI4 = null;
            mapperFromR4 = null;

            switch (destKind)
            {
                case DataKind.Boolean:
                    mapperBL = SchemaHelper.GetConverter<TLabel, bool>(out identity);
                    mapperFromBL = SchemaHelper.GetConverter<bool, TLabel>(out identity);
                    break;
                case DataKind.SByte:
                    mapperU1 = SchemaHelper.GetConverter<TLabel, byte>(out identity);
                    mapperFromU1 = SchemaHelper.GetConverter<byte, TLabel>(out identity);
                    break;
                case DataKind.UInt16:
                    mapperU2 = SchemaHelper.GetConverter<TLabel, ushort>(out identity);
                    mapperFromU2 = SchemaHelper.GetConverter<ushort, TLabel>(out identity);
                    break;
                case DataKind.UInt32:
                    mapperU4 = SchemaHelper.GetConverter<TLabel, uint>(out identity);
                    mapperFromU4 = SchemaHelper.GetConverter<uint, TLabel>(out identity);
                    break;
                case DataKind.Int32:
                    var temp = SchemaHelper.GetConverter<TLabel, float>(out identity);
                    mapperI4 = (in TLabel src, ref int dst) =>
                    {
                        float v = 0f;
                        temp(in src, ref v);
                        dst = (int)v;
                    };
                    var temp2 = SchemaHelper.GetConverter<float, TLabel>(out identity);
                    mapperFromI4 = (in int src, ref TLabel dst) =>
                    {
                        float v = (float)src;
                        temp2(in v, ref dst);
                    };
                    break;
                case DataKind.Single:
                    mapperR4 = SchemaHelper.GetConverter<TLabel, float>(out identity);
                    mapperFromR4 = SchemaHelper.GetConverter<float, TLabel>(out identity);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Unsupported kinds {0} --> {1}", _kind, _destKind);
            }
        }

        public ValueMapper<TLabel, TDest> GetMapper<TDest>()
        {
            var colType = SchemaHelper.GetColumnType<TDest>();
            if (colType.IsVector())
                throw Contracts.ExceptNotImpl($"Not implemented for type {typeof(TDest)}.");
            switch (colType.RawKind())
            {
                case DataKind.Boolean:
                    return mapperBL as ValueMapper<TLabel, TDest>;
                case DataKind.SByte:
                    return mapperU1 as ValueMapper<TLabel, TDest>;
                case DataKind.UInt16:
                    return mapperU2 as ValueMapper<TLabel, TDest>;
                case DataKind.UInt32:
                    return mapperU4 as ValueMapper<TLabel, TDest>;
                case DataKind.Int32:
                    return mapperI4 as ValueMapper<TLabel, TDest>;
                case DataKind.Single:
                    return mapperR4 as ValueMapper<TLabel, TDest>;
                default:
                    throw Contracts.ExceptNotSupp("Unsupported kind {0}", colType.RawKind());
            }
        }

        public ValueMapper<TDest, TLabel> GetMapperFrom<TDest>()
        {
            var colType = SchemaHelper.GetColumnType<TDest>();
            if (colType.IsVector())
                throw Contracts.ExceptNotImpl($"Not implemented for type {typeof(TDest)}.");
            switch (colType.RawKind())
            {
                case DataKind.Boolean:
                    return mapperBL as ValueMapper<TDest, TLabel>;
                case DataKind.SByte:
                    return mapperU1 as ValueMapper<TDest, TLabel>;
                case DataKind.UInt16:
                    return mapperU2 as ValueMapper<TDest, TLabel>;
                case DataKind.UInt32:
                    return mapperU4 as ValueMapper<TDest, TLabel>;
                case DataKind.Single:
                    return mapperR4 as ValueMapper<TDest, TLabel>;
                default:
                    throw Contracts.ExceptNotSupp("Unsupported kind {0}", colType.RawKind());
            }
        }

        public void Save(ModelSaveContext ctx, TLabel value)
        {
            switch (_kind)
            {
                case DataKind.Boolean:
                    mapperBL(in value, ref _bl);
                    ctx.Writer.Write(_bl);
                    break;
                case DataKind.SByte:
                    mapperU1(in value, ref _u1);
                    ctx.Writer.Write(_u1);
                    break;
                case DataKind.UInt16:
                    mapperU2(in value, ref _u2);
                    ctx.Writer.Write(_u2);
                    break;
                case DataKind.UInt32:
                    mapperU4(in value, ref _u4);
                    ctx.Writer.Write(_u4);
                    break;
                case DataKind.Single:
                    mapperR4(in value, ref _r4);
                    ctx.Writer.Write(_r4);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Not supported kind {0}", _kind);
            }
        }

        public void Read(ModelLoadContext ctx, ref TLabel res)
        {
            switch (_kind)
            {
                case DataKind.Boolean:
                    var b = ctx.Reader.ReadBoolByte();
                    mapperFromBL(in _bl, ref res);
                    break;
                case DataKind.SByte:
                    _u1 = ctx.Reader.ReadByte();
                    mapperFromU1(in _u1, ref res);
                    break;
                case DataKind.UInt16:
                    _u2 = ctx.Reader.ReadUInt16();
                    mapperFromU2(in _u2, ref res);
                    break;
                case DataKind.UInt32:
                    _u4 = ctx.Reader.ReadUInt32();
                    mapperFromU4(in _u4, ref res);
                    break;
                case DataKind.Single:
                    _r4 = ctx.Reader.ReadFloat();
                    mapperFromR4(in _r4, ref res);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Not supported kind {0}", _kind);
            }
        }
    }
}

