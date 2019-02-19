// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using Microsoft.Data.DataView;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using MultiClassConvertTransform = Scikit.ML.MultiClass.MultiClassConvertTransform;

[assembly: LoadableClass(MultiClassConvertTransform.Summary, typeof(MultiClassConvertTransform), typeof(MultiClassConvertTransform.Arguments), typeof(SignatureDataTransform),
    "Extended Convert Transform", MultiClassConvertTransform.LoaderSignature, "ExtConv", "mcConv")]

[assembly: LoadableClass(MultiClassConvertTransform.Summary, typeof(MultiClassConvertTransform), null, typeof(SignatureLoadDataTransform),
    "Extended Convert Transform", MultiClassConvertTransform.LoaderSignature, "ExtConv", "mcConv")]


namespace Scikit.ML.MultiClass
{
    /// <summary>
    /// Mostly added to convert a float into a key.
    /// </summary>
    public sealed class MultiClassConvertTransform : OneToOneTransformBase
    {
        public class Column : SchemaHelper.OneToOneColumnForArgument
        {
            public new static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }
        }

        public class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:type:src)", ShortName = "col", SortOrder = 1)]
            public Column[] column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type", SortOrder = 2)]
            public DataKind? resultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyCount keyCount;
        }

        private sealed class ColInfoEx
        {
            public readonly DataKind Kind;
            public readonly bool HasKeyRange;
            public readonly DataViewType TypeDst;
            public readonly VectorType SlotTypeDst;

            public ColInfoEx(DataKind kind, bool hasKeyRange, DataViewType type, VectorType slotType)
            {
                Contracts.AssertValue(type);
                Contracts.AssertValueOrNull(slotType);
                Contracts.Assert(slotType == null || type.ItemType().Equals(slotType.ItemType()));

                Kind = kind;
                HasKeyRange = hasKeyRange;
                TypeDst = type;
                SlotTypeDst = slotType;
            }
        }

        internal const string Summary = "Converts a column to a different type, using standard conversions and specialized for MultiClass.";

        public const string LoaderSignature = "MultiClassConvertTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MCCNVERT",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiClassConvertTransform).Assembly.FullName);
        }

        private const string RegistrationName = "McConvert";
        private readonly ColInfoEx[] _exes;

        public MultiClassConvertTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, "args").column, input, null)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                DataKind kind;
                KeyCount range;
                var col = args.column[i];
                if (col.ResultType != null)
                {
                    kind = col.ResultType.Value;
                    range = col.KeyCount;
                }
                else if (col.KeyCount != null)
                {
                    kind = Infos[i].TypeSrc.IsKey() ? Infos[i].TypeSrc.RawKind() : DataKind.U4;
                    range = col.KeyCount;
                }
                else if (args.resultType != null)
                {
                    kind = args.resultType.Value;
                    range = args.keyCount;
                }
                else if (args.keyCount != null)
                {
                    kind = Infos[i].TypeSrc.IsKey() ? Infos[i].TypeSrc.RawKind() : DataKind.U4;
                    range = args.keyCount;
                }
                else
                {
                    kind = DataKind.Num;
                    range = null;
                }
                Host.CheckUserArg(Enum.IsDefined(typeof(DataKind), kind), "resultType");

                PrimitiveDataViewType itemType;
                if (!TryCreateEx(Host, Infos[i], kind, range, out itemType, out _exes[i]))
                    throw Host.ExceptUserArg("source",
                        "Source column '{0}' with item type '{1}' is not compatible with destination type '{2}'",
                        input.Schema[Infos[i].Source].Name, Infos[i].TypeSrc.ItemType(), itemType);
            }
            SetMetadata();
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var info = Infos[iinfo];
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, info.Source, PassThrough))
                {
                    if (info.TypeSrc.IsBool() && _exes[iinfo].TypeDst.ItemType().IsNumber())
                        bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, true);
                }
            }
            md.Seal();
        }

        /// <summary>
        /// Returns whether metadata of the indicated kind should be passed through from the source column.
        /// </summary>
        private bool PassThrough(string kind, int iinfo)
        {
            var typeSrc = Infos[iinfo].TypeSrc;
            var typeDst = _exes[iinfo].TypeDst;
            switch (kind)
            {
                case MetadataUtils.Kinds.SlotNames:
                    Host.Assert(typeSrc.VectorSize() == typeDst.VectorSize());
                    return typeDst.IsKnownSizeVector();
                case MetadataUtils.Kinds.KeyValues:
                    return typeSrc.ItemType().IsKey() && typeDst.ItemType().IsKey() && typeSrc.ItemType().GetKeyCount() > 0 &&
                        typeSrc.ItemType().GetKeyCount() == typeDst.ItemType().GetKeyCount();
                case MetadataUtils.Kinds.IsNormalized:
                    return typeSrc.ItemType().IsNumber() && typeDst.ItemType().IsNumber();
            }
            return false;
        }

        private MultiClassConvertTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Host.AssertValue(ctx);
            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                byte b = ctx.Reader.ReadByte();
                var kind = (DataKind)(b & 0x7F);
                Host.CheckDecode(Enum.IsDefined(typeof(DataKind), kind));
                KeyCount range = null;
                if ((b & 0x80) != 0)
                {
                    range = new KeyCount();
                    ulong count = ctx.Reader.ReadUInt64();
                    if (count != 0)
                    {
                        if (count < 0 || (ulong)(count - 1) > ulong.MaxValue)
                            throw Host.ExceptDecode("KeyType count too large");
                        range.Count = count;
                    }
                }

                PrimitiveDataViewType itemType;
                if (!TryCreateEx(Host, Infos[i], kind, range, out itemType, out _exes[i]))
                    throw Host.ExceptDecode("source is not of compatible type");
            }
            SetMetadata();
        }

        public static MultiClassConvertTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, "input");
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(float));
                    return new MultiClassConvertTransform(h, ctx, input);
                });
        }

        protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            ctx.Writer.Write(sizeof(float));
            SaveBase(ctx);

            for (int i = 0; i < _exes.Length; i++)
            {
                var ex = _exes[i];
                Host.Assert((DataKind)(byte)ex.Kind == ex.Kind);
                if (!ex.HasKeyRange)
                    ctx.Writer.Write((byte)ex.Kind);
                else
                {
                    Host.Assert(ex.TypeDst.ItemType().IsKey());
                    var key = ex.TypeDst.ItemType().AsKey();
                    byte b = (byte)ex.Kind;
                    b |= 0x80;
                    ctx.Writer.Write(b);
                    ctx.Writer.Write(key.Count);
                }
            }
        }

        private static void ConvIdU4(ref uint src, ref uint dst)
        {
            dst = src;
        }

        private static void CheckRange(long src, uint dst, IExceptionContext ectx = null)
        {
            if (src < 0 || src >= uint.MaxValue)
            {
                if (ectx != null)
                    ectx.Except("Value long {0} cannot be converted into uint.", src);
                else
                    Contracts.Except("Value long {0} cannot be converted into uint.", src);
            }
        }

        private static void CheckRange(long src, ulong dst, IExceptionContext ectx = null)
        {
            if (src < 0)
            {
                if (ectx != null)
                    ectx.Except("Value long {0} cannot be converted into ulong.", src);
                else
                    Contracts.Except("Value long {0} cannot be converted into ulong.", src);
            }
        }

        private static bool TryCreateEx(IExceptionContext ectx, ColInfo info, DataKind kind, KeyCount range,
                                        out PrimitiveDataViewType itemType, out ColInfoEx ex)
        {
            ectx.AssertValue(info);
            ectx.Assert(Enum.IsDefined(typeof(DataKind), kind));

            ex = null;

            var typeSrc = info.TypeSrc;
            if (range != null)
            {
                itemType = TypeParsingUtils.ConstructKeyType(kind, range);
                if (!typeSrc.ItemType().IsKey() && !typeSrc.ItemType().IsText() && typeSrc.ItemType().RawKind() != kind &&
                    !(typeSrc.ItemType().RawKind() == DataKind.I8 && (kind == DataKind.U8 || kind == DataKind.U4)))
                    return false;
            }
            else if (!typeSrc.ItemType().IsKey())
                itemType = ColumnTypeHelper.PrimitiveFromKind(kind);
            else if (!ColumnTypeHelper.IsValidDataKind(kind))
            {
                itemType = ColumnTypeHelper.PrimitiveFromKind(kind);
                return false;
            }
            else
            {
                var key = typeSrc.ItemType().AsKey();
                ectx.Assert(ColumnTypeHelper.IsValidDataKind(key.RawKind()));
                ulong count = key.Count;
                // Technically, it's an error for the counts not to match, but we'll let the Conversions
                // code return false below. There's a possibility we'll change the standard conversions to
                // map out of bounds values to zero, in which case, this is the right thing to do.
                ulong max = kind.ToMaxInt();
                if ((ulong)count > max)
                    count = max;
                itemType = new KeyType(kind.ToType(), count);
            }

            // Ensure that the conversion is legal. We don't actually cache the delegate here. It will get
            // re-fetched by the utils code when needed.
            bool identity;
            Delegate del;
            if (!Conversions.Instance.TryGetStandardConversion(typeSrc.ItemType(), itemType, out del, out identity))
            {
                if (typeSrc.ItemType().RawKind() == itemType.RawKind())
                {
                    switch (typeSrc.ItemType().RawKind())
                    {
                        case DataKind.U4:
                            // Key starts at 1.
                            // MultiClass future issue
                            uint plus = (itemType.IsKey() ? (uint)1 : (uint)0) - (typeSrc.IsKey() ? (uint)1 : (uint)0);
                            identity = false;
                            ValueMapper<uint, uint> map_ = (in uint src, ref uint dst) => { dst = src + plus; };
                            del = (Delegate)map_;
                            if (del == null)
                                throw Contracts.ExceptNotSupp("Issue with casting");
                            break;
                        default:
                            throw Contracts.Except("Not suppoted type {0}", typeSrc.ItemType().RawKind());
                    }
                }
                else if (typeSrc.ItemType().RawKind() == DataKind.I8 && kind == DataKind.U8)
                {
                    ulong plus = (itemType.IsKey() ? (ulong)1 : (ulong)0) - (typeSrc.IsKey() ? (ulong)1 : (ulong)0);
                    identity = false;
                    ValueMapper<long, ulong> map_ = (in long src, ref ulong dst) =>
                    {
                        CheckRange(src, dst, ectx); dst = (ulong)src + plus;
                    };
                    del = (Delegate)map_;
                    if (del == null)
                        throw Contracts.ExceptNotSupp("Issue with casting");
                }
                else if (typeSrc.ItemType().RawKind() == DataKind.I8 && kind == DataKind.U4)
                {
                    // MultiClass future issue
                    uint plus = (itemType.IsKey() ? (uint)1 : (uint)0) - (typeSrc.IsKey() ? (uint)1 : (uint)0);
                    identity = false;
                    ValueMapper<long, uint> map_ = (in long src, ref uint dst) =>
                    {
                        CheckRange(src, dst, ectx); dst = (uint)src + plus;
                    };
                    del = (Delegate)map_;
                    if (del == null)
                        throw Contracts.ExceptNotSupp("Issue with casting");
                }
                else
                    return false;
            }

            DataViewType typeDst = itemType;
            if (typeSrc.IsVector())
                typeDst = new VectorType(itemType, typeSrc.AsVector().Dimensions.ToArray());

            // An output column is transposable iff the input column was transposable.
            VectorType slotType = null;
            if (info.SlotTypeSrc != null)
                slotType = new VectorType(itemType, info.SlotTypeSrc.Dimensions.ToArray());

            ex = new ColInfoEx(kind, range != null, typeDst, slotType);
            return true;
        }

        protected override DataViewType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            Host.Assert(_exes.Length == Infos.Length);
            return _exes[iinfo].TypeDst;
        }

        public static Delegate GetGetterAs(DataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.CheckValue(typeDst, "typeDst");
            Contracts.CheckParam(typeDst.IsPrimitive(), "typeDst");
            Contracts.CheckValue(row, "row");
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, "col");
            Contracts.CheckParam(row.IsColumnActive(col), "col", "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc.IsPrimitive(), "Source column type must be primitive");

            Func<DataViewType, DataViewType, DataViewRow, int, ValueGetter<int>> del = GetGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        private static ValueGetter<TDst> GetGetterAsCore<TSrc, TDst>(DataViewType typeSrc, DataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.RawType);
            Contracts.Assert(typeof(TDst) == typeDst.RawType);

            bool identity;

            if (typeSrc.RawKind() == DataKind.U4 && typeDst.RawKind() == DataKind.U4)
            {
                var getter = row.GetGetter<uint>(col);
                // MultiClass future issue
                uint plus = (typeDst.IsKey() ? (uint)1 : (uint)0) - (typeSrc.IsKey() ? (uint)1 : (uint)0);
                identity = true;
                var src = default(uint);
                ValueGetter<uint> mapu =
                    (ref uint dst) =>
                    {
                        getter(ref src);
                        dst = src + plus;
                    };
                return mapu as ValueGetter<TDst>;
            }
            else if (typeSrc.RawKind() == DataKind.I8 && typeDst.RawKind() == DataKind.U8)
            {
                ulong plus = (typeDst.IsKey() ? (ulong)1 : (ulong)0) - (typeSrc.IsKey() ? (ulong)1 : (ulong)0);
                var getter = row.GetGetter<long>(col);
                identity = true;
                var src = default(long);
                ValueGetter<ulong> mapu =
                    (ref ulong dst) =>
                    {
                        getter(ref src);
                        CheckRange(src, dst);
                        dst = (ulong)src + plus;
                    };
                return mapu as ValueGetter<TDst>;
            }
            else if (typeSrc.RawKind() == DataKind.I8 && typeDst.RawKind() == DataKind.U4)
            {
                // MultiClass future issue
                uint plus = (typeDst.IsKey() ? (uint)1 : (uint)0) - (typeSrc.IsKey() ? (uint)1 : (uint)0);
                var getter = row.GetGetter<long>(col);
                identity = true;
                var src = default(long);
                ValueGetter<uint> mapu =
                    (ref uint dst) =>
                    {
                        getter(ref src);
                        CheckRange(src, dst);
                        dst = (uint)src + plus;
                    };
                return mapu as ValueGetter<TDst>;
            }
            else
            {
                var getter = row.GetGetter<TSrc>(col);
                var conv = Conversions.Instance.GetStandardConversion<TSrc, TDst>(typeSrc, typeDst, out identity);
                if (identity)
                {
                    Contracts.Assert(typeof(TSrc) == typeof(TDst));
                    return (ValueGetter<TDst>)(Delegate)getter;
                }

                var src = default(TSrc);
                return
                    (ref TDst dst) =>
                    {
                        getter(ref src);
                        conv(in src, ref dst);
                    };
            }
        }

        protected override Delegate GetGetterCore(IChannel ch, DataViewRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var typeSrc = Infos[iinfo].TypeSrc;
            var typeDst = _exes[iinfo].TypeDst;

            if (!typeDst.IsVector())
                return GetGetterAs(typeDst, input, Infos[iinfo].Source);
            return RowCursorUtils.GetVecGetterAs(typeDst.AsVector().ItemType(), input, Infos[iinfo].Source);
        }

        protected override VectorType GetSlotTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return _exes[iinfo].SlotTypeDst;
        }
        /*
        protected virtual SlotCursor GetSlotCursorCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.AssertValue(Infos[iinfo].SlotTypeSrc);
            Host.AssertValue(_exes[iinfo].SlotTypeDst);

            SlotCursor cursor = InputTranspose.GetSlotCursor(Infos[iinfo].Source);
            return new SlotCursorImpl(Host, cursor, _exes[iinfo].SlotTypeDst);
        }

        private sealed class SlotCursorImpl : SlotCursor.SynchronizedSlotCursor
        {
            private readonly Delegate _getter;
            private readonly VectorType _type;

            public SlotCursorImpl(IChannelProvider provider, SlotCursor cursor, VectorType typeDst, Delegate getter)
                : base(provider, cursor)
            {
                Contracts.AssertValue(typeDst);
                _getter = getter;
                _type = typeDst;
            }

            public override VectorType GetSlotType()
            {
                return _type;
            }

            public override ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Contracts.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }
        */
    }
}
