// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Data.SignatureLoadDataTransform;
using ResampleTransform = Scikit.ML.RandomTransforms.ResampleTransform;

[assembly: LoadableClass(ResampleTransform.Summary, typeof(ResampleTransform),
    typeof(ResampleTransform.Arguments), typeof(SignatureDataTransform),
    "Resample Transform", ResampleTransform.LoaderSignature, "Resample")]

[assembly: LoadableClass(ResampleTransform.Summary, typeof(ResampleTransform),
    null, typeof(SignatureLoadDataTransform),
    "Resample Transform", ResampleTransform.LoaderSignature, "Resample")]

namespace Scikit.ML.RandomTransforms
{
    /// <summary>
    /// Randomly multiplies rows.
    /// </summary>
    public class ResampleTransform : IDataTransformSingle
    {
        #region identification

        public const string LoaderSignature = "ResampleTransform";  // Not more than 24 letters.
        public const string Summary = "Randomly multiplies rows, the number of multiplication per rows is draws from a Poisson Law.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RESAMPLE",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ResampleTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter lambda of the Poison Law.", ShortName = "l")]
            public float lambda = 1.0f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed", ShortName = "s")]
            public int? seed = null;

            [Argument(ArgumentType.AtMostOnce, ShortName = "c", HelpText = "Cache the random replication. This cache holds in memory. " +
                "You can disable the cache but be aware that a second consecutive run through the view will not have " +
                " the same results.")]
            public bool cache = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Class column, to resample only for a specific class, this column contains the class information (null to resample everything).", ShortName = "col")]
            public string column = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Class to resample (null for all).", ShortName = "cl")]
            public string classValue = null;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(lambda);
                ctx.Writer.Write(seed.HasValue ? seed.Value : -1);
                ctx.Writer.Write(cache ? 1 : 0);
                ctx.Writer.Write(string.IsNullOrEmpty(column) ? string.Empty : column);
                ctx.Writer.Write(string.IsNullOrEmpty(classValue) ? string.Empty : classValue);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                lambda = ctx.Reader.ReadSingle();
                seed = ctx.Reader.ReadInt32();
                if (seed == -1)
                    seed = null;
                cache = ctx.Reader.ReadInt32() == 1;
                column = ctx.Reader.ReadString();
                classValue = ctx.Reader.ReadString();
            }
        }

        #endregion

        #region internal members / accessors

        IDataView _input;
        IDataTransform _transform;          // templated transform (not the serialized version)
        Arguments _args;
        IHost _host;
        Dictionary<DataViewRowId, int> _cacheReplica;

        public IDataView Source { get { return _input; } }

        #endregion

        #region public constructor / serialization / load / save

        public ResampleTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register("ResampleTransform");
            _host.CheckValue(args, "args");                 // Checks values are valid.
            _host.CheckValue(input, "input");
            _host.Check(args.lambda > 0, "lambda must be > 0");
            _input = input;
            _args = args;
            _cacheReplica = null;

            if (!string.IsNullOrEmpty(_args.column))
            {
                int index = SchemaHelper.GetColumnIndex(_input.Schema, _args.column);
                if (string.IsNullOrEmpty(_args.classValue))
                    throw _host.Except("Class value cannot be null.");
            }

            _transform = CreateTemplatedTransform();
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        private ResampleTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            _cacheReplica = null;
            _transform = CreateTemplatedTransform();
        }

        public static ResampleTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ResampleTransform(h, ctx, input));
        }

        #endregion

        #region IDataTransform API

        public DataViewSchema Schema { get { return Source.Schema; } }

        public bool CanShuffle { get { return true; } }
        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            if (_cacheReplica != null)
                return _cacheReplica.Values.Sum();
            else
                return null;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return GetRowCursor(columnsNeeded, rand, (c, r) => _input.GetRowCursor(c, r));
        }

        public DataViewRowCursor GetRowCursorSingle(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return GetRowCursor(columnsNeeded, rand, (c, r) => CursorHelper.GetRowCursorSingle(_input, c, r));
        }

        private DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand,
                                               DelegateGetRowCursor getterCursor)
        {
            int classColumn = -1;
            if (!string.IsNullOrEmpty(_args.column))
                classColumn = SchemaHelper.GetColumnIndex(_input.Schema, _args.column);
            if (_args.cache)
                LoadCache(rand);
            else
                Contracts.Assert(_cacheReplica == null);
            var cursor = getterCursor(columnsNeeded, rand);

            if (classColumn == -1)
                return new ResampleCursor<float>(this, cursor, columnsNeeded, _args.lambda,
                    _args.seed, rand, _cacheReplica, SchemaHelper._dc(classColumn, cursor),
                    float.NaN);

            var newColumns = columnsNeeded.ToList();
            newColumns.Add(Schema.Where(c => c.Index == classColumn).First());

            var type = _input.Schema[classColumn].Type;
            switch (type.RawKind())
            {
                case DataKind.Boolean:
                    bool clbool;
                    if (!bool.TryParse(_args.classValue, out clbool))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return new ResampleCursor<bool>(this, cursor, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, cursor), clbool);
                case DataKind.UInt32:
                    uint cluint;
                    if (!uint.TryParse(_args.classValue, out cluint))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return new ResampleCursor<uint>(this, cursor, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, cursor), cluint);
                case DataKind.Single:
                    float clfloat;
                    if (!float.TryParse(_args.classValue, out clfloat))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return new ResampleCursor<float>(this, cursor, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, cursor), clfloat);
                case DataKind.String:
                    var cltext = new ReadOnlyMemory<char>(_args.classValue.ToCharArray());
                    return new ResampleCursor<ReadOnlyMemory<char>>(this, cursor, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, cursor), cltext);
                default:
                    throw _host.Except("Unsupported type '{0}'", type);
            }
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            int classColumn = -1;
            if (!string.IsNullOrEmpty(_args.column))
                classColumn = SchemaHelper.GetColumnIndex(_input.Schema, _args.column);
            if (_args.cache)
                LoadCache(rand);
            else
                Contracts.Assert(_cacheReplica == null);

            var cursors = _input.GetRowCursorSet(columnsNeeded, n, rand);

            if (classColumn == -1)
                return cursors.Select(c => new ResampleCursor<float>(this,
                    c, columnsNeeded, _args.lambda, _args.seed, rand, _cacheReplica,
                    SchemaHelper._dc(classColumn, c), float.NaN)).ToArray();

            var newColumns = columnsNeeded.ToList();
            newColumns.Add(Schema.Where(c => c.Index == classColumn).First());

            var type = _input.Schema[classColumn].Type;
            switch (type.RawKind())
            {
                case DataKind.Boolean:
                    bool clbool;
                    if (!bool.TryParse(_args.classValue, out clbool))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return cursors.Select(c => new ResampleCursor<bool>(this, c, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, c), clbool)).ToArray();
                case DataKind.UInt32:
                    uint cluint;
                    if (!uint.TryParse(_args.classValue, out cluint))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return cursors.Select(c => new ResampleCursor<uint>(this, c, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, c), cluint)).ToArray();
                case DataKind.Single:
                    float clfloat;
                    if (!float.TryParse(_args.classValue, out clfloat))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return cursors.Select(c => new ResampleCursor<float>(this, c, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, c), clfloat)).ToArray();
                case DataKind.String:
                    var cltext = new ReadOnlyMemory<char>(_args.classValue.ToCharArray());
                    return cursors.Select(c => new ResampleCursor<ReadOnlyMemory<char>>(this, c, newColumns,
                        _args.lambda, _args.seed, rand, _cacheReplica,
                        SchemaHelper._dc(classColumn, c), cltext)).ToArray();
                default:
                    throw _host.Except("Unsupported type '{0}'", type);
            }
        }

        public static int NextPoisson(float lambda, Random rand)
        {
            var L = Math.Exp(-lambda);
            int k = 0;
            float p = 1;
            float u;
            do
            {
                ++k;
                u = rand.NextSingle();
                p *= u;
            }
            while (p > L);
            return k - 1;
        }

        void LoadCache(Random rand)
        {
            if (_cacheReplica != null)
                // Already done.
                return;

            uint? useed = _args.seed.HasValue ? (uint)_args.seed.Value : (uint?)null;
            if (rand == null)
                rand = RandomUtils.Create(useed);

            using (var ch = _host.Start("Resample: fill the cache"))
            {
                var indexClass = SchemaHelper.GetColumnIndexDC(_input.Schema, _args.column, true);

                using (var cur = _input.GetRowCursor(Schema.Where(c => c.Index == indexClass.Index)))
                {
                    if (string.IsNullOrEmpty(_args.column))
                    {
                        _cacheReplica = new Dictionary<DataViewRowId, int>();
                        var gid = cur.GetIdGetter();
                        DataViewRowId did = default(DataViewRowId);
                        int rep;
                        while (cur.MoveNext())
                        {
                            gid(ref did);
                            rep = NextPoisson(_args.lambda, rand);
                            _cacheReplica[did] = rep;
                        }
                    }
                    else
                    {
                        var type = _input.Schema[indexClass.Index].Type;
                        switch (type.RawKind())
                        {
                            case DataKind.Boolean:
                                bool clbool;
                                if (!bool.TryParse(_args.classValue, out clbool))
                                    throw ch.Except("Unable to parse '{0}'.", _args.classValue);
                                LoadCache<bool>(rand, cur, indexClass, clbool, ch);
                                break;
                            case DataKind.UInt32:
                                uint cluint;
                                if (!uint.TryParse(_args.classValue, out cluint))
                                    throw ch.Except("Unable to parse '{0}'.", _args.classValue);
                                LoadCache<uint>(rand, cur, indexClass, cluint, ch);
                                break;
                            case DataKind.Single:
                                float clfloat;
                                if (!float.TryParse(_args.classValue, out clfloat))
                                    throw ch.Except("Unable to parse '{0}'.", _args.classValue);
                                LoadCache<float>(rand, cur, indexClass, clfloat, ch);
                                break;
                            case DataKind.String:
                                var cltext = new ReadOnlyMemory<char>(_args.classValue.ToCharArray());
                                LoadCache<ReadOnlyMemory<char>>(rand, cur, indexClass, cltext, ch);
                                break;
                            default:
                                throw _host.Except("Unsupported type '{0}'", type);
                        }
                    }
                }
            }
        }

        void LoadCache<TClass>(Random rand, DataViewRowCursor cur, DataViewSchema.Column classColumn, TClass valueClass, IChannel ch)
        {
            _cacheReplica = new Dictionary<DataViewRowId, int>();
            var hist = new Dictionary<TClass, long>();
            var gid = cur.GetIdGetter();
            var gcl = cur.GetGetter<TClass>(classColumn);
            DataViewRowId did = default(DataViewRowId);
            TClass cl = default(TClass);
            long nbIn = 0;
            long nbOut = 0;
            int rep;
            while (cur.MoveNext())
            {
                gcl(ref cl);
                gid(ref did);
                if (!hist.ContainsKey(cl))
                    hist[cl] = 1;
                else
                    ++hist[cl];
                if (cl.Equals(valueClass))
                {
                    rep = NextPoisson(_args.lambda, rand);
                    ++nbIn;
                }
                else
                {
                    rep = 1;
                    ++nbOut;
                }
                _cacheReplica[did] = rep;
            }
            if (nbIn == 0)
                ch.Warning(MessageSensitivity.UserData, "Resample on a condition never happened: nbIn={0} nbOut={1}", nbIn, nbOut);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            return this;
        }

        #endregion

        #region Cursor with no cache

        class ResampleCursor<TClass> : DataViewRowCursor
        {
            readonly ResampleTransform _view;
            readonly DataViewRowCursor _inputCursor;
            readonly Random _rand;
            readonly IEnumerable<DataViewSchema.Column> _neededColumns;
            readonly float _lambda;
            readonly int _maxReplica;
            readonly int _shift;
            readonly Dictionary<DataViewRowId, int> _cache;
            readonly ValueGetter<DataViewRowId> _idGetter;
            readonly ValueGetter<TClass> _classGetter;
            readonly TClass _classValue;
            readonly DataViewSchema.Column _classColumn;

            int _copy;
            DataViewRowId _currentId;
            TClass _currentCl;

            public ResampleCursor(ResampleTransform view, DataViewRowCursor cursor, IEnumerable<DataViewSchema.Column> neededColumns,
                                    float lambda, int? seed, Random rand, Dictionary<DataViewRowId, int> cache,
                                    DataViewSchema.Column classColumn, TClass classValue)
            {
                _view = view;
                _inputCursor = cursor;
                _lambda = lambda;
                uint? useed = seed.HasValue ? (uint)seed.Value : (uint?)null;
                _rand = rand == null ? RandomUtils.Create(useed) : rand;
                _neededColumns = neededColumns;
                _cache = cache;
                _maxReplica = cache == null ? Math.Max((int)(lambda * 3 + 1), 1) : _cache.Values.Max();
                int maxReplica = _maxReplica + 1;
                _shift = 0;
                _idGetter = cursor.GetIdGetter();
                while (maxReplica > 0)
                {
                    _shift += 1;
                    maxReplica >>= 1;
                }
                _copy = -1;
                _classValue = classValue;
                _classGetter = classColumn.Index >= 0 && classColumn.Index < int.MaxValue
                    ? _inputCursor.GetGetter<TClass>(classColumn) 
                    : null;
                _classColumn = classColumn;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
#if (DEBUG)
                Dictionary<DataViewRowId, int> localCache = new Dictionary<DataViewRowId, int>();
#endif
                // We do not change the ID (row to row transform).
                var getId = _inputCursor.GetIdGetter();
                return (ref DataViewRowId pos) =>
                {
                    getId(ref pos);
                    if (_shift > 0)
                    {
                        Contracts.Assert(_copy >= 0 && _copy <= _maxReplica);
                        ulong left = pos.Low << _shift;
                        left >>= _shift;
                        left = pos.Low - left;
                        ulong lo = pos.Low << _shift;
                        ulong hi = pos.High << _shift;
                        hi += left >> (64 - _shift);
                        pos = new DataViewRowId(lo + (ulong)_copy, hi);
#if (DEBUG)
                        if (localCache.ContainsKey(pos))
                            throw Contracts.Except("Id already taken: {0}", pos);
#endif
                    }
                    else
                        Contracts.Assert(_copy == 0);
                };
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                // The column is active if is active in the input view or if it the new vector with the polynomial features.
                return _classColumn.Index == col.Index || _inputCursor.IsColumnActive(col);
            }

            public override long Batch { get { return _inputCursor.Batch; } }        // No change.
            public override long Position { get { return _inputCursor.Position; } }  // No change.
            public override DataViewSchema Schema { get { return _view.Schema; } }          // No change.

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveNext()
            {
                if (_copy > 0)
                {
                    --_copy;
                    return true;
                }

                bool r = _inputCursor.MoveNext();
                if (!r)
                    return r;

                if (_cache != null)
                {
                    _idGetter(ref _currentId);
                    _copy = _cache[_currentId];
                }
                else
                {
                    _copy = NextPoisson(_lambda, _rand);
                    _copy = Math.Min(_copy, _maxReplica);
                }

                while (_copy <= 0)
                {
                    r = _inputCursor.MoveNext();
                    if (!r)
                        return r;
                    if (_cache != null)
                    {
                        _idGetter(ref _currentId);
                        _copy = _cache[_currentId];
                    }
                    else if (_classGetter == null)
                    {
                        _copy = NextPoisson(_lambda, _rand);
                        _copy = Math.Min(_copy, _maxReplica);
                    }
                    else
                    {
                        _classGetter(ref _currentCl);
                        _copy = _currentCl.Equals(_classValue) ? NextPoisson(_lambda, _rand) : 1;
                        _copy = Math.Min(_copy, _maxReplica);
                    }
                }
                --_copy;
                return true;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
#if (DEBUG)
                var getter = _inputCursor.GetGetter<TValue>(col);
                return (ref TValue dst) =>
                {
                    getter(ref dst);
                };
#else
                return _inputCursor.GetGetter<TValue>(col);
#endif
            }
        }

        #endregion
    }
}
