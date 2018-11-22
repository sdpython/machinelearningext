﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
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
    public class ResampleTransform : IDataTransform
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
        Dictionary<UInt128, int> _cacheReplica;

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
                int index;
                if (!_input.Schema.TryGetColumnIndex(_args.column, out index))
                    throw _host.Except("Unable to find column '{0}' in\n'{1}'.", _args.column, SchemaHelper.ToString(_input.Schema));
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

        public Schema Schema { get { return Source.Schema; } }

        public bool CanShuffle { get { return true; } }
        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            if (_cacheReplica != null)
                return _cacheReplica.Values.Sum();
            else
                return null;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            int classColumn = -1;
            if (!string.IsNullOrEmpty(_args.column))
            {
                if (!_input.Schema.TryGetColumnIndex(_args.column, out classColumn))
                    throw _host.Except("Unable to find column '{0}'.", _args.column);
            }
            if (_args.cache)
                LoadCache(rand);
            else
                Contracts.Assert(_cacheReplica == null);
            var cursor = _input.GetRowCursor(predicate, rand);

            if (classColumn == -1)
                return new ResampleCursor<float>(this, cursor, predicate, _args.lambda, _args.seed, rand, _cacheReplica, classColumn, float.NaN);

            var type = _input.Schema.GetColumnType(classColumn);
            switch (type.RawKind())
            {
                case DataKind.BL:
                    bool clbool;
                    if (!bool.TryParse(_args.classValue, out clbool))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return new ResampleCursor<bool>(this, cursor, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, clbool);
                case DataKind.U4:
                    uint cluint;
                    if (!uint.TryParse(_args.classValue, out cluint))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return new ResampleCursor<uint>(this, cursor, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, cluint);
                case DataKind.R4:
                    float clfloat;
                    if (!float.TryParse(_args.classValue, out clfloat))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return new ResampleCursor<float>(this, cursor, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, clfloat);
                case DataKind.TX:
                    var cltext = new ReadOnlyMemory<char>(_args.classValue.ToCharArray());
                    return new ResampleCursor<ReadOnlyMemory<char>>(this, cursor, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, cltext);
                default:
                    throw _host.Except("Unsupported type '{0}'", type);
            }
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            int classColumn = -1;
            if (!string.IsNullOrEmpty(_args.column))
            {
                if (!_input.Schema.TryGetColumnIndex(_args.column, out classColumn))
                    throw _host.Except("Unable to find column '{0}'.", _args.column);
            }
            if (_args.cache)
                LoadCache(rand);
            else
                Contracts.Assert(_cacheReplica == null);

            var cursors = _input.GetRowCursorSet(out consolidator, predicate, n, rand);

            if (classColumn == -1)
                return cursors.Select(c => new ResampleCursor<float>(this, c, predicate, _args.lambda, _args.seed, rand, _cacheReplica, classColumn, float.NaN)).ToArray();

            var type = _input.Schema.GetColumnType(classColumn);
            switch (type.RawKind())
            {
                case DataKind.BL:
                    bool clbool;
                    if (!bool.TryParse(_args.classValue, out clbool))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return cursors.Select(c => new ResampleCursor<bool>(this, c, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, clbool)).ToArray();
                case DataKind.U4:
                    uint cluint;
                    if (!uint.TryParse(_args.classValue, out cluint))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return cursors.Select(c => new ResampleCursor<uint>(this, c, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, cluint)).ToArray();
                case DataKind.R4:
                    float clfloat;
                    if (!float.TryParse(_args.classValue, out clfloat))
                        throw _host.Except("Unable to parse '{0}'.", _args.classValue);
                    return cursors.Select(c => new ResampleCursor<float>(this, c, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, clfloat)).ToArray();
                case DataKind.TX:
                    var cltext = new ReadOnlyMemory<char>(_args.classValue.ToCharArray());
                    return cursors.Select(c => new ResampleCursor<ReadOnlyMemory<char>>(this, c, i => i == classColumn || predicate(i),
                        _args.lambda, _args.seed, rand, _cacheReplica, classColumn, cltext)).ToArray();
                default:
                    throw _host.Except("Unsupported type '{0}'", type);
            }
        }

        public static int NextPoisson(float lambda, IRandom rand)
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

        void LoadCache(IRandom rand)
        {
            if (_cacheReplica != null)
                // Already done.
                return;

            uint? useed = _args.seed.HasValue ? (uint)_args.seed.Value : (uint?)null;
            if (rand == null)
                rand = RandomUtils.Create(useed);

            using (var ch = _host.Start("Resample: fill the cache"))
            {
                int indexClass = -1;
                if (!string.IsNullOrEmpty(_args.column) && !_input.Schema.TryGetColumnIndex(_args.column, out indexClass))
                    throw ch.Except("Unable to find column '{0}'.", _args.column);

                using (var cur = _input.GetRowCursor(i => i == indexClass))
                {
                    if (string.IsNullOrEmpty(_args.column))
                    {
                        _cacheReplica = new Dictionary<UInt128, int>();
                        var gid = cur.GetIdGetter();
                        UInt128 did = default(UInt128);
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
                        var type = _input.Schema.GetColumnType(indexClass);
                        switch (type.RawKind())
                        {
                            case DataKind.BL:
                                bool clbool;
                                if (!bool.TryParse(_args.classValue, out clbool))
                                    throw ch.Except("Unable to parse '{0}'.", _args.classValue);
                                LoadCache<bool>(rand, cur, indexClass, clbool, ch);
                                break;
                            case DataKind.U4:
                                uint cluint;
                                if (!uint.TryParse(_args.classValue, out cluint))
                                    throw ch.Except("Unable to parse '{0}'.", _args.classValue);
                                LoadCache<uint>(rand, cur, indexClass, cluint, ch);
                                break;
                            case DataKind.R4:
                                float clfloat;
                                if (!float.TryParse(_args.classValue, out clfloat))
                                    throw ch.Except("Unable to parse '{0}'.", _args.classValue);
                                LoadCache<float>(rand, cur, indexClass, clfloat, ch);
                                break;
                            case DataKind.TX:
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

        void LoadCache<TClass>(IRandom rand, IRowCursor cur, int classColumn, TClass valueClass, IChannel ch)
        {
            _cacheReplica = new Dictionary<UInt128, int>();
            var hist = new Dictionary<TClass, long>();
            var gid = cur.GetIdGetter();
            var gcl = cur.GetGetter<TClass>(classColumn);
            UInt128 did = default(UInt128);
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
                ch.Warning("Resample on a condition never happened: nbIn={0} nbOut={1}", nbIn, nbOut);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            return this;
        }

        #endregion

        #region Cursor with no cache

        class ResampleCursor<TClass> : IRowCursor
        {
            readonly ResampleTransform _view;
            readonly IRowCursor _inputCursor;
            readonly IRandom _rand;
            readonly Func<int, bool> _predicate;
            readonly float _lambda;
            readonly int _maxReplica;
            readonly int _shift;
            readonly Dictionary<UInt128, int> _cache;
            readonly ValueGetter<UInt128> _idGetter;
            readonly ValueGetter<TClass> _classGetter;
            readonly TClass _classValue;

            int _copy;
            UInt128 _currentId;
            TClass _currentCl;

            public ResampleCursor(ResampleTransform view, IRowCursor cursor, Func<int, bool> predicate,
                                    float lambda, int? seed, IRandom rand, Dictionary<UInt128, int> cache,
                                    int classColumn, TClass classValue)
            {
                _view = view;
                _inputCursor = cursor;
                _lambda = lambda;
                uint? useed = seed.HasValue ? (uint)seed.Value : (uint?)null;
                _rand = rand == null ? RandomUtils.Create(useed) : rand;
                _predicate = predicate;
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
                _classGetter = classColumn >= 0 ? _inputCursor.GetGetter<TClass>(classColumn) : null;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
#if (DEBUG)
                Dictionary<UInt128, int> localCache = new Dictionary<UInt128, int>();
#endif
                // We do not change the ID (row to row transform).
                var getId = _inputCursor.GetIdGetter();
                return (ref UInt128 pos) =>
                {
                    getId(ref pos);
                    if (_shift > 0)
                    {
                        Contracts.Assert(_copy >= 0 && _copy <= _maxReplica);
                        ulong left = pos.Lo << _shift;
                        left >>= _shift;
                        left = pos.Lo - left;
                        ulong lo = pos.Lo << _shift;
                        ulong hi = pos.Hi << _shift;
                        hi += left >> (64 - _shift);
                        pos = new UInt128(lo + (ulong)_copy, hi);
#if (DEBUG)
                        if (localCache.ContainsKey(pos))
                            throw Contracts.Except("Id already taken: {0}", pos);
#endif
                    }
                    else
                        Contracts.Assert(_copy == 0);
                };
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool IsColumnActive(int col)
            {
                // The column is active if is active in the input view or if it the new vector with the polynomial features.
                return _predicate(col) && _inputCursor.IsColumnActive(col);
            }

            public CursorState State { get { return _inputCursor.State; } } // No change.
            public long Batch { get { return _inputCursor.Batch; } }        // No change.
            public long Position { get { return _inputCursor.Position; } }  // No change.
            public Schema Schema { get { return _view.Schema; } }          // No change.

            void IDisposable.Dispose()
            {
                _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public bool MoveMany(long count)
            {
                throw Contracts.ExceptNotImpl();
            }

            public bool MoveNext()
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

            public ValueGetter<TValue> GetGetter<TValue>(int col)
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
