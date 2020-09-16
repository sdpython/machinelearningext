// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Transforms;

using LoadableClassAttribute = Microsoft.ML.LoadableClassAttribute;
using ExtLambdaTransform = Scikit.ML.ProductionPrediction.ExtLambdaTransform;

[assembly: LoadableClass(typeof(ITransformer), typeof(ExtLambdaTransform), null, typeof(SignatureLoadModel), "", ExtLambdaTransform.LoaderSignature)]

namespace Scikit.ML.ProductionPrediction
{
    // Licensed to the .NET Foundation under one or more agreements.
    // The .NET Foundation licenses this file to you under the MIT license.
    // See the LICENSE file in the project root for more information.

    using Conditional = System.Diagnostics.ConditionalAttribute;

    public static class SerializableExtLambdaTransform
    {
        // This static class exists so that we can expose the Create loader delegate without having
        // to specify bogus type arguments on the generic class.

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "USERMAPX",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SerializableExtLambdaTransform).Assembly.FullName);
        }

        public const string LoaderSignature = "UserLambdaMapTransform";
        public const string Summary = "Allows the definition of convenient user defined transforms";

        /// <summary>
        /// Creates an instance of the transform from a context.
        /// </summary>
        public static ITransformTemplate Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);
            host.CheckValue(ctx, nameof(ctx));

            host.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: Number of bytes the load method was serialized to
            // byte[n]: The serialized load method info
            // <arbitrary>: Arbitrary bytes saved by the save action

            var loadMethodBytes = ctx.Reader.ReadByteArray();
            host.CheckDecode(Utils.Size(loadMethodBytes) > 0);
            // Attempt to reconstruct the method.
            Exception error;
            var loadFunc = DeserializeStaticDelegateOrNull(host, loadMethodBytes, out error);
            if (loadFunc == null)
            {
                host.AssertValue(error);
                throw error;
            }

            var bytes = ctx.Reader.ReadByteArray() ?? new byte[0];

            using (var ms = new MemoryStream(bytes))
            using (var reader = new BinaryReader(ms))
            {
                var result = loadFunc(reader, env, input);
                env.Check(result != null, "Load method returned null");
                return result;
            }
        }

        /// <summary>
        /// Given a single item function that should be a static method, this builds a serialized version of
        /// that method that should be enough to "recover" it, assuming it is a "recoverable" method (recoverable
        /// here is a loose definition, meaning that <see cref="DeserializeStaticDelegateOrNull"/> is capable
        /// of creating it, which includes among other things that it's static, non-lambda, accessible to
        /// this assembly, etc.).
        /// </summary>
        /// <param name="func">The method that should be "recoverable"</param>
        /// <returns>A string array describing the input method</returns>
        public static byte[] GetSerializedStaticDelegate(ExtLambdaTransform.LoadDelegate func)
        {
            Contracts.CheckValue(func, nameof(func));
            Contracts.CheckParam(func.Target == null, nameof(func), "The load delegate must be static");
            Contracts.CheckParam(Utils.Size(func.GetInvocationList()) <= 1, nameof(func),
                "The load delegate must not be a multicast delegate");

            var meth = func.GetMethodInfo();
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
#if CORECLR
                var m = new CoreHackMethodInfo();
                m.AssemblyName = meth.Module.Assembly.FullName;
                m.MethodName = meth.Name;
                m.ClassName = meth.DeclaringType.ToString();
                formatter.Serialize(ms, m);
#else
                formatter.Serialize(ms, meth);
#endif
                var result = ms.ToArray();
                // I assume it must be impossible to serialize in 0 bytes.
                Contracts.Assert(Utils.Size(result) > 0);
                return result;
            }
        }

        /// <summary>
        /// This is essentially the inverse function to <see cref="GetSerializedStaticDelegate"/>. If the function
        /// is not recoverable for any reason, this will return <c>null</c>, and the error parameter will be set.
        /// </summary>
        /// <param name="ectx">Exception context.</param>
        /// <param name="serialized">The serialized bytes, as returned by <see cref="GetSerializedStaticDelegate"/></param>
        /// <param name="inner">An exception the caller may raise as an inner exception if the return value is
        /// <c>null</c>, else, this itself will be <c>null</c></param>
        /// <returns>The recovered function wrapping the recovered method, or <c>null</c> if it could not
        /// be created, for some reason</returns>
        public static ExtLambdaTransform.LoadDelegate DeserializeStaticDelegateOrNull(IExceptionContext ectx, byte[] serialized, out Exception inner)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertNonEmpty(serialized);
            MethodInfo info = null;
            try
            {
                using (var ms = new MemoryStream(serialized, false))
                {
#if CORECLR
                    var formatter = new BinaryFormatter();
                    object obj = formatter.Deserialize(ms);
                    var hack = obj as CoreHackMethodInfo;
                    var assembly = Assembly.Load(new AssemblyName(hack.AssemblyName));
                    Type t = assembly.GetType(hack.ClassName);
                    info = t.GetTypeInfo().GetDeclaredMethod(hack.MethodName);
#else
                    var formatter = new BinaryFormatter();
                    object obj = formatter.Deserialize(ms);
                    info = obj as MethodInfo;
#endif
                }
            }
            catch (Exception e)
            {
                inner = ectx.ExceptDecode(e, "Failed to deserialize a .NET object");
                return null;
            }
            // Either it's not the right type, or obj itself may be null. Either way we have an error.
            if (info == null)
            {
                inner = ectx.ExceptDecode("Failed to deserialize the method");
                return null;
            }
            if (!info.IsStatic)
            {
                inner = ectx.ExceptDecode("Deserialized method is not static");
                return null;
            }
            try
            {
                var del = info.CreateDelegate(typeof(ExtLambdaTransform.LoadDelegate));
                inner = null;
                return (ExtLambdaTransform.LoadDelegate)del;
            }
            catch (Exception)
            {
                inner = ectx.ExceptDecode("Deserialized method has wrong signature");
                return null;
            }
        }
#if CORECLR
        [Serializable]
        public sealed class CoreHackMethodInfo
        {
            public string MethodName;
            public string AssemblyName;
            public string ClassName;
        }
#endif
    }

    // REVIEW: the current interface to 'state' object may be inadequate: instead of insisting on
    // parameterless constructor, we could take a delegate that would create the state per cursor.
    /// <summary>
    /// This transform is similar to <see cref="CustomMappingTransformer{TSrc,TDst}"/>, but it allows per-cursor state,
    /// as well as the ability to 'accept' or 'filter out' some rows of the supplied <see cref="IDataView"/>.
    /// The downside is that the provided lambda is eagerly called on every row (not lazily when needed), and
    /// parallel cursors are not allowed.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    /// <typeparam name="TState">The type that describes per-cursor state.</typeparam>
    public class ExtStatefulFilterTransform<TSrc, TDst, TState> : ExtLambdaTransformBase, ITransformTemplate
        where TSrc : class, new()
        where TDst : class, new()
        where TState : class, new()
    {
        private const string RegistrationNameTemplate = "ExtStatefulFilterTransform<{0}, {1}>";
        private readonly IDataView _source;
        private readonly Func<TSrc, TDst, TState, bool> _filterFunc;
        private readonly Action<TState> _initStateAction;
        private readonly ColumnBindings _bindings;
        private readonly InternalSchemaDefinition _addedSchema;

        // Memorized input schema definition. Needed for re-apply.
        private readonly SchemaDefinition _inputSchemaDefinition;
        private readonly TypedCursorable<TSrc> _typedSource;

        private static string RegistrationName { get { return string.Format(RegistrationNameTemplate, typeof(TSrc).FullName, typeof(TDst).FullName); } }

        /// <summary>
        /// Create a filter transform that is savable iff <paramref name="saveAction"/> and <paramref name="loadFunc"/> are
        /// not null.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="source">The dataview upon which we construct the transform</param>
        /// <param name="filterFunc">The function by which we transform source to destination columns and decide whether
        /// to keep the row.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="saveAction">An action that allows us to save state to the serialization stream. May be
        /// null simultaneously with <paramref name="loadFunc"/>.</param>
        /// <param name="loadFunc">A function that given the serialization stream and a data view, returns
        /// an <see cref="ITransformTemplate"/>. The intent is, this returned object should itself be a
        /// <see cref="CustomMappingTransformer{TSrc,TDst}"/>, but this is not strictly necessary. This delegate should be
        /// a static non-lambda method that this assembly can legally call. May be null simultaneously with
        /// <paramref name="saveAction"/>.</param>
        /// <param name="inputSchemaDefinition">The schema definition overrides for <typeparamref name="TSrc"/></param>
        /// <param name="outputSchemaDefinition">The schema definition overrides for <typeparamref name="TDst"/></param>
        public ExtStatefulFilterTransform(IHostEnvironment env, IDataView source, Func<TSrc, TDst, TState, bool> filterFunc,
            Action<TState> initStateAction,
            Action<BinaryWriter> saveAction, ExtLambdaTransform.LoadDelegate loadFunc,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, RegistrationName, saveAction, loadFunc)
        {
            Host.AssertValue(source, "source");
            Host.AssertValue(filterFunc, "filterFunc");
            Host.AssertValueOrNull(initStateAction);
            Host.AssertValueOrNull(inputSchemaDefinition);
            Host.AssertValueOrNull(outputSchemaDefinition);

            _source = source;
            _filterFunc = filterFunc;
            _initStateAction = initStateAction;
            _inputSchemaDefinition = inputSchemaDefinition;
            _typedSource = TypedCursorable<TSrc>.Create(Host, Source, false, inputSchemaDefinition);

            var outSchema = InternalSchemaDefinition.Create(typeof(TDst), outputSchemaDefinition);
            _addedSchema = outSchema;
            _bindings = new ColumnBindings(Source.Schema, DataViewConstructionUtils.GetSchemaColumns(outSchema));
        }

        /// <summary>
        /// The 'reapply' constructor.
        /// </summary>
        private ExtStatefulFilterTransform(IHostEnvironment env, ExtStatefulFilterTransform<TSrc, TDst, TState> transform, IDataView newSource)
            : base(env, RegistrationName, transform)
        {
            Host.AssertValue(transform);
            Host.AssertValue(newSource);
            _source = newSource;
            _filterFunc = transform._filterFunc;
            _typedSource = TypedCursorable<TSrc>.Create(Host, newSource, false, transform._inputSchemaDefinition);

            _addedSchema = transform._addedSchema;
            _bindings = new ColumnBindings(newSource.Schema, DataViewConstructionUtils.GetSchemaColumns(_addedSchema));
        }

        public bool CanShuffle => false;

        DataViewSchema IDataView.Schema => OutputSchema;

        public DataViewSchema OutputSchema => _bindings.Schema;

        public long? GetRowCount()
        {
            // REVIEW: currently stateful map is implemented via filter, and this is sub-optimal.
            return null;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var activeInputs = _bindings.GetActiveInput(predicate);
            Func<int, bool> inputPred = c => activeInputs[c];

            var input = _typedSource.GetCursor(inputPred, rand == null ? (int?)null : rand.Next());
            return new Cursor(this, input, columnsNeeded);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Contracts.CheckParam(n >= 0, nameof(n));
            Contracts.CheckValueOrNull(rand);

            // This transform is stateful, its contract is to allocate exactly one state object per cursor and call the filter function
            // on every row in sequence. Therefore, parallel cursoring is not possible.
            return new[] { GetRowCursor(columnsNeeded, rand) };
        }

        public IDataView Source => _source;

        IDataTransform ITransformTemplate.ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(newSource, nameof(newSource));
            return new ExtStatefulFilterTransform<TSrc, TDst, TState>(env, this, newSource);
        }

        private sealed class Cursor : RootCursorBase
        {
            private readonly ExtStatefulFilterTransform<TSrc, TDst, TState> _parent;

            private readonly RowCursor<TSrc> _input;
            // This is used to serve getters for the columns we produce.
            private readonly DataViewRow _appendedRow;

            private readonly TSrc _src;
            private readonly TDst _dst;
            private readonly TState _state;

            private bool _disposed;

            public override long Batch => _input.Batch;

            public Cursor(ExtStatefulFilterTransform<TSrc, TDst, TState> parent, RowCursor<TSrc> input, IEnumerable<DataViewSchema.Column> columnsNeeded)
                : base(parent.Host)
            {
                Ch.AssertValue(input);
                Ch.AssertValue(columnsNeeded);

                _parent = parent;
                _input = input;

                _src = new TSrc();
                _dst = new TDst();
                _state = new TState();

                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _src, Ch);
                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _dst, Ch);
                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _state, Ch);

                parent._initStateAction?.Invoke(_state);

                var appendedDataView = new DataViewConstructionUtils.SingleRowLoopDataView<TDst>(parent.Host, _parent._addedSchema);
                appendedDataView.SetCurrentRowObject(_dst);

                var columnNames = columnsNeeded.Select(c => c.Name);
                _appendedRow = appendedDataView.GetRowCursor(appendedDataView.Schema.Where(c => !c.IsHidden && columnNames.Contains(c.Name)));
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    if (_state is IDisposable disposableState)
                        disposableState.Dispose();
                    if (_src is IDisposable disposableSrc)
                        disposableSrc.Dispose();
                    if (_dst is IDisposable disposableDst)
                        disposableDst.Dispose();
                    _input.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return _input.GetIdGetter();
            }

            protected override bool MoveNextCore()
            {
                bool isAccepted = false;
                while (!isAccepted)
                {
                    if (!_input.MoveNext())
                        return false;
                    RunLambda(out isAccepted);
                }
                return true;
            }

            private void RunLambda(out bool isRowAccepted)
            {
                _input.FillValues(_src);
                // REVIEW: what if this throws? Maybe swallow the exception?
                isRowAccepted = _parent._filterFunc(_src, _dst, _state);
            }

            public override DataViewSchema Schema => _parent._bindings.Schema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Contracts.CheckParam(column.Index < Schema.Count, nameof(column));
                bool isSrc;
                int iCol = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return _input.IsColumnActive(_input.Schema[iCol]);
                return _appendedRow.IsColumnActive(_appendedRow.Schema[iCol]);
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Contracts.CheckParam(column.Index < Schema.Count, nameof(column));
                bool isSrc;
                int iCol = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                return isSrc ?
                    _input.GetGetter<TValue>(_input.Schema[iCol])
                    : _appendedRow.GetGetter<TValue>(_appendedRow.Schema[iCol]);
            }
        }
    }

    /// <summary>
    /// Utility class for creating transforms easily.
    /// </summary>
    public static class ExtLambdaTransform
    {
        /// <summary>
        /// A delegate type to create a persistent transform, utilized by the creation functions
        /// as a callback to reconstitute a transform from binary data.
        /// </summary>
        /// <param name="reader">The binary stream from which the load is persisted</param>
        /// <param name="env">The host environment</param>
        /// <param name="input">The dataview this transform should be persisted on</param>
        /// <returns>A transform of the input data, as parameterized by the binary input
        /// stream</returns>
        public delegate ITransformTemplate LoadDelegate(BinaryReader reader, IHostEnvironment env, IDataView input);

        public const string LoaderSignature = "ExtCustomTransformer";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EUSTOMXF",
                //verWrittenCur: 0x00010001,  // Initial
                verWrittenCur: 0x00010002,  // Added name of assembly in which the contractName is present
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ExtLambdaTransform).Assembly.FullName);
        }

        private const uint VerAssemblyNameSaved = 0x00010002;

        public static void SaveCustomTransformer(IExceptionContext ectx, ModelSaveContext ctx, string contractName, string contractAssembly)
        {
            ectx.CheckValue(ctx, nameof(ctx));
            ectx.CheckValue(contractName, nameof(contractName));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveString(contractName);
            ctx.SaveString(contractAssembly);
        }

        // Factory for SignatureLoadModel.
        private static ITransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var contractName = ctx.LoadString();
            if (ctx.Header.ModelVerWritten >= VerAssemblyNameSaved)
            {
                var contractAssembly = ctx.LoadString();
                Assembly assembly = Assembly.Load(contractAssembly);
                env.ComponentCatalog.RegisterAssembly(assembly);
            }

            object factoryObject = env.ComponentCatalog.GetExtensionValue(env, typeof(CustomMappingFactoryAttributeAttribute), contractName);
            if (!(factoryObject is ICustomMappingFactory mappingFactory))
            {
                throw env.Except($"The class with contract '{contractName}' must derive from '{typeof(CustomMappingFactory<,>).FullName}' or from '{typeof(StatefulCustomMappingFactory<,,>).FullName}'.");
            }

            return mappingFactory.CreateTransformer(env, contractName);
        }

        /// <summary>
        /// This is a 'stateful non-savable' version of the map transform: the mapping function is guaranteed to be invoked once per
        /// every row of the data set, in sequence; one user-defined state object will be allocated per cursor and passed to the
        /// map function every time. If <typeparamref name="TSrc"/>, <typeparamref name="TDst"/>, or
        /// <typeparamref name="TState"/> implement the <see cref="IDisposable" /> interface, they will be disposed after use.
        /// </summary>
        /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the
        /// input <see cref="IDataView"/>.</typeparam>
        /// <typeparam name="TState">The type of the state object to allocate per cursor.</typeparam>
        /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
        /// <param name="env">The host environment to use.</param>
        /// <param name="source">The input data to apply transformation to.</param>
        /// <param name="mapAction">The function that performs the transformation. The function should transform its <typeparamref name="TSrc"/>
        /// argument into its <typeparamref name="TDst"/> argument and can utilize the per-cursor <typeparamref name="TState"/> state.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TDst"/> type.</param>
        public static ITransformTemplate CreateMap<TSrc, TDst, TState>(IHostEnvironment env, IDataView source,
            Action<TSrc, TDst, TState> mapAction, Action<TState> initStateAction,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            where TState : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            env.CheckValue(mapAction, nameof(mapAction));
            env.CheckValueOrNull(initStateAction);
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);

            return new ExtStatefulFilterTransform<TSrc, TDst, TState>(env, source,
                (src, dst, state) =>
                {
                    mapAction(src, dst, state);
                    return true;
                }, initStateAction, null, null, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// This creates a filter transform that can 'accept' or 'decline' any row of the data based on the contents of the row
        /// or state of the cursor.
        /// This is a 'stateful non-savable' version of the filter: the filter function is guaranteed to be invoked once per
        /// every row of the data set, in sequence (non-parallelizable); one user-defined state object will be allocated per cursor and passed to the
        /// filter function every time.
        /// If <typeparamref name="TSrc"/> or <typeparamref name="TState"/> implement the <see cref="IDisposable" /> interface, they will be disposed after use.
        /// </summary>
        /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the
        /// input <see cref="IDataView"/>.</typeparam>
        /// <typeparam name="TState">The type of the state object to allocate per cursor.</typeparam>
        /// <param name="env">The host environment to use.</param>
        /// <param name="source">The input data to apply transformation to.</param>
        /// <param name="filterFunc">The user-defined function that determines whether to keep the row or discard it. First parameter
        /// is the current row's contents, the second parameter is the cursor-specific state object.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <returns></returns>
        public static ITransformTemplate CreateFilter<TSrc, TState>(IHostEnvironment env, IDataView source,
            Func<TSrc, TState, bool> filterFunc, Action<TState> initStateAction, SchemaDefinition inputSchemaDefinition = null)
            where TSrc : class, new()
            where TState : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            env.CheckValue(filterFunc, nameof(filterFunc));
            env.CheckValueOrNull(initStateAction);
            env.CheckValueOrNull(inputSchemaDefinition);

            return new ExtStatefulFilterTransform<TSrc, object, TState>(env, source,
                (src, dst, state) => filterFunc(src, state), initStateAction, null, null, inputSchemaDefinition);
        }

        /// <summary>
        /// This creates a filter transform that can 'accept' or 'decline' any row of the data based on the contents of the row
        /// or state of the cursor.
        /// This is a 'stateful savable' version of the filter: the filter function is guaranteed to be invoked once per
        /// every row of the data set, in sequence (non-parallelizable); one user-defined state object will be allocated per cursor and passed to the
        /// filter function every time; save and load routines must be provided.
        /// If <typeparamref name="TSrc"/> or <typeparamref name="TState"/> implement the <see cref="IDisposable" /> interface, they will be disposed after use.
        /// </summary>
        /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the
        /// input <see cref="IDataView"/>.</typeparam>
        /// <typeparam name="TState">The type of the state object to allocate per cursor.</typeparam>
        /// <param name="env">The host environment to use.</param>
        /// <param name="source">The input data to apply transformation to.</param>
        /// <param name="filterFunc">The user-defined function that determines whether to keep the row or discard it. First parameter
        /// is the current row's contents, the second parameter is the cursor-specific state object.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="saveAction">An action that allows us to save state to the serialization stream</param>
        /// <param name="loadFunc">A function that given the serialization stream and a data view, returns
        /// an <see cref="ITransformTemplate"/>. The intent is, this returned object should itself be the same
        /// as if we had recreated it using this method, but this is impossible to enforce. This transform
        /// will do its best to save a description of this method through assembly qualified names of the defining
        /// class, method name, and generic type parameters (if any), and then recover this same method on load,
        /// so it should be a static non-lambda method that this assembly can legally call.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <returns></returns>
        public static ITransformTemplate CreateFilter<TSrc, TState>(IHostEnvironment env, IDataView source,
            Func<TSrc, TState, bool> filterFunc, Action<TState> initStateAction,
            Action<BinaryWriter> saveAction, LoadDelegate loadFunc,
            SchemaDefinition inputSchemaDefinition = null)
            where TSrc : class, new()
            where TState : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            env.CheckValue(filterFunc, nameof(filterFunc));
            env.CheckValue(initStateAction, nameof(initStateAction));
            env.CheckValue(saveAction, nameof(saveAction));
            env.CheckValue(loadFunc, nameof(loadFunc));
            env.CheckValueOrNull(inputSchemaDefinition);

            return new ExtStatefulFilterTransform<TSrc, object, TState>(env, source,
                (src, dst, state) => filterFunc(src, state), initStateAction, saveAction, loadFunc, inputSchemaDefinition);
        }
    }

    /// <summary>
    /// Defines common ancestor for various flavors of lambda-based user-defined transforms that may or may not be
    /// serializable.
    ///
    /// In order for the transform to be serializable, the user should specify a save and load delegate.
    /// Specifically, for this the user has to provide the following things:
    ///  * a custom save action that serializes the transform 'state' to the binary writer.
    ///  * a custom load action that de-serializes the transform from the binary reader. This must be a public static method of a public class.
    /// </summary>
    public abstract class ExtLambdaTransformBase : ICanSaveModel
    {
        private readonly Action<BinaryWriter> _saveAction;
        private readonly byte[] _loadFuncBytes;
        protected readonly IHost Host;
        protected ExtLambdaTransformBase(IHostEnvironment env, string name, Action<BinaryWriter> saveAction, ExtLambdaTransform.LoadDelegate loadFunc)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);

            Host.Assert((saveAction == null) == (loadFunc == null));

            if (saveAction != null)
            {
                _saveAction = saveAction;
                // First, verify as best we can, that we can recover the function, by attempting to do it once.
                _loadFuncBytes = SerializableExtLambdaTransform.GetSerializedStaticDelegate(loadFunc);
                Exception error;
                var recoveredLoadFunc = SerializableExtLambdaTransform.DeserializeStaticDelegateOrNull(Host, _loadFuncBytes, out error);
                if (recoveredLoadFunc == null)
                {
                    Host.AssertValue(error);
                    throw Host.Except(error, "Load function does not appear recoverable");
                }
            }

            AssertConsistentSerializable();
        }

        /// <summary>
        /// The 'reapply' constructor.
        /// </summary>
        protected ExtLambdaTransformBase(IHostEnvironment env, string name, ExtLambdaTransformBase source)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            _saveAction = source._saveAction;
            _loadFuncBytes = source._loadFuncBytes;

            AssertConsistentSerializable();
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Check(CanSave(), "Cannot save this transform as it was not specified as being savable");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(SerializableExtLambdaTransform.GetVersionInfo());

            // *** Binary format ***
            // int: Number of bytes the load method was serialized to
            // byte[n]: The serialized load method info
            // <arbitrary>: Arbitrary bytes saved by the save action

            Host.AssertNonEmpty(_loadFuncBytes);
            ctx.Writer.WriteByteArray(_loadFuncBytes);

            using (var ms = new MemoryStream())
            {
                using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
                    _saveAction(writer);
                ctx.Writer.WriteByteArray(ms.ToArray());
            }
        }

        private bool CanSave()
        {
            return _saveAction != null;
        }

        [Conditional("DEBUG")]
        private void AssertConsistentSerializable()
        {
#if DEBUG
            // This class can be either serializable, or not. Some fields should
            // be null iff the transform is not savable.
            bool canSave = CanSave();
            Host.Assert(canSave == (_saveAction != null));
            Host.Assert(canSave == (_loadFuncBytes != null));
#endif
        }
    }
}


