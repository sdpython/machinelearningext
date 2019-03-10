//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------


using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Microsoft.ML.CommandLine;


namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Common class to pass through transform.
    /// </summary>
    public abstract class AbstractSimpleTransformTemplate : IDataTransform
    {
        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Delay the initialization when the first cursor is requested.", ShortName = "dinit")]
            public bool delayInit = false;
        }

        /// <summary>
        /// The secondary source. Received as an input but cursors will be created on _sourcePipe.
        /// </summary>
        protected IDataView _sourceCtx;

        /// <summary>
        /// The output source.
        /// </summary>
        protected IDataView _sourcePipe;

        /// <summary>
        /// Needed by DelayedInitialization.
        /// This method is called when the first cursor is requested.
        /// </summary>
        protected object _lock;

        /// <summary>
        /// Host.
        /// </summary>
        protected readonly IHost _host;

        /// <summary>
        /// The secondary source. Received as an input but cursors will be created on SourceEnd.
        /// </summary>
        public virtual IDataView Source { get { return _sourceCtx; } }

        /// <summary>
        /// The output source.
        /// </summary>
        public virtual IDataView SourceEnd { get { return _sourcePipe; } }

        /// <summary>
        /// DataViewSchema.
        /// </summary>
        public virtual DataViewSchema Schema
        {
            get
            {
                if (_sourcePipe == null)
                    throw _host.Except("The transform was not initialized.");
                return _sourcePipe.Schema;
            }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="input">input source stored as the secondary source</param>
        /// <param name="name">name of the transform</param>
        public AbstractSimpleTransformTemplate(IHostEnvironment env, IDataView input, string name)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(name);
            _host.CheckValue(input, "input");
            _host.Check(!string.IsNullOrEmpty(name));
            _sourceCtx = input;
            _sourcePipe = null;
            _lock = new object();
        }

        /// <summary>
        /// Loading constructor.
        /// </summary>
        /// <param name="ctx">reading context</param>
        /// <param name="env">environment</param>
        /// <param name="input">input source stored as the secondary source</param>
        /// <param name="name">name of the transform</param>
        public AbstractSimpleTransformTemplate(IHost host, ModelLoadContext ctx, IDataView input, string name)
        {
            Contracts.CheckValue(host, "env");
            _host = host;
            _host.CheckValue(ctx, "env");
            _host.CheckValue(input, "input");
            _host.Check(!string.IsNullOrEmpty(name));
            _sourceCtx = input;
            _sourcePipe = null;
            _lock = new object();
        }

        /// <summary>
        /// Initialization of the class when the first cursor is requested.
        /// </summary>
        protected virtual void DelayedInitialisationLockFree()
        {
            // This method is needed when the transform cannot return its schema without
            // being run. However, we still delay that execution.
            Contracts.CheckValue(_host, "_host");
            throw _host.Except("This method was not overridden. Internal error.");
        }

        /// <summary>
        /// Tells if the class is initialized. Otherwise, calls DelayedInitialisation.
        /// </summary>
        protected virtual bool IsInitialized()
        {
            return _sourcePipe != null;
        }

        public virtual bool CanShuffle { get { return _sourcePipe.CanShuffle; } }
        public virtual long? GetRowCount()
        {
            _host.CheckValue(_sourceCtx, "_sourceCtx");
            if (!IsInitialized())
            {
                lock (_lock)
                    if (!IsInitialized())
                        DelayedInitialisationLockFree();
            }
            _host.CheckValue(_sourcePipe, "_sourcePipe");
            return _sourcePipe.GetRowCount();
        }

        public virtual DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.CheckValue(_sourceCtx, "_sourceCtx");
            if (!IsInitialized())
            {
                lock (_lock)
                    if (!IsInitialized())
                        DelayedInitialisationLockFree();
            }
            _host.CheckValue(_sourcePipe, "_sourcePipe");
            return _sourcePipe.GetRowCursor(columnsNeeded, rand);
        }

        public virtual DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            _host.AssertValue(_sourceCtx, "_sourceCtx");
            _host.AssertValue(_sourcePipe, "_sourcePipe");
            return _sourcePipe.GetRowCursorSet(columnsNeeded, n, rand);
        }

        protected abstract void SaveModel(ModelSaveContext ctx);

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);
    }
}
