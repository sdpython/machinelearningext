﻿// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Model;


namespace Scikit.ML.PipelineHelper
{
    public abstract class PredictorBase<OutputType> : IPredictorProducing<OutputType>
    {
        IHostEnvironment _env;
        public IHost Host;

        public PredictorBase(IHostEnvironment env, string registrationName)
        {
            Host = env.Register(registrationName);
            _env = env;
        }

        public PredictorBase(IHostEnvironment env, string registrationName, ModelLoadContext ctx)
        {
            Host = env.Register(registrationName);
            Host.AssertValue(ctx);
            var over = GetType().GetMethod("GetVersionInfo").Invoke(null, null);
            var ver = (VersionInfo)over;
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(ver);
        }

        public virtual PredictionKind PredictionKind { get { throw new NotImplementedException(); } }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(_env);
            _env.CheckValue(ctx, nameof(ctx));
            SaveCore(ctx);
        }
    }
}
