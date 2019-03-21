// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;


namespace Scikit.ML.ProductionPrediction
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    public class PassThroughEnvironment : HostEnvironmentBase<PassThroughEnvironment>
    {
        protected class Channel : ChannelBase
        {
            public readonly Stopwatch Watch;
            public Channel(PassThroughEnvironment root, ChannelProviderBase parent, string shortName,
                Action<IMessageSource, ChannelMessage> dispatch)
                : base(root, parent, shortName, dispatch)
            {
                Watch = Stopwatch.StartNew();
                Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel started"));
            }

            private void ChannelDisposed()
                => Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel finished. Elapsed { 0:c }."));

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ChannelDisposed();
                    Watch.Stop();
                    Dispatch(this, new ChannelMessage(ChannelMessageKind.Trace, MessageSensitivity.None, "Channel disposed"));
                }
                base.Dispose(disposing);
            }
        }

        private sealed class Host : HostBase
        {
            public Host(HostEnvironmentBase<PassThroughEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
                : base(source, shortName, parentFullName, rand, verbose)
            {
                IsCanceled = source.IsCanceled;
            }

            protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Channel(Root, parent, name, GetDispatchDelegate<ChannelMessage>());
            }

            protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
            }

            protected override IHost RegisterCore(HostEnvironmentBase<PassThroughEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
            {
                return new Host(source, shortName, parentFullName, rand, verbose);
            }
        }

        IHostEnvironment _parent;

        public PassThroughEnvironment(IHostEnvironment source,
                                    Random rand = null, bool verbose = false,
                                    string shortName = null,
                                    string parentFullName = null)
            : base(rand, verbose, shortName, parentFullName)
        {
            _parent = source;
        }

        protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.AssertNonEmpty(name);
            return new Channel(this, parent, name, GetDispatchDelegate<ChannelMessage>());
        }

        protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.AssertNonEmpty(name);
            return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
        }

        protected override IHost RegisterCore(HostEnvironmentBase<PassThroughEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose);
        }
    }
}
