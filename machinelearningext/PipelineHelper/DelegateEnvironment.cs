// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Text;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    public interface ILogWriter
    {
        void Write(string text);
        void WriteLine(string line);
        TextWriter AsTextWriter { get; }
    }

    public delegate void WriteType(string line);

    public class LogWriter : ILogWriter
    {
        public class LikeTextWriter : TextWriter
        {
            public ILogWriter _parent;
            public override Encoding Encoding => Encoding.UTF8;
            public LikeTextWriter(ILogWriter parent) : base() { _parent = parent; }
            public override void Write(string text) { _parent.Write(text); }
            public override void WriteLine(string text) { _parent.WriteLine(text); }
        }
        private readonly WriteType _out;
        private readonly LikeTextWriter _asTextWriter;
        public LogWriter(WriteType fout) { _out = fout; _asTextWriter = new LikeTextWriter(this); }
        public void Write(string text) { _out(text); }
        public void WriteLine(string line) { _out(line + "\n"); }
        public TextWriter AsTextWriter => _asTextWriter;
    }

    public class DelegateEnvironment : HostEnvironmentBase<DelegateEnvironment>
    {
        private bool _elapsed;
        public bool Elapsed => _elapsed;
        public void SetPrintElapsed(bool e) { _elapsed = e; }

        /// <summary>
        /// Creates an environment
        /// </summary>
        public static DelegateEnvironment Create(int? seed = null, int verbose = 0,
                                    MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                                    ILogWriter outWriter = null, ILogWriter errWriter = null)
        {
            return new DelegateEnvironment(seed: seed, verbose: verbose, sensitivity: sensitivity,
                                           outWriter: outWriter, errWriter: errWriter);
        }

        public static void Delete(DelegateEnvironment env)
        {
        }

        private sealed class OutErrLogWriter
        {
            private readonly object _lock;
            private readonly DelegateEnvironment _parent;
            private ILogWriter _out;
            private ILogWriter _err;
            private int _verbose;

            // Progress reporting. Print up to 50 dots, if there's no meaningful (checkpoint) events.
            // At the end of 50 dots, print current metrics.
            private const int _maxDots = 50;
            private int _dots;

            public OutErrLogWriter(DelegateEnvironment parent, ILogWriter outWriter, ILogWriter errWriter, int verbose)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(outWriter);
                Contracts.AssertValue(errWriter);
                _lock = new object();
                _parent = parent;
                _out = outWriter;
                _err = errWriter;
                _verbose = verbose;
            }

            public void PrintMessage(IMessageSource sender, ChannelMessage msg)
            {
                if (_verbose == 0)
                    return;
                bool isError = false;
                switch (msg.Kind)
                {
                    case ChannelMessageKind.Trace:
                        if (!sender.Verbose)
                            return;
                        if (_verbose < 4)
                            return;
                        break;
                    case ChannelMessageKind.Info:
                        if (_verbose < 2)
                            return;
                        break;
                    case ChannelMessageKind.Warning:
                        if (_verbose < 2)
                            return;
                        isError = true;
                        break;
                    default:
                        Contracts.Assert(msg.Kind == ChannelMessageKind.Error);
                        isError = true;
                        break;
                }

                lock (_lock)
                {
                    EnsureNewLine(isError);
                    var wr = isError ? _err : _out;

                    string prefix = WriteAndReturnLinePrefix(msg.Sensitivity, wr);
                    var commChannel = sender as PipeBase<ChannelMessage>;
                    if (commChannel?.Verbose == true)
                        WriteHeader(wr, commChannel);
                    if (msg.Kind == ChannelMessageKind.Warning)
                        wr.Write("Warning: ");
                    _parent.PrintMessageNormalized(wr.AsTextWriter, msg.Message, true, prefix);
                }
            }

            private string LinePrefix(MessageSensitivity sensitivity)
            {
                if (_parent._sensitivityFlags == MessageSensitivity.All || ((_parent._sensitivityFlags & sensitivity) != MessageSensitivity.None))
                    return null;
                return "SystemLog:";
            }

            private string WriteAndReturnLinePrefix(MessageSensitivity sensitivity, ILogWriter writer)
            {
                string prefix = LinePrefix(sensitivity);
                if (prefix != null)
                    writer.Write(prefix);
                return prefix;
            }

            private void WriteHeader(ILogWriter wr, PipeBase<ChannelMessage> commChannel)
            {
                Contracts.Assert(commChannel.Verbose);
                if (_verbose > 2)
                {
                    wr.Write(new string(' ', commChannel.Depth * 2));
                    WriteName(wr, commChannel);
                }
            }

            private void WriteName(ILogWriter wr, ChannelProviderBase provider)
            {
                var channel = provider as Channel;
                if (channel != null)
                    WriteName(wr, channel.Parent);
                wr.Write(string.Format("{0}: ", provider.ShortName));
            }

            public void ChannelStarted(Channel channel)
            {
                if (!channel.Verbose)
                    return;

                lock (_lock)
                {
                    EnsureNewLine();
                    WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                    if (_verbose > 2)
                    {
                        WriteHeader(_out, channel);
                        _out.WriteLine("Started.");
                    }
                }
            }

            public void ChannelDisposed(Channel channel)
            {
                if (!channel.Verbose)
                    return;

                if (_parent.Elapsed)
                    lock (_lock)
                    {
                        EnsureNewLine();
                        WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                        WriteHeader(_out, channel);
                        _out.WriteLine("Finished.");
                        EnsureNewLine();
                        WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                        WriteHeader(_out, channel);
                        _out.WriteLine(string.Format("Elapsed {0:c}.", channel.Watch.Elapsed));
                    }
            }

            /// <summary>
            /// Query all progress and:
            /// * If there's any checkpoint/start/stop event, print all of them.
            /// * If there's none, print a dot.
            /// * If there's <see cref="_maxDots"/> dots, print the current status for all running calculations.
            /// </summary>
            public void GetAndPrintAllProgress(ProgressReporting.ProgressTracker progressTracker)
            {
                Contracts.AssertValue(progressTracker);

                var entries = progressTracker.GetAllProgress();
                if (entries.Count == 0)
                {
                    // There's no calculation running. Don't even print a dot.
                    return;
                }

                var checkpoints = entries.Where(
                    x => x.Kind != ProgressReporting.ProgressEvent.EventKind.Progress || x.ProgressEntry.IsCheckpoint);

                lock (_lock)
                {
                    bool anyCheckpoint = false;
                    foreach (var ev in checkpoints)
                    {
                        anyCheckpoint = true;
                        EnsureNewLine();
                        // We assume that things like status counters, which contain only things
                        // like loss function values, counts of rows, counts of items, etc., are
                        // not sensitive.
                        WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                        switch (ev.Kind)
                        {
                            case ProgressReporting.ProgressEvent.EventKind.Start:
                                PrintOperationStart(_out, ev);
                                break;
                            case ProgressReporting.ProgressEvent.EventKind.Stop:
                                PrintOperationStop(_out, ev);
                                break;
                            case ProgressReporting.ProgressEvent.EventKind.Progress:
                                _out.Write(string.Format("[{0}] ", ev.Index));
                                PrintProgressLine(_out, ev);
                                break;
                        }
                    }
                    if (anyCheckpoint)
                    {
                        // At least one checkpoint has been printed, so there's no need for dots.
                        return;
                    }

                    if (PrintDot())
                    {
                        // We need to print an extended status line. At this point, every event should be
                        // a non-checkpoint progress event.
                        bool needPrepend = entries.Count > 1;
                        foreach (var ev in entries)
                        {
                            Contracts.Assert(ev.Kind == ProgressReporting.ProgressEvent.EventKind.Progress);
                            Contracts.Assert(!ev.ProgressEntry.IsCheckpoint);
                            if (needPrepend)
                            {
                                EnsureNewLine();
                                WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                                _out.Write(string.Format("[{0}] ", ev.Index));
                            }
                            else
                            {
                                // This is the only case we are printing something at the end of the line of dots.
                                // So, we need to reset the dots counter.
                                _dots = 0;
                            }
                            PrintProgressLine(_out, ev);
                        }
                    }
                }
            }

            private static void PrintOperationStart(ILogWriter writer, ProgressReporting.ProgressEvent ev)
            {
                writer.WriteLine(string.Format("[{0}] '{1}' started.", ev.Index, ev.Name));
            }

            private static void PrintOperationStop(ILogWriter writer, ProgressReporting.ProgressEvent ev)
            {
                writer.WriteLine(string.Format("[{0}] '{1}' finished in {2}.", ev.Index, ev.Name, ev.EventTime - ev.StartTime));
            }

            private void PrintProgressLine(ILogWriter writer, ProgressReporting.ProgressEvent ev)
            {
                // Elapsed time.
                var elapsed = ev.EventTime - ev.StartTime;
                if (elapsed.TotalMinutes < 1)
                    writer.Write(string.Format("(00:{0:00.00})", elapsed.TotalSeconds));
                else if (elapsed.TotalHours < 1)
                    writer.Write(string.Format("({0:00}:{1:00.0})", elapsed.Minutes, elapsed.TotalSeconds - 60 * elapsed.Minutes));
                else
                    writer.Write(string.Format("({0:00}:{1:00}:{2:00})", elapsed.Hours, elapsed.Minutes, elapsed.Seconds));

                // Progress units.
                bool first = true;
                for (int i = 0; i < ev.ProgressEntry.Header.UnitNames.Count; i++)
                {
                    if (ev.ProgressEntry.Progress[i] == null)
                        continue;
                    writer.Write(first ? "\t" : ", ");
                    first = false;
                    writer.Write(string.Format("{0}", ev.ProgressEntry.Progress[i]));
                    if (ev.ProgressEntry.ProgressLim[i] != null)
                        writer.Write(string.Format("/{0}", ev.ProgressEntry.ProgressLim[i].Value));
                    writer.Write(string.Format(" {0}", ev.ProgressEntry.Header.UnitNames[i]));
                }

                // Metrics.
                for (int i = 0; i < ev.ProgressEntry.Header.MetricNames.Count; i++)
                {
                    if (ev.ProgressEntry.Metrics[i] == null)
                        continue;
                    writer.Write(string.Format("\t{0}: {1}", ev.ProgressEntry.Header.MetricNames[i], ev.ProgressEntry.Metrics[i].Value));
                }

                writer.WriteLine(string.Empty);
            }

            /// <summary>
            /// If we printed any dots so far, finish the line. This call is expected to be protected by _lock.
            /// </summary>
            private void EnsureNewLine(bool isError = false)
            {
                if (_dots == 0)
                    return;

                // If _err and _out is the same writer, we need to print new line as well.
                // If _out and _err writes to Console.Out and Console.Error respectively,
                // in the general user scenario they ends up with writing to the same underlying stream,.
                // so write a new line to the stream anyways.
                if (isError && _err != _out && (_out != Console.Out || _err != Console.Error))
                    return;

                _out.WriteLine(string.Empty);
                _dots = 0;
            }

            /// <summary>
            /// Print a progress dot. Returns whether it is 'time' to print more info. This call is expected
            /// to be protected by _lock.
            /// </summary>
            private bool PrintDot()
            {
                _out.Write(".");
                _dots++;
                return (_dots == _maxDots);
            }
        }

        private sealed class Channel : ChannelBase
        {
            public readonly Stopwatch Watch;
            public Channel(DelegateEnvironment root, ChannelProviderBase parent, string shortName,
                Action<IMessageSource, ChannelMessage> dispatch)
                : base(root, parent, shortName, dispatch)
            {
                Watch = Stopwatch.StartNew();
                Root._outErrWriter.ChannelStarted(this);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    Watch.Stop();
                    Root._outErrWriter.ChannelDisposed(this);
                }
                base.Dispose(disposing);
            }
        }

        private volatile OutErrLogWriter _outErrWriter;
        private readonly MessageSensitivity _sensitivityFlags;
        private readonly int _verbose;

        public int VerboseLevel => _verbose;

        public DelegateEnvironment(int? seed = null, int verbose = 0,
            MessageSensitivity sensitivity = MessageSensitivity.All,
            ILogWriter outWriter = null, ILogWriter errWriter = null)
            : this(RandomUtils.Create(seed), verbose, sensitivity, outWriter, errWriter)
        {
            _elapsed = true;
        }

        /// <summary>
        /// This takes ownership of the random number generator.
        /// </summary>
        public DelegateEnvironment(Random rand, int verbose = 0,
            MessageSensitivity sensitivity = MessageSensitivity.All,
            ILogWriter outWriter = null, ILogWriter errWriter = null)
            : base(rand, verbose > 0, nameof(DelegateEnvironment))
        {
            Contracts.CheckValue(outWriter, nameof(outWriter));
            Contracts.CheckValue(errWriter, nameof(errWriter));
            Contracts.CheckParam(verbose >= 0 && verbose <= 4, nameof(verbose), "verbose must be in [[0, 4]]");
            _outErrWriter = new OutErrLogWriter(this, outWriter, errWriter, verbose);
            _sensitivityFlags = sensitivity;
            _verbose = verbose;
            AddListener<ChannelMessage>(PrintMessage);
        }

        /// <summary>
        /// This takes ownership of the random number generator.
        /// </summary>
        public DelegateEnvironment(Random rand, int verbose = 0,
            MessageSensitivity sensitivity = MessageSensitivity.All,
            WriteType outWriter = null, WriteType errWriter = null)
            : base(rand, verbose > 0, nameof(DelegateEnvironment))
        {
            Contracts.CheckValueOrNull(outWriter);
            Contracts.CheckValueOrNull(errWriter);
            Contracts.CheckParam(verbose >= 0 && verbose <= 4, nameof(verbose), "verbose must be in [[0, 4]]");
            _outErrWriter = new OutErrLogWriter(this, new LogWriter(outWriter), new LogWriter(errWriter), verbose);
            _sensitivityFlags = sensitivity;
            _verbose = verbose;
            AddListener<ChannelMessage>(PrintMessage);
        }

        /// <summary>
        /// Pull running calculations for their progress and output all messages to the console.
        /// If no messages are available, print a dot.
        /// If a specified number of dots are printed, print an ad-hoc status of all running calculations.
        /// </summary>
        public void PrintProgress()
        {
            Root._outErrWriter.GetAndPrintAllProgress(ProgressTracker);
        }

        private void PrintMessage(IMessageSource src, ChannelMessage msg)
        {
            Root._outErrWriter.PrintMessage(src, msg);
        }

        protected override IHost RegisterCore(HostEnvironmentBase<DelegateEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose);
        }

        protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is DelegateEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Channel(this, parent, name, GetDispatchDelegate<ChannelMessage>());
        }

        protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is DelegateEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
        }

        private sealed class Host : HostBase
        {
            public Host(HostEnvironmentBase<DelegateEnvironment> source, string shortName, string parentFullName, Random rand, bool verbose)
                : base(source, shortName, parentFullName, rand, verbose)
            {
                IsCancelled = source.IsCancelled;
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

            protected override IHost RegisterCore(HostEnvironmentBase<DelegateEnvironment> source, string shortName, 
                                                  string parentFullName, Random rand, bool verbose)
            {
                return new Host(source, shortName, parentFullName, rand, verbose);
            }
        }
    }
}
