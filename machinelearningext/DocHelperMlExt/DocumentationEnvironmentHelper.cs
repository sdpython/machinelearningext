// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tools;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DocHelperMlExt
{
    /// <summary>
    /// Mostly taken from the location below and modified to add custom logging.
    /// https://github.com/dotnet/machinelearning/tree/master/src/Microsoft.ML.Core/ComponentModel
    /// </summary>
    public class DocumentationEnvironmentHelper
    {
        #region private

        private static Assembly LoadAssembly(IHostEnvironment env, string path)
        {
            Assembly assembly = null;
            try
            {
                assembly = Assembly.LoadFrom(path);
            }
            catch (Exception)
            {
                return null;
            }

            if (assembly != null)
            {
                try
                {
                    TryRegisterAssembly(env.ComponentCatalog, assembly);
                }
                catch (Exception e)
                {
                    throw new Exception($"Unable to register DLL '{path}'.", e);
                }
            }

            return assembly;
        }

        private static bool ShouldSkipPath(string path)
        {
            string name = Path.GetFileName(path).ToLowerInvariant();
            switch (name)
            {
                case "cpumathnative.dll":
                case "cqo.dll":
                case "dnnanalyzer.dll":
                case "fasttreenative.dll":
                case "factorizationmachinenative.dll":
                case "libiomp5md.dll":
                case "ldanative.dll":
                case "libvw.dll":
                case "matrixinterf.dll":
                case "microsoft.ml.neuralnetworks.gpucuda.dll":
                case "mklimports.dll":
                case "microsoft.research.controls.decisiontrees.dll":
                case "microsoft.ml.neuralnetworks.sse.dll":
                case "neuraltreeevaluator.dll":
                case "optimizationbuilderdotnet.dll":
                case "parallelcommunicator.dll":
                case "microsoft.ml.runtime.runtests.dll":
                case "symsgdnative.dll":
                case "tbb.dll":
                    return true;
            }

            var _filePrefixesToAvoid = new string[] {
                "api-ms-win",
                "clr",
                "coreclr",
                "dbgshim",
                "ext-ms-win",
                "microsoft.bond.",
                "microsoft.cosmos.",
                "microsoft.csharp",
                "microsoft.data.",
                "microsoft.hpc.",
                "microsoft.live.",
                "microsoft.platformbuilder.",
                "microsoft.visualbasic",
                "microsoft.visualstudio.",
                "microsoft.win32",
                "microsoft.windowsapicodepack.",
                "microsoft.windowsazure.",
                "mscor",
                "msvc",
                "petzold.",
                "roslyn.",
                "sho",
                "sni",
                "sqm",
                "system.",
                "zlib",
            };

            foreach (var s in _filePrefixesToAvoid)
                if (name.StartsWith(s, StringComparison.OrdinalIgnoreCase))
                    return true;

            return false;
        }

        private static void LoadAssembliesInDir(IHostEnvironment env, string dir, bool filter)
        {
            if (!Directory.Exists(dir))
                return;

            // Load all dlls in the given directory.
            var paths = Directory.EnumerateFiles(dir, "*.dll");
            foreach (string path in paths)
            {
                if (filter && ShouldSkipPath(path))
                    continue;
                LoadAssembly(env, path);
            }
        }

        public static IDisposable CreateAssemblyRegistrar(IHostEnvironment env, string loadAssembliesPath = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValueOrNull(loadAssembliesPath);
            return new AssemblyRegistrar(env, loadAssembliesPath);
        }

        public static void RegisterCurrentLoadedAssemblies(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            foreach (Assembly a in AppDomain.CurrentDomain.GetAssemblies())
                TryRegisterAssembly(env.ComponentCatalog, a);
        }

        private sealed class AssemblyRegistrar : IDisposable
        {
            private readonly IHostEnvironment _env;

            public AssemblyRegistrar(IHostEnvironment env, string path)
            {
                _env = env;

                RegisterCurrentLoadedAssemblies(_env);

                if (!string.IsNullOrEmpty(path))
                {
                    LoadAssembliesInDir(_env, path, true);
                    path = Path.Combine(path, "AutoLoad");
                    LoadAssembliesInDir(_env, path, true);
                }

                AppDomain.CurrentDomain.AssemblyLoad += CurrentDomainAssemblyLoad;
            }

            public void Dispose()
            {
                AppDomain.CurrentDomain.AssemblyLoad -= CurrentDomainAssemblyLoad;
            }

            private void CurrentDomainAssemblyLoad(object sender, AssemblyLoadEventArgs args)
            {
                TryRegisterAssembly(_env.ComponentCatalog, args.LoadedAssembly);
            }
        }

        private static void TryRegisterAssembly(ComponentCatalog catalog, Assembly assembly)
        {
            if (assembly.IsDynamic)
                return;
            if (!CanContainComponents(assembly))
                return;
            catalog.RegisterAssembly(assembly);
        }

        private static bool CanContainComponents(Assembly assembly)
        {
            var targetFullName = typeof(LoadableClassAttributeBase).Assembly.GetName().FullName;

            bool found = false;
            foreach (var name in assembly.GetReferencedAssemblies())
            {
                if (name.FullName == targetFullName)
                {
                    found = true;
                    break;
                }
            }

            return found;
        }

        private static void TrackProgress(DelegateEnvironment env, CancellationToken ct)
        {
            try
            {
                while (!ct.IsCancellationRequested)
                {
                    TimeSpan interval = TimeSpan.FromSeconds(0.6);
                    if (ct.WaitHandle.WaitOne(interval))
                        return;
                    env.PrintProgress();
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("Progress tracking terminated with an exception");
                PrintExceptionData(Console.Error, ex, false);
                Console.Error.WriteLine("Progress tracking is terminated.");
            }
        }

        private static void PrintExceptionData(IChannel ch, Exception ex, bool includeComponents)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(ex);

            var sb = new StringBuilder();
            using (var sw = new StringWriter(sb, CultureInfo.InvariantCulture))
                PrintExceptionData(sw, ex, includeComponents);

            if (sb.Length > 0)
                ch.Error(ex.Sensitivity(), sb.ToString());
        }

        private static void PrintExceptionData(TextWriter writer, Exception ex, bool includeComponents)
        {
            bool anyDataPrinted = false;
            foreach (DictionaryEntry kvp in ex.Data)
            {
                if (Contracts.IsMarkedKey.Equals(kvp.Key))
                    continue;
                if (Contracts.SensitivityKey.Equals(kvp.Key))
                    continue;
                if (!anyDataPrinted)
                {
                    writer.WriteLine();
                    writer.WriteLine("Exception context:");
                }

                if (ConsoleEnvironment.ComponentHistoryKey.Equals(kvp.Key))
                {
                    if (kvp.Value is string[] createdComponents)
                    {
                        if (!includeComponents)
                            continue;

                        writer.WriteLine("    Created components:");
                        foreach (var name in createdComponents)
                            writer.WriteLine("        {0}", name);

                        anyDataPrinted = true;
                        continue;
                    }
                }

                writer.WriteLine("    {0}: {1}", kvp.Key, kvp.Value);
                anyDataPrinted = true;
            }

            if (anyDataPrinted)
                writer.WriteLine();
        }

        /// <summary>
        /// Runs a script.
        /// </summary>
        /// <param name="args">script or command line arguments</param>
        /// <param name="env">environment, created if null</param>
        /// <returns>exit code</returns>
        public static int MainWithProgress(string args, DelegateEnvironment env = null)
        {
            string currentDirectory = Path.GetDirectoryName(typeof(Maml).Module.FullyQualifiedName);
            //bool dispose = false;
            var keepOut = new StringBuilder();
            var keepErr = new StringBuilder();
            if (env == null)
            {
                ILogWriter logout = new LogWriter((string s) =>
                {
                    keepOut.Append(s);
                    //if (env.VerboseLevel <= 2 && s.Contains("Elapsed"))
                    //    throw new Exception(string.Format("{0}\n---\n{1}", s, keepOut.ToString()));
                    Console.Write(s);
                });
                ILogWriter logerr = new LogWriter((string s) =>
                {
                    keepErr.Append(s);
                    if (env.VerboseLevel <= 2 && s.Contains("Elapsed"))
                    {
                        // We do nothing.
                    }
                    if (s.Contains("Elapsed"))
                        Console.Write(s);
                    else
                        Console.Error.Write(s);
                });
                env = new DelegateEnvironment(verbose: 2, outWriter: logout, errWriter: logerr);
                //dispose = true;
            }

            int ret = 0;

            using (var progressCancel = new CancellationTokenSource())
            using (CreateAssemblyRegistrar(env, currentDirectory))
            {
                var progressTrackerTask = Task.Run(() => TrackProgress(env, progressCancel.Token));
                try
                {
                    ret = MainCore(env, args, true);
                }
                finally
                {
                    progressCancel.Cancel();
                    progressTrackerTask.Wait();
                    // If the run completed so quickly that the progress task was cancelled before it even got a chance to start,
                    // we need to gather the checkpoints.
                    env.PrintProgress();
                }
            }
            //if (dispose)
            //env.Dispose();
            return ret;
        }

        static int MainCore(DelegateEnvironment env, string args, bool alwaysPrintStacktrace)
        {
            var mainHost = env.Register("Main");
            using (var ch = mainHost.Start("Main"))
            {
                int result;
                try
                {
                    if (!CmdParser.TryGetFirstToken(args, out string kind, out string settings))
                    {
                        Usage();
                        return -1;
                    }
                    if (!ComponentCatalog.TryCreateInstance<ICommand, SignatureCommand>(mainHost, out ICommand cmd, kind, settings))
                    {
                        ch.Error("Unknown command: '{0}'", kind);
                        Usage();
                        return -1;
                    }
                    var helpC = cmd as HelpCommand;
                    if (helpC == null)
                    {
                        env.SetPrintElapsed(true);
                        cmd.Run();
                    }
                    else
                    {
                        int width = 80;
                        try
                        {
                            width = Console.BufferWidth;
                        }
                        catch (Exception)
                        {
                        }
                        env.SetPrintElapsed(false);
                        helpC.Run(width);
                    }
                    result = 0;
                }
                catch (Exception ex)
                {
                    int count = 0;
                    for (var e = ex; e != null; e = e.InnerException)
                    {
                        // Telemetry: Log the exception
                        if (e.IsMarked())
                        {
                            ch.Error(e.Sensitivity(), e.Message);
                            PrintExceptionData(ch, e, false);
                            count++;
                        }
                    }
                    if (count == 0)
                        ch.Error(MessageSensitivity.None, "Unexpected failure.");
                    if (count == 0 || alwaysPrintStacktrace)
                    {
                        ch.Error(MessageSensitivity.None, "===== Begin detailed dump =====");
                        PrintFullExceptionDetails(ch, ex);
                        if (env.VerboseLevel >= 3)
                        {
                            ch.Error("= LoadedAssemblies =");
                            var assemblies = AppDomain.CurrentDomain.GetAssemblies().Select(x => InfoAssembly(x)).OrderBy(c => c);
                            foreach (var a in assemblies)
                                ch.Error(a);
                        }
                        ch.Error(MessageSensitivity.None, "====== End detailed dump =====");
                    }

                    // Return a negative result code so AEther recognizes this as a failure.
                    result = count > 0 ? -1 : -2;
                }
                finally
                {
                }
                return result;
            }
        }

        private static string InfoAssembly(Assembly ass)
        {
            string name = ass.FullName;
            try
            {
                return name + " - " + ass.Location;
            }
            catch (Exception e)
            {
                return name + " - " + e.ToString();
            }
        }

        private static void PrintFullExceptionDetails(TextWriter writer, Exception ex)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValue(ex);

            int index = 0;
            for (var e = ex; e != null; e = e.InnerException)
            {
                index++;
                writer.WriteLine("({0}) Unexpected exception: {1}, '{2}'", index, e.Message, e.GetType());
                PrintExceptionData(writer, e, true);
                writer.WriteLine(e.StackTrace);
            }
        }

        private static void Usage()
        {
            Console.WriteLine("Usage: maml <cmd> <args>");
            Console.WriteLine("       To get a list of commands: maml ?");
        }

        private static void PrintFullExceptionDetails(IChannel ch, Exception ex)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(ex);
            int index = 0;
            for (var e = ex; e != null; e = e.InnerException)
            {
                index++;
                ch.Error(e.Sensitivity(), "({0}) Unexpected exception: {1}, '{2}'", index, e.Message, e.GetType());
                PrintExceptionData(ch, e, true);
                // While the message can be sensitive, we suppose the stack trace itself is not.
                if (e.StackTrace != null && e.StackTrace.Length > 0)
                    ch.Error(MessageSensitivity.None, e.StackTrace);
            }
        }

        #endregion
    }
}
