// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Scikit.ML.DataManipulation;
using Scikit.ML.ScikitAPI;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DocHelperMlExt
{
    /// <summary>
    /// Raised when a script cannot be executed.
    /// </summary>
    public class MamlException : Exception
    {
        public MamlException(string msg) : base(msg)
        {
        }
    }

    public class ExampleVector
    {
        [VectorType(3)]
        public float[] X;
    }

    /// <summary>
    /// Helpers to run scripts through maml.
    /// </summary>
    public static class MamlHelper
    {
        #region command line

        /// <summary>
        /// Runs a script. Can change the level of desired information.
        /// </summary>
        /// <param name="script">script to run</param>
        /// <param name="catch_output">catches standard output</param>
        /// <param name="verbose">2 is default</param>
        /// <returns>standard outputs</returns>
        public static string MamlScriptConsole(string script, bool catch_output, int verbose = 2)
        {
            ILogWriter logout = new LogWriter((string s) =>
            {
                // if (verbose <= 2 && s.Contains("Elapsed"))
                //    throw new Exception(s);
                Console.Write(s);
            });
            ILogWriter logerr = new LogWriter((string s) =>
            {
                // if (verbose <= 2 && s.Contains("Elapsed"))
                //    throw new Exception(s);
                if (s.Contains("Elapsed"))
                    Console.Write(s);
                else
                    Console.Error.Write(s);
            });
            using (var env = new DelegateEnvironment(verbose: verbose, outWriter: logout, errWriter: logerr))
                return MamlScript(script, catch_output, env);
        }

        /// <summary>
        /// Runs a command line with ML.Net.
        /// </summary>
        /// <param name="script">script to run</param>
        /// <param name="catch_output">capture output</param>
        /// <param name="env">delegate environment</param>
        /// <returns>output and error if captured</returns>
        public static string MamlScript(string script, bool catch_output, DelegateEnvironment env = null)
        {
            int errCode;
            string res;
            if (catch_output)
            {
                using (var capture = new StdCapture())
                {
                    errCode = DocumentationEnvironmentHelper.MainWithProgress(script, env);
                    var sout = capture.StdOut;
                    var serr = capture.StdErr;
                    if (string.IsNullOrEmpty(serr))
                        res = sout;
                    else
                        res = $"--OUT--\n{sout}\n--ERR--\n{serr}";
                }
            }
            else
            {
                errCode = DocumentationEnvironmentHelper.MainWithProgress(script, env);
                res = string.Empty;
            }
            if (errCode != 0)
                throw new MamlException($"Unable to run script, error code={errCode}\n{script}" + (string.IsNullOrEmpty(res) ? string.Empty : $"\n{res}"));
            return res;
        }

        #endregion

        #region helpers

        /// <summary>
        /// Retrieves the list of added components.
        /// </summary>
        public static Assembly[] GetAssemblies()
        {
            return ComponentHelper.GetAssemblies();
        }

        static IEnumerable<Assembly> FromAssemblyDependencies(Assembly assembly)
        {
            var assemblies = new List<Assembly> { assembly };
            var dependencyNames = assembly.GetReferencedAssemblies();
            foreach (var dependencyName in dependencyNames)
            {
                try
                {
                    assemblies.Add(Assembly.Load(dependencyName));
                }
                catch
                {
                }
            }

            foreach (var a in assemblies)
                yield return a;
        }

        public static Assembly[] GetReferencedAssemblies()
        {
            return FromAssemblyDependencies(Assembly.GetExecutingAssembly()).ToArray();
        }

        #endregion

        #region tests

        public static void TestScikitAPITrain(string name = null)
        {
            var script = "train data=__NAME__ loader=text{col=Label:R4:0 col=Features:R4:1-4 header=+} tr=mlr{maxiter=5} out=logistic_regression.zip";
            name = name ?? Path.Combine("..", "..", "..", "..", "..", "docs", "source", "iris.txt");
            name = Path.GetFullPath(name);
            if (!File.Exists(name))
            {
                TestScikitAPI2();
                return;
            }
            script = script.Replace("__NAME__", name);
            var res = MamlScript(script, true);
            if (string.IsNullOrEmpty(res))
                throw new Exception("Empty output.");
        }

        /// <summary>
        /// Runs a simple test.
        /// </summary>
        public static void TestScikitAPI()
        {
            var inputs = new[] {
                new ExampleVector() { X = new float[] { 1, 10, 100 } },
                new ExampleVector() { X = new float[] { 2, 3, 5 } },
                new ExampleVector() { X = new float[] { 2, 4, 5 } },
                new ExampleVector() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleVector() { X = new float[] { -1, -10, -100 } },
                new ExampleVector() { X = new float[] { -2, -3, -5 } },
                new ExampleVector() { X = new float[] { 3, 4, 5 } },
                new ExampleVector() { X = new float[] { 3, 4, 7 } },
            };

            using (var host = new ConsoleEnvironment(conc: 1))
            {
                ComponentHelper.AddStandardComponents(host);
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{clustersCount=2}", host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    if (predictor == null)
                        throw new Exception("Test failed: no predictor.");
                    var data2 = host.CreateStreamingDataView(inputs2);
                    var predictions = pipe.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    if (df.Shape.Item1 != 4 || df.Shape.Item2 != 12)
                        throw new Exception("Test failed: prediction failed.");
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    if (!dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"))
                        throw new Exception("Test failed: prediction failed (header).");
                }
            }
        }

        /// <summary>
        /// Runs a simple test.
        /// </summary>
        public static void TestScikitAPI2()
        {
            var res = MamlScript("? ap", true);
            if (string.IsNullOrEmpty(res))
                throw new Exception("Empty output.");
        }

        #endregion

        #region inspect

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

        public static string[] GetLoadedAssemblies(bool env = true)
        {
            if (env)
            {
                using (var e = new ConsoleEnvironment())
                {
                    ComponentHelper.AddStandardComponents(e);
                    var assemblies = AppDomain.CurrentDomain.GetAssemblies().Select(x => InfoAssembly(x)).OrderBy(c => c);
                    return assemblies.ToArray();
                }
            }
            else
            {
                var assemblies = AppDomain.CurrentDomain.GetAssemblies().Select(x => InfoAssembly(x)).OrderBy(c => c);
                return assemblies.ToArray();
            }
        }

        private static string GetAssemblyLocation(Assembly a)
        {
            try
            {
                return a.Location;
            }
            catch (Exception)
            {
                return null;
            }
        }

        public static string[] GetLoadedAssembliesLocation(bool env = true)
        {
            if (env)
            {
                using (var e = new ConsoleEnvironment())
                {
                    ComponentHelper.AddStandardComponents(e);
                    var assemblies = AppDomain.CurrentDomain.GetAssemblies()
                                                            .Select(x => GetAssemblyLocation(x))
                                                            .Where(x => !string.IsNullOrEmpty(x))
                                                            .ToArray();
                    return assemblies.ToArray();
                }
            }
            else
            {
                var assemblies = AppDomain.CurrentDomain.GetAssemblies()
                                                        .Select(x => GetAssemblyLocation(x))
                                                        .Where(x => !string.IsNullOrEmpty(x))
                                                        .ToArray();
                return assemblies.ToArray();
            }
        }

        /// <summary>
        /// Returns all kinds of models.
        /// </summary>
        public static string[] GetAllKinds()
        {
            using (var env = new ConsoleEnvironment())
            {
                ComponentHelper.AddStandardComponents(env);
                var sigs = env.ComponentCatalog.GetAllSignatureTypes();
                var typeSig = sigs.Select(t => ComponentCatalog.SignatureToString(t).ToLowerInvariant());
                return new HashSet<string>(typeSig.OrderBy(c => c)).ToArray();
            }
        }

        public class ComponentDescription
        {
            public class Argument
            {
                public string Name;
                public string Help;
                public string ShortName;
                public string DefaultValue;
                public CmdParser.ArgInfo.Arg Arg;
            }

            public string Name;
            public string ComponentName;
            public object Args;
            public string Description;
            public string[] Aliases;
            public ComponentCatalog.LoadableClassInfo Info;
            public Assembly Assembly;
            public string AssemblyName;
            public string Namespace;
            public Argument[] Arguments;
            public bool IsArgumentClass => Arguments == null;

            public ComponentDescription(ComponentCatalog.LoadableClassInfo info, object args, Assembly asse,
                                        IEnumerable<Argument> arguments)
            {
                Args = args;
                Name = info.UserName;
                Description = info.Summary;
                Info = info;
                Aliases = info.LoadNames.ToArray();
                if (asse == null)
                    asse = info.LoaderType.Assembly;
                Assembly = asse;
                AssemblyName = asse == null ? null : asse.ManifestModule.Name;
                Arguments = arguments == null ? null : arguments.OrderBy(c => c.Name).ToArray();
                Namespace = info.Type.Namespace;
                if (arguments == null)
                {
                    ComponentName = info.LoaderType.ReflectedType != null ? info.LoaderType.ReflectedType.Name : info.LoaderType.Name;

                    // Default Values.
                    object obj = null;
                    foreach (var cst in info.LoaderType.GetConstructors())
                    {
                        try
                        {
                            obj = cst.Invoke(null);
                        }
                        catch (Exception)
                        {
                        }
                    }
                    Dictionary<string, object> values = new Dictionary<string, object>();
                    if (obj != null)
                    {
                        foreach (var field in obj.GetType().GetFields())
                        {
                            var name = field.Name;
                            var value = field.GetValue(obj);
                            if (value != null)
                                values[name] = value;
                        }
                        Args = obj;
                    }

                    // Fields.
                    var fields = info.LoaderType.GetFields();
                    var largs = new List<Argument>();
                    foreach (var field in fields)
                    {
                        var arg = new Argument();
                        arg.Name = field.Name;
                        foreach (var c in field.CustomAttributes)
                        {
                            foreach (var d in c.NamedArguments)
                            {
                                if (d.MemberName.Contains("Help"))
                                    arg.Help = d.ToString();
                                else if (d.MemberName == "ShortName")
                                    arg.ShortName = d.ToString();
                            }
                        }
                        object vv = null;
                        if (values.TryGetValue(arg.Name, out vv))
                            arg.DefaultValue = vv.ToString();
                        if (!string.IsNullOrEmpty(arg.Help))
                            largs.Add(arg);
                    }
                    Arguments = largs.Count > 0 ? largs.ToArray() : null;
                }
                else
                    ComponentName = info.Type.Name;
                if (Name.Contains("+"))
                    Name = Name.Split('.').Last().Replace("+", ".");
            }

            public DataFrame GetArgsAsDataFrame()
            {
                var df = new DataFrame();
                if (Arguments != null)
                {
                    df.AddColumn("Name", Arguments.Select(c => c.Name).ToArray());
                    df.AddColumn("ShortName", Arguments.Select(c => c.ShortName).ToArray());
                    df.AddColumn("DefaultValue", Arguments.Select(c => c.DefaultValue).ToArray());
                    df.AddColumn("Help", Arguments.Select(c => c.Help).ToArray());
                }
                else if (Info.LoaderType != null)
                {
                    // Arguments class.
                    var atts = Info.LoaderType.CustomAttributes;
                    var names = new List<string>();
                    var shortNames = new List<string>();
                    var defaultValues = new List<string>();
                    var help = new List<string>();
                    foreach (var p in atts)
                    {
                        var n = "";
                        var s = "";
                        var d = "";
                        var h = "";
                        foreach (var at in p.NamedArguments)
                        {
                            if (at.MemberName == "Name")
                                n = at.ToString();
                            else if (at.MemberName == "Alias")
                                s = at.ToString();
                            else if (at.MemberName == "Default")
                                d = at.ToString();
                            else if (at.MemberName == "Desc")
                                h = at.ToString();
                        }
                        names.Add(n);
                        shortNames.Add(s);
                        defaultValues.Add(d);
                        help.Add(h);
                    }
                    var decl = Info.LoaderType;
                    df.AddColumn("Name", names.ToArray());
                    df.AddColumn("ShortName", shortNames.ToArray());
                    df.AddColumn("DefaultValue", defaultValues.ToArray());
                    df.AddColumn("Help", help.ToArray());
                }
                return df;
            }
        }

        /// <summary>
        /// Returns all kinds of parameters.
        /// </summary>
        private static IEnumerable<ComponentDescription> EnumerateComponentsParameter(bool commands)
        {
            foreach (var comp in EnumerateComponents(null))
            {
                var name = comp.Name;
                if (commands)
                {
                    if (comp.ComponentName.Contains("Command"))
                        yield return comp;
                }
                else
                {
                    if (comp.Name.EndsWith("Arguments"))
                        yield return comp;
                }
            }
        }

        /// <summary>
        /// Returns all kinds of models.
        /// </summary>
        public static IEnumerable<ComponentDescription> EnumerateComponents(string kind)
        {
            if (kind == "argument")
                foreach (var comp in EnumerateComponentsParameter(false))
                    yield return comp;
            else if (kind == "command")
                foreach (var comp in EnumerateComponentsParameter(true))
                    yield return comp;
            else
            {
                var kinds = GetAllKinds();
                if (!string.IsNullOrEmpty(kind) && !kinds.Where(c => c == kind).Any())
                    throw new ArgumentException($"Unable to find kind '{kind}' in\n{string.Join("\n", kinds)}.");

                using (var env = new ConsoleEnvironment())
                {
                    ComponentHelper.AddStandardComponents(env);
                    var sigs = env.ComponentCatalog.GetAllSignatureTypes();
                    var typeRes = typeof(object);
                    Type[] typeSigs;
                    if (string.IsNullOrEmpty(kind))
                        typeSigs = sigs.ToArray();
                    else
                        typeSigs = new[] { sigs.FirstOrDefault(t => ComponentCatalog.SignatureToString(t).ToLowerInvariant() == kind) };
                    foreach (var typeSig in typeSigs)
                    {
                        var infos = env.ComponentCatalog.GetAllDerivedClasses(typeRes, typeSig)
                            .Where(x => !x.IsHidden)
                            .OrderBy(x => x.LoadNames[0].ToLowerInvariant());
                        foreach (var info in infos)
                        {
                            var args = info.CreateArguments();
                            if (args == null)
                                yield return new ComponentDescription(info, args, null, null);
                            else
                            {
                                var asse = args.GetType().Assembly;

                                var parsedArgs = CmdParser.GetArgInfo(args.GetType(), args).Args;
                                var arguments = new List<ComponentDescription.Argument>();
                                foreach (var arg in parsedArgs)
                                {
                                    var a = new ComponentDescription.Argument()
                                    {
                                        Name = arg.LongName,
                                        ShortName = arg.ShortNames == null || !arg.ShortNames.Any() ? null : arg.ShortNames.First(),
                                        DefaultValue = arg.DefaultValue == null ? null : arg.DefaultValue.ToString(),
                                        Help = arg.HelpText,
                                        Arg = arg,
                                    };
                                    arguments.Add(a);
                                }

                                var cmp = new ComponentDescription(info, args, asse, arguments);
                                yield return cmp;
                            }
                        }
                    }
                }
            }
        }

        #endregion
    }
}
