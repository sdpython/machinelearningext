import os
from pyquickhelper.filehelper.synchelper import explore_folder_iterfile
folder = "machinelearning/src"

#################
# source
################
rep = [
    #(' [BestFriend]', ' /*[BestFriend]*/'),
    #(' internal ', ' /*internal*/public '),
    #(' private protected ', ' /*private*/ protected '),
    #('(Utils.Size(', '(Microsoft.ML.Internal.Utilities.Utils.Size('),
    #(' Utils.Size(', ' Microsoft.ML.Internal.Utilities.Utils.Size('),
    #(')Hashing.HashString(', ')Microsoft.ML.Internal.Utilities.Hashing.HashString('),
    #(' private ', ' /*private*/public '),
    # ('private sealed class Mapper : OneToOneMapperBase',
    # '/*private*/protected sealed class Mapper : OneToOneMapperBase'),
    #('private sealed class Mapper<TCalibrator> : MapperBase',
    # '/*private*/protected sealed class Mapper<TCalibrator> : MapperBase'),
    ('internal abstract class SourceNameColumnBase', 
     '/*internal*/ public abstract class SourceNameColumnBase'),
    ('private protected virtual bool TryParse(string str)',
     '/*private*/ protected virtual bool TryParse(string str)'),
    ('private protected override bool TryParse(string str)',
     '/*private*/ protected override bool TryParse(string str)'),
    ('private protected bool TryParse(string str, out string extra)',
     '/*private*/ protected bool TryParse(string str, out string extra)'),
    ('public readonly int Length;',
     'public readonly int Length;\n        public int Count => _count;'
     '\n        public T[] Values => _values;\n        '
     'public int[] Indices => _indices;'),
    ('internal Column(string name, int index, bool isHidden, DataViewType type, Annotations annotations)',
     '/*internal*/public Column(string name, int index, bool isHidden, DataViewType type, Annotations annotations)'),
    ('[BestFriend]\n    internal interface IValueMapper',
     '/*[BestFriend]\n    internal*/ public interface IValueMapper'),
    ('[BestFriend]\n    internal interface IDataTransformSource',
     '/*[BestFriend]\n    internal*/ public interface IDataTransformSource'),
    ('[BestFriend]\n    internal delegate void ValueMapper<',
     '/*[BestFriend]\n    internal*/ public delegate void ValueMapper<'),
    ('[BestFriend]\n    internal abstract class Repository',
     '/*[BestFriend]\n    internal*/ public abstract class Repository'),
    ('[BestFriend]\n    internal sealed class RepositoryReader : Repository',
     '/*[BestFriend]\n    internal*/ public sealed class RepositoryReader : Repository'),
    ('[BestFriend]\n    internal interface IDataTransform :',
     '/*[BestFriend]\n    internal*/ public interface IDataTransform :'),
    ('[BestFriend]\n    internal readonly struct VersionInfo',
     '/*[BestFriend]\n    internal*/ public readonly struct VersionInfo'),
    ('[BestFriend]\n    internal sealed partial class ModelLoadContext :',
     '/*[BestFriend]\n    internal*/ public sealed partial class ModelLoadContext :'),
    ('internal sealed partial class ModelLoadContext : IDisposable',
     '/*internal*/ public sealed partial class ModelLoadContext : IDisposable'),
    ('[BestFriend]\n    internal interface IPredictor',
     '/*[BestFriend]\n    internal*/ public interface IPredictor'),
    ('[BestFriend]\n    internal enum PredictionKind',
     '/*[BestFriend]\n    internal*/ public enum PredictionKind'),
    ('[BestFriend]\n    internal interface IHaveFeatureWeights',
     '/*[BestFriend]\n    internal*/ public interface IHaveFeatureWeights'),
    ('[BestFriend]\n    internal enum InternalDataKind',
     '/*[BestFriend]\n    internal*/ public enum InternalDataKind'),
    ('private protected DataViewType(Type rawType)',
     '/*private*/ protected DataViewType(Type rawType)'),
    ('[BestFriend]\n    internal sealed class RoleMappedData',
     '/*[BestFriend]\n    internal*/ public sealed class RoleMappedData'),
    ('[BestFriend]\n    internal sealed class RoleMappedSchema',
     '/*[BestFriend]\n    internal*/ public sealed class RoleMappedSchema'),
    ('[BestFriend]\n    internal interface ITrainer',
     '/*[BestFriend]\n    internal*/ public interface ITrainer'),
    ('[BestFriend]\n    internal sealed class TrainContext',
     '/*[BestFriend]\n    internal*/ public sealed class TrainContext'),
    ('[BestFriend]\n    internal abstract class OneToOneColumn',
     '/*[BestFriend]\n    internal*/ public abstract class OneToOneColumn'),
    ('[BestFriend]\n    internal enum NormalizeOption',
     '/*[BestFriend]\n    internal*/ public enum NormalizeOption'),
    ('[BestFriend]\n    internal enum CachingOptions',
     '/*[BestFriend]\n    internal*/ public enum CachingOptions'),
    ('[BestFriend]\n    internal abstract class TransformBase :',
     '/*[BestFriend]\n    internal*/ public abstract class TransformBase :'),
    ('[BestFriend]\n    internal abstract class OnnxContext',
     '/*[BestFriend]\n    internal*/ public abstract class OnnxContext'),
    ('[BestFriend]\n    internal abstract class OnnxNode',
     '/*[BestFriend]\n    internal*/ public abstract class OnnxNode'),
    ('[BestFriend]\n    internal delegate void Signature',
     '/*[BestFriend]\n    internal*/ public delegate void Signature'),
    ('[BestFriend]\n    internal interface ISchemaBoundMapper',
     '/*[BestFriend]\n    internal*/ public interface ISchemaBoundMapper'),
    ('[BestFriend]\n    internal interface ISchemaBindableMapper',
     '/*[BestFriend]\n    internal*/ public interface ISchemaBindableMapper'),
    ('[BestFriend]\n    internal enum OnnxVersion {',
     '/*[BestFriend]\n    internal*/ public enum OnnxVersion {'),
    ('[BestFriend]\n    internal interface IDataSaver',
     '/*[BestFriend]\n    internal*/ public interface IDataSaver'),
    ('[BestFriend]\n    internal static class TrainAndScoreTransformer',
     '/*[BestFriend]\n    internal*/ public static class TrainAndScoreTransformer'),
    ('[BestFriend]\n    internal interface ICalibratorTrainer',
     '/*[BestFriend]\n    internal*/ public interface ICalibratorTrainer'),
    ('[BestFriend]\n    internal interface IDataScorerTransform',
     '/*[BestFriend]\n    internal*/ public interface IDataScorerTransform'),
    ('[BestFriend]\n    internal interface ITransformTemplate',
     '/*[BestFriend]\n    internal*/ public interface ITransformTemplate'),
    ('[BestFriend]\n    internal interface ILegacyDataLoader',
     '/*[BestFriend]\n    internal*/ public interface ILegacyDataLoader'),
    ('internal class OnnxCSharpToProtoWrapper',
     '/*internal*/ public class OnnxCSharpToProtoWrapper'),
    ('[BestFriend]\n    internal abstract class OneToOneTransformBase',
     '/*[BestFriend]\n    internal*/ public abstract class OneToOneTransformBase'),
    ('[BestFriend]\n    internal interface ICommandLineComponentFactory',
     '/*[BestFriend]\n    internal*/ public interface ICommandLineComponentFactory'),
    ('[BestFriend]\n    internal abstract class RowToRowMapperTransformBase',
     '/*[BestFriend]\n    internal*/ public abstract class RowToRowMapperTransformBase'),
    ('[BestFriend]\n    internal abstract class RowToRowTransformBase',
     '/*[BestFriend]\n    internal*/ public abstract class RowToRowTransformBase'),
    ('Contracts.Check(!IsCollection || AllowMultiple, "Collection arguments must allow multiple");',
     '// Contracts.Check(!IsCollection || AllowMultiple, "Collection arguments must allow multiple");'),
]

for name in explore_folder_iterfile(folder, pattern=".*[.]cs$"):
    if "ArrayUtils.cs" in name:
        continue
    try:
        with open(name, 'r', encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Unicode issue with file '{}'".format(name))
        continue
    if "public int Count => _count;" in content:
        continue
    content0 = content
    for k, v in rep:
        content = content.replace(k, v)
    if content0 != content:
        print("Modified: '{}'".format(name))
        with open(name, 'w', encoding="utf-8") as f:
            f.write(content)

#################
# AssemblyInfo
################
libs = []
for k in os.listdir("machinelearningext"):
    if os.path.isfile(k) or k[0].upper() != k[0] or '.' in k:
        continue
    libs.append(k)
pattern = '[assembly: InternalsVisibleTo(assemblyName: "Scikit.ML.{}")]'
patterns = [pattern.format(k) for k in libs]
patterns.append('[assembly: InternalsVisibleTo(assemblyName: "TestMachineLearningExt")]')
patterns.append('[assembly: InternalsVisibleTo(assemblyName: "TestProfileBenchmark")]')
addition = "\n".join(patterns)


rep = [
    ('[assembly: WantsToBeBestFriends]',
     '{}\n\n[assembly: WantsToBeBestFriends]'.format(addition)),
]
for name in explore_folder_iterfile(folder, pattern=".*AssemblyInfo[.]cs$"):
    try:
        with open(name, 'r', encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Unicode issue with file '{}'".format(name))
        continue
    if '[assembly: InternalsVisibleTo(assemblyName: "TLC"' not in content and \
       '[assembly: InternalsVisibleTo("TLC"' not in content and \
       '[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests"' not in content and \
       '[assembly: InternalsVisibleTo(assemblyName: "RunTests"' not in content and \
       '[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.EntryPoints"' not in content:
        continue
    if "Scikit.ML." in content:
        continue
    content0 = content
    if "[assembly: WantsToBeBestFriends]" in content:
        for k, v in rep:
            content = content.replace(k, v)
    else:
        content += "\n" + addition + "\n"
    if content0 != content:
        print("Modified: '{}'".format(name))
        with open(name, 'w', encoding="utf-8") as f:
            f.write(content)

for fold in [folder,
             'machinelearning/tools-local']:
    for name in explore_folder_iterfile(fold, pattern=".*AssemblyInfo[.]cs$"):
        try:
            with open(name, 'r', encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print("Unicode issue with file '{}'".format(name))
            continue
        content0 = content
        content = content.replace('[assembly: InternalsVisibleTo("Microsoft.ML.CodeAnalyzer.Tests, PublicKey',
                                  '[assembly: InternalsVisibleTo("Microsoft.ML.CodeAnalyzer.Tests")] //, PublicKey')
        if content0 != content:
            print("Modified: '{}'".format(name))
            with open(name, 'w', encoding="utf-8") as f:
                f.write(content)
            
for name in explore_folder_iterfile(folder, pattern=".*PublicKey[.]cs$"):
    try:
        with open(name, 'r', encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Unicode issue with file '{}'".format(name))
        continue
    content0 = content
    content = content.replace('Value = ",', 'Value = ""; //",')
    if content0 != content:
        print("Modified: '{}'".format(name))
        with open(name, 'w', encoding="utf-8") as f:
            f.write(content)

#################
# props
################

