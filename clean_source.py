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
    ('internal sealed partial class ModelLoadContext : IDisposable',
     '/*internal*/public sealed partial class ModelLoadContext : IDisposable'),
    ('[BestFriend]\n    internal interface IValueMapper',
     '/*[BestFriend]\n    internal*/ public interface IValueMapper'),
    ('[BestFriend]\n    internal delegate void ValueMapper<',
     '/*[BestFriend]\n    internal*/ public delegate void ValueMapper<'),
    ('[BestFriend]\n    internal abstract class Repository',
     '/*[BestFriend]\n    internal*/ public abstract class Repository'),
    ('[BestFriend]\n    interface IDataTransform :',
     '/*[BestFriend]\n    internal*/ public IDataTransform :'),
    ('[BestFriend]\n    internal readonly struct VersionInfo',
     '/*[BestFriend]\n    internal*/ public readonly struct VersionInfo'),
    ('[BestFriend]\n    internal sealed partial class ModelLoadContext :',
     '/*[BestFriend]\n    internal*/ public sealed partial class ModelLoadContext :'),
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
    if os.path.isfile(k) or k[0].upper() != k[0] or 'Test' in k or '.' in k:
        continue
    libs.append(k)
pattern = '[assembly: InternalsVisibleTo(assemblyName: "Scikit.ML.{}")]'
addition = "\n".join(pattern.format(k) for k in libs)


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
       '[assembly: InternalsVisibleTo("TLC"' not in content:
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

