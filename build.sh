echo

echo [build.sh] clean sources
python3 -u clean_source.py

echo [build.sh] build machinelearning
cd machinelearning
bash build.sh -release
bash build.sh -debug
cd ..

mkdir machinelearning/dist
mkdir machinelearning/dist/Debug
mkdir machinelearning/dist/Release

echo
echo [build.sh] copy release binaries
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.Ensemble/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.FastTree/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.ImageAnalytics/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.KMeansClustering/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.Maml/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.Mkl.Components/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.Mkl.Components.StaticPipe/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.LightGbm/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.OnnxConverter/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.OnnxTransformer/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.OnnxTransformer.StaticPipe/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.PCA/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.StandardTrainers/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.Sweeper/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.TensorFlow/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/AnyCPU.Release/Microsoft.ML.TimeSeries/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/packages/lightgbm/2.2.3/runtimes/linux-x64/native/*.so machinelearning/bin/x64.Release/Native
cp machinelearning/packages/mlnetmkldeps/0.0.0.9/runtimes/linux-x64/native/*.so machinelearning/bin/x64.Release/Native
cp machinelearning/packages/google.protobuf/3.5.1/lib/netstandard1.0/*.* machinelearning/bin/x64.Release/Native
cp machinelearning/packages/system.codedom/4.5.0/lib/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/x64.Release/Native/*.* machinelearning/dist/Release

echo
echo [build.sh] copy debug binaries
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.Ensemble/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.FastTree/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.ImageAnalytics/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.KMeansClustering/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.Maml/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.Mkl.Components/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.Mkl.Components.StaticPipe/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.LightGbm/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.OnnxConverter/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.OnnxTransformer/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.OnnxTransformer.StaticPipe/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.PCA/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.StandardTrainers/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.Sweeper/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.TensorFlow/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/bin/AnyCPU.Debug/Microsoft.ML.TimeSeries/netstandard2.0/*.dll machinelearning/dist/Debug
cp machinelearning/packages/lightgbm/2.2.3/runtimes/linux-x64/native/*.so machinelearning/bin/x64.Release/Native
cp machinelearning/packages/mlnetmkldeps/0.0.0.9/runtimes/linux-x64/native/*.so machinelearning/bin/x64.Release/Native
cp machinelearning/packages/google.protobuf/3.5.1/lib/netstandard1.0/*.* machinelearning/bin/x64.Release/Native
cp machinelearning/packages/system.codedom/4.5.0/lib/netstandard2.0/*.dll machinelearning/dist/Release
cp machinelearning/bin/x64.Release/Native/*.* machinelearning/dist/Release

echo
echo [build.sh] copy native binaries
cp machinelearning/bin/x64.Debug/Native/*.* machinelearning/dist/Debug
cp machinelearning/bin/x64.Release/Native/*.* machinelearning/dist/Release

echo
echo [build.sh] build machinelearningext
cd machinelearningext
dotnet build -c Debug
dotnet build -c Release
cd ..

echo
echo [build.sh] final copy
cp machinelearning/bin/x64.Debug/Native/*.* machinelearningext/bin/AnyCPU.Debug/TestProfileBenchmark/netcoreapp2.1
cp machinelearning/bin/x64.Debug/Native/*.* machinelearningext/bin/AnyCPU.Debug/TestMachineLearningExt/netcoreapp2.1
cp machinelearning/bin/x64.Debug/Native/*.* machinelearningext/bin/AnyCPU.Debug/DocHelperMlExt/netstandard2.0
cp machinelearning/bin/x64.Release/Native/*.* machinelearningext/bin/AnyCPU.Release/TestProfileBenchmark/netcoreapp2.1
cp machinelearning/bin/x64.Release/Native/*.* machinelearningext/bin/AnyCPU.Release/TestMachineLearningExt/netcoreapp2.1
cp machinelearning/bin/x64.Release/Native/*.* machinelearningext/bin/AnyCPU.Release/DocHelperMlExt/netstandard2.0
