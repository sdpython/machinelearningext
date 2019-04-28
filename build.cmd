@echo off

@echo [build.cmd] build machinelearning
cd machinelearning
if "%1"=="ml" goto compileml:
if exist bin\x64.Release goto mldeb:
:compileml:
git submodule update --init

cmd /C build.cmd
cmd /C build.cmd -Release
:mldeb:
cd ..

if "%1"=="ml" goto docopy:
if not exist machinelearning\bin\x64.Debug goto end:

:docopy:
@echo [build.cmd] Publish Release
if not exist machinelearning\dist\Release mkdir machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Ensemble\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.FastTree\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.ImageAnalytics\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.KMeansClustering\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Legacy\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.LightGBM\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Mkl.Components\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Mkl.Components.StaticPipe\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.OnnxConverter\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.OnnxTransformer\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.PCA\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.PipelineInference\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.StandardTrainers\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Sweeper\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.TensorFlow\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.TimeSeries\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\packages\lightgbm\2.2.3\runtimes\win-x64\native machinelearning\bin\x64.Release\Native
copy machinelearning\packages\google.protobuf\3.5.1\lib\netstandard1.0 machinelearning\bin\x64.Release\Native
copy machinelearning\packages\mlnetmkldeps\0.0.0.9\runtimes\win-x64\native machinelearning\bin\x64.Release\Native
copy machinelearning\packages\system.codedom\4.5.0\lib\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\x64.Release\Native\*.dll machinelearning\dist\Release

@echo [build.cmd] Publish Debug
if not exist machinelearning\dist\Debug mkdir machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Ensemble\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.FastTree\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.KMeansClustering\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Legacy\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.LightGBM\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.ImageAnalytics\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Mkl.Components\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Mkl.Components.StaticPipe\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.OnnxConverter\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.OnnxTransformer\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.PCA\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.PipelineInference\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.StandardTrainers\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Sweeper\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.TimeSeries\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.TensorFlow\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\packages\lightgbm\2.2.3\runtimes\win-x64\native machinelearning\bin\x64.Debug\Native
copy machinelearning\packages\google.protobuf\3.5.1\lib\netstandard1.0 machinelearning\bin\x64.Debug\Native
copy machinelearning\packages\mlnetmkldeps\0.0.0.9\runtimes\win-x64\native machinelearning\bin\x64.Debug\Native
copy machinelearning\packages\system.codedom\4.5.0\lib\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearning\dist\Debug

@echo [build.cmd] build machinelearningext
cd machinelearningext
cmd /C dotnet build -c Debug
cmd /C dotnet build -c Release
cd ..

:finalcopy:
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearningext\bin\AnyCPU.Debug\TestProfileBenchmark\netcoreapp2.1
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearningext\bin\AnyCPU.Debug\TestMachineLearningExt\netcoreapp2.1
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearningext\bin\AnyCPU.Debug\DocHelperMlExt\netstandard2.0

copy machinelearning\bin\x64.Release\Native\*.dll machinelearningext\bin\AnyCPU.Release\TestProfileBenchmark\netcoreapp2.1
copy machinelearning\bin\x64.Release\Native\*.dll machinelearningext\bin\AnyCPU.Release\TestMachineLearningExt\netcoreapp2.1
copy machinelearning\bin\x64.Release\Native\*.dll machinelearningext\bin\AnyCPU.Release\DocHelperMlExt\netstandard2.0

rem 4.5.0 or 4.4.0...
copy machinelearning\packages\system.codedom\4.5.0\lib\netstandard2.0\*.dll machinelearningext\bin\AnyCPU.Release\DocHelperMlExt\netstandard2.0
copy machinelearning\packages\system.codedom\4.5.0\lib\netstandard2.0\*.dll machinelearningext\bin\AnyCPU.Debug\DocHelperMlExt\netstandard2.0

:end:
if not exist machinelearning\bin\x64.Debug @echo [build.cmd] Cannot build.
@echo [build.cmd] Completed.

