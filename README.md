# Custom Extensions to ML.net

This project proposes some extension to
[machinelearning](https://github.com/dotnet/machinelearning)
written in C#.
Work in progress.

[![TravisCI](https://travis-ci.org/xadupre/machinelearningext.svg?branch=master)](https://travis-ci.org/xadupre/machinelearningext)
[![Build status](https://ci.appveyor.com/api/projects/status/cb0xos4p3xe1bqmg?svg=true)](https://ci.appveyor.com/project/xadupre/machinelearningext)
[![CircleCI](https://circleci.com/gh/xadupre/machinelearningext.svg?style=svg)](https://circleci.com/gh/xadupre/machinelearningext)

[![TravisCI](https://travis-ci.org/sdpython/machinelearningext.svg?branch=master)](https://travis-ci.org/sdpython/machinelearningext)
[![Build status](https://ci.appveyor.com/api/projects/status/uwanivg3b5qibncs?svg=true)](https://ci.appveyor.com/project/sdpython/machinelearningext)
[![CircleCI](https://circleci.com/gh/sdpython/machinelearningext.svg?style=svg)](https://circleci.com/gh/sdpython/machinelearningext)

## Build

On windows: ``build.cmd`` or ``build ml`` to force rebuilding *machinelearning*.

On Linux: ``build.sh``.

The documentation can be build with: ``doxygen conf.dox``.

## Documentation

* [machinelearning](https://github.com/dotnet/machinelearning/tree/master/docs)
* EntryPoints
  * [Entry Points And Helper Classes](https://github.com/dotnet/machinelearning/blob/master/docs/code/EntryPoints.md)
  * [Entry Point JSON Graph format](https://github.com/dotnet/machinelearning/blob/master/docs/code/GraphRunner.md)
* Data View ansd Cursors
  * [IDataView Design Principles](https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewDesignPrinciples.md)
  * [IDataView Implementation](https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewImplementation.md)
  * [IDataView Type System](https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewTypeSystem.md)
  * [Key Values](https://github.com/dotnet/machinelearning/blob/master/docs/code/KeyValues.md)
  * [VBuffer Care and Feeding](https://github.com/dotnet/machinelearning/blob/master/docs/code/VBufferCareFeeding.md)
* [IDV File Format](https://github.com/dotnet/machinelearning/blob/master/docs/code/IdvFileFormat.md)
* [machinelearningext](docs/README.rst)

### Example 1: Inner API

This example relies on the inner API, mostly used
inside components of ML.net.

```CSharp
var env = new TlcEnvironment();
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrameIO.ReadCsv(iris, sep: '\t',
                             dtypes: new DataKind?[] { DataKind.R4 });

// We add a transform to concatenate two features in one vector columns.
var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);

// We create training data by mapping roles to columns.
var trainingData = env.CreateExamples(conc, "Feature", label: "Label");

// We create a trainer, here a One Versus Rest with a logistic regression as inner model.
var trainer = env.CreateTrainer("ova{p=lr}");

using (var ch = env.Start("test"))
{
    // We train the model.
    var pred = trainer.Train(env, ch, trainingData);

    // We compute the prediction (here with the same training data but it should not be the same).
    var scorer = ScoreUtils.GetScorer(pred, trainingData, env, null);

    // We store the predictions on a file.
    DataFrame.ViewToCsv(scorer, "iris_predictions.txt", host: env);

    // Or we could put the predictions into a dataframe.
    var dfout = DataFrameIO.ReadView(scorer);

    // And access one value...
    var v = dfout.iloc[0, 7];
    Console.WriteLine("PredictedLabel: {0}", v);
}
```

The current interface of
[DataFrame](https://github.com/xadupre/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs)
is not rich. It will improve in the future.

### Example 2: Inner API like Scikit-Learn

This is the same example but with a *ScikitPipeline* which
looks like *scikit-learn*.

```CSharp
var env = new TlcEnvironment();
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrameIO.ReadCsv(iris, sep: '\t',
                             dtypes: new DataKind?[] { DataKind.R4 });

var pipe = new ScikitPipeline(new[] { "Concat{col=Feature:Sepal_length,Sepal_width}" }, "ova{p=lr}");
pipe.Train(df, feature: "Feature", label: "Label");

var scorer = pipe.Predict(df);

var dfout = DataFrameIO.ReadView(scorer);

// And access one value...
var v = dfout.iloc[0, 7];
Console.WriteLine("PredictedLabel: {0}", v);
```

### Example 3: DataFrame in C#

The class ``DataFrame`` replicates some functionalities
datascientist are used to in others languages such as
*Python* or *R*. It is possible to do basic operations
on columns:

```CSharp
var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
var df = DataFrameIO.ReadStr(text);
df["AA+BB"] = df["AA"] + df["BB"];
Console.WriteLine(df.ToString());
```

```
AA,BB,CC,AA+BB
0,1,text,1
1,1.1,text2,2.1
```

Or:

```CSharp
df["AA2"] = df["AA"] + 10;
Console.WriteLine(df.ToString());
```

```
AA,BB,CC,AA+BB,AA2
0,1,text,1,10
1,1.1,text2,2.1,11
```

The next instructions change one value
based on a condition.

```CSharp
df.loc[df["AA"].Filter<DvInt4>(c => (int)c == 1), "CC"] = "changed";
Console.WriteLine(df.ToString());
```

```
AA,BB,CC,AA+BB,AA2
0,1,text,1,10
1,1.1,changed,2.1,11
```

A specific set of columns or rows can be extracted:

```CSharp
var view = df[df.ALL, new [] {"AA", "CC"}];
Console.WriteLine(view.ToString());
```

```
AA,CC
0,text
1,changed
```

The dataframe also allows basic filtering:

```CSharp
var view = df[df["AA"] == 0];
Console.WriteLine(view.ToString());
```

```
AA,BB,CC,AA+BB,AA2
0,1,text,1,10
```
