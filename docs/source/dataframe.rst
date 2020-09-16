
.. _l-dataframe-cs:

================
DataFrames in C#
================

.. contents::
    :local:

*Dataframes* are very common in many applications
to manipulate data. :epkg:`ML.net` API implements a kind
of *StreamingDataFrame* which basically of to go through
huge volume of data by implementing a kind of 
*map/reduce* API. Most of the time, the data holds in memory
and it becomes quite convenient to manipulate it
with pseudo *SQL* methods. That's what the class
:ref:`l-cs-dataframe` implements.
Many examples can be found in unit test
`TestDataManipulation.cs <https://github.com/sdpython/machinelearningext/blob/master/machinelearningext/TestMachineLearningExt/TestDataManipulation.cs>`_.

StreamingDataFrame
==================

.. index:: SteamingDataFrame

Class *StreamingDataView* is a wrapper around :epkg:`IDataView` interface.
It adds easy conversions to :ref:`l-cs-dataframe` and easy to parse
a file or multiple files.



.. code-block:: C#

    var sdf = StreamingDataFrame.ReadCsv(new"iris.txt", sep: '\t');
    var sdf2 = StreamingDataFrame.ReadCsv(new [] {"part1.txt", "part2.txt"}, sep: '\t');

.. _l-cs-dataframe:
    
DataFrame
=========

.. index:: DataFrame

The class :epkg:`DataFrame` replicates some functionalities
datascientist are used to in others languages such as
:epkg:`Python` or :epkg:`R`. It is possible to do basic operations
on columns:

.. code-block:: C#

    var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
    var df = DataFrameIO.ReadStr(text);
    df["AA+BB"] = df["AA"] + df["BB"];
    Console.WriteLine(df.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB
    0,1,text,1
    1,1.1,text2,2.1

Or:

.. code-block:: C#

    df["AA2"] = df["AA"] + 10;
    Console.WriteLine(df.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB,AA2
    0,1,text,1,10
    1,1.1,text2,2.1,11

The next instructions change one value
based on a condition.

.. code-block:: C#

    df.loc[df["AA"].Filter<DvInt4>(c => (int)c == 1), "CC"] = "changed";
    Console.WriteLine(df.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB,AA2
    0,1,text,1,10
    1,1.1,changed,2.1,11

A specific set of columns or rows can be extracted:

.. code-block:: C#

    var view = df[df.ALL, new [] {"AA", "CC"}];
    Console.WriteLine(view.ToString());

.. code-block:: text

    AA,CC
    0,text
    1,changed

The dataframe also allows basic filtering:

.. code-block:: C#

    var view = df[df["AA"] == 0];
    Console.WriteLine(view.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB,AA2
    0,1,text,1,10
    