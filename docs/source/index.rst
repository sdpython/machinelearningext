
===========================
Custom Extensions to ML.net
===========================

This documentation goes through some elements useful
to whoever wants to extend :epkg:`ML.net`
which is a open source machine learning library mostly
written in :epkg:`C#` and implemented by :epkg:`Microsoft`.
It contains documentation pages about the :epkg:`ML.net`
itself taken from the main repository or automatically
built from it extended with custom code or experiments.
These extensions are not yet available as a nuget package
but the source can obtained by forking
`sdpython/machinelearningext <https://github.com/sdpython/machinelearningext>`_.

.. toctree::
    :maxdepth: 2

    introduction
    dataframe
    commandline
    components/index
    machinelearning_docs
    concepts
    apicsharpdoc
    aonnx
    incompatibilities    

You can also navigate through the documentation
with the :ref:`genindex` or the :ref:`search`.
The documentation was generating on:

.. runpython::

    from datetime import datetime
    print(str(datetime.now()).split()[0])

The nuget package
`Scikit.ML.DataFrame <https://www.nuget.org/packages/Scikit.ML.DataFrame/>`_
(`github <https://github.com/sdpython/machinelearning_dataframe>`_)
was built from an early version of this patchwork
which still includes
:ref:`Scikit.ML.DataManipulation.DataFrame <l-dataframe-cs>`.


.. image:: https://travis-ci.org/sdpython/machinelearningext.svg?branch=master
    :target: https://travis-ci.org/sdpython/machinelearningext
    :alt: Build status
.. image:: https://ci.appveyor.com/api/projects/status/uwanivg3b5qibncs?svg=true
    :target: https://ci.appveyor.com/project/sdpython/machinelearningext
    :alt: Build Status Windows
.. image:: https://circleci.com/gh/sdpython/machinelearningext/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/machinelearningext/tree/master
