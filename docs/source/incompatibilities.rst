

====================
ML.net customization
====================

This repository includes :epkg:`ML.net` as a submodule,
it does not directly points to the main repository but
to modified version of it which will be eventually merged.
The submodule points to branch *ext* from
`xadupre/machinelearning <https://github.com/xadupre/machinelearning/tree/ext>`_.
Many changes changes were introduced and the custom extensions probably 
would be compile against the current nuget package 
`Microoft.ML <https://www.nuget.org/packages/Microsoft.ML/>`_
without a significant amount of work.
This will wait until :epkg:`ML.net`'s API stabilizes.

.. contents::
    :local:

Warning as errors
=================

The compilation failed due to a couple of warnings treated as error
on appveyor and Visual Studio 2015. The option was removed:
`Remove option /WX for native libraries <https://github.com/xadupre/machinelearning/commit/a7eb9efb54a0849bb76279a807ab4fef7b8752d2>`_.

Internal to public
==================

The `PR 1 <https://github.com/sdpython/machinelearning/pull/1>`_
sumarizes one part of the changes, the script
`clean_source.py <https://github.com/sdpython/machinelearningext/blob/master/clean_source.py>`_
holds the rest of the needed modifications.


