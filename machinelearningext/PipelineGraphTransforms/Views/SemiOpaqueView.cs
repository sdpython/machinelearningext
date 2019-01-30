//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Similar to OpaqueDataView implementation. It provides a barrier for data pipe optimizations.
    /// Used in cross validatation to generate the train/test pipelines for each fold.
    /// However, it gives access to previous tag. The class can overwrite a schema.
    /// </summary>
    public sealed class SemiOpaqueDataView : IDataView
    {
        private readonly IDataView _source;
        private readonly Schema _newSchema;

        public IDataView SourceTags { get { return _source; } }

        public SemiOpaqueDataView(IDataView source, Schema newSchema = null)
        {
            _source = source;
            _newSchema = newSchema;
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public Schema Schema
        {
            get { return _newSchema == null ? _source.Schema : _newSchema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
        {
            return _source.GetRowCursor(columnsNeeded, rand);
        }

        public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
        {
            return _source.GetRowCursorSet(columnsNeeded, n, rand);
        }
    }
}
