﻿// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Ext.PipelineHelper
{
    /// <summary>
    /// The transform is trainable: it must be trained on the training data.
    /// </summary>
    public interface ITrainableTransform
    {
        void Estimate();
    }
}