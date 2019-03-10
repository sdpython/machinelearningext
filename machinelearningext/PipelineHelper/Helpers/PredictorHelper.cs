// See the LICENSE file in the project root for more information.

using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Basic operations with predictors.
    /// </summary>
    public static class PredictorHelper
    {
        /// <summary>
        /// Computes the predictions using a default scorer.
        /// </summary>
        public static IDataScorerTransform Predict(IHostEnvironment env, IPredictor predictor,
                                                   RoleMappedData data, RoleMappedSchema trainSchema = null)
        {
            return CreateDefaultScorer(env, data, predictor, trainSchema);
        }


        /// <summary>
        /// Changes the default scorer for ExePythonPredictor.
        /// </summary>
        public static IDataScorerTransform CreateDefaultScorer(IHostEnvironment env,
                                                IDataView view, string featureColumn, string groupColumn,
                                                IPredictor ipredictor, RoleMappedSchema trainSchema = null)
        {
            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            if (string.IsNullOrEmpty(featureColumn))
                throw env.Except("featureColumn cannot be null");
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, featureColumn));
            if (!string.IsNullOrEmpty(groupColumn))
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Group, groupColumn));
            var data = new RoleMappedData(view, roles);
            return CreateDefaultScorer(env, data, ipredictor, trainSchema);
        }

        /// <summary>
        /// Implements a different behaviour when the predictor inherits from 
        /// IPredictorScorer (the predictor knows the scorer to use).
        /// </summary>
        public static IDataScorerTransform CreateDefaultScorer(IHostEnvironment env,
                                                RoleMappedData roles, IPredictor ipredictor,
                                                RoleMappedSchema trainSchema = null)
        {
            IDataScorerTransform scorer;
            env.CheckValue(ipredictor, "IPredictor");
            var iter = roles.Schema.GetColumnRoleNames().Where(c =>
                                        c.Key.Value != RoleMappedSchema.ColumnRole.Feature.Value &&
                                        c.Key.Value != RoleMappedSchema.ColumnRole.Group.Value);
            if (ipredictor.PredictionKind == PredictionKind.MultiClassClassification && ipredictor is IValueMapperDist)
            {
                // There is an issue with the code creating the default scorer. It expects to find a Float
                // as the output of DistType (from by IValueMapperDist)
                var newPred = new WrappedPredictorWithNoDistInterface(ipredictor);
                scorer = ScoreUtils.GetScorer(null, newPred, roles.Data, roles.Schema.Feature.Value.Name,
                                              roles.Schema.Group == null ? null : roles.Schema.Group.Value.Name,
                                              iter, env, trainSchema);
            }
            else
                scorer = ScoreUtils.GetScorer(null, ipredictor, roles.Data,
                                              roles.Schema.Feature == null ? null : roles.Schema.Feature.Value.Name,
                                              roles.Schema.Group == null ? null : roles.Schema.Group.Value.Name,
                                              iter, env, trainSchema);
            return scorer;
        }
    }
}
