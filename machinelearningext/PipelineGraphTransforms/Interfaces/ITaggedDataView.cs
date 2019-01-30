//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineTransforms;


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// A view in a pipeline can be tagged.
    /// Tagged view can retrieve and put back in the pipeline.
    /// </summary>
    public interface ITaggedDataView : IDataView
    {
        /// <summary>
        /// Retrieve all views.
        /// </summary>
        /// <param name="recursive">look into previous views. Works only if it is a transform.</param>
        IEnumerable<Tuple<string, ITaggedDataView>> EnumerateTaggedView(bool recursive = true);

        /// <summary>
        /// When the user selects another view, we need to keep a pointer on this view and the view set aside.
        /// </summary>
        IEnumerable<Tuple<string, ITaggedDataView>> ParallelViews { get; }

        /// <summary>
        /// References new tagged views.
        /// </summary>
        void AddRange(IEnumerable<Tuple<string, ITaggedDataView>> tagged);

        /// <summary>
        /// A tag view can host a predictor.
        /// </summary>
        IPredictor TaggedPredictor { get; }
    }

    /// <summary>
    /// Helpers for tags.
    /// </summary>
    public static class TagHelper
    {
        public enum GraphPositionEnum
        {
            first = 0,
            middle = 1,
            last = 2
        };

        /// <summary>
        /// Enumerates all views in a pipeline between two views.
        /// </summary>
        /// <param name="first">consider as the first view, nothing beyond this point is considered</param>
        /// <param name="view">last view</param>
        /// <param name="depth">depth: distance from the last node</param>
        public static IEnumerable<Tuple<IDataView, GraphPositionEnum>> EnumerateAllViews(IDataView view, IDataView[] first = null, int depth = 0)
        {
            bool isfirst = false;
            if (first != null)
            {
                foreach (var v in first)
                {
                    if (v == view)
                    {
                        isfirst = true;
                        break;
                    }
                }
            }

            if (isfirst)
                yield return new Tuple<IDataView, GraphPositionEnum>(view, GraphPositionEnum.first);
            else
            {
                yield return new Tuple<IDataView, GraphPositionEnum>(view, depth == 0 ? GraphPositionEnum.last : GraphPositionEnum.middle);
                var tagged = view as ITaggedDataView;
                if (tagged != null)
                {
                    foreach (var par in tagged.ParallelViews)
                        foreach (var v in EnumerateAllViews(par.Item2, first, depth + 1))
                            yield return v;
                }
                else
                {
                    var trans = view as IDataTransform;
                    if (trans != null)
                        foreach (var v in EnumerateAllViews(trans.Source, first, depth + 1))
                            yield return v;
                }
            }
        }

        /// <summary>
        /// Implements get tagged view.
        /// </summary>
        /// <param name="recursive">recursive enumeration</param>
        /// <param name="view">IDataView</param>
        public static IEnumerable<Tuple<string, ITaggedDataView>> EnumerateTaggedView(bool recursive, IDataView view)
        {
            var taggedCheck = new Dictionary<string, ITaggedDataView>();
            var taggedList = new List<IDataView>();
            taggedList.Add(view);
            int position = 0;
            while (position < taggedList.Count)
            {
                view = taggedList[position];
                ++position;

                var taggedView = view as ITaggedDataView;
                if (taggedView != null && taggedView.ParallelViews != null)
                {
                    foreach (var pv in taggedView.ParallelViews)
                    {
                        if (taggedCheck.ContainsKey(pv.Item1))
                        {
                            if (!taggedCheck[pv.Item1].Equals(pv.Item2))
                                throw Contracts.Except("Tag '{0}' is used to tagged more than one view", pv.Item1);
                            continue;
                        }
                        taggedCheck[pv.Item1] = pv.Item2;
                        yield return pv;

                        if (recursive)
                            taggedList.Add(pv.Item2);
                    }
                }

                if (recursive)
                {
                    var xf = view as IDataTransform;
                    if (xf != null && xf.Source != null)
                        taggedList.Add(xf.Source);
                    var ov = view as SemiOpaqueDataView;
                    if (ov != null && ov.SourceTags != null)
                        taggedList.Add(ov.SourceTags);
                    var xa = view as AbstractSimpleTransformTemplate;
                    if (xa != null && xa.SourceEnd != null)
                        taggedList.Add(xa.SourceEnd);
                }
            }
        }

        /// <summary>
        /// Looks for a predictor among tagged predictors.
        /// If the tag ends by .zip, it assumes it is a file.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="input">IDataView</param>
        /// <param name="tag">tag name</param>
        /// <returns>predictor</returns>
        public static IPredictor GetTaggedPredictor(IHostEnvironment env, IDataView input, string tag)
        {
            if (string.IsNullOrEmpty(tag))
                throw env.Except("tag must not be null.");
            if (tag.EndsWith(".zip"))
            {
                using (Stream modelStream = new FileStream(tag, FileMode.Open, FileAccess.Read))
                {
                    var ipred = ComponentCreation.LoadPredictorOrNull(env, modelStream);
                    return ipred;
                }
            }
            else
            {
                var tagged = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == tag);
                if (!tagged.Any())
                    throw env.Except("Unable to find any view with tag '{0}'.", tag);
                if (tagged.Skip(1).Any())
                {
                    var allTagged = TagHelper.EnumerateTaggedView(true, input).ToArray();
                    throw env.Except("Ambiguous tag '{0}' - {1}/{2}.", tag,
                        allTagged.Where(c => c.Item1 == tag).Count(), allTagged.Length);
                }
                var predictor = tagged.First().Item2.TaggedPredictor;
                if (predictor == null)
                    env.Except("Tagged view '{0}' does not host a predictor.", tag);
                return predictor;
            }
        }

        /// <summary>
        /// Implements get tagged view.
        /// </summary>
        public static List<Tuple<string, ITaggedDataView>> Reconcile(IEnumerable<Tuple<string, ITaggedDataView>> viewSet)
        {
            foreach (var pair in viewSet)
            {
                var tag = pair.Item2 as ITaggedDataView;
                if (tag == null)
                    throw Contracts.Except("Tagged view '{0}' is not ITaggedDataView.", pair.Item1);
            }
            var tagged = new Dictionary<string, ITaggedDataView>();
            foreach (var item in viewSet)
            {
                if (!tagged.ContainsKey(item.Item1))
                    tagged[item.Item1] = item.Item2;
                else
                {
                    var o = tagged[item.Item1];
                    if (!o.Equals(item.Item2))
                        throw Contracts.Except("Tags cannot be reconciled for tag='{0}'.", item.Item1);
                }
            }
            return tagged.Select(c => new Tuple<string, ITaggedDataView>(c.Key, c.Value)).ToList();
        }
    }
}
