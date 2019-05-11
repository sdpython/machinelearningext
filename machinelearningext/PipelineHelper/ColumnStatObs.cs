// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Retains statistiques computed in a streaming mode.
    /// </summary>
    public class ColumnStatObs
    {
        public enum StatKind
        {
            min = 1,
            max = 2,
            sum = 3,
            nb = 4,
            sum2 = 5,
            hist = 6
        }

        public StatKind kind;
        public VBuffer<double> stat;
        public Dictionary<string, long> text;
        public Dictionary<string, long> distText;
        public Dictionary<double, long> distDouble;

        public ColumnStatObs(StatKind kind)
        {
            this.kind = kind;
            stat = new VBuffer<double>();
            text = null;
            distText = null;
            distDouble = null;
        }

        #region read write

        public void Write(ModelSaveContext ctx)
        {
            ctx.Writer.Write((int)kind);
            IOHelper.Write(ctx, stat);
            IOHelper.Write(ctx, text);
            IOHelper.Write(ctx, distText);
            IOHelper.Write(ctx, distDouble);
        }

        internal ColumnStatObs(ModelLoadContext ctx)
        {
            kind = (StatKind)ctx.Reader.ReadInt32();
            stat = IOHelper.ReadVBufferDouble(ctx);
            text = IOHelper.ReadDictStringLong(ctx);
            distText = IOHelper.ReadDictStringLong(ctx);
            distDouble = IOHelper.ReadDictDoubleLong(ctx);
        }

        #endregion

        #region ToString

        public string ToJson(Dictionary<string, long> stat)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            foreach (var pair in stat)
                sb.Append(string.Format("\"{0}\":{1}, ", pair.Key.Replace("\"", "\\\""), pair.Value));
            sb.Append("}");
            return sb.ToString();
        }

        public string ToJson(Dictionary<float, long> stat)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            foreach (var pair in stat)
                sb.Append(string.Format("{0}:{1}, ", pair.Key, pair.Value));
            sb.Append("}");
            return sb.ToString();
        }

        public string ToJson(Dictionary<double, long> stat)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            foreach (var pair in stat)
                sb.Append(string.Format("{0}:{1}, ", pair.Key, pair.Value));
            sb.Append("}");
            return sb.ToString();
        }

        public string ToString(bool jsonFormat)
        {
            if (kind == StatKind.hist && distDouble == null && distText == null)
                throw Contracts.Except("No distribution.");
            var allRows = new List<string>();

            if (jsonFormat)
            {
                if (text != null && text.Count > 0)
                    allRows.Add(string.Format("{{\"{0}\": {1}}}", "stat", ToJson(text)));
                if (distDouble != null && distDouble.Count > 0)
                    allRows.Add(string.Format("{{\"{0}\": {1}}}", "distFloat", ToJson(distDouble)));
                if (distText != null && distText.Count > 0)
                    allRows.Add(string.Format("{{\"{0}\": {1}}}", "distText", ToJson(distText)));
            }
            else
            {
                // Statistics.
                if (text != null && text.Count > 0)
                {
                    var rows = text.OrderBy(c => c.Key).Select(c => string.Format("{0}:\"{1}\"", c.Value, c.Key));
                    allRows.Add(string.Format("{0}({1}): {2}", kind.ToString(), text.Count, string.Join(" ", rows)));
                }
                else if (kind != StatKind.hist)
                {
                    if (stat.Count <= 1)
                        allRows.Add(string.Format("{0}: {1}", kind.ToString(), stat.Values[0]));
                    else
                    {
                        var rows = new List<string>();
                        for (int i = 0; i < stat.Count; ++i)
                        {
                            if (!double.IsNaN(stat.Values[i]))
                                rows.Add(string.Format("{0}:{1}", i, stat.Values[i]));
                        }
                        allRows.Add(string.Format("{0}[{1}]: {2}", kind.ToString(), stat.Count, string.Join(" ", rows)));
                    }
                }

                // Histograms.
                if (distDouble != null && distDouble.Any())
                {
                    var rows = new List<string>();
                    rows.Add("Distribution: ");
                    foreach (var pair in distDouble.OrderBy(c => -c.Value))
                    {
                        if ((long)pair.Key == pair.Key)
                            rows.Add(string.Format("{0}:{1}", (long)pair.Key, pair.Value));
                        else
                            rows.Add(string.Format("{0}:{1}", pair.Key, pair.Value));
                    }
                    allRows.Add(string.Join(" ", rows));
                }
                if (distText != null && distText.Any())
                {
                    var rows = new List<string>();
                    rows.Add("Distribution: ");
                    foreach (var pair in distText.OrderBy(c => -c.Value))
                        rows.Add(string.Format("{0}:{1}", pair.Key, pair.Value));
                    allRows.Add(string.Join(" ", rows));
                }
            }

            // Final.
            return jsonFormat
                    ? string.Format("[{0}]", string.Join(", ", allRows))
                    : string.Join("\n", allRows);
        }

        #endregion

        #region init

        bool Init(double value)
        {
            if (stat.Count == 0)
            {
                stat = new VBuffer<double>(1, new double[] { double.NaN });
                if (kind == StatKind.hist)
                    distDouble = new Dictionary<double, long>();
                return true;
            }
            return false;
        }

        bool Init(string value)
        {
            if (text == null)
            {
                text = new Dictionary<string, long>();
                if (kind == StatKind.hist)
                    distText = new Dictionary<string, long>();
                return true;
            }
            return false;
        }

        bool Init(VBuffer<double> value)
        {
            if (stat.Count == 0)
            {
                stat = new VBuffer<double>();
                value.CopyToDense(ref stat);
                for (int i = 0; i < stat.Count; ++i)
                    stat.Values[i] = double.NaN;
                return true;
            }
            return false;
        }

        bool Init(VBuffer<float> value_)
        {
            if (stat.Count == 0)
            {
                var value = new VBuffer<float>();
                value_.CopyToDense(ref value);
                double[] values = value.Values == null ? null : new double[value.Values.Length];
                for (int i = 0; i < stat.Count; ++i)
                    values[i] = value.Values[i];
                stat = new VBuffer<double>(value.Length, value.Count, values, value.Indices);
                return true;
            }
            return false;
        }

        #endregion

        #region update for a float

        void Update(double value, int i, bool r)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
                return;
            if (kind != StatKind.hist && stat.Count == 1 && double.IsNaN(stat.Values[0]))
                r = true;
            if (r)
            {
                switch (kind)
                {
                    case StatKind.min:
                    case StatKind.max:
                    case StatKind.sum:
                        stat.Values[i] = value;
                        break;
                    case StatKind.sum2:
                        stat.Values[i] = value * value;
                        break;
                    case StatKind.nb:
                        stat.Values[i] = 1;
                        break;
                    case StatKind.hist:
                        distDouble[value] = 1;
                        break;
                    default:
                        throw Contracts.ExceptNotImpl();
                }
            }
            else
            {
                switch (kind)
                {
                    case StatKind.min:
                        stat.Values[i] = Math.Min(value, stat.Values[i]);
                        break;
                    case StatKind.max:
                        stat.Values[i] = Math.Max(value, stat.Values[i]);
                        break;
                    case StatKind.sum:
                        stat.Values[i] += value;
                        break;
                    case StatKind.sum2:
                        stat.Values[i] += value * value;
                        break;
                    case StatKind.nb:
                        stat.Values[i] += 1;
                        break;
                    case StatKind.hist:
                        distDouble[value] = distDouble.ContainsKey(value) ? distDouble[value] + 1 : 1;
                        break;
                    default:
                        throw Contracts.ExceptNotImpl();
                }
            }
        }

        #endregion

        #region update for a string

        void Update(string value, int i, bool r)
        {
            if (text.ContainsKey(value))
                ++text[value];
            else if (text.Count < 50)
                text[value] = 1;
            else
            {
                string[] keys;
                switch (kind)
                {
                    case StatKind.min:
                        keys = text.OrderBy(c => c.Key).Select(c => c.Key).ToArray();
                        for (int j = 0; j < keys.Length - 10; ++j)
                            text.Remove(keys[j]);
                        break;
                    case StatKind.max:
                        keys = text.OrderBy(c => c.Key).Select(c => c.Key).ToArray();
                        for (int j = 0; j < 10; ++j)
                            text.Remove(keys[j]);
                        break;
                    case StatKind.sum:
                    case StatKind.sum2:
                    case StatKind.nb:
                    case StatKind.hist:
                        keys = text.OrderBy(c => -c.Value).Select(c => c.Key).ToArray();
                        var sum = keys.Take(20).Select(k => text[k]).Sum();
                        for (int j = 0; j < 20; ++j)
                            text.Remove(keys[j]);
                        text["#QUEUE#"] = sum;
                        break;
                    default:
                        throw Contracts.ExceptNotImpl();
                }

                text[value] = 1;
            }
        }

        #endregion

        #region small functions

        public void Update(float value)
        {
            bool r = Init(value);
            Update(value, 0, r);
        }

        public void Update(double value)
        {
            bool r = Init(value);
            Update(value, 0, r);
        }

        public void Update(uint value)
        {
            bool r = Init(value);
            Update(value, 0, r);
        }

        public void Update(bool? value)
        {
            int i = value.HasValue ? (value.Value ? 1 :  0) : -10;
            bool r = Init(i);
            Update(i, 0, r);
        }

        public void Update(string value)
        {
            bool r = Init(value);
            Update(value, 0, r);
        }

        public void Update(VBuffer<float> value)
        {
            bool r = Init(value);
            if (value.IsDense)
            {
                for (int i = 0; i < stat.Count; ++i)
                    Update(value.Values[i], i, r);
            }
            else
            {
                for (int i = 0; i < value.Count; ++i)
                    Update(value.Values[i], value.Indices[i], r);
            }
        }

        public void Update(VBuffer<double> value)
        {
            bool r = Init(value);
            if (value.IsDense)
            {
                for (int i = 0; i < stat.Count; ++i)
                    Update(value.Values[i], i, r);
            }
            else
            {
                for (int i = 0; i < value.Count; ++i)
                    Update(value.Values[i], value.Indices[i], r);
            }
        }

        #endregion
    }
}