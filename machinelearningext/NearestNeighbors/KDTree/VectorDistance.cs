// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;


namespace Scikit.ML.NearestNeighbors
{
    public static class VectorDistanceHelper
    {
        /// <summary>
        ///  Computes the L2 distance between two spare vectors.
        /// </summary>
        public static float L2(VBuffer<float> v1, VBuffer<float> v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException(string.Format("Incompatible point dimensions: expected {0}, got {1}", v1.Length, v2.Length));
            float dist = 0;
            float d;
            if (v1.IsDense && v2.IsDense)
            {
                Contracts.Assert(v1.Count == v2.Count);
                for (int i = 0; i < v1.Count; ++i)
                {
                    d = v1.Values[i] - v2.Values[i];
                    dist += d * d;
                }
            }
            else
            {
                if (v1.Count == 0 || v2.Count == 0)
                    return 0;
                int i = 0;
                int j = 0;
                while (true)
                {
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++i;
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++j;
                    if (i < v1.Count)
                    {
                        if (j < v2.Count)
                        {
                            d = v1.Values[i] - v2.Values[j];
                            dist += d * d;
                            ++i;
                            ++j;
                        }
                        else
                        {
                            d = v1.Values[i];
                            dist += d * d;
                            ++i;
                        }
                    }
                    else if (j < v2.Count)
                    {
                        d = v2.Values[j];
                        dist += d * d;
                        ++j;
                    }
                    else
                        break;
                }
            }
            return (float)Math.Sqrt(dist);
        }

        /// <summary>
        ///  Computes the L1 distance distance between two spare vectors.
        /// </summary>
        public static float L1(VBuffer<float> v1, VBuffer<float> v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException(string.Format("Incompatible point dimensions: expected {0}, got {1}", v1.Length, v2.Length));
            float dist = 0;
            float d;
            if (v1.IsDense && v2.IsDense)
            {
                Contracts.Assert(v1.Count == v2.Count);
                for (int i = 0; i < v1.Count; ++i)
                {
                    d = v1.Values[i] - v2.Values[i];
                    dist += d > 0 ? d : -d;
                }
            }
            else
            {
                if (v1.Count == 0 || v2.Count == 0)
                    return 0;
                int i = 0;
                int j = 0;
                while (true)
                {
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++i;
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++j;
                    if (i < v1.Count)
                    {
                        if (j < v2.Count)
                        {
                            d = v1.Values[i] - v2.Values[j];
                            dist += d > 0 ? d : -d;
                            ++i;
                            ++j;
                        }
                        else
                        {
                            d = v1.Values[i];
                            dist += d > 0 ? d : -d;
                            ++i;
                        }
                    }
                    else if (j < v2.Count)
                    {
                        d = v2.Values[j];
                        dist += d > 0 ? d : -d;
                        ++j;
                    }
                    else
                        break;
                }
            }
            return dist;
        }

        /// <summary>
        ///  Computes the cosine distance between two spare vectors.
        /// </summary>
        public static float Cosine(VBuffer<float> v1, VBuffer<float> v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException(string.Format("Incompatible point dimensions: expected {0}, got {1}", v1.Length, v2.Length));
            float cos = 0;
            float n1 = 0;
            float n2 = 0;
            float d;
            if (v1.IsDense && v2.IsDense)
            {
                Contracts.Assert(v1.Count == v2.Count);
                for (int i = 0; i < v1.Count; ++i)
                {
                    d = v1.Values[i] * v2.Values[i];
                    cos += d;
                    d = v1.Values[i];
                    n1 += d * d;
                    d = v2.Values[i];
                    n2 += d * d;
                }
                float pnd = n1 * n2;
                if (pnd > 0)
                    cos /= (float)Math.Sqrt(pnd);
                return cos > 1 ? 0 : 1 - cos;
            }
            else
            {
                if (v1.Count == 0 || v2.Count == 0)
                    return 0;
                int i = 0;
                int j = 0;
                while (true)
                {
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++i;
                    while (j < v2.Count && i < v1.Count && v1.Indices[i] < v2.Indices[j])
                        ++j;
                    if (i < v1.Count)
                    {
                        if (j < v2.Count)
                        {
                            d = v1.Values[i] * v2.Values[j];
                            cos += d;
                            d = v1.Values[i];
                            n1 += d * d;
                            d = v2.Values[j];
                            n2 += d * d;
                            ++i;
                            ++j;
                        }
                        else
                        {
                            d = v1.Values[i];
                            n1 += d * d;
                            ++i;
                        }
                    }
                    else if (j < v2.Count)
                    {
                        d = v2.Values[j];
                        n2 += d * d;
                        ++j;
                    }
                    else
                        break;
                }
            }
            float pn = n1 * n2;
            if (pn > 0)
                cos /= (float)Math.Sqrt(pn);
            return cos > 1 ? 0 : 1 - cos;
        }

        public static float L2Norm(VBuffer<float> v1, VBuffer<float> v2)
        {
            float d = L2(v1, v2);
            return (float)Math.Sqrt(d * d / v1.Length);
        }
    }
}
