﻿// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements operator for DataFrame for many types.
    /// </summary>
    public static class DataFrameOpMultiplicationHelper
    {
        public const string OperationName = "Multiplication";

        #region Operation between a column and a value.

        static void Operation<T1, T3>(NumericColumn c1, out T1[] a, out DataColumn<T3> res)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var c1o = c1.Column as DataColumn<T1>;
            if (c1o is null)
                throw new DataTypeError(string.Format("{0} not implemented for type {1}.", OperationName, c1.GetType()));
            res = new DataColumn<T3>(c1.Length);
            a = c1o.Data;
        }

        public static NumericColumn Operation(NumericColumn c1, int value)
        {
            if (c1.Kind.IsVector())
                throw new NotImplementedException();
            else
            {
                switch (c1.Kind.RawKind())
                {
                    case DataKind.I4:
                        {
                            int[] a;
                            DataColumn<int> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.I8:
                        {
                            long[] a;
                            DataColumn<long> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R4:
                        {
                            float[] a;
                            DataColumn<float> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * (float)value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R8:
                        {
                            double[] a;
                            DataColumn<double> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * (double)value);
                            return new NumericColumn(res);
                        }
                    default:
                        throw new DataTypeError(string.Format("{0} not implemented for column {1}.", OperationName, c1.Kind));
                }
            }
        }

        public static NumericColumn Operation(NumericColumn c1, long value)
        {
            if (c1.Kind.IsVector())
                throw new NotImplementedException();
            else
            {
                switch (c1.Kind.RawKind())
                {
                    case DataKind.I4:
                        {
                            long[] a;
                            DataColumn<long> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.I8:
                        {
                            long[] a;
                            DataColumn<long> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R4:
                        {
                            float[] a;
                            DataColumn<float> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * (float)value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R8:
                        {
                            double[] a;
                            DataColumn<double> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * (double)value);
                            return new NumericColumn(res);
                        }
                    default:
                        throw new DataTypeError(string.Format("{0} not implemented for column {1}.", OperationName, c1.Kind));
                }
            }
        }

        public static NumericColumn Operation(NumericColumn c1, float value)
        {
            if (c1.Kind.IsVector())
                throw new NotImplementedException();
            else
            {
                switch (c1.Kind.RawKind())
                {
                    case DataKind.I4:
                        {
                            int[] a;
                            DataColumn<float> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, (int)a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.I8:
                        {
                            long[] a;
                            DataColumn<float> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, (Int64)a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R4:
                        {
                            float[] a;
                            DataColumn<float> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R8:
                        {
                            double[] a;
                            DataColumn<double> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    default:
                        throw new DataTypeError(string.Format("{0} not implemented for column {1}.", OperationName, c1.Kind));
                }
            }
        }

        public static NumericColumn Operation(NumericColumn c1, double value)
        {
            if (c1.Kind.IsVector())
                throw new NotImplementedException();
            else
            {
                switch (c1.Kind.RawKind())
                {
                    case DataKind.I4:
                        {
                            int[] a;
                            DataColumn<double> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, (int)a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R4:
                        {
                            float[] a;
                            DataColumn<double> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    case DataKind.R8:
                        {
                            double[] a;
                            DataColumn<double> res;
                            Operation(c1, out a, out res);
                            for (int i = 0; i < res.Length; ++i)
                                res.Set(i, a[i] * value);
                            return new NumericColumn(res);
                        }
                    default:
                        throw new DataTypeError(string.Format("{0} not implemented for column {1}.", OperationName, c1.Kind));
                }
            }
        }

        #endregion

        #region Operation between two columns.

        static void Operation<T1, T2, T3>(NumericColumn c1, NumericColumn c2,
                                         out T1[] a, out T2[] b, out DataColumn<T3> res)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var c1o = c1.Column as DataColumn<T1>;
            var c2o = c2.Column as DataColumn<T2>;
            if (c1o is null || c2o is null)
                throw new DataTypeError(string.Format("{0} not implemented for {1}, {2}.", OperationName, c1.Kind, c2.Kind));
            res = new DataColumn<T3>(c1.Length);
            a = c1o.Data;
            b = c2o.Data;
        }

        public static NumericColumn Operation(NumericColumn c1, NumericColumn c2)
        {
            if (c1.Kind.IsVector())
                throw new NotImplementedException();
            else
            {
                switch (c1.Kind.RawKind())
                {
                    case DataKind.I4:
                        if (c2.Kind.IsVector())
                            throw new NotImplementedException();
                        else
                        {
                            switch (c2.Kind.RawKind())
                            {
                                case DataKind.I4:
                                    {
                                        int[] a;
                                        int[] b;
                                        DataColumn<int> res;
                                        Operation(c1, c2, out a, out b, out res);
                                        for (int i = 0; i < res.Length; ++i)
                                            res.Set(i, a[i] * b[i]);
                                        return new NumericColumn(res);
                                    }
                                case DataKind.R4:
                                    {
                                        int[] a;
                                        float[] b;
                                        DataColumn<float> res;
                                        Operation(c1, c2, out a, out b, out res);
                                        for (int i = 0; i < res.Length; ++i)
                                            res.Set(i, (int)a[i] * b[i]);
                                        return new NumericColumn(res);
                                    }
                                default:
                                    throw new DataTypeError(string.Format("{0} not implemented for {1}, {2}.", OperationName, c1.Kind, c2.Kind));
                            }
                        }
                    case DataKind.R4:
                        if (c2.Kind.IsVector())
                            throw new NotImplementedException();
                        else
                        {
                            switch (c2.Kind.RawKind())
                            {
                                case DataKind.I4:
                                    {
                                        float[] a;
                                        int[] b;
                                        DataColumn<float> res;
                                        Operation(c1, c2, out a, out b, out res);
                                        for (int i = 0; i < res.Length; ++i)
                                            res.Set(i, a[i] * (int)b[i]);
                                        return new NumericColumn(res);
                                    }
                                case DataKind.R4:
                                    {
                                        float[] a;
                                        float[] b;
                                        DataColumn<float> res;
                                        Operation(c1, c2, out a, out b, out res);
                                        for (int i = 0; i < res.Length; ++i)
                                            res.Set(i, a[i] * b[i]);
                                        return new NumericColumn(res);
                                    }
                                default:
                                    throw new DataTypeError(string.Format("{0} not implemented for {1}, {2}.", OperationName, c1.Kind, c2.Kind));
                            }
                        }
                    default:
                        throw new DataTypeError(string.Format("{0} not implemented for {1} for left element.", OperationName, c1.Kind));
                }
            }
        }

        #endregion
    }
}
