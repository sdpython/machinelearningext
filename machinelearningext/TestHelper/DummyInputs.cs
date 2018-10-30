﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.ProductionPrediction;


namespace Scikit.ML.TestHelper
{
    public class InputOutput
    {
        [VectorType(2)]
        public float[] X;
        public int Y;

        public static InputOutput[] CreateInputs()
        {
            var inputs = new InputOutput[] {
                new InputOutput() { X = new float[] { 0, 1 }, Y = 1 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 0 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 2 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 3 },
            };
            return inputs;
        }
    }

    public class InputOutput2
    {
        [VectorType(2)]
        public float[] X2;
    }

    public class InputOutputU : IClassWithGetter<InputOutputU>, IClassWithSetter<InputOutputU>
    {
        [VectorType(2)]
        public float[] X;
        public uint Y;

        public Delegate GetGetter(int col)
        {
            switch (col)
            {
                case 0:
                    {
                        ValueGetterInstance<InputOutputU, float[]> dele = (ref InputOutputU self, ref float[] x) => { x = self.X; };
                        return dele;
                    }
                case 1:
                    {
                        ValueGetterInstance<InputOutputU, uint> dele = (ref InputOutputU self, ref uint y) => { y = self.Y; };
                        return dele;
                    }
                default:
                    throw new System.Exception($"No field number {col}.");
            }
        }

        public Delegate[] GetCursorGetter(IRowCursor cursor)
        {
            return new Delegate[]
            {
                cursor.GetGetter<float[]>(0),
                cursor.GetGetter<uint>(1),
            };
        }

        public void Set(Delegate[] delegates)
        {
            var del1 = delegates[0] as ValueGetter<float[]>;
            del1(ref X);
            var del2 = delegates[1] as ValueGetter<uint>;
            del2(ref Y);
        }
    }

    public class ExampleA0
    {
        public float X;
    }

    public class ExampleA
    {
        [VectorType(3)]
        public float[] X;
    }

    public class ExampleXY
    {
        public float X;
        public float Y;
    }

    public class ExampleASparse
    {
        [VectorType(5)]
        public VBuffer<float> X;
    }

    public class ExampleValueMapper : IValueMapper
    {
        public ColumnType InputType { get { return new VectorType(NumberType.R4, 2); } }
        public ColumnType OutputType { get { return NumberType.R4; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, float> GetMapper_()
        {
            return (ref VBuffer<float> X, ref float y) => { y = X.Values.Sum(); };
        }
    }

    public class SHExampleA
    {
        [VectorType(2)]
        public float[] X;
    }

    public class SHExampleASparse
    {
        [VectorType()]
        public VBuffer<float> X;
    }

    public class SHExampleValueMapper : IValueMapper
    {
        public ColumnType InputType { get { return new VectorType(NumberType.R4, 2); } }
        public ColumnType OutputType { get { return NumberType.R4; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, float> GetMapper_()
        {
            return (ref VBuffer<float> X, ref float y) => { y = X.Values.Sum(); };
        }
    }

    public class ExampleValueMapperVector : IValueMapper
    {
        public ColumnType InputType { get { return new VectorType(NumberType.R4, 2); } }
        public ColumnType OutputType { get { return new VectorType(NumberType.R4); } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper_()
        {
            return (ref VBuffer<float> X, ref VBuffer<float> y) =>
            {
                y = new VBuffer<float>(X.Length, X.Values.Select(c => c).ToArray());
            };
        }
    }

    public class SentimentData : IClassWithGetter<SentimentData>
    {
        [ColumnName("Label")]
        public bool Sentiment;
        public string SentimentText;

        public Delegate GetGetter(int col)
        {
            switch (col)
            {
                case 0:
                    {
                        ValueGetterInstance<SentimentData, float> dele =
                            (ref SentimentData self, ref float x) => { x = self.Sentiment ? 1f : 0f; };
                        return dele;
                    }
                case 1:
                    {
                        ValueGetterInstance<SentimentData, ReadOnlyMemory<char>> dele =
                            (ref SentimentData self, ref ReadOnlyMemory<char> x) => { x = new ReadOnlyMemory<char>(self.SentimentText.ToCharArray()); };
                        return dele;
                    }
                default:
                    throw new Exception($"No available column for index {col}.");
            }
        }
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
        public float Score;
    }
}
