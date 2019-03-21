// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
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

    public class InputOutputOut
    {
        [VectorType(2)]
        public float[] Xout;
        public int Yout;

        public static InputOutputOut[] CreateInputs()
        {
            var inputs = new InputOutputOut[] {
                new InputOutputOut() { Xout = new float[] { 0, 1 }, Yout = 1 },
                new InputOutputOut() { Xout = new float[] { 0, 1 }, Yout = 0 },
                new InputOutputOut() { Xout = new float[] { 0, 1 }, Yout = 2 },
                new InputOutputOut() { Xout = new float[] { 0, 1 }, Yout = 3 },
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

        private DataViewSchema.Column _dc(int i)
        {
            return new DataViewSchema.Column(null, i, false, null, null);
        }

        public Delegate[] GetCursorGetter(DataViewRowCursor cursor)
        {
            return new Delegate[]
            {
                cursor.GetGetter<float[]>(_dc(0)),
                cursor.GetGetter<uint>(_dc(1)),
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
        public DataViewType InputType { get { return new VectorType(NumberDataViewType.Single, 2); } }
        public DataViewType OutputType { get { return NumberDataViewType.Single; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, float> GetMapper_()
        {
            return (in VBuffer<float> X, ref float y) => { y = X.Values.Sum(); };
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
        public DataViewType InputType { get { return new VectorType(NumberDataViewType.Single, 2); } }
        public DataViewType OutputType { get { return NumberDataViewType.Single; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, float> GetMapper_()
        {
            return (in VBuffer<float> X, ref float y) => { y = X.Values.Sum(); };
        }
    }

    public class ExampleValueMapperVector : IValueMapper
    {
        public DataViewType InputType { get { return new VectorType(NumberDataViewType.Single, 2); } }
        public DataViewType OutputType { get { return new VectorType(NumberDataViewType.Single); } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper_()
        {
            return (in VBuffer<float> X, ref VBuffer<float> y) =>
            {
                y = new VBuffer<float>(X.Length, X.Values.Select(c => c).ToArray());
            };
        }
    }

    public class SentimentDataBoolFloat : IClassWithGetter<SentimentDataBoolFloat>
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
                        ValueGetterInstance<SentimentDataBoolFloat, float> dele =
                            (ref SentimentDataBoolFloat self, ref float x) => { x = self.Sentiment ? 1f : 0f; };
                        return dele;
                    }
                case 1:
                    {
                        ValueGetterInstance<SentimentDataBoolFloat, ReadOnlyMemory<char>> dele =
                            (ref SentimentDataBoolFloat self, ref ReadOnlyMemory<char> x) => { x = new ReadOnlyMemory<char>(self.SentimentText.ToCharArray()); };
                        return dele;
                    }
                default:
                    throw new Exception($"No available column for index {col}.");
            }
        }
    }

    public class SentimentDataFloat : IClassWithGetter<SentimentDataFloat>
    {
        [ColumnName("Label")]
        public float Sentiment;
        public string SentimentText;

        public Delegate GetGetter(int col)
        {
            switch (col)
            {
                case 0:
                    {
                        ValueGetterInstance<SentimentDataFloat, float> dele =
                            (ref SentimentDataFloat self, ref float x) => { x = self.Sentiment; };
                        return dele;
                    }
                case 1:
                    {
                        ValueGetterInstance<SentimentDataFloat, ReadOnlyMemory<char>> dele =
                            (ref SentimentDataFloat self, ref ReadOnlyMemory<char> x) => { x = new ReadOnlyMemory<char>(self.SentimentText.ToCharArray()); };
                        return dele;
                    }
                default:
                    throw new Exception($"No available column for index {col}.");
            }
        }
    }

    public class SentimentDataBool : IClassWithGetter<SentimentDataBool>
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
                        ValueGetterInstance<SentimentDataBool, bool> dele =
                            (ref SentimentDataBool self, ref bool x) => { x = self.Sentiment; };
                        return dele;
                    }
                case 1:
                    {
                        ValueGetterInstance<SentimentDataBool, ReadOnlyMemory<char>> dele =
                            (ref SentimentDataBool self, ref ReadOnlyMemory<char> x) => { x = new ReadOnlyMemory<char>(self.SentimentText.ToCharArray()); };
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
