// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Getter type for a specific instance.
    /// </summary>
    /// <typeparam name="TRow">a class type</typeparam>
    /// <typeparam name="TValue">a column type</typeparam>
    /// <param name="row">instance (input)</param>
    /// <param name="value">column (output)</param>
    public delegate void ValueGetterInstance<TRow, TValue>(ref TRow row, ref TValue value);

    /// <summary>
    /// Declares the method to implement for a <see cref="ValueMapperFromTransform" />
    /// for the input type. This could be automatically created.
    ///
    /// <pre>
    ///     public class DummyExample : IClassWithGetter<DummyExample>
    /// {
    ///     [VectorType(2)]
    ///     public float[] X;
    /// 
    ///     public uint Y;
    /// 
    ///     public Delegate GetGetter(int col)
    ///     {
    ///         switch (col)
    ///         {
    ///             case 0:
    ///                 {
    ///                     ValueGetterInstance<DummyExample, float[]> dele = (ref DummyExample self, ref float[] x) => { x = self.X; };
    ///                     return dele;
    ///                 }
    ///             case 1:
    ///                 {
    ///                     ValueGetterInstance<DummyExample, uint> dele = (ref DummyExample self, ref uint y) => { y = self.Y; };
    ///                     return dele;
    ///                 }
    ///             default:
    ///                 throw new System.Exception($"No field number {col}.");
    ///         }
    ///     }
    /// }
    /// </pre>
    /// </summary>
    /// <typeparam name="TRow">itself</typeparam>
    public interface IClassWithGetter<TRow>
    {
        // Delegate is of type ValueGetterInstance<TRow, TValue>;
        Delegate GetGetter(int col);
    }

    /// <summary>
    /// Declares the method to implement for a <see cref="ValueMapperFromTransform" />
    /// for the output type. This could be automatically created.
    /// 
    /// <pre>
    /// public class DummyExample : IClassWithSetter<DummyExample>
    /// {
    ///     [VectorType(2)]
    ///     public float[] X;
    ///     public uint Y;
    ///     
    ///     public Delegate[] GetCursorGetter(DataViewRowCursor cursor)
    ///     {
    ///         return new Delegate[]
    ///         {
    ///             cursor.GetGetter<float[]>(0),
    ///             cursor.GetGetter<uint>(1),
    ///         };
    ///     }
    ///     
    ///     public void Set(Delegate[] delegates)
    ///     {
    ///         var del1 = delegates[0] as ValueGetter<float[]>;
    ///         del1(ref X);
    ///         var del2 = delegates[1] as ValueGetter<uint>;
    ///         del2(ref Y);
    ///     }
    ///     }
    /// </pre>
    /// </summary>
    /// <typeparam name="TRow">itself</typeparam>
    public interface IClassWithSetter<TRow>
    {
        Delegate[] GetCursorGetter(DataViewRowCursor cursor);
        void Set(Delegate[] delegates);
    }

    /// <summary>
    /// Used by <see cref="ValueMapperFromTransform" />.
    /// </summary>
    public static class GetterSetterHelper
    {
        public static Dictionary<string, Delegate> GetGetter<TRow>()
            where TRow : IClassWithGetter<TRow>, new()
        {
            var inst = new TRow();
            var schema = SchemaDefinition.Create(typeof(TRow), SchemaDefinition.Direction.Read);
            var res = new Dictionary<string, Delegate>();
            for (int i = 0; i < schema.Count; ++i)
            {
                var name = schema[i].ColumnName;
                res[name] = inst.GetGetter(i);
            }
            return res;
        }
    }
}
