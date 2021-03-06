﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Defines the shape of a <see cref="DataFrame"/>.
    /// </summary>
    public struct ShapeType : IComparable<ShapeType>, IEquatable<ShapeType>
    {
        public int Length => 2;
        public int nlin => Item1;
        public int ncol => Item2;
        public int Item1;
        public int Item2;

        /// <summary>
        /// Creates a shape for a dataframe.
        /// </summary>
        public ShapeType(int nlin, int ncol) { Item1 = nlin; Item2 = ncol; }
        public ShapeType(Tuple<int, int> shape) { Item1 = shape.Item1; Item2 = shape.Item2; }
        public override string ToString() { return $"({Item1},{Item2})"; }

        /// <summary>
        /// Retrieves either the number of lines or the number of columns.
        /// </summary>
        public int this[int i]
        {
            get
            {
                if (i == 0) return Item1;
                if (i == 1) return Item2;
                throw new IndexOutOfRangeException($"ShapeType has only two dimension.");
            }
        }

        public int CompareTo(ShapeType other)
        {
            if (Item1 != other.Item1)
                return Item1 < other.Item1 ? -1 : 1;
            if (Item2 != other.Item2)
                return Item2 < other.Item2 ? -1 : 1;
            return 0;
        }

        public bool Equals(ShapeType other)
        {
            return Item1 == other.Item1 && Item2 == other.Item2;
        }

        public override int GetHashCode()
        {
            return (int)CombineHash((uint)Item1.GetHashCode(), (uint)Item2.GetHashCode());
        }

        private static uint CombineHash(uint u1, uint u2)
        {
            return ((u1 << 7) | (u1 >> 25)) ^ u2;
        }

        public override bool Equals(object other)
        {
            if (other is ShapeType)
                return Equals((ShapeType)other);
            else if (other is Tuple<int, int>)
                return Equals(new ShapeType((Tuple<int, int>)other));
            else
                throw new DataTypeError($"Cannot compare a ShapeType to {other.GetType()}.");
        }

        public static bool operator ==(ShapeType c1, ShapeType c2)
        {
            return c1.Equals(c2);
        }

        public static bool operator !=(ShapeType c1, ShapeType c2)
        {
            return !c1.Equals(c2);
        }
    }
}