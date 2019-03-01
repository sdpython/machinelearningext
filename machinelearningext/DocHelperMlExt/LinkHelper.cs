// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using Microsoft.ML.Transforms;


namespace Scikit.ML.DocHelperMlExt
{
    public static class LinkHelper
    {
        public static void _Immutable()
        {
            var res = ImmutableArray.Create<float>(0f);
            if (res.Length == 0)
                throw new Exception("No immutable");
        }

        public static void _Memory()
        {
            var res = new ReadOnlyMemory<char>();
            if (!res.IsEmpty)
                throw new Exception("No memory");
        }

        public static void _Normalize()
        {
            var args = new NormalizeTransform.MinMaxArguments()
            {
                Columns = new[]
                {
                    NormalizeTransform.AffineColumn.Parse("A"),
                    new NormalizeTransform.AffineColumn() { Name = "B", Source = "B", FixZero = false },
                },
                FixZero = true,
                MaxTrainingExamples = 1000
            };
            if (args == null)
                throw new Exception("No NormalizeTransform.");
        }
    }
}
