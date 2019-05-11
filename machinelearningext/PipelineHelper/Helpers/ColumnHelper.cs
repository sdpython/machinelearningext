// See the LICENSE file in the project root for more information.

using System.Text;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;


namespace Scikit.ML.PipelineHelper
{
    internal class Column1x1 : OneToOneColumn
    {
        public static Column1x1 Parse(string str)
        {
            Contracts.AssertNonEmpty(str);

            var res = new Column1x1();
            if (res.TryParse(str))
                return res;
            return null;
        }

        public bool TryUnparse(StringBuilder sb)
        {
            Contracts.AssertValue(sb);
            return TryUnparseCore(sb);
        }

        public static Column1x1[] ParseMulti(string sr)
        {
            var spl = sr.Split(',');
            var columns = new Column1x1[spl.Length];
            for (int i = 0; i < spl.Length; ++i)
            {
                var sub = spl[i].Split(':');
                if (sub.Length != 2)
                    throw Contracts.Except("Unable to parse '{0}'.", spl[i]);
                columns[i] = new Column1x1() { Name = sub[0], Source = sub[1] };
            }
            return columns;
        }

        public string ToLine()
        {
            return string.Format("{0}:{1}", Name, Source);
        }

        public static string ArrayToLine(Column1x1[] columns)
        {
            return string.Join(",", columns.Select(c => c.ToLine()));
        }
    }
}
