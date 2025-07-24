using Autodesk.AutoCAD.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZktCad.Models
{
    /// <summary>
    /// 智能布点请求参数
    /// </summary>
    public class AiDrillParam
    {
        /// <summary>
        /// 默认间距
        /// </summary>
        public double DefaultDistance { get; set; }

        /// <summary>
        /// 最大间距
        /// </summary>
        public double MaxDistance { get; set; }

        /// <summary>
        /// 场地边界
        /// </summary>
        public List<HktPoint> Boundary { get; set; }

        /// <summary>
        /// 角点约束，可选，可以有多个角点约束，最内层元素格式为X,Y
        /// </summary>
        public List<List<HktPoint>> Corner { get; set; }
    }

    public class HktPoint
    {
        public double X { get; set; }
        public double Y { get; set; }
    }

    /// <summary>
    /// 智能布点响应参数
    /// </summary>
    public class AiDrillDto
    {
        public List<HktPoint> Drill { get; set; }
    }
}
