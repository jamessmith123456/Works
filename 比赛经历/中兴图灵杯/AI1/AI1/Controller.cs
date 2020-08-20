using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HackerInterface;
using ljqLog;

namespace AI4
{    
    ////////////////////////////////////////////////////////////////////
    ///A*算法
    /// ///////////////////////////////////////////////////////////////
    public class Point
    {
        public Point ParentPoint { get; set; }
        public int F { get; set; }  //F=G+H
        public int G { get; set; }
        public int H { get; set; }
        public int X { get; set; }
        public int Y { get; set; }

        public Point(int x, int y)
        {
            this.X = x;
            this.Y = y;
        }
        public void CalcF()
        {
            this.F = this.G + this.H;
        }
    }
    public static class ListHelper
    {
        public static bool Exists(this List<Point> points, Point point)
        {
            foreach (Point p in points)
                if ((p.X == point.X) && (p.Y == point.Y))
                    return true;
            return false;
        }
        public static bool Exists(this List<Point> points, int x, int y)
        {
            foreach (Point p in points)
                if ((p.X == x) && (p.Y == y))
                    return true;
            return false;
        }
        public static Point MinPoint(this List<Point> points)
        {
            points = points.OrderBy(p => p.F).ToList();
            return points[0];
        }
        public static void Add(this List<Point> points, int x, int y)
        {
            Point point = new Point(x, y);
            points.Add(point);
        }
        public static Point Get(this List<Point> points, Point point)
        {
            foreach (Point p in points)
                if ((p.X == point.X) && (p.Y == point.Y))
                    return p;
            return null;
        }
        public static void Remove(this List<Point> points, int x, int y)
        {
            foreach (Point point in points)
            {
                if (point.X == x && point.Y == y)
                    points.Remove(point);
            }
        }
    }
    class Maze
    {
        public const int STEP = 10;
        public int[,] MazeArray { get; private set; }
        List<Point> CloseList;
        List<Point> OpenList;

        public Maze(int[,] maze)
        {
            this.MazeArray = maze;
            OpenList = new List<Point>(MazeArray.Length);
            CloseList = new List<Point>(MazeArray.Length);
        }
        public Point FindPath(Point start, Point end, bool IsIgnoreCorner)
        {
            OpenList.Add(start);
            while (OpenList.Count != 0)
            {
                //var tempStart = OpenList.MinPoint();
                OpenList = OpenList.OrderBy(p => p.F).ToList();
                var tempStart = OpenList[0];
                OpenList.RemoveAt(0);
                CloseList.Add(tempStart);
                var surroundPoints = SurrroundPoints(tempStart, IsIgnoreCorner);
                foreach (Point point in surroundPoints)
                {
                    if (OpenList.Exists(point))
                        FoundPoint(tempStart, point);//计算G值, 如果比原来的大, 就什么都不做, 否则设置它的父节点为当前点,并更新G和F
                    else
                        NotFoundPoint(tempStart, end, point);//如果它们不在开始列表里, 就加入, 并设置父节点,并计算GHF
                }
                if (OpenList.Get(end) != null)
                    return OpenList.Get(end);
            }
            return OpenList.Get(end);
        }
        private void FoundPoint(Point tempStart, Point point)//FoundPoint()函数用于根据 当前点及周围点 更新周围点的F值 每个点的H值不变 所以只需要比较G值就可以了
        {
            var G = CalcG(tempStart, point);
            if (G < point.G)
            {
                point.ParentPoint = tempStart;
                point.G = G;
                point.CalcF();
            }
        }
        private void NotFoundPoint(Point tempStart, Point end, Point point)//如果不在开启列表里 就假如开启列表 并将其父节点设置为当前的节点 计算它的F值
        {
            point.ParentPoint = tempStart;
            point.G = CalcG(tempStart, point);
            point.H = CalcH(end, point);
            point.CalcF();
            OpenList.Add(point);
        }
        private int CalcG(Point start, Point point)
        {
            //int G = (Math.Abs(point.X - start.X) + Math.Abs(point.Y - start.Y)) == 2 ? STEP : OBLIQUE;//三元表达式 如果为2就是10不然就是14
            int G = STEP;
            int parentG = point.ParentPoint != null ? point.ParentPoint.G : 0;//如果某个节点有父节点 那么该点的G就是父节点的G
            return G + parentG;
        }
        private int CalcH(Point end, Point point)
        {
            int step = Math.Abs(point.X - end.X) + Math.Abs(point.Y - end.Y);//这里我觉得没毛病 不需要改 而且H并不影响F的比较 因为H是固定的
            return step * STEP;
        }
        public List<Point> SurrroundPoints(Point point, bool IsIgnoreCorner)//获取某个点周围可以到达的点
        {
            var surroundPoints = new List<Point>(9);
            for (int x = point.X - 1; x <= point.X + 1; x++)
                for (int y = point.Y - 1; y <= point.Y + 1; y++)
                {
                    if (x >= 0 && x <= 21 && y >= 0 && y <= 19)//越界判断 这个必须要有的
                    {
                        if (CanReach(point, x, y, IsIgnoreCorner))
                            surroundPoints.Add(x, y);
                    }
                }
            return surroundPoints;
        }
        private bool CanReach(int x, int y)
        {
            return MazeArray[x, y] == 0 || MazeArray[x, y] == 3; //MazeArray数组的对应位置如果为0 则返回True 否则返回False
        }
        public bool CanReach(Point start, int x, int y, bool IsIgnoreCorner)
        {
            if (!CanReach(x, y) || CloseList.Exists(x, y)) //如果某个位置x y已经在关闭列表里 或者 CanReach()返回为False
                return false;
            else
            {
                if (Math.Abs(x - start.X) + Math.Abs(y - start.Y) == 1) //判断上下左右方向
                    return true;
                else
                    return false;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////
    ///比赛部分
    /// ///////////////////////////////////////////////////////////////
    public class Controller : HackerInterface.IControl
    {
        Log logger = new Log("1");
        private bool x ;
        //队名录入位置
        public string GetTeamName()
        {
            return "一去二三里";
        }

        public int[,] GetMaze(Hacker hacker, int z)
        {
            int[] map_info;
            map_info = hacker.GetMapInfo();
            int[,] maze = new int[map_info[2], map_info[0]];//2,22,20
            for (int i = 0; i < map_info[0]; i++)
            {
                for (int j = 0; j < map_info[2]; j++)
                {
                    int? pos_type;
                    pos_type = hacker.GetMapType(i, z, j);//0：普通地板 1：不可用位置 3：出口处 注意这个函数的第二个输入值为1或者2 表示楼层
                    maze[map_info[2] - j - 1, i] = (int)pos_type; //强制类型转换 将int?类型转换为int类型
                }
            }
            return maze;
        }
        #region //原本打算用于坐标转换的，冗余，不用
        public int[] OfficeToMyself(int[] map_info, int[] Office)
        {
            int[] Myself;
            Myself = new int[] { map_info[2] - 1 - Office[2], Office[1], Office[0] };
            return Myself;
        }
        public int[] MyselfToOffice(int[] map_info, int[] Myself)
        {
            int[] Office;
            Office = new int[] { Myself[2], Myself[1], map_info[2] - 1 - Myself[0] };
            return Office;
        }
        #endregion

        public int GetLength(Point point, Point start)//这个函数用于获取寻找到的路径的长度
        {
            int length = 0;
            if (point == null)
            {
                return length;
            }
            Point temp;
            while (point != null)
            {
                temp = point;
                point = point.ParentPoint;
                length = length + 1;
                if (point.X == start.X && point.Y == start.Y)
                {
                    return length;
                }
            }
            return length;
        }
        public Point GetNext(Point point, Point start)//这个函数用于获取寻找到的路径的紧接着下一步
        {
            if (point == null)
            {
                Console.WriteLine("Not Found.");
            }
            Point temp;
            while (point != null)
            {
                temp = point;
                point = point.ParentPoint;
                if (point.X == start.X && point.Y == start.Y)
                {
                    return temp;
                }
            }
            return null;
        }
        public int Get_Remove_Elevator_Length(Hacker hacker, int[] map_info, int index)//获取小偷移动到电梯index的距离
        {
            int length = 0;
            int[] self_pos = hacker.GetPosition();
            int[] target_pos = hacker.GetElevatorPosition(index);//为0表示门 注意电梯位置 都是2维数组
            int[,] a = GetMaze(hacker, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[1] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }
        public int Get_Remove_Gate_Length_same(Hacker hacker, int[] map_info, int index)//这个函数针对的是同一层情况下 获取小偷距离某个门index的距离
        {
            int length = 0;
            int[] self_pos = hacker.GetPosition();
            int[] target_pos = hacker.GetExitPosition(index);//
            int[,] a = GetMaze(hacker, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }
        public int Get_Remove_Key_Length_same(Hacker hacker, int[] map_info, int index)//这个函数针对的是同一层情况下 获取小偷距离某个钥匙index的距离
        {
            int length = 0;
            int[] self_pos = hacker.GetPosition();
            int[] target_pos = hacker.GetKeysPosition(index);//
            int[,] a = GetMaze(hacker, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }

        public int Get_Remove_Key_Length(Hacker hacker, int[] map_info, int index) //该小偷移动到某个钥匙的距离
        {
            int[] self_pos = hacker.GetPosition();
            int[] target_pos = hacker.GetKeysPosition(index);
            int dis_final;
            if (self_pos[1] == target_pos[1])
            {
                dis_final = Get_Remove_Key_Length_same(hacker, map_info, index);
            }
            else
            {
                int[] elevator1 = hacker.GetElevatorPosition(1);
                int[] elevator2 = hacker.GetElevatorPosition(2);
                int elevator1_dis = Get_Remove_Elevator_Length(hacker, map_info, 1); //这个函数返回的是两个元素的数组！
                int elevator2_dis = Get_Remove_Elevator_Length(hacker, map_info, 2);
                int dis1;
                int[] elevator;
                if (elevator1_dis < elevator2_dis)
                {
                    dis1 = elevator1_dis;
                    elevator = elevator1;
                }
                else
                {
                    dis1 = elevator2_dis;
                    elevator = elevator2;
                }
                int[,] a = GetMaze(hacker, target_pos[1]); //二维的当前层地图
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - elevator[1] - 1, elevator[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                int dis2 = GetLength(next_point, start);
                dis_final = dis1 + dis2;
            }
            return dis_final;
        }
        public int Get_Remove_Gate_Length(Hacker hacker, int[] map_info, int index)
        {
            int[] self_pos = hacker.GetPosition();
            int[] target_pos = hacker.GetExitPosition(index);
            int dis_final;
            if (self_pos[1] == target_pos[1])
            {
                dis_final = Get_Remove_Gate_Length_same(hacker, map_info, index);
            }
            else
            {
                int[] elevator1 = hacker.GetElevatorPosition(1);
                int[] elevator2 = hacker.GetElevatorPosition(2);
                int elevator1_dis = Get_Remove_Elevator_Length(hacker, map_info, 1); //这个函数返回的是两个元素的数组！
                int elevator2_dis = Get_Remove_Elevator_Length(hacker, map_info, 2);
                int dis1;
                int[] elevator;
                if (elevator1_dis < elevator2_dis)
                {
                    dis1 = elevator1_dis;
                    elevator = elevator1;
                }
                else
                {
                    dis1 = elevator2_dis;
                    elevator = elevator2;
                }
                int[,] a = GetMaze(hacker, target_pos[1]); //二维的当前层地图
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - elevator[1] - 1, elevator[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                int dis2 = GetLength(next_point, start);
                dis_final = dis1 + dis2;
            }
            return dis_final;
        }

        public void Escape(Hacker hacker, int[] map_info)
        {
            if (hacker.Warning()) //每次remove完之后 都要判断是否有危险 如果发现危险
            {
                int[] police_pos = hacker.GetPolicePosition();
                int[] hacker_pos = hacker.GetPosition();

                //一共6种情况
                int x = hacker_pos[0];
                int z = hacker_pos[2];
                int x1 = -1, x2 = -1, z1 = -1, z2 = -1;
                #region //更新x1 x2 z1 z2
                if (hacker_pos[0] > 0 && hacker_pos[0] < (map_info[0] - 1))
                {
                    x1 = hacker_pos[0] - 1;
                    x2 = hacker_pos[0] + 1;
                }
                if (hacker_pos[0] == 0)
                {
                    x2 = hacker_pos[0] + 1;
                }
                if (hacker_pos[0] == map_info[0] - 1)
                {
                    x1 = hacker_pos[0] - 1;
                }
                if (hacker_pos[2] > 0 && hacker_pos[2] < (map_info[2] - 1))
                {
                    z1 = hacker_pos[2] - 1;
                    z2 = hacker_pos[2] + 1;
                }
                if (hacker_pos[2] == 0)
                {
                    z2 = hacker_pos[2] + 1;
                }
                if (hacker_pos[2] == map_info[2] - 1)
                {
                    z1 = hacker_pos[2] - 1;
                }
                #endregion

                #region //未达到四个边界
                if (x1 != -1 && x2 != -1 && z1 != -1 && z2 != -1)
                {
                    int disWest = System.Math.Abs(police_pos[0] - x1) + System.Math.Abs(police_pos[2] - x);
                    int disEast = System.Math.Abs(police_pos[0] - x2) + System.Math.Abs(police_pos[2] - x);
                    int disSout = System.Math.Abs(police_pos[0] - z1) + System.Math.Abs(police_pos[2] - z);
                    int disNort = System.Math.Abs(police_pos[0] - z2) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min_four(disWest, disEast, disSout, disNort);
                    if (next == 1)
                    {
                        hacker.MoveWest();
                    }
                    if (next == 2)
                    {
                        hacker.MoveEast();
                    }
                    if (next == 3)
                    {
                        hacker.MoveSouth();
                    }
                    if (next == 4)
                    {
                        hacker.MoveNorth();
                    }
                }
                #endregion

                #region //达到左边界
                if (x1 == -1 && x2 != -1 && z1 != -1 && z2 != -1)
                {
                    int disEast = System.Math.Abs(police_pos[0] - x2) + System.Math.Abs(police_pos[2] - x);
                    int disSout = System.Math.Abs(police_pos[0] - z1) + System.Math.Abs(police_pos[2] - z);
                    int disNort = System.Math.Abs(police_pos[0] - z2) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min(disEast, disSout, disNort);
                    if (next == 1)
                    {
                        hacker.MoveEast();
                    }
                    if (next == 2)
                    {
                        hacker.MoveSouth();
                    }
                    if (next == 3)
                    {
                        hacker.MoveNorth();
                    }
                }
                #endregion

                #region //达到右边界
                if (x1 != -1 && x2 == -1 && z1 != -1 && z2 != -1)
                {
                    int disWest = System.Math.Abs(police_pos[0] - x1) + System.Math.Abs(police_pos[2] - x);
                    int disSout = System.Math.Abs(police_pos[0] - z1) + System.Math.Abs(police_pos[2] - z);
                    int disNort = System.Math.Abs(police_pos[0] - z2) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min(disWest, disSout, disNort);
                    if (next == 1)
                    {
                        hacker.MoveWest();
                    }
                    if (next == 2)
                    {
                        hacker.MoveSouth();
                    }
                    if (next == 3)
                    {
                        hacker.MoveNorth();
                    }
                }
                #endregion

                #region //达到下边界
                if (x1 != -1 && x2 != -1 && z1 == -1 && z2 != -1)
                {
                    int disWest = System.Math.Abs(police_pos[0] - x1) + System.Math.Abs(police_pos[2] - x);
                    int disEast = System.Math.Abs(police_pos[0] - x2) + System.Math.Abs(police_pos[2] - x);
                    int disNort = System.Math.Abs(police_pos[0] - z2) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min(disWest, disEast, disNort);
                    if (next == 1)
                    {
                        hacker.MoveWest();
                    }
                    if (next == 2)
                    {
                        hacker.MoveEast();
                    }
                    if (next == 3)
                    {
                        hacker.MoveNorth();
                    }
                }
                #endregion

                #region //达到上边界
                if (x1 != -1 && x2 != -1 && z1 != -1 && z2 == -1)
                {
                    int disWest = System.Math.Abs(police_pos[0] - x1) + System.Math.Abs(police_pos[2] - x);
                    int disEast = System.Math.Abs(police_pos[0] - x2) + System.Math.Abs(police_pos[2] - x);
                    int disSout = System.Math.Abs(police_pos[0] - z1) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min(disWest, disEast, disSout);
                    if (next == 1)
                    {
                        hacker.MoveWest();
                    }
                    if (next == 2)
                    {
                        hacker.MoveEast();
                    }
                    if (next == 3)
                    {
                        hacker.MoveSouth();
                    }
                }
                #endregion

                #region //达到左边界及下边界
                if (x1 == -1 && x2 != -1 && z1 == -1 && z2 != -1)
                {
                    int disEast = System.Math.Abs(police_pos[0] - x2) + System.Math.Abs(police_pos[2] - x);
                    int disNort = System.Math.Abs(police_pos[0] - z2) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min_two(disEast, disNort);
                    if (next == 1)
                    {
                        hacker.MoveEast();
                    }
                    if (next == 2)
                    {
                        hacker.MoveNorth();
                    }
                }
                #endregion

                #region //达到右边界及下边界
                if (x1 != -1 && x2 == -1 && z1 == -1 && z2 != -1)
                {
                    int disWest = System.Math.Abs(police_pos[0] - x1) + System.Math.Abs(police_pos[2] - x);
                    int disNort = System.Math.Abs(police_pos[0] - z2) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min_two(disWest, disNort);
                    if (next == 1)
                    {
                        hacker.MoveWest();
                    }
                    if (next == 2)
                    {
                        hacker.MoveNorth();
                    }
                }
                #endregion

                #region //达到左边界及上边界
                if (x1 == -1 && x2 != -1 && z1 != -1 && z2 == -1)
                {
                    int disEast = System.Math.Abs(police_pos[0] - x2) + System.Math.Abs(police_pos[2] - x);
                    int disSout = System.Math.Abs(police_pos[0] - z1) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min_two(disEast, disSout);
                    if (next == 1)
                    {
                        hacker.MoveEast();
                    }
                    if (next == 2)
                    {
                        hacker.MoveSouth();
                    }
                }
                #endregion

                #region //达到右边界及上边界
                if (x1 != -1 && x2 == -1 && z1 != -1 && z2 == -1)
                {
                    int disWest = System.Math.Abs(police_pos[0] - x1) + System.Math.Abs(police_pos[2] - x);
                    int disSout = System.Math.Abs(police_pos[0] - z1) + System.Math.Abs(police_pos[2] - z);
                    int next = get_min_two(disWest, disSout);
                    if (next == 1)
                    {
                        hacker.MoveWest();
                    }
                    if (next == 2)
                    {
                        hacker.MoveSouth();
                    }
                }
                #endregion

            }
        }
        public bool remove(Hacker hacker, int[] pre, int[] aft)//根据下个点与当前点坐标判断移动函数
        {
            if ((aft[0] - pre[0]) > 0)
                hacker.MoveEast();
            else
            {
                if ((aft[0] - pre[0]) < 0)
                    hacker.MoveWest();
            }

            if ((aft[1] - pre[1]) > 0)
                hacker.MoveNorth();
            else
            {
                if ((aft[1] - pre[1]) < 0)
                    hacker.MoveSouth();
            }

            //在这里添加一个判断 warning!每次移动完之后 如果遇到警察 都要做出躲避！
            int[] result_pos = hacker.GetPosition();
            if (result_pos[0] == aft[0] && result_pos[2] == aft[1])
                return true;
            else
                return false;
        }
        public bool Get_Remove(Hacker hacker, int[] map_info, int index, int logo)//经过检测 Get_Remove()到达四个门0123以及三个小偷123都是没问题的
        {
            //logo=0表示门 logo = 1表示钥匙
            bool success = false;
            int[] self_pos = hacker.GetPosition();
            int[] target_pos = { 0, 0, 0 };
            if (logo == 0)
            {
                target_pos = hacker.GetExitPosition(index);//对的，这里就是门！而不是电梯！
            }
            else
            {
                target_pos = hacker.GetKeysPosition(index);
            }

            if (self_pos[1] == target_pos[1])//如果小偷和 门/钥匙 在一层
            {
                int[,] a = GetMaze(hacker, self_pos[1]);
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);
                var target_point = GetNext(next_point, start);
                int[] bs_start = { start.Y, map_info[2] - 1 - start.X };
                int[] bs_end = { target_point.Y, map_info[2] - 1 - target_point.X };
                success = remove(hacker, bs_start, bs_end);
                Escape(hacker, map_info);
                return success;
            }
            else
            {
                int[,] a = GetMaze(hacker, self_pos[1]);
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
                int[] elevator1 = hacker.GetElevatorPosition(1);
                int[] elevator2 = hacker.GetElevatorPosition(2);
                int elevator1_dis = Get_Remove_Elevator_Length(hacker, map_info, 1); //这个函数返回的是两个元素的数组！
                int elevator2_dis = Get_Remove_Elevator_Length(hacker, map_info, 2);
                int[] dis = { 0, 0, 0 };
                if (elevator1_dis < elevator2_dis)
                {
                    target_pos[0] = elevator1[0]; //将target_pos设置为最近的电梯位置
                    target_pos[1] = self_pos[1];
                    target_pos[2] = elevator1[1];
                }
                else
                {
                    target_pos[0] = elevator2[0];
                    target_pos[1] = self_pos[1];
                    target_pos[2] = elevator2[1];
                }
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                var target_point = GetNext(next_point, start);
                int[] bs_start = { start.Y, map_info[2] - 1 - start.X };
                int[] bs_end = { target_point.Y, map_info[2] - 1 - target_point.X };
                success = remove(hacker, bs_start, bs_end); //诀窍理解：因为update()函数本身就是不断更新帧的 所以这里没必要调用自己了 走到电梯就可以了
                Escape(hacker, map_info);
                return success;
            }
        }

        public int get_min_two(int a, int b) //求ab两个数的最小值 a最小返回1 b最小返回2
        {
            int min;
            if (a > b)
            {
                min = b;
            }
            else
            {
                min = a;
            }
            int index = 0;
            if (min == a)
            {
                index = 1;
            }
            if (min == b)
            {
                index = 2;
            }
            return index;
        }
        public int get_min(int a, int b, int c) //求abc三个数的最小值 a最小返回1 b最小返回2 c最小返回3
        {
            int min;
            if (a > b)
            {
                min = b;
            }
            else
            {
                min = a;
            }
            if (c < min)
            {
                min = c;
            }

            int index = 0;
            if (min == a)
            {
                index = 1;
            }
            if (min == b)
            {
                index = 2;
            }
            if (min == c)
            {
                index = 3;
            }
            return index;
        }
        public int get_min_four(int a, int b, int c, int d) //求abcd四个数的最小值
        {
            int min;
            if (a > b)
            {
                min = b;
            }
            else
            {
                min = a;
            }
            if (c < min)
            {
                min = c;
            }
            if (d < min)
            {
                min = d;
            }

            int index = 0;
            if (min == a)
            {
                index = 1;
            }
            if (min == b)
            {
                index = 2;
            }
            if (min == c)
            {
                index = 3;
            }
            if (min == d)
            {
                index = 4;
            }
            return index;
        }
        //突然想到一个致命的地方：需要考虑游戏剩余时间吗？(可能不需要，因为只要选择两个门中最近的跑就行了;要是最近的门赶不上，那么更远的也赶不上)
        #region
        //public bool on_time(Hacker hacker, int[] map_info, int thief_index) 
        //{
        //    
        //    bool success = false;
        //    int? key_index = hacker.GetHackerKey(thief_index);
        //    int self_thief_dis = cal_police_thief_dis(hacker, map_info, thief_index);
        //    int thief_gate_dis0 = Get_Thief_Gate_Length(hacker, map_info, thief_index, 0);
        //    int thief_gate_dis1 = 10000;
        //    if (key_index != null)
        //    {
        //        thief_gate_dis1 = Get_Thief_Gate_Length(hacker, map_info, thief_index, (int)key_index);//int? 强制转化为int
        //    }
        //    //小偷到达门的时间：thief_gate_dis0/4 警察抓到小偷时间self_thief_dis/0.5
        //    if (((thief_gate_dis0 * 0.25) < (self_thief_dis * 2)) || ((thief_gate_dis1 * 0.25) < (self_thief_dis * 2)))
        //    {
        //        success = false;
        //    }
        //    else
        //    {
        //        success = true;
        //    }
        //    return success;
        //}
        #endregion
        public void get_key(Hacker hacker, int[] map_info)
        {
            int[] key1 = hacker.GetKeysPosition(1);
            int[] key2 = hacker.GetKeysPosition(2);
            int[] key3 = hacker.GetKeysPosition(3);
            bool success = false;
            #region//钥匙123都没有被获取 那么就去获取最近的一把钥匙
            if (key1 != null && key2 != null && key3 != null)
            {
                int key1_dis = Get_Remove_Key_Length(hacker, map_info, 1);//这个逻辑是正确的：只有先确定钥匙是否被获取 才能计算距离
                int key2_dis = Get_Remove_Key_Length(hacker, map_info, 2);
                int key3_dis = Get_Remove_Key_Length(hacker, map_info, 3);
                int next_key_index = get_min(key1_dis, key2_dis, key3_dis);
                success = Get_Remove(hacker, map_info, next_key_index, 1);//0表示门 1表示钥匙
            }
            #endregion
            #region //如果两把未被获取 则去获取最近的一把
            if (key1 != null && key2 != null && key3 == null)//钥匙12未被获取 3已被获取 则获取12中最近的钥匙
            {
                int key1_dis = Get_Remove_Key_Length(hacker, map_info, 1);
                int key2_dis = Get_Remove_Key_Length(hacker, map_info, 2);
                if (key1_dis < key2_dis)
                {
                    success = Get_Remove(hacker, map_info, 1, 1);//0表示门 1表示钥匙
                }
                else
                {
                    success = Get_Remove(hacker, map_info, 2, 1);//0表示门 1表示钥匙
                }
            }
            if (key1 != null && key2 == null && key3 != null)//钥匙13未被获取 2已被获取 则获取13中最近的钥匙
            {
                int key1_dis = Get_Remove_Key_Length(hacker, map_info, 1);
                int key3_dis = Get_Remove_Key_Length(hacker, map_info, 3);
                if (key1_dis < key3_dis)
                {
                    success = Get_Remove(hacker, map_info, 1, 1);//0表示门 1表示钥匙
                }
                else
                {
                    success = Get_Remove(hacker, map_info, 3, 1);//0表示门 1表示钥匙
                }
            }
            if (key1 == null && key2 != null && key3 != null)//钥匙23未被获取 1已被获取  则获取23中最近的钥匙
            {
                int key2_dis = Get_Remove_Key_Length(hacker, map_info, 2);
                int key3_dis = Get_Remove_Key_Length(hacker, map_info, 3);
                if (key2_dis < key3_dis)
                {
                    success = Get_Remove(hacker, map_info, 2, 1);//0表示门 1表示钥匙
                }
                else
                {
                    success = Get_Remove(hacker, map_info, 3, 1);//0表示门 1表示钥匙
                }
            }
            #endregion
            #region//如果仅剩一把未被获取
            if (key1 != null && key2 == null && key3 == null)//钥匙1未被获取 23已被获取
            {
                success = Get_Remove(hacker, map_info, 1, 1);//0表示门 1表示钥匙
            }
            if (key1 == null && key2 != null && key3 == null)//钥匙2未被获取 13已被获取
            {
                success = Get_Remove(hacker, map_info, 2, 1);//0表示门 1表示钥匙
            }
            if (key1 == null && key2 == null && key3 != null)//钥匙3未被获取 12已被获取
            {
                success = Get_Remove(hacker, map_info, 3, 1);//0表示门 1表示钥匙
            }
            #endregion
        }

        public void make_choice(Hacker hacker, int[] map_info)
        {
            //小偷涉及到两个方面的策略：1.钥匙 2.门
            //1.如果三把钥匙都在  2.还有两把在 3.只有一把在 4.直接奔着0号门去
            //4.直接奔着0号门去，不建议这样；因为一开始警察肯定出生在0号门附近
            bool success = false;

            //如果没有钥匙 就先去获取钥匙
            int? mykey_index = hacker.GetHackerKey(1);//AI2 是 2号吧？
            if (mykey_index != null && mykey_index == 0) //如果没有钥匙且没有被抓 则去获取钥匙
            {
                get_key(hacker, map_info);
            }

            if (mykey_index != null && mykey_index != 0)
            {
                int gate0_length = Get_Remove_Gate_Length(hacker, map_info, 0);
                int gate1_length = Get_Remove_Gate_Length(hacker, map_info, (int)mykey_index);
                if (gate0_length < gate1_length)
                {
                    Get_Remove(hacker, map_info, 0, 0);
                }
                else
                {
                    Get_Remove(hacker, map_info, (int)mykey_index, 0);
                }
            }

        }
        //逻辑代码编写
        public void Update(Hacker hacker)
        {
            #region //原本的部分
            //int[] position = hacker.GetPosition();
            //logger.info(x);
            //if (!hacker.isMoving())//不在移动
            //{
            //    if (hacker.GetMapType(position[0], 1, position[2] + 1) != 0 && !x)
            //    {

            //        x = true;
            //    }
            //    else if (hacker.GetMapType(position[0], 1, position[2] - 1) == 0 && x)
            //    {
            //        x = false;
            //    }
            //    if (!x)
            //    {
            //        hacker.MoveNorth();
            //    }
            //    else
            //    {

            //        hacker.MoveSouth();
            //    }
            //}
            #endregion
            int[] map_info = hacker.GetMapInfo();
            make_choice(hacker, map_info);
        }
    }
}
