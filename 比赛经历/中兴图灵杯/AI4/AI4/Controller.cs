using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PoliceInterface;
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
        public int[,] MazeArray { get; private set;}
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
    public class Controller : PoliceInterface.IControl
    {
        Log logger = new Log("4");
        public string GetTeamName()
        {
            return "一去二三里";
        }
        public int[,] GetMaze(Police police, int z)
        {
            int[] map_info;
            map_info = police.GetMapInfo();
            int[,] maze = new int[map_info[2], map_info[0]];//2,22,20
            for (int i = 0; i < map_info[0]; i++)
            {
                for (int j = 0; j < map_info[2]; j++)
                {
                    int? pos_type;
                    pos_type = police.GetMapType(i, z, j);//0：普通地板 1：不可用位置 3：出口处 注意这个函数的第二个输入值为1或者2 表示楼层
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
        public int Get_Remove_Elevator_Length(Police police, int[] map_info, int index)//获取警察移动到电梯index的距离
        {
            int length = 0;
            int[] self_pos = police.GetPosition();
            int[] target_pos = police.GetElevatorPosition(index);//为0表示门 注意电梯位置 都是2维数组
            int[,] a = GetMaze(police, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[1] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }
        public int Get_Remove_Thief_Length_same(Police police, int[] map_info, int index)//这个函数针对的是同一层情况下 获取警察距离某个小偷index的距离
        {
            int length = 0;
            int[] self_pos = police.GetPosition();
            int[] target_pos = police.HackerPosition(index);//
            int[,] a = GetMaze(police, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }
        public int Get_Remove_Gate_Length_same(Police police, int[] map_info, int index)//这个函数针对的是同一层情况下 获取警察距离某个门index的距离
        {
            int length = 0;
            int[] self_pos = police.GetPosition();
            int[] target_pos = police.GetExitPosition(index);//
            int[,] a = GetMaze(police, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }

        public bool remove(Police police, int[] pre, int[] aft)//根据下个点与当前点坐标判断移动函数
        {
            if ((aft[0] - pre[0]) > 0)
                police.MoveEast();
            else
            {
                if ((aft[0] - pre[0]) < 0)
                    police.MoveWest();
            }

            if ((aft[1] - pre[1]) > 0)
                police.MoveNorth();
            else
            {
                if ((aft[1] - pre[1]) < 0)
                    police.MoveSouth();
            }

            int[] result_pos = police.GetPosition();
            if (result_pos[0] == aft[0] && result_pos[2] == aft[1])
                return true;
            else
                return false;
        }
        public bool Get_Remove(Police police, int[] map_info, int index, int logo)//经过检测 Get_Remove()到达四个门0123以及三个小偷123都是没问题的
        {
            //logo=0表示门 logo = 1表示贼
            //如果在同一楼层，则警察向小偷移动；如果不在同一楼层，则警察先移动到最近的电梯，再向小偷移动
            bool success = false;
            int[] self_pos = police.GetPosition();
            int[] target_pos = { 0, 0, 0 };
            if (logo == 0)
            {
                target_pos = police.GetExitPosition(index);//对的，这里就是门！而不是电梯！
            }
            else
            {
                target_pos = police.HackerPosition(index);
            }

            if (self_pos[1] == target_pos[1])//如果警察和 小偷/门 在一层
            {
                int[,] a = GetMaze(police, self_pos[1]); //二维的当前层地图
                Maze current_maze = new Maze(a);
                //int[] self_pos_my = OfficeToMyself(map_info, self_pos);
                //int[] target_pos_my = OfficeToMyself(map_info, target_pos);
                Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                var target_point = GetNext(next_point, start);
                //logger.info("target_point:" + target_point.X + ',' + target_point.Y);
                int[] bs_start = { start.Y, map_info[2] - 1 - start.X };
                int[] bs_end = { target_point.Y, map_info[2] - 1 - target_point.X };
                success = remove(police, bs_start, bs_end);
                return success;
            }
            else
            {
                int[,] a = GetMaze(police, self_pos[1]); //二维的当前层地图
                Maze current_maze = new Maze(a);
                //int[] self_pos_my = OfficeToMyself(map_info, self_pos);
                //int[] target_pos_my = OfficeToMyself(map_info, target_pos);
                Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
                int[] elevator1 = police.GetElevatorPosition(1);
                int[] elevator2 = police.GetElevatorPosition(2);
                int elevator1_dis = Get_Remove_Elevator_Length(police, map_info, 1); //这个函数返回的是两个元素的数组！
                int elevator2_dis = Get_Remove_Elevator_Length(police, map_info, 2);
                int[] dis = { 0, 0, 0 };
                if (elevator1_dis < elevator2_dis)
                {
                    //dis = elevator1;
                    //bool get_elevator = Get_Remove_Elevator(police, map_info, 1);
                    target_pos[0] = elevator1[0];
                    target_pos[1] = self_pos[1];
                    target_pos[2] = elevator1[1];
                }
                else
                {
                    //bool get_elevator = Get_Remove_Elevator(police, map_info, 2);
                    target_pos[0] = elevator2[0];
                    target_pos[1] = self_pos[1];
                    target_pos[2] = elevator2[1];
                }
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                var target_point = GetNext(next_point, start);
                //logger.info("target_point:" + target_point.X + ',' + target_point.Y);
                int[] bs_start = { start.Y, map_info[2] - 1 - start.X };
                int[] bs_end = { target_point.Y, map_info[2] - 1 - target_point.X };
                success = remove(police, bs_start, bs_end); //诀窍理解：因为update()函数本身就是不断更新帧的 所以这里没必要调用自己了 走到电梯就可以了
                return success;
            }
        }

        //我突然想到一个致命的地方：在计算两个点距离的时候，并不是欧氏距离；
        //而且，也不是x/y作差相减 应该是A*算法寻找的路径的长度 不然不准确！
        public int cal_police_thief_dis(Police police, int[] map_info, int index) //警察距离小偷的距离 需要考虑是否同层
        {
            int[] self_pos = police.GetPosition();
            int[] target_pos = police.HackerPosition(index);


            int dis_final;
            if (self_pos[1] == target_pos[1])
            {
                dis_final = Get_Remove_Thief_Length_same(police, map_info, index);
            }
            else
            {
                int[] elevator1 = police.GetElevatorPosition(1);
                int[] elevator2 = police.GetElevatorPosition(2);
                int elevator1_dis = Get_Remove_Elevator_Length(police, map_info, 1); //这个函数返回的是两个元素的数组！
                int elevator2_dis = Get_Remove_Elevator_Length(police, map_info, 2);
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
                int[,] a = GetMaze(police, target_pos[1]); //二维的当前层地图
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - elevator[1] - 1, elevator[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                int dis2 = GetLength(next_point, start);
                dis_final = dis1 + dis2;
            }
            return dis_final;
        }
        public int cal_police_gate_dis(Police police, int[] map_info, int index) //警察距离门的距离 需要考察是否同层
        {
            int[] self_pos = police.GetPosition();
            int[] target_pos = police.GetExitPosition(index);
            int dis_final;
            if (self_pos[1] == target_pos[1])
            {
                dis_final = Get_Remove_Gate_Length_same(police, map_info, index);
            }
            else
            {
                int[] elevator1 = police.GetElevatorPosition(1);
                int[] elevator2 = police.GetElevatorPosition(2);
                int elevator1_dis = Get_Remove_Elevator_Length(police, map_info, 1); //这个函数返回的是两个元素的数组！
                int elevator2_dis = Get_Remove_Elevator_Length(police, map_info, 2);
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
                int[,] a = GetMaze(police, target_pos[1]); //二维的当前层地图
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - elevator[1] - 1, elevator[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                int dis2 = GetLength(next_point, start);
                dis_final = dis1 + dis2;
            }
            return dis_final;
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

        public int Get_Thief_Elevator_Length(Police police, int[] map_info, int index_thief, int index_elevator) //小偷index_thief距离电梯index_elevator的距离
        {
            int length = 0;
            int[] self_pos = police.HackerPosition(index_thief);
            int[] target_pos = police.GetElevatorPosition(index_elevator);//为0表示门 注意电梯位置 都是2维数组
            int[,] a = GetMaze(police, self_pos[1]);
            Maze current_maze = new Maze(a);
            Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
            Point end = new Point(map_info[2] - target_pos[1] - 1, target_pos[0]);
            var next_point = current_maze.FindPath(start, end, false);
            length = GetLength(next_point, start);
            return length;
        }
        public int Get_Thief_Gate_Length(Police police, int[] map_info, int index_thief, int index_gate) //小偷index_thief距离门index_gate的距离
        {
            int length = 0;
            int[] self_pos = police.HackerPosition(index_thief);//小偷位置
            int[] target_pos = police.GetExitPosition(index_gate);//门位置
            if (self_pos[1] != target_pos[1])
            {
                int elevator1_dis = Get_Thief_Elevator_Length(police, map_info, index_thief, 1);
                int elevator2_dis = Get_Thief_Elevator_Length(police, map_info, index_thief, 2);
                int[] elevator1 = police.GetElevatorPosition(1);
                int[] elevator2 = police.GetElevatorPosition(2);
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
                int[,] a = GetMaze(police, target_pos[1]);
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - elevator[1] - 1, elevator[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);//false表示忽略绊脚点
                int dis2 = GetLength(next_point, start);
                length = dis1 + dis2;
                return length;
            }
            else
            {
                int[,] a = GetMaze(police, self_pos[1]);
                Maze current_maze = new Maze(a);
                Point start = new Point(map_info[2] - self_pos[2] - 1, self_pos[0]);
                Point end = new Point(map_info[2] - target_pos[2] - 1, target_pos[0]);
                var next_point = current_maze.FindPath(start, end, false);
                length = GetLength(next_point, start);
                return length;
            }
        }

        public bool on_time(Police police, int[] map_info, int thief_index)
        {
            bool success = false;
            int? key_index = police.GetHackerKey(thief_index);
            int self_thief_dis = cal_police_thief_dis(police, map_info, thief_index);
            int thief_gate_dis0 = Get_Thief_Gate_Length(police, map_info, thief_index, 0);
            int thief_gate_dis1 = 10000;
            if (key_index != null)
            {
                thief_gate_dis1 = Get_Thief_Gate_Length(police, map_info, thief_index, (int)key_index);//int? 强制转化为int
            }
            //小偷到达门的时间：thief_gate_dis0/4 警察抓到小偷时间self_thief_dis/0.5
            if (police.GetGameTime() > 10)
            {
                if (((thief_gate_dis0 / 3) < (self_thief_dis * 2)) || ((thief_gate_dis1 /3) < (self_thief_dis * 2)))
                {
                    success = false;
                }
                else
                {
                    success = true;
                }
            }
            else
            {
                if (((thief_gate_dis0 / 3) < (self_thief_dis * 1)) || ((thief_gate_dis1 / 3) < (self_thief_dis * 1)))
                {
                    success = false;
                }
                else
                {
                    success = true;
                }
            }
            return success;
        }

        public bool case_two_online(Police police, int[] map_info, int online_index1, int online_index2, int[] self_thief_all, int[] self_gate_all)
        {
            bool success = false;
            bool on_time1 = on_time(police, map_info, online_index1);
            bool on_time2 = on_time(police, map_info, online_index2);
            if ((on_time1 && on_time2) || (on_time1 == false && on_time2 == false)) //如果两个都来得及或者都来不及 就挑近的（都来不及考虑到万一两个智障 总得抓一个）
            {
                if (self_thief_all[online_index1 - 1] < self_thief_all[online_index2 - 1])
                {

                    success = Get_Remove(police, map_info, online_index1, 1); //Get_Remove()函数0表示门 1表示贼
                }
                else
                {
                    success = Get_Remove(police, map_info, online_index2, 1); //Get_Remove()函数0表示门 1表示贼
                }
            }
            if (on_time1 == false && on_time2)//2来得及 1来不及
            {
                success = Get_Remove(police, map_info, online_index2, 1);
            }
            if (on_time2 == false && on_time1)//1来得及 2来不及
            {
                success = Get_Remove(police, map_info, online_index1, 1);
            }
            return success;
        }

        public bool case_three_online(Police police, int[] map_info, int online_index1, int online_index2, int online_index3, int[] self_thief_all, int[] self_gate_all)
        {
            bool success = false;
            bool on_time1 = on_time(police, map_info, online_index1);
            bool on_time2 = on_time(police, map_info, online_index2);
            bool on_time3 = on_time(police, map_info, online_index3);
            if ((on_time1 && on_time2 && on_time3) || (on_time1 == false && on_time2 == false && on_time3 == false)) //如果三个都来得及或者都来不及 就挑近的（都来不及考虑到万一两个智障 总得抓一个）
            {
                int temp_index = get_min(self_thief_all[online_index1 - 1], self_thief_all[online_index2 - 1], self_thief_all[online_index3 - 1]);
                if (temp_index == 1)
                {
                    success = Get_Remove(police, map_info, online_index1, 1); //Get_Remove()函数0表示门 1表示贼
                }
                if (temp_index == 2)
                {
                    success = Get_Remove(police, map_info, online_index2, 1); //Get_Remove()函数0表示门 1表示贼
                }
                if (temp_index == 3)
                {
                    success = Get_Remove(police, map_info, online_index3, 1); //Get_Remove()函数0表示门 1表示贼
                }
            }

            if (on_time1 == false && on_time2 && on_time3)//1来不及 23来得及 
            {
                if (self_thief_all[online_index2 - 1] < self_thief_all[online_index3 - 1])
                {
                    success = Get_Remove(police, map_info, online_index2, 1);
                }
                else
                {
                    success = Get_Remove(police, map_info, online_index3, 1);
                }
            }

            if (on_time2 == false && on_time1 && on_time3)//2来不及 13来得及
            {
                if (self_thief_all[online_index1 - 1] < self_thief_all[online_index3 - 1])
                {
                    success = Get_Remove(police, map_info, online_index1, 1);
                }
                else
                {
                    success = Get_Remove(police, map_info, online_index3, 1);
                }
            }

            if (on_time3 == false && on_time2 && on_time1)//3来不及 12来得及
            {
                if (self_thief_all[online_index2 - 1] < self_thief_all[online_index1 - 1])
                {
                    success = Get_Remove(police, map_info, online_index2, 1);
                }
                else
                {
                    success = Get_Remove(police, map_info, online_index1, 1);
                }
            }

            if (on_time3 == false && on_time2 == false && on_time1)//1来得及 23来不及
            {
                success = Get_Remove(police, map_info, online_index1, 1);
            }

            if (on_time3 == false && on_time1 == false && on_time2)//2来得及 13来不及
            {
                success = Get_Remove(police, map_info, online_index2, 1);
            }

            if (on_time1 == false && on_time2 == false && on_time3)//3来得及 12来不及
            {
                success = Get_Remove(police, map_info, online_index3, 1);
            }

            return success;
        }

        public void make_choice(Police police, int[] map_info)
        {
            int? hacker1_key = police.GetHackerKey(1);
            int? hacker2_key = police.GetHackerKey(2);
            int? hacker3_key = police.GetHackerKey(3);

            int[] hacker1_online = police.HackerPosition(1);
            int[] hacker2_online = police.HackerPosition(2);
            int[] hacker3_online = police.HackerPosition(3);

            int self_thief_1_dis = 10000;
            if (hacker1_online != null)
            {
                self_thief_1_dis = cal_police_thief_dis(police, map_info, 1);
            }
            int self_thief_2_dis = 10000;
            if (hacker2_online != null)
            {
                self_thief_2_dis = cal_police_thief_dis(police, map_info, 2);
            }
            int self_thief_3_dis = 10000;
            if (hacker3_online != null)
            {
                self_thief_3_dis = cal_police_thief_dis(police, map_info, 3);
            }
            //int self_thief_1_dis = cal_police_thief_dis(police, map_info, 1);
            //int self_thief_2_dis = cal_police_thief_dis(police, map_info, 2);
            //int self_thief_3_dis = cal_police_thief_dis(police, map_info, 3);
            int self_gate_0_dis = cal_police_gate_dis(police, map_info, 0);
            int self_gate_1_dis = cal_police_gate_dis(police, map_info, 1);
            int self_gate_2_dis = cal_police_gate_dis(police, map_info, 2);
            int self_gate_3_dis = cal_police_gate_dis(police, map_info, 3);

            int[] self_thief_all = new int[] { self_thief_1_dis, self_thief_2_dis, self_thief_3_dis };
            int[] self_gate_all = new int[] { self_gate_0_dis, self_gate_1_dis, self_gate_2_dis, self_gate_3_dis };

            #region //注释1
            //logger.info("hacker1_online is not null:");
            //if (hacker1_online != null)
            //{
            //    logger.info(hacker1_online);//!=null为真 ==null为假
            //}
            //else
            //{
            //    logger.info(hacker1_online != null);//!=null为真 ==null为假
            //}

            //logger.info("hacker2_online is not null:");
            //if (hacker2_online != null)
            //{
            //    logger.info(hacker2_online);//!=null为真 ==null为假
            //}
            //else
            //{
            //    logger.info(hacker2_online != null);//!=null为真 ==null为假
            //}

            //logger.info("hacker3_online is not null:");
            //if (hacker3_online != null)
            //{
            //    logger.info(hacker3_online);//!=null为真 ==null为假
            //}
            //else
            //{
            //    logger.info(hacker3_online != null);//!=null为真 ==null为假
            //}
            #endregion


            #region //注释2
            //logger.info("hacker2_online:" + hacker2_online != null);
            //logger.info("hacker3_online:" + hacker3_online != null);
            //logger.info("self_thief_1_dis:" + self_thief_1_dis);
            //logger.info("self_thief_2_dis:" + self_thief_2_dis);
            //logger.info("self_thief_3_dis:" + self_thief_3_dis);
            //logger.info("self_gate_0_dis:" + self_gate_0_dis);
            //logger.info("self_gate_1_dis:" + self_gate_1_dis);
            //logger.info("self_gate_2_dis:" + self_gate_2_dis);
            //logger.info("self_gate_3_dis:" + self_gate_3_dis);
            #endregion

            logger.info("Again.");
            bool success = false;

            #region//首先，最简单的情况；如果两个人都不在了,也就是只剩一个人，那就直接抓他
            if (hacker1_online != null && hacker2_online == null && hacker3_online == null)
            {
                logger.info("1 exist. 2 3 not exist.");
                success = Get_Remove(police, map_info, 1, 1);
            }
            if (hacker1_online == null && hacker2_online != null && hacker3_online == null)
            {
                logger.info("2 exist. 1 3 not exist.");
                success = Get_Remove(police, map_info, 2, 1);
            }
            if (hacker1_online == null && hacker2_online == null && hacker3_online != null)
            {
                logger.info("3 exist. 1 2 not exist.");
                success = Get_Remove(police, map_info, 3, 1);
            }
            #endregion

            #region
            //下面，如果有一个不在了，还有两个人在；
            //1.两人都没钥匙（i.两人都离公共门足够远；则先抓离自己近的；ii.两人一个离公共门近，一个离得远，则先抓离公共门近的；iii.两人都离公共门很近，则抓离公共门最近的）
            if (hacker1_online != null && hacker2_online != null && hacker3_online == null)
            {
                logger.info("1 2 exist.");
                success = case_two_online(police, map_info, 1, 2, self_thief_all, self_gate_all);//这里面不用细分了 只要判断来不来得及就行了（来得及将公共门与钥匙门统一起来了）
            }
            if (hacker1_online != null && hacker2_online == null && hacker3_online != null)
            {
                logger.info("1 3 exist.");
                success = case_two_online(police, map_info, 1, 3, self_thief_all, self_gate_all);
            }
            if (hacker1_online == null && hacker2_online != null && hacker3_online != null)
            {
                logger.info("2 3 exist.");
                success = case_two_online(police, map_info, 2, 3, self_thief_all, self_gate_all);
            }
            #endregion

            #region
            //下面是三个人都在的情况
            if (hacker1_online != null && hacker2_online != null && hacker3_online != null)
            {
                logger.info("Three all exist.");
                //success = Get_Remove(police, map_info, 0, 1); //Get_Remove()函数0表示门 1表示贼
                success = case_three_online(police, map_info, 1, 2, 3, self_thief_all, self_gate_all);
            }
            //else
            //{
            //    logger.info("fuck.");
            //    //logger.info(hacker1_online != null);
            //    //logger.info(hacker2_online != null);
            //    //logger.info(hacker3_online != null);
            //}
            #endregion


        }


        public void Update(Police police)
        {
            int[] map_info;
            map_info = police.GetMapInfo();
            logger.info("地图大小:");
            logger.info(map_info);
            bool success = Get_Remove(police, map_info, 1,0);
            //make_choice(police, map_info);
            //logger.info(police.GetExitPosition(2));

            //if (police.HackerPosition(3) != null)
            //{
            //    logger.info(police.HackerPosition(3));
            //}
            //else
            //{
            //    logger.info(police.HackerPosition(3) != null);
            //}


            //if (police.HackerPosition(3) != null)
            //{
            //    logger.info(police.HackerPosition(3));
            //}
            //else
            //{
            //    logger.info("hacker3_online2:" + "null");
            //}

        }
    }
}
