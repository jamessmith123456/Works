from tkinter import *
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import xlwt
import xlrd
class Application(Frame):  # 从Frame派生出Application类，它是所有widget的父容器
    def __init__(self, master=None):  # master即是窗口管理器，用于管理窗口部件，如按钮标签等，顶级窗口master是None，即自己管理自己
        Frame.__init__(self, master)
        self.pack()  # 将widget加入到父容器中并实现布局
        self.createWidgets()
        self.mean = None
        self.variance = None


        #self.create_form(self.figure1)



    def createWidgets(self):
        self.helloLabel = Label(self, text='卡尔曼滤波数据处理演示系统').grid(row=0,column=7)  # 创建一个标签显示内容到窗口
        self.blankLabel1 = Label(self, text='').grid(row=2, column=7)

        self.Label_mean = Label(self, text='Mean:').grid(row=4,sticky=E)
        self.input_mean = Entry(self)  # 创建一个输入框，以输入均值
        self.input_mean.grid(row=4,column=1)

        self.Label_variance = Label(self, text='Variance:').grid(row=5,sticky=E)
        self.input_variance = Entry(self)  # 创建一个输入框，以输入标准差
        self.input_variance.grid(row=5, column=1)

        self.Label_niter = Label(self, text='iters:').grid(row=6,sticky=E)
        self.input_niter = Entry(self)  # 创建一个输入框，以输入标准差
        self.input_niter.grid(row=6, column=1)

        self.dataButton = Button(self, text='GetParameter', command=self.getData)  # 创建一个hello按钮，点击调用hello方法，实现输出
        self.dataButton.grid(row=4,column=7)

        self.plotButton = Button(self, text='PlotFigure', command=self.plot)
        self.plotButton.grid(row=5, column=7)

        self.quitButton = Button(self, text='Quit', command=self.quit)  # 创建一个Quit按钮，实现点击即退出窗口
        self.quitButton.grid(row=6, column=7)

        # self.canvas1 = tk.Canvas()
        # self.canvas1.get_tk_widget().grid(row=3,rowspan=5,column=9)
        # self.figure1 = Figure(figsize=(5, 4), dpi=50)
        # self.canvas2 = tk.Canvas()
        # self.canvas2.get_tk_widget().grid(row=3,rowspan=5,column=17)
        # self.figure2 = Figure(figsize=(5, 4), dpi=50)

        # f = Figure(figsize=(5, 4), dpi=100)
        # a = f.add_subplot(111)  # 添加子图:1行1列第1个
        #
        # # 生成用于绘sin图的数据
        # x = np.arange(0, 3, 0.01)
        # y = np.sin(2 * np.pi * x)
        #
        # # 在前面得到的子图上绘图
        # a.plot(x, y)
        #
        # # 将绘制的图形显示到tkinter:创建属于root的canvas画布,并将图f置于画布上
        # canvas = FigureCanvasTkAgg(f, master=self)
        # canvas.draw()


        self.f1 = plt.figure(num=1, figsize=(5, 4), dpi=50)
        #f1 = Figure(figsize=(5, 4), dpi=50)
        self.canvas1 = FigureCanvasTkAgg(self.f1, master=self)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(row=3,rowspan=5,column=9)

        self.blankLabel2 = Label(self, text='').grid(row=10, column=5)


        f2 = plt.figure(num=2, figsize=(5, 4), dpi=50)
        self.canvas2 = FigureCanvasTkAgg(f2, master=self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=3,rowspan=5,column=17)

        self.blankLabel3 = Label(self, text='').grid(column=17)

    def getData(self):
        self.mean = float(self.input_mean.get())  # 获取输入的内容
        self.variance = float(self.input_variance.get())  # 获取输入的内容
        self.n_iter = int(self.input_niter.get())
        messagebox.showinfo('Data Information', 'Mean:'+str(self.mean)+', Variance:'+str(self.variance))  # 显示输出
        self.sz = (self.n_iter,)
        self.z = np.random.normal(self.mean, self.variance, size=self.sz)

        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('data', cell_overwrite_ok=True)
        ii = 0
        sheet.write(ii, 0, '观测值')
        for i in range(self.z.shape[0]):
            sheet.write(ii, 0,self.z[i])
            ii += 1
        book.save('./data.xls')

    def updateplot(self):
        data = xlrd.open_workbook('./data.xls')
        table = data.sheets()[0]
        first_col = table.col_values(0)
        self.z = np.array(first_col)
        self.n_iter = self.z.shape[0]
        self.sz = (self.n_iter,)
        f1 = plt.figure(num=1, figsize=(5, 4), dpi=50)
        self.canvas1 = FigureCanvasTkAgg(f1, master=self)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(row=3,rowspan=5,column=9)

        f2 = plt.figure(num=2, figsize=(5, 4), dpi=50)
        self.canvas2 = FigureCanvasTkAgg(f2, master=self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=3,rowspan=5,column=17)

    def plot(self):
        plt.close(1)
        plt.close(2)
        #self.f1.close()
        data = xlrd.open_workbook('./data.xls')
        table = data.sheets()[0]
        first_col = table.col_values(0)
        self.z = np.array(first_col)
        self.n_iter = self.z.shape[0]
        self.sz = (self.n_iter)
        Q = 1e-5  # 处理方差
        xhat = np.zeros(self.sz)  # 滤波估计值
        P = np.zeros(self.sz)  # 滤波估计协方差矩阵
        xhatminus = np.zeros(self.sz)  # 估计值
        Pminus = np.zeros(self.sz)  # a估计协方差矩阵
        K = np.zeros(self.sz)
        R = 0.1 ** 2  # 估计时的测量方差 影响滤波效果

        #初始化
        xhat[0] = 0.0
        P[0] = 1.0
        for k in range(1, self.n_iter):
            # 预测
            xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
            Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
            # 更新
            K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
            xhat[k] = xhatminus[k] + K[k] * (self.z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
            P[k] = (1 - K[k]) * Pminus[k]

        self.f1 = plt.figure(num=1, figsize=(5, 4), dpi=50)
        #plt.figure()
        plt.plot(self.z, 'ko', label='noisy measurements')  # 观测值
        plt.plot(xhat, 'b-', label='a posteri estimate')  # 滤波估计值
        if not self.mean is None:
            plt.axhline(self.mean, color='r', label='truth value')  # 真实值
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Voltage')
        #plt.show()
        self.canvas1 = FigureCanvasTkAgg(self.f1, master=self)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(row=3,rowspan=5,column=9)

        f2 = plt.figure(num=2, figsize=(5, 4), dpi=50)
        #plt.figure()
        valid_iter = range(1, self.n_iter)  # Pminus not valid at step 0
        plt.plot(valid_iter, Pminus[valid_iter], label='a priori error estimate')
        plt.xlabel('Iteration')
        plt.ylabel('$(Voltage)^2$')
        plt.setp(plt.gca(), 'ylim', [0, .01])
        #plt.show()
        self.canvas2 = FigureCanvasTkAgg(f2, master=self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=3,rowspan=5,column=17)
        self.mean = None

app = Application()
app.master.title("Ensemble Kalman filter")  # 窗口标题

app.mainloop()  # 主消息循环