from tkinter import *
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import xlwt
import xlrd
from similarity import get_alld, get_similraity_index
from query import Query

class Application(Frame):  # 从Frame派生出Application类，它是所有widget的父容器
    def __init__(self, master=None):  # master即是窗口管理器，用于管理窗口部件，如按钮标签等，顶级窗口master是None，即自己管理自己
        Frame.__init__(self, master)
        self.pack()  # 将widget加入到父容器中并实现布局
        self.createWidgets()
        self.mean = None
        self.variance = None

    def createWidgets(self):
        self.helloLabel = Label(self, text='烟叶质量综合评价系统').grid(row=0,column=7)  # 创建一个标签显示内容到窗口
        self.blankLabel1 = Label(self, text='').grid(row=2, column=7)

        self.question_mean = Label(self, text='输入问题:').grid(row=5,column=7,sticky=E)
        self.input_question = Entry(self)  # 创建一个输入框，以输入均值
        self.input_question.grid(row=5,column=7)

        self.dataButton = Button(self, text='分析问题', command=self.getData)  # 创建一个hello按钮，点击调用hello方法，实现输出
        self.dataButton.grid(row=4,column=1)

        self.plotButton = Button(self, text='获得答案', command=self.plot)
        self.plotButton.grid(row=5, column=1)

        self.quitButton = Button(self, text='退出', command=self.quit)  # 创建一个Quit按钮，实现点击即退出窗口
        self.quitButton.grid(row=6, column=1)

        self.str = StringVar()

        # self.question_answer = Label(self, textvariable=str).grid(row=5,sticky=E, column=10, columnspan=6, rowspan=5)
        # self.str.set('显示答案！')

        self.t = Text(self, width=100, height=10)
        self.t.grid(row=5, column=10)
        self.t.insert("insert", "答案显示:")
        #self.t.insert("end", "Python.com!")
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


        self.blankLabel3 = Label(self, text='').grid(column=17)

    def getData(self):
        self.raw_question = self.input_question.get()  # 获取输入的内容
        messagebox.showinfo('输入信息', '问题:'+str(self.raw_question))# 显示输出

    def plot(self):
        self.t.delete(1.0,"insert")
        all_d = get_alld("question_best.csv")
        queryer = Query()
        target_question, cos_target_question = get_similraity_index(all_d, self.raw_question)
        answer = queryer.run(target_question)
        self.t.insert("insert", answer)
        print(answer)
        print("finished!")
        # self.str.set(answer)
        #pass

    def quit(self):
        """点击退出按钮时调用这个函数"""
        self.master.quit()
        self.master.destroy()
        #self.master.quit()
        #self.destroy()
        #root.quit()  # 结束主循环
        #root.destroy()  # 销毁窗口

app = Application()
app.master.title("问答系统")  # 窗口标题

app.mainloop()  # 主消息循环