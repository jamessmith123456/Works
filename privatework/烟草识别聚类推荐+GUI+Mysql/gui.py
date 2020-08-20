from tkinter import *
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import xlwt
import xlrd
from preprocess import *

class Application(Frame):  # 从Frame派生出Application类，它是所有widget的父容器
    def __init__(self, master=None):  # master即是窗口管理器，用于管理窗口部件，如按钮标签等，顶级窗口master是None，即自己管理自己
        Frame.__init__(self, master)
        self.pack()  # 将widget加入到父容器中并实现布局
        self.createWidgets()

        self.all_example_chemistry = None
        self.all_example_score = None
        self.all_example_feature = None
        self.all_example_human = None

    def createWidgets(self):
        self.helloLabel = Label(self, text='烟叶质量综合评价系统').grid(row=0,column=7)  # 创建一个标签显示内容到窗口
        self.blankLabel1 = Label(self, text='').grid(row=2, column=7)

        self.question_mean1 = Label(self, text='Excel:').grid(row=4,sticky=E)
        self.input_path = Entry(self)  # 创建一个输入框，以输入均值
        self.input_path.grid(row=4,column=1)

        self.question_mean3 = Label(self, text='Save:').grid(row=5,sticky=E)
        self.save_logo = Entry(self)  # 创建一个输入框，以输入均值
        self.save_logo.grid(row=5,column=1)

        self.question_mean2 = Label(self, text='Table:').grid(row=6,sticky=E)
        self.input_table = Entry(self)  # 创建一个输入框，以输入均值
        self.input_table.grid(row=6,column=1)

        self.examplename1 = Label(self, text='Example1').grid(row=8,sticky=E)
        self.example1 = Entry(self)  # 创建一个输入框，以输入均值
        self.example1.grid(row=8,column=1)

        self.examplename2 = Label(self, text='Example2').grid(row=9,sticky=E)
        self.example2 = Entry(self)  # 创建一个输入框，以输入均值
        self.example2.grid(row=9,column=1)

        self.dataButton = Button(self, text='读取Excel', command=self.getData)  # 创建一个hello按钮，点击调用hello方法，实现输出
        self.dataButton.grid(row=4,column=7,sticky=W)

        self.dataButton = Button(self, text='读取Mysql', command=self.getMySql)  # 创建一个hello按钮，点击调用hello方法，实现输出
        self.dataButton.grid(row=6,column=7,sticky=W)

        self.quitButton = Button(self, text='PCA处理人工特征', command=self.pca)  # 创建一个Quit按钮，实现点击即退出窗口
        self.quitButton.grid(row=7, column=7,sticky=W)

        self.quitButton = Button(self, text='Fuz处理化学特征', command=self.fuz)  # 创建一个Quit按钮，实现点击即退出窗口
        self.quitButton.grid(row=8, column=7,sticky=W)

        self.quitButton = Button(self, text='神经网络', command=self.neu)  # 创建一个Quit按钮，实现点击即退出窗口
        self.quitButton.grid(row=9, column=7,sticky=W)

        self.quitButton = Button(self, text='退出', command=self.quit)  # 创建一个Quit按钮，实现点击即退出窗口
        self.quitButton.grid(row=10, column=1,sticky=W)

        self.str = StringVar()

        # self.question_answer = Label(self, textvariable=str).grid(row=5,sticky=E, column=10, columnspan=6, rowspan=5)
        # self.str.set('显示答案！')

        self.t = Text(self, width=100, height=10)
        self.t.grid(row=4,rows=6, column=10)
        self.t.insert("insert", "处理显示:")
        self.blankLabel3 = Label(self, text='').grid(column=17)

    def getData(self):
        try:
            self.raw_path = self.input_path.get()  # 获取输入的内容
            self.save = self.save_logo.get()  # 获取输入的内容
            if "感官质量" in self.raw_path:
                self.all_example_feature, self.chemistry_name, self.human_name, self.all_example_chemistry, self.all_example_human, self.all_example_score = get_data_1(self.raw_path)
                self.all_example_chemistry = remove_nan(self.all_example_chemistry)  # (91,111)
                self.all_example_human = remove_nan(self.all_example_human)  # (91,8)  #all_example_score的shape是(91,)
                if self.all_example_chemistry.shape[1]>10:
                    self.plot(str(self.all_example_chemistry[:,0:10]))
                else:
                    self.plot(str(self.all_example_chemistry))
                if self.save.upper()=='Y':
                    feature_id = [item['example_name'] for item in self.all_example_feature]
                    write_to_mysql(feature_id, self.all_example_chemistry, self.chemistry_name, 'chemistry_feature')
                    write_to_mysql(feature_id, self.all_example_human, self.human_name, 'human_feature')
            else:
                self.all_example_feature, self.chemistry_name, self.all_example_chemistry = get_data_2(self.raw_path)
                self.all_example_chemistry = remove_nan(self.all_example_chemistry)  # (43,103)
                if self.all_example_chemistry.shape[1]>10:
                    self.plot(str(self.all_example_chemistry[:,0:10]))
                else:
                    self.plot(str(self.all_example_chemistry))
                if self.save.upper() == 'Y':
                    feature_id = [item['example_name'] for item in self.all_example_feature]
                    write_to_mysql(feature_id, self.all_example_chemistry, self.chemistry_name, 'chemistry_feature2')

            messagebox.showinfo('文件读取', '文件读取成功')# 显示输出
        except:
            messagebox.showinfo('异常提示', '检查excel文件存在路径')

    def getMySql(self):
        try:
            self.raw_table = self.input_table.get()  # 获取输入的内容
            if self.raw_table=='human_feature':
                self.feature_id1, self.all_example_human = read_from_mysql(self.raw_table) #chemistry_feature   human_feature   chemistry_feature2
                if self.all_example_human.shape[1]>10:
                    self.plot(str(self.all_example_human[:,0:10]))
                else:
                    self.plot(str(self.all_example_human))
            else:
                self.feature_id1, self.all_example_chemistry = read_from_mysql(self.raw_table) #chemistry_feature   human_feature   chemistry_feature2
                if self.all_example_chemistry.shape[1]>10:
                    self.plot(str(self.all_example_chemistry[:,0:10]))
                else:
                    self.plot(str(self.all_example_chemistry))
        except:
            messagebox.showinfo('异常提示', '检查数据库名称或网络连接')

    def plot(self, result):
        self.t.delete(1.0,"insert")
        self.t.insert("insert", result)
        print(result)
        print("finished!")

    def pca(self):
        try:
            if self.all_example_human.any() !=None:
                new_all_example_human, pca_components_, pca_explained_variance_ratio_ = pca_analysis(self.all_example_human)
                if new_all_example_human.shape[1] > 10:
                    self.plot(str(new_all_example_human[:, 0:10]))
                else:
                    self.plot(str(new_all_example_human))
        except:
            messagebox.showinfo('异常提示', '未读取人工特征')

    def get_example(self,path):
        data = xlrd.open_workbook(path)
        table = data.sheet_by_index(0)
        all_example = []
        for i in range(table.nrows):
            temp_example = []
            for j in range(table.ncols):
                temp_example.append(table.cell(i, j).value)
            all_example.append(temp_example)
        all_example = np.array(all_example)
        return all_example

    def fuz(self):
        try:
            self.pre_example1 = self.example1.get()
            example = self.get_example(self.pre_example1)
        except:
            messagebox.showinfo('异常提示', '未提供测试样本')

        try:
            if self.all_example_chemistry.any() != None and self.all_example_score.any() != None:
                all_score = []
                for i in range(example.shape[0]):
                    score = fuzzy_comprehensive_evaluation(self.all_example_chemistry, self.all_example_score,
                                                           example[i, :])
                    all_score.append(str(score))
                self.plot(str(','.join(all_score)))
        except:
            messagebox.showinfo('异常提示', '未读取化学特征及得分')

    def neu(self):
        try:
            self.pre_example2 = self.example2.get()
            example = self.get_example(self.pre_example2)
        except:
            messagebox.showinfo('异常提示', '未提供测试样本')
        try:
            if self.all_example_chemistry.any() !=None and self.all_example_score.any() !=None:
                all_score = []
                for i in range(example.shape[0]):
                    x_test = example[i,:]
                    x_test = x_test[np.newaxis, :]
                    predict = network(self.all_example_chemistry, self.all_example_score, x_test)
                    all_score.append(str(predict))
                self.plot(str(','.join(all_score)))
        except:
            messagebox.showinfo('异常提示', '未读取化学特征及得分')

    def quit(self):
        """点击退出按钮时调用这个函数"""
        self.master.quit()
        self.master.destroy()
        #self.master.quit()
        #self.destroy()
        #root.quit()  # 结束主循环
        #root.destroy()  # 销毁窗口

app = Application()
app.master.title("烟叶综合评价系统")  # 窗口标题

app.mainloop()  # 主消息循环