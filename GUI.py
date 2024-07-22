# ('南京林业‘）
import pandas as pd
from PIL import ImageTk, Image
import numpy as np
import tkinter as tk
from tkinter import ttk
import xgboost as xgb

print(tk.TkVersion)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML predication model")
        self.geometry("940x668")
        self.tabControl = ttk.Notebook(self)
        self.tabControl.pack(expand=1, fill="both")

        # 添加第一个功能选项卡
        self.build_tab_1()

        # 添加第二个功能选项卡
        self.build_tab_2()

        self.thick = tk.StringVar()
        self.thick.set("1.5")

    def build_tab_1(self):
        self.tab1 = tk.Frame(self.tabControl)

        self.frame = tk.Frame(self.tab1, bd=0, relief="solid",highlightbackground="white", highlightthickness=2,bg="#2F82A4")
        self.frame.grid(column=0, row=0, padx=0, pady=10, sticky="w")

        #添加第一个功能的输入框
        tk.Label(self.frame, text="Failure mode prediction", font=("Arial", 16,"bold",),bg="#2F82A4",fg="white",bd=0).grid(
            column=0, row=0, pady=15, padx=0,ipadx=350)

        tk.Label(self.tab1, text="f_t (MPa)  ", font=("Arial", 11)).grid(
            column=0, row=1, padx=20,pady=10, sticky='w')
        self.pf1 = tk.Entry(self.tab1, width=20)
      #  self.pf1 = tk.Entry(self.tab1,width=20)
        self.pf1.place(x=210, y=98)
        #self.pf1.grid(column=1, row=1, pady=5, padx=5, sticky='w')

        tk.Label(self.tab1, text="t_f (mm)   ", font=("Arial", 11)).grid(
            column=0, row=2,padx=20, pady=10, sticky='w')
        self.f_t1 = tk.Entry(self.tab1,width=20)
        self.f_t1.place(x=210, y=138)
       # self.f_t1.grid(column=1, row=2, pady=5, padx=5, sticky='w')
        tk.Label(self.tab1, text="b_f (mm)  ", font=("Arial", 11)).grid(column=0,  row=3,padx=20,pady=10,sticky='w')
        self.t_f1 = tk.Entry(self.tab1,width=20)
        self.t_f1.place(x=210, y=178)
        #self.t_f1.grid(column=1, row=3, pady=5, padx=5, sticky='w')
        tk.Label(self.tab1, text="f_y (MPa)  ", font=("Arial", 11)).grid(
            column=0, row=4, padx=20,pady=10, sticky='w')
        self.E_f1 = tk.Entry(self.tab1,width=20)
        self.E_f1.place(x=210, y=218)
        #self.E_f1.grid(column=1, row=4, pady=5, padx=5, sticky='w')
        tk.Label(self.tab1, text="rhol ", font=("Arial", 11)).grid(
            column=0, row=5,padx=20, pady=10, sticky='w')
        self.f_y1 = tk.Entry(self.tab1,width=20)
        self.f_y1.place(x=210, y=258)
        #self.f_y1.grid(column=1, row=5, pady=5, padx=5, sticky='w')
        tk.Label(self.tab1, text="alpha / d   ",
                 font=("Arial", 11)).grid(column=0, row=6, padx=20,pady=5, sticky='w')
        self.A_s1 = tk.Entry(self.tab1,width=20)
        self.A_s1.place(x=210, y=298)
        #self.A_s1.grid(column=1, row=6, pady=5, padx=5, sticky='w')
        tk.Label(self.tab1, text="anchor (0 for no,1 for yes)  ",
                 font=("Arial", 11)).grid(column=0, row=7, padx=20,pady=10, sticky='w')
        self.a1 = tk.Entry(self.tab1,width=20)
        self.a1.place(x=210, y=338)
        #self.a1.grid(column=1, row=7, pady=5, padx=5, sticky='w')
       # tk.Label(self.tab1, text="Anchor(0 for no,1 for yes)",
        #         font=("Times New Romans", 11)).grid(column=0, row=8, pady=5, sticky='w')
        #self.Anchor1 = tk.Entry(self.tab1,width=20)
        #self.Anchor1.place(x=210, y=298)
       # self.Anchor1.grid(column=1, row=8, pady=5, padx=5, sticky='w')
        tk.Label(self.tab1, text="E_s(GPa)  ",
                 font=("Arial", 11)).grid(column=0, row=8, padx=20,pady=10, sticky='w')
        self.E_s1 = tk.Entry(self.tab1,width=20)
        self.E_s1.place(x=210, y=378)
        #self.E_s1.grid(column=1, row=9, pady=5, padx=5, sticky='w')

        tk.Label(self.tab1, text="", font=("Arial", 13), fg='green').grid(column=0, row=12, padx=20,pady=5,
                                                                                     sticky='w')
        tk.Label(self.tab1, text="\tThis GUI is developed by department of Civil Engineering for Nanjing Forestry University", font=("Arial", 12,"bold"),
                 fg='#2F82A4', height=3).grid(column=0, row=12, pady=50, sticky='w')


        # 添加第一个功能的结果框
        tk.Label(self.tab1, text="Prediction Result:", font=("Arial", 12)).grid(column=0, row=10, pady=30, padx=20, sticky='w')
        self.pred1 = tk.Label(self.tab1, text="", font=("Arial", 13,'bold'), fg='red')
        self.pred1.place(x=180,y=445)


        # 添加第一个功能的“预测”按钮
        button = tk.Button(self.tab1, text="Predict", command=self.predict_fail_mode,font=("Arial", 12,"bold"),bg='#2F82A4',fg="#FFFFFF")
        button.grid(column=0, row=11, pady=0, padx=20, sticky='w')

        # 将第一个选项卡添加到选项卡控件
        self.tabControl.add(self.tab1, text="Failure mode")

        image = Image.open('figure3.png')
        resized_image = image.resize((500, 500))  # 将图片调整为200x200的尺寸
        # 创建PhotoImage对象并在标签中显示图片
        photo2 = ImageTk.PhotoImage(resized_image)
        label = tk.Label(width=500, height=500)
        label.image = photo2  # 保存对图像的引用
        label.configure(image=photo2)
        label.place(x=400, y=110)

        image2 = Image.open('NJFU.png')  # 替换为你想要添加的图片
        resized_image = image2.resize((60, 60))  # 将图片调整为200x200的尺寸
        photo3 = ImageTk.PhotoImage(resized_image)
        label = tk.Label(width=60, height=60)
        label.image = photo3  # 保存对图像的引用
        label.configure(image=photo3)
        label.place(x=10, y=600)



    def build_tab_2(self):
        self.tab2 = tk.Frame(self.tabControl)

        self.frame2 = tk.Frame(self.tab2, bd=0, relief="solid",highlightbackground="white", highlightthickness=2,bg="#006400")
        self.frame2.grid(column=0, row=0, padx=0, pady=10, sticky="w")
        # 添加第二个功能的输入框
        tk.Label(self.frame2, text="flexural strnegth", font=("Arial", 16,"bold"),bg='#006400',fg="white",bd=0).grid(
            column=0, row=0, pady=15, padx=0,ipadx=350)

        tk.Label(self.tab2, text="rhol ", font=("Arial", 11)).grid(
            column=0, row=1,padx=0,pady=10, sticky='w')
        self.pf2 = tk.Entry(self.tab2,width=20)
        self.pf2.place(x=210, y=90)
        #self.pf2.grid(column=1, row=1, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="alpha / d   ", font=("Arial", 11)).grid(
            column=0, row=2, padx=0,pady=10, sticky='w')
        self.f_t2 = tk.Entry(self.tab2,width=20)
        self.f_t2.place(x=210, y=133)
        #self.f_t2.grid(column=1, row=2, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="f_y(MPa)  ",
                 font=("Arial", 11)).grid(column=0, row=3,padx=0,pady=10, sticky='w')
        self.a2 = tk.Entry(self.tab2,width=20)
        self.a2.place(x=210, y=176)
        #self.a2.grid(column=1, row=3, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="f_t(MPa)  ", font=("Arial", 11)).grid(
            column=0, row=4, padx=0,pady=10, sticky='w')
        self.E_f2 = tk.Entry(self.tab2,width=20)
        self.E_f2.place(x=210, y=219)
        #self.E_f2.grid(column=1, row=4, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="f_fu(MPa)  ", font=("Arial", 11)).grid(
            column=0, row=5, padx=0,pady=10, sticky='w')
        self.f_y2 = tk.Entry(self.tab2,width=20)
        self.f_y2.place(x=210, y=262)
        #self.f_y2.grid(column=1, row=5, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="anchor(0 for no,1 for yes)",
                 font=("Arial", 11)).grid(column=0, row=6, padx=0,pady=10, sticky='w')
        self.A_s2 = tk.Entry(self.tab2,width=20)
        self.A_s2.place(x=210, y=305)
        #self.A_s2.grid(column=1, row=6, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="rou_f",
                 font=("Arial", 11)).grid(column=0, row=7, padx=0,pady=10, sticky='w')
        self.Anchor2 = tk.Entry(self.tab2,width=20)
        self.Anchor2.place(x=210, y=348)
        #self.Anchor2.grid(column=1, row=7, pady=5, padx=5, sticky='w')
        tk.Label(self.tab2, text="Failure Mode", font=("Arial", 11)).grid(column=0, row=8, padx=0,pady=10,
                                                                                     sticky='w')
        tk.Label(self.tab2, text="(0 for CC, 1 for FR，2 for IC, 3 for PE)", font=("Arial", 11)).grid(column=0, row=9, pady=5,
                                                                                     sticky='w')


        self.FM2 = tk.Entry(self.tab2,width=20)
        self.FM2.place(x=210, y=391)
        #self.FM2.grid(column=1, row=8, pady=5, padx=5, sticky='w')

        tk.Label(self.tab2,
                 text="\tThis GUI is developed by department of Civil Engineering for Nanjing Forestry University",
                 font=("Arial", 12, "bold"),
                 fg='#006400', height=3).grid(column=0, row=12, pady=50, sticky='w')

        # 添加第二个功能的结果框
        tk.Label(self.tab2, text="Prediction Result:", font=("Arial", 12)).grid(column=0, row=10, pady=30, padx=0, sticky='w')
        self.pred2 = tk.Label(self.tab2, text="", font=("Arial", 13), fg='blue')
        self.pred2.place(x=180,y=465)
        #self.pred2.grid(column=1, row=9, pady=5, padx=10, sticky='w')

        # 添加第二个功能的“预测”按钮
        button = tk.Button(self.tab2, text="Predict", command=self.predict_load_capacity,font=("Arial", 12, "bold") ,bg='#006400', fg="white")
        button.grid(column=0, row=11, pady=0, padx=0, sticky='w')



        # 将第二个选项卡添加到选项卡控件
        self.tabControl.add(self.tab2, text="Ultimate flexural strength of FRP beams")

    def predict_fail_mode(self):
        '''
       预测零件失效模式
        '''
        # 获取用户输入的特征值
        pf = float(self.pf1.get())
        f_t = float(self.f_t1.get())
        t_f = float(self.t_f1.get())
        E_f = float(self.E_f1.get())
        f_y = float(self.f_y1.get())
        A_s = float(self.A_s1.get())
        a = float(self.a1.get())
       # Anchor = float(self.Anchor1.get())
        E_s = float(self.E_s1.get())
        # 构建待预测的特征向量
        y = np.array([[pf, f_t, t_f, E_f, f_y, A_s, a, E_s]])

        # 加载分类模型，并预测零件失效模式

        clf = xgb.XGBClassifier()
        clf.load_model('classifier.model')
        y_pred = clf.predict(y)[0]
        proba = clf.predict_proba(y)
        # get the index of the highest probability
        label = clf.classes_[proba.argmax()]

        # 根据预测的数字标签，转换成相应的文字标签
        if label == 0:
            result = "Concrete Crush(CC)"
        elif label == 1:
            result = "FRP Rupture(FR)"
        elif label == 2:
            result = "Intermediate Crack \nDebonding(IC)"
        else:
            result = "End debonding(PE)"

        # 将预测结果显示在窗口中
        self.pred1.configure(text=result)

    def predict_load_capacity(self):
        '''
        预测梁的最大荷载
        '''
        # 获取用户输入的特征值
        PF = float(self.pf2.get())
        f_t = float(self.f_t2.get())
        a = float(self.a2.get())
        E_f = float(self.E_f2.get())
        f_y = float(self.f_y2.get())
        A_s = float(self.A_s2.get())
        Anchor = float(self.Anchor2.get())
        FM = float(self.FM2.get())
        # 构建待预测的特征向量
        x = np.array([[PF, f_t, a, E_f, f_y, A_s, Anchor, FM]])

        # 加载回归模型，并预测最大荷载
        reg = xgb.XGBRegressor()
        reg.load_model('regressor.model')
        X = np.array(x)
        y_pred_reg = reg.predict(X)[0]

        # 将预测结果显示在窗口中
        self.pred2.configure(text="{:.2f} kN*m".format(y_pred_reg))

    # 主程序入口


if __name__ == "__main__":
    # 加载分类模型
    clf = xgb.XGBClassifier()
    clf.load_model('classifier.model')

    # 加载回归模型
    reg = xgb.XGBRegressor()
    reg.load_model('regressor.model')

    # 启动 Tkinter 窗口程序
    app = App()
    app.mainloop()