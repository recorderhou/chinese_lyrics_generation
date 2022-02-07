import sys
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QMessageBox
from PyQt5.QtWidgets import QTextEdit,QComboBox,QLineEdit,QMainWindow,QHBoxLayout,QVBoxLayout
from PyQt5.QtGui import QPalette,QBrush,QPixmap,QFont
from PyQt5.QtCore import QCoreApplication
from random import randint
from Final_Shallow_Model import shallow_generate_lyrics
from Final_Deep_Model import generate_lyrics_attention
import tensorflow as tf

song_list = ['五言绝句','七言绝句','七言律诗','燕园情','青春大概','我的一八九八','未名湖是个海洋','转身之间','少年游','我喜欢-北大版','春风十里-北大版','夜空中最亮的星']

def Generate(source,template):
    with tf.Graph().as_default() as g:
        temp = generate_lyrics_attention(source)
    result = shallow_generate_lyrics(template,temp)
    return result


class MyWidget(QWidget):

    def __init__(self):

        super().__init__()
        self.initUI()

    def initUI(self):

        self.setGeometry(300,200,660,480)
        self.setWindowTitle('歌词生成结果')
        self.setObjectName("MyWidget")
        palette = QPalette()
        palette.setBrush(self.backgroundRole(), QBrush(QPixmap("./gui/7.jpg")))
        self.setPalette(palette)
        self.setAutoFillBackground(False)
        self.text1 = QTextEdit(self)
        self.text1.setGeometry(200,50,300,400)
        
        self.text1.setStyleSheet("background-image:url(./gui/7.jpg)")

        self.bt1 = QPushButton('再来一首',self)
        self.bt1.setStyleSheet("QPushButton{background-image: url(./gui/6.jpg)}")
        self.bt1.clicked.connect(self.close)

        self.bt2 = QPushButton('直接退出',self)
        self.bt2.setStyleSheet("QPushButton{background-image: url(./gui/6.jpg)}")
        self.bt2.clicked.connect(QCoreApplication.quit)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.text1)
        hbox1 = QHBoxLayout()
        hbox1.addStretch(10)
        hbox1.addWidget(self.bt1)
        hbox1.addStretch(1)
        hbox1.addWidget(self.bt2)
        vbox1.addLayout(hbox1)

        self.setLayout(vbox1)

    def Print(self,Lyrics):
        self.text1.setText(Lyrics)


class MyMainWindow(QWidget):

    def __init__(self,child):

        super().__init__()
        self.child = child
        self.ininUI()
    
    def ininUI(self):

        self.setGeometry(300,200,660,480)
        self.setWindowTitle('奉码填词')
        self.setObjectName("MyMainWindow")
        palette = QPalette()
        palette.setBrush(self.backgroundRole(), QBrush(QPixmap("./gui/3.jpg")))
        #palette.setColor(QPalette.Background, Qt.red)
        self.setPalette(palette)
        self.setAutoFillBackground(False)
        #self.setStyleSheet("#MyMainWindow{border-image:url(/Users/sunny2001/Desktop/喜欢的小东西/电脑壁纸/1.jpg);}")
        self.bt1 = QPushButton('生成歌词',self)
        self.bt1.setGeometry(100,150,140,100)
        self.bt1.setToolTip('<b>点击这里生成歌词：</b>')
        self.bt1.setStyleSheet("QPushButton{background-image: url(./gui/2.jpg)}")
        self.bt1.clicked.connect(self.generator)
        self.bt2 = QPushButton('退出',self)
        self.bt2.setStyleSheet("QPushButton{background-image: url(./gui/2.jpg)}")
        self.bt2.clicked.connect(QCoreApplication.quit)

        self.text1 = QTextEdit('请在这里输入你想要转化成歌词的文字',self)
        self.text1.selectAll()
        self.text1.setFocus()
        self.text1.setGeometry(80,60,150,30)
        self.text1.setStyleSheet("background-image:url(./gui/3.jpg)")

        self.text2 = QTextEdit('请在这里输入你想要生成歌词的格式',self)
        self.text2.selectAll()
        self.text2.setFocus()
        self.text2.setGeometry(220,60,150,30)
        self.text2.setStyleSheet("background-image:url(./gui/3.jpg)")

        self.combobox1 = QComboBox(self, minimumWidth=200)
        self.combobox1.addItem("")#先添加一个下拉菜单空位
        self.combobox1.addItems(song_list)
        self.combobox1.currentIndexChanged[str].connect(self.change_template)
        self.combobox1.setGeometry(400,10,150,30)

        self.layout = QHBoxLayout()

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.text1)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.bt1)
        hbox1.addStretch(1)
        hbox1.addWidget(self.bt2)
        hbox1.addStretch(3)
        vbox1.addLayout(hbox1)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.combobox1)
        vbox2.addWidget(self.text2)
        
        self.layout.addLayout(vbox1)
        self.layout.addLayout(vbox2)

        self.setLayout(self.layout) #设置窗体布局
        self.show()

    def change_template(self,song_name):
        if song_name == "":
            self.text2.clear()
            return
        path = "./gui/" + song_name
        with open(path,"r",encoding='utf-8') as rf:
            template = rf.read()
            self.text2.setText(template)

    def generator(self):
        source = self.text1.toPlainText()
        template = self.text2.toPlainText()
        print(template)
        Lyrics = Generate(source,template)
        self.child.Print(Lyrics)
        self.child.show()
        self.text1.clear()
        self.text2.clear()


if __name__=='__main__':
    app = QApplication(sys.argv)
    child = MyWidget()
    win = MyMainWindow(child)
    sys.exit(app.exec_())
    