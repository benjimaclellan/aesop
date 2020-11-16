from bokeh.models import ColumnDataSource
import numpy as np
import ray
import time

DATA = dict(source=ColumnDataSource(data=dict(x=np.random.random(100), y=np.random.random(100))),
            resources=ColumnDataSource(data=dict(x=np.random.random(100), y=np.random.random(100))))

def new_random_data():
    return np.random.random(100)

@ray.remote
def f(x):
    print('in remote function', x)
    time.sleep(2)
    return x * x


from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout

import sys

class DashBoard(object):
    def __init__(self):
        self._data = 20
        return

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def create_dashboard(self):
        app = QApplication([])
        w = QWidget()
        w.setWindowTitle("Musketeers")

        def button1_callback():
            print('yahadadahdadh')

        btn1 = QPushButton("Athos")
        btn1.clicked.connect(button1_callback)
        hbox = QHBoxLayout(w)

        hbox.addWidget(btn1)

        print('starting ray')
        ray.init(include_dashboard=False)

        futures = [f.remote(i) for i in range(30)]
        print(ray.get(futures))

        w.show()

        sys.exit(app.exec_())
        return

    def update_dashboard(self):
        return



d = DashBoard()
d.create_dashboard()
for i in range(100):
    print(i)