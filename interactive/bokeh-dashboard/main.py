from bokeh.layouts import column
from bokeh.models import Tabs, Panel
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure, output_file, show
from bokeh.io import curdoc
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from bokeh.server.server import Server
from bokeh.themes import Theme
import time
import numpy as np
from tornado.ioloop import IOLoop, PeriodicCallback
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio

import pathlib, sys
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.append(parent_dir)
from functions import DATA, new_random_data



def tab_current_generation(dash_data):
    source = dash_data['source']
    plot = figure(title="test")
    plot.line('x', 'y', source=source)

    def callback(attr, old, new):
        source.data = dict(x=np.random.random(100), y=np.random.random(100))

    slider = Slider(start=0, end=30, value=0, step=1, title="Widget")
    slider.on_change('value', callback)

    tab = Panel(child=column(slider, plot), title='Current Generation')
    return tab

def tab_resource_usage(dash_data):
    source = dash_data['resources']

    plot = figure(title="test")
    plot.line('x', 'y', source=source)

    def callback(attr, old, new):
        source.data = dict(x=np.random.random(100), y=np.random.random(100))

    slider = Slider(start=0, end=30, value=0, step=1, title="Widget")
    slider.on_change('value', callback)

    tab = Panel(child=column(slider, plot), title='Resources')
    return tab

def dashboard(doc):
    # data = dict(source=ColumnDataSource(data=dict(x=np.random.random(100), y=np.random.random(100))),
    #             resources=ColumnDataSource(data=dict(x=np.random.random(100), y=np.random.random(100))))

    tab1 = tab_current_generation(DATA)
    tab2 = tab_resource_usage(DATA)
    doc.add_root(Tabs(tabs=[tab1, tab2]))
    doc.add_periodic_callback(callback_data, 1000)

def optimization_dummy(n_gens=10):
    print('starting the dummy call')
    for i in range(300):
        time.sleep(2)
        print('next')
        print(DATA)

def callback_data():
    # print('in callback_data')
    DATA['source'].data = dict(x=np.random.random(100), y=np.random.random(100)) #new_random_data()

import os

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server = Server({'/': dashboard})
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.add_callback(optimization_dummy)
    server.io_loop.start()

    # server.io_loop.initialize()
    for i in range(10):
        server.io_loop.current()
        time.sleep(1)
        print(i)
    server.io_loop.stop()
    print()

    # print('after start')
    # server.io_loop.stop()
    # # server.io_loop.start()
    # # server.io_loop.stop()

    # server.io_loop.add_callback(server.show, "/")
    # server.io_loop.add
    # server.io_loop.add_callback(callback_data)
    # server.show('/')
    # server.io_loop.start()
    # server.io_loop.stop()
    # for i in range(10):
    #     time.sleep(1)
    #     print(f"i : {i}")
    #     server.io_loop.add_callback_from_signal(callback_data)
    #     # server.io_loop
    #     # print(server.get_sessions('/'))
    #     # server.io_loop.add_callback(callback_data)

    # server.io_loop.start()
        # server.
    # optimization_dummy(server, n_gens=10)



