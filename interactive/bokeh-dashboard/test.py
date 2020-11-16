from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio

doc = curdoc()

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1234")
socket.setsockopt(zmq.SUBSCRIBE, b"")

def update(new_data):
    source.stream(new_data, rollover=50)

async def loop():
    while True:
        new_data = await socket.recv_pyobj()
        doc.add_next_tick_callback(partial(update, new_data))

source = ColumnDataSource(data=dict(x=[0], y=[0]))

plot = figure(height=300)
plot.line(x='x', y='y', source=source)

doc.add_root(plot)
IOLoop.current().spawn_callback(loop)



import time
import random
import zmq

context = zmq.Context.instance()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://127.0.0.1:1234")

t = 0
y = 0

while True:
    time.sleep(1.0)
    t += 1
    y += random.normalvariate(0, 1)
    pub_socket.send_pyobj(dict(x=[t], y=[y]))