''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''

import numpy as np
import copy
import random
import string
import math

from bokeh.layouts import gridplot, column, row
from bokeh.models import BoxSelectTool, LassoSelectTool
from bokeh.plotting import curdoc, figure

import networkx as nx

from bokeh.io import output_file, show
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,)
from bokeh.models import NodesOnly
import bokeh.models.tools as tools
from bokeh.models import GraphRenderer, StaticLayoutProvider
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx
from bokeh.models import ColumnDataSource
from bokeh.models import Button, CustomJS, FileInput, TextInput, Select, Slider, Div

#%% callbacks
DEBUG = False
def update():
    plots['prop_time']['source'].data = dict(new_points().data)
    plots['prop_freq']['source'].data = dict(new_points().data)
    plots['tran_time']['source'].data = dict(new_points().data)
    plots['tran_freq']['source'].data = dict(new_points().data)

def edge_callback(attr,old,new):
    if DEBUG: print(f"edge select {attr, old, new}")
    update()
    update_textbox(attr, old, new)

def button_callback():
    G = random_graph()
    update_node_positions(G)
    div.text = f"something new {random.choice(string.ascii_lowercase)}"

def file_input_callback(attr,old,new):
    if DEBUG: print(f'file input {attr, old, new}')

def update_textbox(attr, old, new):
    div.text = f"New edge selected: edge index {new}"

def slider_callback(attr, old, new):
    if DEBUG: print(f'slider {attr, old, new}')

def dropdown_callback(attr, old, new):
    if DEBUG: print(f'select {attr, old, new}')

def new_points():
    x = np.linspace(-5, 5, 500)
    y = 5 * np.exp(-x**2) + np.random.uniform(0.0, 1.5, x.shape[0])
    return ColumnDataSource(data=dict(x=x, y=y))

def convert_from_networkx(graph_nx):
    pass

def random_graph():
    G = nx.random_regular_graph(np.random.randint(1, 4), np.random.choice([6,8,10]))
    return G

def update_node_positions(_graph):
    node_indices = [node for node in _graph.nodes]
    graph_renderer.node_renderer.data_source.data = dict(index=node_indices, type=[random.choice(string.ascii_lowercase) for _ in _graph.nodes])

    graph_layout = nx.kamada_kawai_layout(_graph)
    data = dict(start=[edge[0] for edge in _graph.edges],
                end=[edge[1] for edge in _graph.edges],
                type=['q' for _ in _graph.edges])
    graph_renderer.edge_renderer.data_source.data = data

    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)



#%%
# add the buttons and user-interface components
button = Button(label="Load graph", button_type="success")
button.on_click(button_callback)

filepath_input = TextInput(value='path/to/graph/file/graph.pkl', title='')
filepath_input.on_change('value', file_input_callback)

dropdown = Select(title="Parameter to control:", value="option1", options=["option1", "option2", "option3"])
dropdown.on_change('value', dropdown_callback)

slider = Slider(start=0.1, end=1.0, value=1, step=.1, title="Parameter value")
slider.on_change('value', slider_callback)

div = Div(text="""this is a HTML Div where the node info will be printed when an edge or node is selected in the graph""", sizing_mode='stretch_both')

widgets = column(filepath_input, button, dropdown, slider, width=200, height=100)

#%% main graph
graph = random_graph()

plot_graph = Plot(sizing_mode="stretch_both", width=600,
                  x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
hover = HoverTool(line_policy='interp', tooltips=[("type", "@type"),])
plot_graph.add_tools(hover, TapTool(), BoxSelectTool())

graph_renderer = GraphRenderer()
update_node_positions(graph)


## set interaction visuals
graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

# graph_renderer.inspection_policy = EdgesAndLinkedNodes()
graph_renderer.inspection_policy = NodesOnly()
graph_renderer.selection_policy = EdgesAndLinkedNodes()
graph_renderer.edge_renderer.data_source.selected.on_change("indices", edge_callback)

plot_graph.renderers.append(graph_renderer)


#%% temporal electric field

plots = {
    'prop_time':{'title':'Optical Intensity', 'xlabel':'Time', 'ylabel':'a.u.', 'source':new_points()},
    'prop_freq':{'title':'', 'xlabel':'Freq', 'ylabel':'a.u.', 'source':new_points()},
    'tran_time':{'title':'Transfer Functions', 'xlabel':'Time', 'ylabel':'a.u.', 'source':new_points()},
    'tran_freq':{'title':'', 'xlabel':'Freq', 'ylabel':'a.u.', 'source':new_points()},
}
for plot, info in plots.items():
    pfig = figure(tools=[tools.PanTool(), tools.BoxZoomTool(), tools.ResetTool(), tools.SaveTool()],
                  sizing_mode="stretch_both", height=300, min_border=10, min_border_left=50,
                  x_axis_label=info['xlabel'], y_axis_label=info['ylabel'],
                  toolbar_location="left", title=info['title'], title_location='above')
    pfig.background_fill_color = "#fafafa"
    r = pfig.line(x='x', y='y', source=info['source'], alpha=0.6)
    info['plot'] = pfig

plots['tran_time']['plot'].x_range = plots['prop_time']['plot'].x_range
plots['tran_freq']['plot'].x_range = plots['prop_freq']['plot'].x_range

layout = column(row(widgets, row(div, width=200), plot_graph, height=300),
                row(plots['prop_time']['plot'], plots['prop_freq']['plot'], height=300),
                row(plots['tran_time']['plot'], plots['tran_freq']['plot'], height=300), sizing_mode='scale_height')

curdoc().add_root(layout)
curdoc().title = "ASOPE Photonic Circuit Design"
