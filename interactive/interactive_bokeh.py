''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''

import sys
import pathlib
import os
import platform
import copy
import dill
import numpy as np
import copy
import random
import string
import math
import networkx as nx
import itertools

from bokeh.layouts import gridplot, column, row
from bokeh.models import BoxSelectTool, LassoSelectTool
from bokeh.plotting import curdoc, figure

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
from bokeh.palettes import Spectral6
from bokeh.models import Paragraph, FileInput
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn, HTMLTemplateFormatter, NumberFormatter

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from model import AccessGraph
from problems.example.assets.functions import fft_, ifft_, psd_, power_
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

#%% callbacks

def edge_callback(attr,old,new):
    global main_graph

    if DEBUG: print(f"edge select {attr, old, new}")

    text = 'Selected edge models:\n'

    data = {'parameters':[], 'parameter_names': [], 'model': []}
    for index in new:
        edge = graph_renderer.edge_renderer.data_source.data['edge'][index]
        text +=  f"\t{main_graph.edges[edge]['model'].node_acronym}\n\n"

        data['parameters'] += main_graph.edges[edge]['model'].parameters
        data['parameter_names'] += main_graph.edges[edge]['model'].parameter_names
        data['model'] += [main_graph.edges[edge]['model'].node_acronym] * len(data['parameter_names'])
    table_source.data = data

        # dropdown.options = main_graph.edges[edge]['model'].parameter_names

    # here we change which lines are visible based on the selection policy of the graph
    for plot_id, plot in plots.items():
        for index in new:
            edge = graph_renderer.edge_renderer.data_source.data['edge'][index]
            if edge in plot['lines'].keys():
                for line in plot['lines'][edge]:
                    line.visible = True

        for index in list(set(old) - set(new)):
            edge = graph_renderer.edge_renderer.data_source.data['edge'][index]
            if edge in plot['lines'].keys():
                for line in plot['lines'][edge]:
                    line.visible = False

    text_display.value = text
    # text_display.text = text


def text_summary_node(model):
    summary = f"Parameters {model.parameters}\n"
    return summary


def node_callback(attr, old, new):
    if DEBUG: print(f"node select {attr, old, new}")
    text = ''
    for node in new:
        text += text_summary_node(main_graph.nodes[node]['model'])


def update_plots():
    global plots
    global main_graph

    main_graph.propagate(propagator, save_transforms=True)

    for edge in main_graph.edges:
        state = main_graph.measure_propagator(edge)

        plots['prop_time']['sources'][edge][0].data['y'] = power_(state).astype('float')
        # plots['prop_time']['sources'][edge][0] = ColumnDataSource(data=dict(x=propagator.f, y=power_(state).astype('float')))
        plots['prop_freq']['sources'][edge][0].data['y'] = np.log10(psd_(state, dt=propagator.dt, df=propagator.df).astype('float'))
        # plots['prop_freq']['sources'][edge][0] = ColumnDataSource(data=dict(x=propagator.f, y=np.log10(psd_(state, dt=propagator.dt, df=propagator.df).astype('float'))))

        if main_graph.edges[edge]['model'].transform is not None:
            for i, (dof, transform, label) in enumerate(main_graph.edges[edge]['model'].transform):
                if dof == 't':
                    plots['tran_time']['sources'][edge][i].data['y'] = transform.astype('float')
                    # plots['tran_time']['sources'][edge][i] = ColumnDataSource(data=dict(x=propagator.t, y=transform.astype('float')))
                elif dof == 'f':
                    plots['tran_freq']['sources'][edge][i].data['y'] = transform.astype('float')
                    # plots['tran_freq']['sources'][edge][i] = ColumnDataSource(data=dict(x=propagator.f, y=transform.astype('float')))


def load_graph():
    global propagator
    global main_graph
    global plots
    global palette
    global styles

    for plot_id, plot in plots.items():
        for id, lines in plot['lines'].items():
            for line in lines:
                plot['plot'].renderers.remove(line)
        plot['sources'] = {}
        plot['lines'] = {}


    filepath = filepath_input.value
    verbose = True
    if verbose: print(f'Loading graph from {filepath}')
    with open(filepath, 'rb') as file:
        full_graph = dill.load(file)
    main_graph = nx.convert_node_labels_to_integers(full_graph, first_label=0, ordering='default', label_attribute=None)

    prop_filepath = pathlib.Path(filepath).parent.joinpath('propagator.pkl')
    if verbose: print(f"Loading propagator from {prop_filepath}")
    with open(prop_filepath, 'rb') as file:
        propagator = dill.load(file)

    main_graph.propagate(propagator, save_transforms=True)

    for edge in main_graph.edges:

        state = main_graph.measure_propagator(edge)

        plots['prop_time']['sources'][edge] = [ColumnDataSource(data=dict(x=propagator.t, y=power_(state).astype('float')))]
        plots['prop_freq']['sources'][edge] = [ColumnDataSource(data=dict(x=propagator.f, y=np.log10(psd_(state, dt=propagator.dt, df=propagator.df).astype('float'))))]

        plots['tran_time']['sources'][edge] = []
        plots['tran_freq']['sources'][edge] = []

        if main_graph.edges[edge]['model'].transform is not None:
            for i, (dof, transform, label) in enumerate(main_graph.edges[edge]['model'].transform):
                if dof == 't':
                    plots['tran_time']['sources'][edge].append(ColumnDataSource(data=dict(x=propagator.t, y=transform.astype('float'))))
                elif dof == 'f':
                    plots['tran_freq']['sources'][edge].append(ColumnDataSource(data=dict(x=propagator.f, y=transform.astype('float'))))


    for plot_id, plot in plots.items():
        for i, (id, sources) in enumerate(plot['sources'].items()):
            color = palette[i % len(palette)]
            plot['lines'][id] = []
            for j, new_source in enumerate(sources):
                style = styles[j % len(styles)]
                new_line = plot['plot'].line(x='x', y='y', source=new_source, alpha=0.8, line_color=color, line_width=3.0, visible=False)
                plot['lines'][id] += [new_line]

    layout, node_dict, edge_dict = disp_graph.export_as_dict(main_graph)
    update_node_positions(layout, node_dict, edge_dict)

    return


def button_callback():
    button.label = '... Loading graph'
    load_graph()
    button.label = 'Load graph'

def slider_callback(attr, old, new):
    if DEBUG: print(f'slider {attr, old, new}')


def dropdown_callback(attr, old, new):
    if DEBUG: print(f'select {attr, old, new}')


def update_node_positions(layout, node_dict, edge_dict):
    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=layout)

    graph_renderer.node_renderer.data_source.data = node_dict
    graph_renderer.edge_renderer.data_source.data = edge_dict


#%%
# add the buttons and user-interface components
button = Button(label="Load graph", button_type="success")
button.on_click(button_callback)

filepath_input = TextInput(value=r'C:\Users\benjamin\Documents\INRS - Projects\asope_data\interactive\example\test_graph.pkl', title='')

dropdown = Select(title="Parameter to control:", value="-", options=["-",])
dropdown.on_change('value', dropdown_callback)

slider = Slider(start=0.1, end=1.0, value=1, step=.1, title="Parameter value")
slider.on_change('value', slider_callback)

text_display = TextInput(value='', title='')

table_source = ColumnDataSource(data=dict(parameter_names=[],
                                          parameters=[],
                                          model=[],
                                          ))

template="""
<b><div style="background:<%= "NavajoWhite" %>;">
<%= (value).toExponential(4) %></div></b>
"""
formatter = HTMLTemplateFormatter(template=template)
columns = [TableColumn(field="parameters", title="Parameter", formatter=formatter),
           TableColumn(field="parameter_names", title="Parameter Name"),
           TableColumn(field="model", title="Model Type"),]
data_table = DataTable(source=table_source, columns=columns, width=400, height=300, sizing_mode='stretch_both')

div = Paragraph(text="""this is a HTML Div where the node info will be printed when an edge or node is selected in the graph""", sizing_mode='stretch_both')

widgets = column(filepath_input, button, text_display, width=200, height=300)

#%% main graph

DEBUG = False
palette = Spectral6
styles = ['solid','dashed','dotted','dotdash','dashdot']

propagator = None
main_graph = Graph()
disp_graph = AccessGraph()


plot_graph = Plot(sizing_mode="stretch_both", width=600,
                  x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1)
                  )
hover = HoverTool(line_policy='interp', tooltips=[("Component", "@component"),])
plot_graph.add_tools(hover, TapTool(), BoxSelectTool())

graph_renderer = GraphRenderer()

node_dict, edge_dict = disp_graph.init_dict()
update_node_positions(disp_graph.get_graph_positions(main_graph), node_dict, edge_dict)

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
# graph_renderer.selection_policy = NodesOnly()
graph_renderer.node_renderer.data_source.selected.on_change("indices", node_callback)
graph_renderer.edge_renderer.data_source.selected.on_change("indices", edge_callback)

plot_graph.renderers.append(graph_renderer)


#%% temporal electric field

plots = {
    'prop_time':{'title':'Optical Intensity', 'xlabel':'Time', 'ylabel':'a.u.', 'sources':{}, 'lines':{}},
    'prop_freq':{'title':'', 'xlabel':'Freq', 'ylabel':'a.u.', 'sources':{}, 'lines':{}},
    'tran_time':{'title':'Transfer Functions', 'xlabel':'Time', 'ylabel':'a.u.', 'sources':{}, 'lines':{}},
    'tran_freq':{'title':'', 'xlabel':'Freq', 'ylabel':'a.u.', 'sources':{}, 'lines':{}},
}
for plot, info in plots.items():
    pfig = figure(tools=[tools.PanTool(), tools.BoxZoomTool(), tools.ResetTool(), tools.SaveTool()],
                  sizing_mode="stretch_both", height=300, min_border=10, min_border_left=50,
                  x_axis_label=info['xlabel'], y_axis_label=info['ylabel'],
                  toolbar_location="left", title=info['title'], title_location='above')
    pfig.background_fill_color = "#fafafa"
    info['plot'] = pfig

# this will lock the ranges together for comparing plots
plots['tran_time']['plot'].x_range = plots['prop_time']['plot'].x_range
plots['tran_freq']['plot'].x_range = plots['prop_freq']['plot'].x_range

layout = column(row(widgets, data_table, plot_graph, height=300),
                row(plots['prop_time']['plot'], plots['prop_freq']['plot'], height=300),
                row(plots['tran_time']['plot'], plots['tran_freq']['plot'], height=300), sizing_mode='scale_height')

curdoc().add_root(layout)
curdoc().title = "ASOPE Photonic Circuit Design"

# load_graph()