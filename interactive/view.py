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


class View(object):
    """
    View class for the interactive ASOPE dashboard.
    View interacts with Bokeh, defining visual styles, setting callbacks, and updating page elements.
    It recieves a Control class instance, but should never touch the Model directly or ASOPE code
    """

    verbose = False

    def __init__(self, control):
        self.control = control

        self.palette = Spectral6
        self.styles = ['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']

        """
            Add widgets for interaction 
        """
        button = Button(label="Load graph", button_type="success", width=100, sizing_mode='fixed')
        button.on_click(self.button_callback)

        filepath_input = TextInput(
            value=r'C:\Users\benjamin\Documents\INRS - Projects\asope_data\interactive\example\test_graph.pkl',
            title='', sizing_mode='stretch_both')

        table_source = ColumnDataSource(data=self.control.table_data)

        template = """
        <b><div style="background:<%= "NavajoWhite" %>;">
        <%= (value).toExponential(4) %></div></b>
        """
        formatter = HTMLTemplateFormatter(template=template)
        columns = [TableColumn(field="parameters", title="Parameter", formatter=formatter),
                   TableColumn(field="parameter_names", title="Parameter Name"),
                   TableColumn(field="model", title="Model Type"), ]
        data_table = DataTable(source=table_source, columns=columns, sizing_mode='stretch_both')

        # store widgets as class variables for access by callbacks
        self.button = button
        self.filepath_input = filepath_input
        self.table_source = table_source

        """
            Add figure for displaying the graph 
        """
        plot_graph = Plot(sizing_mode="stretch_both", width=600,
                          x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1)
                          )
        hover = HoverTool(line_policy='interp', tooltips=[("Component", "@component"), ])
        plot_graph.add_tools(hover, TapTool(), BoxSelectTool())

        self.graph_renderer = GraphRenderer()
        self.set_graph_renderer_policies(self.graph_renderer)
        plot_graph.renderers.append(self.graph_renderer)

        """
            Add interactive plots to visualize optical field and transformations on edges/nodes
        """
        plots = {
            'prop_time': {'title': 'Optical Intensity', 'xlabel': 'Time', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
            'prop_freq': {'title': '', 'xlabel': 'Freq', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
            'tran_time': {'title': 'Transfer Functions', 'xlabel': 'Time', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
            'tran_freq': {'title': '', 'xlabel': 'Freq', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
        }

        for plot_id, plot in plots.items():
            pfig = figure(tools=[tools.PanTool(), tools.BoxZoomTool(), tools.ResetTool(), tools.SaveTool()],
                          sizing_mode="stretch_both", height=300, min_border=10, min_border_left=50,
                          x_axis_label=plot['xlabel'], y_axis_label=plot['ylabel'],
                          toolbar_location="left", title=plot['title'], title_location='above')
            pfig.background_fill_color = "#fafafa"
            plot['plot'] = pfig

        plots['tran_time']['plot'].x_range = plots['prop_time']['plot'].x_range
        plots['tran_freq']['plot'].x_range = plots['prop_freq']['plot'].x_range
        self.plots = plots

        """
        Build layout using grid
        """
        layout = column(row(column(row(filepath_input, button, height=30), data_table), plot_graph, height=400),
                        row(plots['prop_time']['plot'], plots['prop_freq']['plot'], height=300),
                        row(plots['tran_time']['plot'], plots['tran_freq']['plot'], height=300),
                        sizing_mode='scale_height')
        self.layout = layout
        return

    def set_graph_renderer_policies(self, graph_renderer):
        self.update_node_positions()

        # set interaction visuals
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
        graph_renderer.node_renderer.data_source.selected.on_change("indices", self.node_selected_callback)
        graph_renderer.edge_renderer.data_source.selected.on_change("indices", self.edge_selected_callback)

        return graph_renderer

    def update_node_positions(self):
        self.graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=self.control.graph_layout)
        self.graph_renderer.node_renderer.data_source.data = self.control.graph_node_data
        self.graph_renderer.edge_renderer.data_source.data = self.control.graph_edge_data

        if self.verbose: print(f'node_data {self.control.graph_node_data}, edge_data {self.control.graph_edge_data}')

    def clear_plots(self):
        for plot_id, plot in self.plots.items():
            for _, lines in plot['lines'].items():
                for line in lines:
                    plot['plot'].renderers.remove(line)
            plot['sources'] = {}
            plot['lines'] = {}

    """
    Add the sources and lines to the interactive plots (propagator, transfer functions)
    """
    def add_sources(self):
        for (plot_id, plot_data) in self.control.plot_edge_data.items():
            for (index, source_data_lst) in plot_data.items():
                self.plots[plot_id]['sources'][index] = []
                for source in source_data_lst:
                    self.plots[plot_id]['sources'][index].append(ColumnDataSource(data=source))

    def add_lines(self):
        for plot_id, plot in self.plots.items():
            for j, (index, source_data_lst) in enumerate(plot['sources'].items()):
                color = self.palette[j % len(self.palette)]
                plot['lines'][index] = []
                for k, source in enumerate(source_data_lst):
                    new_line = plot['plot'].line(x='x', y='y', source=source, alpha=0.9, line_color=color, line_width=2.0, visible=False)
                    self.plots[plot_id]['lines'][index].append(new_line)
        if self.verbose: print(self.plots['prop_time']['sources'])

    """
    Callbacks for the graph inspection and selection
    """
    def node_selected_callback(self, attr, old, new):
        return

    def edge_selected_callback(self, attr, old, new):
        self.control.update_table_data(new)
        self.table_source.data = self.control.table_data

        # here we change which lines are visible based on the selection policy of the graph
        for plot_id, plot in self.plots.items():
            for index in new:
                if index in plot['lines'].keys():
                    for line in plot['lines'][index]:
                        line.visible = True

            for index in list(set(old) - set(new)):
                if index in plot['lines'].keys():
                    for line in plot['lines'][index]:
                        line.visible = False

        return

    """
    Callbacks for the widget interactions
    """
    def button_callback(self):
        self.button.label = '... Loading graph'
        self.clear_plots()

        self.control.load_new_graph(self.filepath_input.value)

        self.add_sources()
        self.add_lines()
        self.update_node_positions()
        self.button.label = 'Load graph'
        return
