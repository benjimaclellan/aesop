import numpy as np
import matplotlib.pyplot as plt
from bokeh.layouts import gridplot, column, row
from bokeh.models import BoxSelectTool, LassoSelectTool
from bokeh.plotting import curdoc, figure

from bokeh.io import output_file, show
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,)
from bokeh.colors import RGB
from bokeh.models import NodesOnly
import bokeh.models.tools as tools
from bokeh.models import GraphRenderer, StaticLayoutProvider
from bokeh.palettes import Spectral4, Magma256, Cividis256
from bokeh.plotting import from_networkx
from bokeh.models import ColumnDataSource
from bokeh.models import Button, CustomJS, FileInput, TextInput, Select, Slider, Div
from bokeh.palettes import Spectral6
from bokeh.models import Paragraph, FileInput
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn, HTMLTemplateFormatter, NumberFormatter
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, LogColorMapper, PrintfTickFormatter,)
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.models import Range1d

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
        button = Button(label="Load graph", button_type="success", sizing_mode='stretch_both')
        button.on_click(self.button_callback)

        folder_input = TextInput(
            value=r'C:\Users\benjamin\Documents\INRS - Projects\asope_data\20210421__multiple_pulse_rep_runs\20210421_2dE6_pulse_rep_rate',
            title='', sizing_mode='stretch_both')

        file_input = TextInput(
            value=r'graph_hof0.pkl',
            title='', sizing_mode='stretch_both')

        table_source = ColumnDataSource(data=self.control.table_data)

        template = """
        <b><div style="background:<%= "NavajoWhite" %>;">
        <%= (value).toExponential(4) %></div></b>
        """
        formatter = HTMLTemplateFormatter(template=template)
        columns = [TableColumn(field="parameters", title="Parameter"), #, formatter=formatter),
                   TableColumn(field="parameter_names", title="Parameter Name"),
                   TableColumn(field="model", title="Model Type"), ]

        data_table = DataTable(source=table_source, columns=columns, sizing_mode='stretch_both')
        self.data_table = data_table

        # store widgets as class variables for access by callbacks
        self.button = button
        self.folder_input = folder_input
        self.file_input = file_input
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
        plots = self.create_prop_tran_plots()
        self.plots = plots
        self.heatmap, self.heatmap_colormap = self.create_heatmap()
        self.eigen_plot = self.create_eigen_plot()

        # Create two panels, one for each conference
        panel1_child = column(row(plots['prop_time']['plot'], plots['prop_freq']['plot'], height=250),
                              row(plots['tran_time']['plot'], plots['tran_freq']['plot'], height=250))
        panel1 = Panel(child=panel1_child, title='Propagation & Transfer Functions')
        panel2 = Panel(child=row(self.heatmap, self.eigen_plot, height=500), title='Sensitivity Analysis')

        # Assign the panels to Tabs
        tabs = Tabs(tabs=[panel1, panel2])

        """
        Build layout using grid
        """
        layout = column(row(column(row(folder_input, height=30),
                                   row(file_input, height=30),
                                   row(button, height=30),
                            data_table), plot_graph, height=300),
                        tabs,
                        sizing_mode='scale_height')
        self.layout = layout
        return

    def create_prop_tran_plots(self):
        plots = {
            'prop_time': {'title': 'Optical Intensity', 'xlabel': 'Time', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
            'prop_freq': {'title': '', 'xlabel': 'Freq', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
            'tran_time': {'title': 'Transfer Functions', 'xlabel': 'Time', 'ylabel': 'a.u.', 'sources': {},
                          'lines': {}},
            'tran_freq': {'title': '', 'xlabel': 'Freq', 'ylabel': 'a.u.', 'sources': {}, 'lines': {}},
        }

        for plot_id, plot in plots.items():
            p = figure(tools=[tools.PanTool(), tools.BoxZoomTool(), tools.ResetTool(), tools.SaveTool()],
                          sizing_mode="stretch_both", height=300, min_border=10, min_border_left=50,
                          x_axis_label=plot['xlabel'], y_axis_label=plot['ylabel'],
                          toolbar_location="left", title=plot['title'], title_location='above')
            p.background_fill_color = "#fafafa"
            plot['plot'] = p

        plots['tran_time']['plot'].x_range = plots['prop_time']['plot'].x_range
        plots['tran_freq']['plot'].x_range = plots['prop_freq']['plot'].x_range
        return plots


    def create_heatmap(self):
        scale = 'log'
        if scale == 'lin':
            mapper = LinearColorMapper(palette=Cividis256)
        elif scale == 'log':
            mapper = LogColorMapper(palette=Cividis256)

        p = figure(plot_width=600, plot_height=600, title="Local Hessian Matrix",
                   x_axis_location="below", y_axis_location="left", min_border=10, min_border_left=50,
                   x_axis_label='Parameter i', y_axis_label='Parameter j',
                   tools="hover,save,pan,box_zoom,reset,wheel_zoom", toolbar_location='above',
                   tooltips=[('value', '@value'), ('i-th parameter', '@x_name'), ('j-th parameter', '@y_name')],
                   sizing_mode="stretch_both",
                   )

        source = ColumnDataSource(data=dict(value=[], x=[], y=[]))
        s = 0.0
        mat = p.rect(x="x", y="y", width=1 - s, height=1 - s, source=source,
                     line_color=None, fill_color=transform('value' if scale == 'lin' else 'log_corrected_value', mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=10))

        p.add_layout(color_bar, 'right')

        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "7px"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 1.0

        self.hessian_heatmap_mat = mat
        return p, mapper

    def create_eigen_plot(self):
        p = figure(plot_width=600, plot_height=600, title="Eigen-analysis",title_location='above',
                   x_axis_location="below", y_axis_location="left", sizing_mode="stretch_both",
                   x_axis_label='Parameter', y_axis_label='', min_border=10, min_border_left=50,
                   tools="hover,save,pan,box_zoom,reset,wheel_zoom", toolbar_location='above',
                   tooltips=[('eigen-value', '@eigen_value'), ('parameter', '@parameter_name'), ('basis weight', '@weight')]
                   )
        p.axis.visible = False
        data_vectors, data_total = self.initialize_eigen_analysis_data()
        bar_total = p.vbar(source=ColumnDataSource(data=data_total),
                           x=-2.0, width=2.0, bottom='bottom', top='top', alpha=0.3, color='grey')
        bar = p.vbar(source=ColumnDataSource(data=data_vectors),
                     x='x', width=0.5, bottom='bottom', top='top', alpha=1.0,color='color')

        self.eigen_plot_bars = {'bar': bar, 'bar_total': bar_total}
        return p

    def initialize_eigen_analysis_data(self):
        data_vectors = dict(x=[], top=[], bottom=[], parameter_name=[], eigen_value=[], weight=[], color=[])
        data_total = dict(top=[], bottom=[], eigen_value=[], weight=[], parameter_name=[], )
        return data_vectors, data_total

    def add_lha_eigenvectors(self):
        def map_eigenvalue_to_color(value, cmap, vmax=255, vmin=0):
            color = tuple([int(round(255 * c)) for c in  cmap((value - vmin) / (vmax - vmin))])
            new_color = RGB(color[0], color[1], color[2])
            return new_color

        cmap = plt.get_cmap('summer')
        eig_vals, eig_vecs = self.control.graph_lha_data
        graph_hessian_data = self.control.graph_hessian_data  # this is a dictionary

        data_vectors, data_total = self.initialize_eigen_analysis_data()
        data_vectors['parameter_name'] = graph_hessian_data['x_name']

        for i, (eig_val) in enumerate(eig_vals):
            eig_vec = eig_vecs[:, i]
            base = 2.0*i

            data_total['bottom'].append(base - 0.95)
            data_total['top'].append(base + 0.95)
            data_total['eigen_value'].append(eig_val)
            data_total['weight'].append('-')
            data_total['parameter_name'].append('-')

            # color = map_eigenvalue_to_color(eig_val, cmap, vmax=np.max(eig_vals), vmin=np.min(eig_vals))
            color = map_eigenvalue_to_color(i, cmap, vmax=eig_vals.shape[0], vmin=0.0)
            for j, eig_vec_val in enumerate(eig_vec):
                data_vectors['x'].append(j)
                data_vectors['bottom'].append(base)
                data_vectors['top'].append(base + eig_vec_val)
                data_vectors['weight'].append(eig_vec_val)
                data_vectors['eigen_value'].append(eig_val)
                data_vectors['color'].append(color)

        self.eigen_plot_bars['bar'].data_source.data = data_vectors
        self.eigen_plot_bars['bar_total'].data_source.data = data_total

        self.eigen_plot.x_range = Range1d(start=-3, end=eig_vals.shape[0]+1)
        self.eigen_plot.y_range = Range1d(start=-1, end=eig_vals.shape[0]+2)
        return

    def add_hessian_matrix(self):
        graph_hessian_data = self.control.graph_hessian_data  # this is a dictionary
        self.hessian_heatmap_mat.data_source.data = graph_hessian_data
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
        self.table_source.data = {}
        self.control.update_table_data(new)
        print(self.control.table_data)
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



        self.control.load_new_graph(self.folder_input.value, self.file_input.value)

        self.add_sources()
        self.add_lines()
        self.update_node_positions()
        # self.add_hessian_matrix()
        # self.add_lha_eigenvectors()
        self.button.label = 'Load graph'
        return
