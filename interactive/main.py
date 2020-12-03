"""
The interactive dashboard allows simple exploration of systems proposed by ASOPE.
A saved graph is loaded and the parameters, propagation, and transfer functions on each component in the system
can be visualized individually or aggregate.

To run, ensure bokeh is installed (pip install bokeh). We use the app serve functionality (i.e. a full directory
rather than one script). To run, use
    bokeh serve interactive/ --show
which will open a local HTML page.
"""

#%%
from view import View
from model import Model
from controller import Controller
from bokeh.plotting import curdoc, figure

#%%
"""
We use the Model-View-Control design. 
    View: stores all visual elements and interacts with Bokeh
    Model: stores all the data, such as the graph, propagator, and calls ASOPE functions
    Control: an intermediary between View and Model - however currently does not add much functionality
"""
model = Model()
controller = Controller(model)
view = View(controller)

# view.button_callback()

curdoc().add_root(view.layout)
curdoc().title = "ASOPE Photonic System Design"
