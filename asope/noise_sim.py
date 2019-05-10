import numpy as np

def update_error_attributes(experiment):
    exp = experiment
    for node in exp.node():
        if exp.node()[node]['info'].N_EPARAMETERS != 0:
            component = exp.node()[node]['info']
            new_at = component.error_model()
            component.update_error_attributes(new_at)