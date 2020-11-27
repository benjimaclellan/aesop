from autograd.numpy.numpy_boxes import ArrayBox

def unwrap_arraybox_list(x):
    for i in range(len(x)):
        while type(x[i]) == ArrayBox:
            x[i] = x[i]._value
    return x

def unwrap_arraybox_val(x):
    while type(x) == ArrayBox:
        x = x._value
    return x