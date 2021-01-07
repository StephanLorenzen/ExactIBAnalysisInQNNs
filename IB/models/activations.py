def get_activation_bound(act_func):
    act_func = act_func.lower()
    act_funcs = {
        "tanh":(-1.0,1.0),
        "relu":(0.0,None),
        "relu6":(0.0,6.0),
    }
    if act_func not in act_funcs:
        raise Exception("Unknown activation function: '"+str(act_func)+"'")
    return act_funcs[act_func]
