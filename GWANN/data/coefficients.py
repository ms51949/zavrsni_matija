import math

def beta0():
    return 1

def beta1(ui, vi):
    return 1 + (ui+vi)/12

def beta12(ui, vi):
    return 1 + 2 * (math.cos((math.pi*ui)/24) * math.cos((math.pi*vi)/24))

def beta22(ui, vi):
    return 1 + 2 * (math.cos((math.pi*ui)/12) * math.cos((math.pi*vi)/12))