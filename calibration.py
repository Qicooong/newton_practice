import newton
import numpy as np
def f(x):
    return x**3
newton.optimize(1,f(x)) 
newton.optimize(2.5, np.cos)


