import numpy as np

epsilon = 1e-5

def f_prime(x, f):
    return (f(x + epsilon) - f(x)) / epsilon

def f2_prime(x, f):
    return (f_prime(x + epsilon, f) - f_prime(x, f)) / epsilon

tol = 1e-7
max_iter = 500

def optimize(x0, f):
    x = float(x0)
    
    for _ in range(max_iter):
        fp = f_prime(x, f)
        fpp = f2_prime(x, f)
        
        if abs(fp) < tol:
            return x
        
        if abs(fpp) < 1e-12:
            raise ValueError("Second derivative too close to zero.")
        
        x_new = x - fp / fpp
        
        if abs(x_new - x) < tol:
            return x_new
        
        x = x_new
    
    raise RuntimeError("Max iterations reached without convergence.")
