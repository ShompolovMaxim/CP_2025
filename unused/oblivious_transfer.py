import random

m = 10**9 + 7
g = 2


def get_A():
    global m, g
    a = random.randint(0, m - 1)
    return g**a%m

def get_B(A, c):
    global m, g
    b = random.randint(0, m - 1)
    if c == 0:
        return g**b%m
    if c == 1:
        return A*g**b%m
    return None

def get_keys_A(a, B):
    return B**a%m, (B*get_module_inverse(A))**a%m

def get_key_B(A, b):
    global m
    return A**b%m
    
