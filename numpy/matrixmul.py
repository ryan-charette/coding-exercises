import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
              
    arr_a = np.array(a)
    arr_b = np.array(b)
              
    if arr_a.shape[1] != arr_b.shape[0]:
      return -1
                
    return np.matmul(arr_a, arr_b).tolist()
