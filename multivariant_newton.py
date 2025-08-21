# multivariant_newton.py
import numpy as np
import numdifftools as nd
from numpy.linalg import inv, LinAlgError

# ... (你之前的单变量 optimize 函数可以保留在这里) ...

def optimize_multi(f, x0, tol=1e-7, max_iter=100):
    """
    使用牛顿法寻找多变量函数的最小值。
    
    Args:
        f (function): 目标函数。它应该接受一个 numpy 数组并返回一个标量值。
        x0 (np.array): 优化的初始猜测点 (一个向量)。
        tol (float): 收敛的容忍度。
        max_iter (int): 最大迭代次数。
        
    Returns:
        np.array: 找到的最优解 x。
    """
    x = np.array(x0, dtype=float)

    # 创建梯度和海森矩阵的计算器
    grad_func = nd.Gradient(f)
    hess_func = nd.Hessian(f)

    print(f"开始优化... 初始点: {x}")

    for i in range(max_iter):
        # 计算当前点的梯度和海森矩阵
        grad = grad_func(x)
        hess = hess_func(x)

        # 防御性编程：检查海森矩阵是否可逆
        try:
            # 求解线性方程组 H * step = -grad, 这比直接求逆更稳定
            # 但为了教学清晰，我们这里使用求逆
            inv_hess = inv(hess)
        except LinAlgError:
            print("警告: 海森矩阵是奇异的（不可逆）。算法无法继续。")
            break
            
        # 计算牛顿步长
        # 使用 @ 进行矩阵-向量乘法
        step = inv_hess @ grad
        
        # 更新 x
        x_new = x - step
        
        # 检查收敛条件 (使用向量的范数来判断距离)
        if np.linalg.norm(x_new - x) < tol:
            print(f"在 {i+1} 次迭代后收敛。")
            return x_new
            
        x = x_new
        if (i+1) % 10 == 0: # 每10次迭代打印一次进度
            print(f"迭代 {i+1}: x = {x}")

    print(f"警告: 在 {max_iter} 次迭代后未收敛。")
    return x

# --- 测试用例 ---
if __name__ == '__main__':
    # 定义 Rosenbrock 函数
    def rosenbrock_func(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    # 设置初始点
    initial_guess = np.array([0.0, 0.0])

    # 运行多变量优化器
    optimal_x = optimize_multi(rosenbrock_func, initial_guess)

    print("\n------------------")
    print(f"优化完成。")
    print(f"找到的最优解在: {optimal_x}")
    print(f"真实的最优解在: [1. 1.]")