import numpy as np

def quat2rotm(q):
    """
    四元数转旋转矩阵
    q: 四元数 [w, x, y, z]
    返回: 3x3 旋转矩阵 R
    """
    w, x, y, z = q  # 注意四元数的顺序是 [w, x, y, z]
    
    # 公式计算
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])
    return R

# 示例使用
if __name__ == "__main__":
    q = [0.709606, -0.048774, -0.039987, -0.701770]  # 3.27展示前jaka最后手眼标定的四元数
    R = quat2rotm(q)
    print("旋转矩阵 R:")
    print(R)