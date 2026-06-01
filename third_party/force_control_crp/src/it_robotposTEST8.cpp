#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <thread>
#include <chrono>
#include <mutex>
#include <cmath>
#include <condition_variable>
#include "CSDKLoader.h"
#include "IRobotService.h"
#include "IMotionService.h"
#include "IModelService.h"
#include "IFileService.h"
#include <vector>
#include "ForceControl.h"
// 键盘控制所需头文件
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// 数据记录所需头文件
#include <fstream>
#include <iomanip>
#include <atomic>
#include <sys/stat.h> // for stat() and mkdir()
#include <ctime>      // for std::strftime (如果尚未包含)

#include <poll.h>

#ifdef __cplusplus
extern "C" {
#endif

// 提供python调用的函数列表
int ForceControl_demo(void); //测试函数
int Forcecontrol_ChargeIn(void);  //插枪函数
int ForceControl_Poseadjust(void); //姿态保持
int Forcecontrol_ChargeOut(void); //拔枪函数
int PowerOn_PPmode(); //PP模式启动

int GetForcecontrolState(void); //获取力控状态

int ForceControl_OpenCover();
int return_the_gun();
void AttachExternalCrpServices(void *robot, void *motion, void *file);
void DetachExternalCrpServices();


#ifdef __cplusplus
}
#endif

std::ofstream g_pose_log;          // 位姿日志文件
std::mutex g_log_mtx;              // 保护日志写入的互斥锁
std::atomic<bool> g_logging{true}; // 控制日志是否继续


//chargedfinished  0-插枪中 1-插枪成功 2-插枪失败 3-拔枪中 4-拔枪成功 6-姿态保持 8-默认（运控）  10-归枪中 11-归枪成功 12-归枪失败 15-开盖中 16-开盖成功 17-开盖失败

//******************************逆运动学部分**********************************************

using namespace std;

// 3维向量实现
struct Vector3
{
    double x, y, z;

    Vector3() : x(0), y(0), z(0) {}
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3 operator-(const Vector3 &other) const
    {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    Vector3 operator+(const Vector3 &other) const
    {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator*(double s) const
    {
        return Vector3(x * s, y * s, z * s);
    }

    // 新增：标量除法
    Vector3 operator/(double s) const
    {
        return Vector3(x / s, y / s, z / s);
    }

    // 叉乘
    Vector3 cross(const Vector3 &other) const
    {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x);
    }

    // 点乘
    double dot(const Vector3 &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    // 向量范数
    double norm() const
    {
        return sqrt(x * x + y * y + z * z);
    }

    // 归一化
    Vector3 normalized() const
    {
        double n = norm();
        if (n < 1e-10)
            return Vector3(1, 0, 0);
        return Vector3(x / n, y / n, z / n);
    }
};

// 3x3矩阵实现（旋转矩阵）
struct Matrix3
{
    double m[3][3];

    Matrix3()
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i][j] = 0;
    }

    static Matrix3 Identity()
    {
        Matrix3 m;
        m.m[0][0] = 1;
        m.m[1][1] = 1;
        m.m[2][2] = 1;
        return m;
    }

    // 矩阵乘向量
    Vector3 operator*(const Vector3 &v) const
    {
        return Vector3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    // 矩阵乘矩阵
    Matrix3 operator*(const Matrix3 &other) const
    {
        Matrix3 res;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                res.m[i][j] = m[i][0] * other.m[0][j] +
                              m[i][1] * other.m[1][j] +
                              m[i][2] * other.m[2][j];
            }
        }
        return res;
    }

    // 矩阵转置
    Matrix3 transpose() const
    {
        Matrix3 res;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                res.m[i][j] = m[j][i];
            }
        }
        return res;
    }

    // 旋转矩阵转轴角
    void toAngleAxis(Vector3 &axis, double &angle) const
    {
        double tr = m[0][0] + m[1][1] + m[2][2];
        angle = acos(max(-1.0, min(1.0, (tr - 1) / 2)));

        if (angle < 1e-6)
        {
            angle = 0;
            axis = Vector3(1, 0, 0);
            return;
        }

        axis.x = m[2][1] - m[1][2];
        axis.y = m[0][2] - m[2][0];
        axis.z = m[1][0] - m[0][1];
        axis = axis.normalized();
    }

    // 轴角转旋转矩阵
    static Matrix3 fromAngleAxis(const Vector3 &axis, double angle)
    {
        double c = cos(angle);
        double s = sin(angle);
        double t = 1 - c;
        Vector3 a = axis.normalized();
        double x = a.x, y = a.y, z = a.z;

        Matrix3 m;
        m.m[0][0] = t * x * x + c;
        m.m[0][1] = t * x * y - z * s;
        m.m[0][2] = t * x * z + y * s;
        m.m[1][0] = t * x * y + z * s;
        m.m[1][1] = t * y * y + c;
        m.m[1][2] = t * y * z - x * s;
        m.m[2][0] = t * x * z - y * s;
        m.m[2][1] = t * y * z + x * s;
        m.m[2][2] = t * z * z + c;
        return m;
    }

    // ZYX欧拉角转旋转矩阵（输入rx,ry,rz，弧度）
    static Matrix3 fromEulerZYX(double rx, double ry, double rz)
    {
        double cx = cos(rx), sx = sin(rx);
        double cy = cos(ry), sy = sin(ry);
        double cz = cos(rz), sz = sin(rz);

        Matrix3 m;
        // R = Rz * Ry * Rx
        m.m[0][0] = cz * cy;
        m.m[0][1] = cz * sy * sx - sz * cx;
        m.m[0][2] = cz * sy * cx + sz * sx;
        m.m[1][0] = sz * cy;
        m.m[1][1] = sz * sy * sx + cz * cx;
        m.m[1][2] = sz * sy * cx - cz * sx;
        m.m[2][0] = -sy;
        m.m[2][1] = cy * sx;
        m.m[2][2] = cy * cx;
        return m;
    }

    // 旋转矩阵转ZYX欧拉角（输出rx,ry,rz，弧度）
    void toEulerZYX(double &rx, double &ry, double &rz) const
    {
        ry = asin(max(-1.0, min(1.0, -m[2][0])));

        if (fabs(ry - M_PI / 2) < 1e-6)
        {
            // 万向锁情况
            rz = 0;
            rx = atan2(m[0][1], m[0][2]);
        }
        else if (fabs(ry + M_PI / 2) < 1e-6)
        {
            // 万向锁情况
            rz = 0;
            rx = atan2(-m[0][1], -m[0][2]);
        }
        else
        {
            double cy = cos(ry);
            rz = atan2(m[1][0] / cy, m[0][0] / cy);
            rx = atan2(m[2][1] / cy, m[2][2] / cy);
        }
    }
};

// 4x4齐次变换矩阵
struct Matrix4
{
    double m[4][4];

    Matrix4()
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = 0;
    }

    static Matrix4 Identity()
    {
        Matrix4 m;
        for (int i = 0; i < 4; i++)
            m.m[i][i] = 1;
        return m;
    }

    // 矩阵乘矩阵
    Matrix4 operator*(const Matrix4 &other) const
    {
        Matrix4 res;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                res.m[i][j] = 0;
                for (int k = 0; k < 4; k++)
                {
                    res.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return res;
    }

    // 获取位置部分（最后一列前3个元素）
    Vector3 getPosition() const
    {
        return Vector3(m[0][3], m[1][3], m[2][3]);
    }

    // 获取旋转部分（左上角3x3）
    Matrix3 getRotation() const
    {
        Matrix3 r;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                r.m[i][j] = m[i][j];
            }
        }
        return r;
    }

    // 获取Z轴（旋转部分第三列）
    Vector3 getZAxis() const
    {
        return Vector3(m[0][2], m[1][2], m[2][2]);
    }
};

// 6维向量
struct Vector6
{
    double v[6];

    Vector6()
    {
        for (int i = 0; i < 6; i++)
            v[i] = 0;
    }

    double &operator[](int i) { return v[i]; }
    double operator[](int i) const { return v[i]; }

    Vector6 operator-(const Vector6 &other) const
    {
        Vector6 res;
        for (int i = 0; i < 6; i++)
            res.v[i] = v[i] - other.v[i];
        return res;
    }

    Vector6 operator+(const Vector6 &other) const
    {
        Vector6 res;
        for (int i = 0; i < 6; i++)
            res.v[i] = v[i] + other.v[i];
        return res;
    }

    Vector6 operator*(double s) const
    {
        Vector6 res;
        for (int i = 0; i < 6; i++)
            res.v[i] = v[i] * s;
        return res;
    }

    double norm() const
    {
        double sum = 0;
        for (int i = 0; i < 6; i++)
            sum += v[i] * v[i];
        return sqrt(sum);
    }
};

// 6x6矩阵（用于雅可比和阻尼最小二乘）
struct Matrix6
{
    double m[6][6];

    Matrix6()
    {
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                m[i][j] = 0;
    }

    static Matrix6 Identity()
    {
        Matrix6 m;
        for (int i = 0; i < 6; i++)
            m.m[i][i] = 1;
        return m;
    }

    double &operator()(int i, int j) { return m[i][j]; }
    double operator()(int i, int j) const { return m[i][j]; }

    Matrix6 operator+(const Matrix6 &other) const
    {
        Matrix6 res;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                res.m[i][j] = m[i][j] + other.m[i][j];
            }
        }
        return res;
    }

    // 矩阵乘矩阵
    Matrix6 operator*(const Matrix6 &other) const
    {
        Matrix6 res;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                res.m[i][j] = 0;
                for (int k = 0; k < 6; k++)
                {
                    res.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return res;
    }

    // 新增：矩阵乘标量
    Matrix6 operator*(double s) const
    {
        Matrix6 res;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                res.m[i][j] = m[i][j] * s;
            }
        }
        return res;
    }

    // 矩阵乘向量
    Vector6 operator*(const Vector6 &v) const
    {
        Vector6 res;
        for (int i = 0; i < 6; i++)
        {
            res.v[i] = 0;
            for (int j = 0; j < 6; j++)
            {
                res.v[i] += m[i][j] * v.v[j];
            }
        }
        return res;
    }

    // 矩阵转置
    Matrix6 transpose() const
    {
        Matrix6 res;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                res.m[i][j] = m[j][i];
            }
        }
        return res;
    }

    // 高斯消元法解线性方程组 Ax = b
    Vector6 solve(const Vector6 &b) const
    {
        double aug[6][7];
        // 构造增广矩阵
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                aug[i][j] = m[i][j];
            }
            aug[i][6] = b.v[i];
        }

        // 高斯消元
        for (int col = 0; col < 6; col++)
        {
            // 选主元
            int pivot = col;
            for (int row = col; row < 6; row++)
            {
                if (fabs(aug[row][col]) > fabs(aug[pivot][col]))
                {
                    pivot = row;
                }
            }
            // 交换行
            if (pivot != col)
            {
                for (int j = col; j <= 6; j++)
                {
                    std::swap(aug[col][j], aug[pivot][j]);
                }
            }
            // 归一化主元行
            double div = aug[col][col];
            if (fabs(div) < 1e-10)
                break;
            for (int j = col; j <= 6; j++)
            {
                aug[col][j] /= div;
            }
            // 消去其他行
            for (int row = 0; row < 6; row++)
            {
                if (row != col && fabs(aug[row][col]) > 1e-10)
                {
                    double factor = aug[row][col];
                    for (int j = col; j <= 6; j++)
                    {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }

        // 提取解
        Vector6 x;
        for (int i = 0; i < 6; i++)
        {
            x.v[i] = aug[i][6];
        }
        return x;
    }
};

// DH参数（与原Python版本完全一致）
const double a_list[] = {0, 621.899, 559.067, 0, 0, 0};
const double d_list[] = {0, 0, 0, -165.124, 119.425, 115.0};
const double alpha_deg_list[] = {90, 0, 0, 90, 90, 0};
const double offset_deg_list[] = {0, 0, -90, 90, -90, 0};
const int N_JOINT = 6;

/**
 * @brief 标准DH齐次变换矩阵
 */
Matrix4 dh_transform(double a, double alpha_rad, double d, double theta_rad)
{
    double sa = sin(alpha_rad);
    double ca = cos(alpha_rad);
    double st = sin(theta_rad);
    double ct = cos(theta_rad);

    Matrix4 T;
    T.m[0][0] = ct;
    T.m[0][1] = -st * ca;
    T.m[0][2] = st * sa;
    T.m[0][3] = a * ct;
    T.m[1][0] = st;
    T.m[1][1] = ct * ca;
    T.m[1][2] = -ct * sa;
    T.m[1][3] = a * st;
    T.m[2][0] = 0;
    T.m[2][1] = sa;
    T.m[2][2] = ca;
    T.m[2][3] = d;
    T.m[3][0] = 0;
    T.m[3][1] = 0;
    T.m[3][2] = 0;
    T.m[3][3] = 1;
    return T;
}

/**
 * @brief 正运动学计算
 */
Vector6 forward_kinematics(const Vector6 &joints, const string &representation = "euler")
{
    Matrix4 T = Matrix4::Identity();

    for (int i = 0; i < N_JOINT; ++i)
    {
        double a = a_list[i];
        double d = d_list[i];
        double alpha = alpha_deg_list[i] * M_PI / 180.0;
        double offset = offset_deg_list[i] * M_PI / 180.0;
        double theta = joints[i] * M_PI / 180.0 + offset;

        T = T * dh_transform(a, alpha, d, theta);
    }

    Vector3 pos = T.getPosition();
    Matrix3 rot = T.getRotation();

    if (representation == "euler")
    {
        double rx, ry, rz;
        rot.toEulerZYX(rx, ry, rz);
        rx = rx * 180.0 / M_PI;
        ry = ry * 180.0 / M_PI;
        rz = rz * 180.0 / M_PI;

        Vector6 result;
        result[0] = pos.x;
        result[1] = pos.y;
        result[2] = pos.z;
        result[3] = rx;
        result[4] = ry;
        result[5] = rz;
        return result;
    }
    else if (representation == "rotvec")
    {
        Vector3 axis;
        double angle;
        rot.toAngleAxis(axis, angle);
        Vector3 rotvec = (axis * angle) * 180.0 / M_PI;

        Vector6 result;
        result[0] = pos.x;
        result[1] = pos.y;
        result[2] = pos.z;
        result[3] = rotvec.x;
        result[4] = rotvec.y;
        result[5] = rotvec.z;
        return result;
    }
    else
    {
        throw invalid_argument("representation must be euler, rotvec or matrix");
    }
}

/**
 * @brief 数值逆运动学（阻尼最小二乘法）
 */
Vector6 inverse_kinematics(const Vector6 &target_pose, const Vector6 *initial_joints = nullptr,
                           const string &representation = "euler", int max_iter = 200, double tol = 1e-9)
{
    Vector6 joints;
    if (initial_joints != nullptr)
    {
        joints = *initial_joints;
    }

    // 解析目标位姿
    Vector3 target_pos(target_pose[0], target_pose[1], target_pose[2]);
    Matrix3 target_R;

    if (representation == "euler")
    {
        double rx = target_pose[3] * M_PI / 180.0;
        double ry = target_pose[4] * M_PI / 180.0;
        double rz = target_pose[5] * M_PI / 180.0;
        target_R = Matrix3::fromEulerZYX(rx, ry, rz);
    }
    else if (representation == "rotvec")
    {
        double rx = target_pose[3] * M_PI / 180.0;
        double ry = target_pose[4] * M_PI / 180.0;
        double rz = target_pose[5] * M_PI / 180.0;
        Vector3 rotvec(rx, ry, rz);
        double angle = rotvec.norm();
        if (angle < 1e-6)
        {
            target_R = Matrix3::Identity();
        }
        else
        {
            target_R = Matrix3::fromAngleAxis(rotvec, angle);
        }
    }
    else
    {
        throw invalid_argument("Unsupported representation");
    }

    for (int it = 0; it < max_iter; ++it)
    {
        Matrix4 T = Matrix4::Identity();
        vector<Matrix4> T_list;
        T_list.reserve(N_JOINT);

        // 正向计算累积变换
        for (int i = 0; i < N_JOINT; ++i)
        {
            double a = a_list[i];
            double d = d_list[i];
            double alpha = alpha_deg_list[i] * M_PI / 180.0;
            double offset = offset_deg_list[i] * M_PI / 180.0;
            double theta = joints[i] * M_PI / 180.0 + offset;

            Matrix4 T_i = dh_transform(a, alpha, d, theta);
            T = T * T_i;
            T_list.push_back(T);
        }

        // 当前位姿
        Vector3 current_pos = T.getPosition();
        Matrix3 current_R = T.getRotation();

        // 计算误差
        Vector3 err_pos = target_pos - current_pos;
        Matrix3 R_diff = target_R * current_R.transpose();
        Vector3 axis;
        double angle;
        R_diff.toAngleAxis(axis, angle);
        Vector3 err_rot = axis * angle;

        Vector6 error;
        error[0] = err_pos.x;
        error[1] = err_pos.y;
        error[2] = err_pos.z;
        error[3] = err_rot.x;
        error[4] = err_rot.y;
        error[5] = err_rot.z;

        // 检查收敛
        // if (error.norm() < tol)
        // {
        //     cout << "IK 收敛，迭代次数: " << it + 1 << endl;
        //     break;
        // }

        // 构建雅可比矩阵
        Matrix6 J;
        Vector3 p_end = current_pos;

        for (int i = 0; i < N_JOINT; ++i)
        {
            Vector3 z, p;
            if (i == 0)
            {
                z = Vector3(0, 0, 1);
                p = Vector3(0, 0, 0);
            }
            else
            {
                Matrix4 &Ti = T_list[i - 1];
                z = Ti.getZAxis();
                p = Ti.getPosition();
            }

            Vector3 p_diff = p_end - p;
            Vector3 cross = z.cross(p_diff);

            // 填充雅可比
            J(0, i) = cross.x;
            J(1, i) = cross.y;
            J(2, i) = cross.z;
            J(3, i) = z.x;
            J(4, i) = z.y;
            J(5, i) = z.z;
        }

        // 阻尼最小二乘法求解
        double lambda = 0.005;
        Matrix6 I = Matrix6::Identity();
        Matrix6 JJT = J * J.transpose() + I * lambda;
        Vector6 delta_theta = J.transpose() * JJT.solve(error);

        // 更新关节角
        for (int i = 0; i < 6; i++)
        {
            joints[i] += delta_theta[i] * 180.0 / M_PI;
        }
    }

    return joints;
}

//******************************逆运动学部分**********************************************

// 键盘监听
//  Linux 下无回车键盘监听 工具函数
static struct termios old_term, new_term;

void init_keyboard(void)
{
    tcgetattr(STDIN_FILENO, &old_term);
    new_term = old_term;
    new_term.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
}

void close_keyboard(void)
{
    tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
}

int kbhit(void)
{
    struct timeval tv = {0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &fds);
}

char get_key(void)
{
    char c;
    read(STDIN_FILENO, &c, 1);
    return c;
}

// sendpath测试程序/已增加六维力数据读取/已增加力控算法

// 限幅参数：可根据机器人实际性能调整
double MAX_POS_DELTA = 0.05;  // 单次力控周期内，坐标最大变化量，单位mm，防止突变
double MAX_ORI_DELTA = 0.005; // 单次力控周期内，姿态最大变化量，单位度，防止突变

constexpr char const *ROBOT_SERVICE_DLL = ROBOT_SERVICE_NAME;

Crp::CSDKLoader gLoader(ROBOT_SERVICE_DLL);
Crp::IRobotService *g_robot = nullptr;
Crp::IMotionService *g_motion = nullptr;
std::atomic<bool> gStopping{false};
Crp::IFileService *g_file = nullptr;
Crp::IModelService *g_model = nullptr;
std::atomic<bool> g_sdk_ready{false};
std::atomic<bool> g_external_services{false};

extern "C"
{
    // 声明C里的全局变量，类型必须和C里完全一致！
    extern double ChargeFinished;
    extern double g_MoveDistance;
    void ResetForceControlRuntimeState(void);
}

float g_force[6] = {0};

static bool ensure_sdk_services()
{
    if (g_external_services.load())
    {
        return g_robot && g_motion && g_file;
    }

    if (!g_sdk_ready.load())
    {
        if (!gLoader.initialize())
        {
            printf("SDK初始化失败\n");
            return false;
        }
        g_sdk_ready = true;
    }

    if (!g_robot)
        g_robot = gLoader.getService<Crp::IRobotService>(ID_ROBOT_SERVICE);
    if (!g_motion)
        g_motion = gLoader.getService<Crp::IMotionService>(ID_MOTION_SERVICE);
    if (!g_file)
        g_file = gLoader.getService<Crp::IFileService>(ID_FILE_SERVICE);
    if (!g_model)
        g_model = gLoader.getService<Crp::IModelService>(ID_MODEL_SERVICE);

    if (!g_robot || !g_motion || !g_file)
    {
        printf("获取服务失败\n");
        return false;
    }

    return true;
}

static bool ensure_robot_session(const char *ip, bool ensure_servo_ready)
{
    if (!ensure_sdk_services())
        return false;

    if (g_external_services.load())
    {
        if (!g_robot->isConnected())
        {
            printf("外部CRP上下文未连接机器人\n");
            return false;
        }
        printf("复用外部CRP上下文\n");
    }
    else if (!g_robot->isConnected())
    {
        if (!g_robot->connect(ip))
        {
            printf("连接机器人失败: %s\n", ip);
            return false;
        }
        printf("已连接机器人: %s\n", ip);
    }
    else
    {
        printf("复用已存在的机器人连接: %s\n", ip);
    }

    if (ensure_servo_ready)
    {
        g_robot->clearError();
        if (!g_robot->isServoOn())
        {
            if (!g_robot->servoPowerOn())
            {
                printf("伺服上电失败\n");
                return false;
            }
            for (int i = 0; i < 50; ++i)
            {
                if (g_robot->isServoOn())
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        printf("伺服已就绪\n");
    }

    return true;
}

static bool ensure_servo_ready_for_program_start(const char *context)
{
    if (!g_robot)
    {
        printf("[%s] robot service is null\n", context);
        return false;
    }

    if (!g_robot->isServoOn())
    {
        printf("[%s] 检测到伺服未上电，准备重新上电\n", context);
        if (!g_robot->servoPowerOn())
        {
            printf("[%s] 伺服上电失败\n", context);
            return false;
        }
    }

    for (int i = 0; i < 50; ++i)
    {
        if (g_robot->isServoOn())
        {
            printf("[%s] 伺服状态确认完成\n", context);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    printf("[%s] 等待伺服就绪超时\n", context);
    return false;
}

static bool start_program_allow_reuse(const char *program, int line, const char *context)
{
    if (!ensure_servo_ready_for_program_start(context))
    {
        return false;
    }

    if (g_robot->startProgram(program, line))
    {
        return true;
    }

    const char *program_now = g_robot->getProgramPath();
    if (program_now != nullptr && std::strstr(program_now, program) != nullptr)
    {
        printf("[%s] 检测到程序已在运行，复用当前程序: %s\n", context, program_now);
        return true;
    }

    return false;
}

void AttachExternalCrpServices(void *robot, void *motion, void *file)
{
    g_robot = static_cast<Crp::IRobotService *>(robot);
    g_motion = static_cast<Crp::IMotionService *>(motion);
    g_file = static_cast<Crp::IFileService *>(file);
    g_external_services = (g_robot != nullptr && g_motion != nullptr && g_file != nullptr);
}

void DetachExternalCrpServices()
{
    g_external_services = false;
    g_robot = nullptr;
    g_motion = nullptr;
    g_file = nullptr;
}

typedef std::vector<Crp::SRobotPosition> pos_vector_t;
typedef std::vector<Crp::SJointPosition> jnt_vector_t;

int g_CFuncflagtest = 0; // 控制指令  1-插枪 2-保持 3-拔枪  0-默认
int charge_in_time = 0;
int charge_addjust_time = 0;
int charge_out_time = 0;

// ===================== 力传感器缓冲区 =====================
float g_force_data[6] = {0};
std::mutex g_force_mtx; // 定义互斥锁

// 调用外部程序 simple_test（传入网卡名）来读取力传感器数据。实时解析输出行中的六维力信息，写入全局缓冲区 g_force，并用互斥锁保护。
void force_read_thread(const char *eth)
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "sudo ./simple_test %s 2>/dev/null", eth);
    printf("[力传感器] 启动: %s\n", cmd);

    FILE *pipe = popen(cmd, "r");
    if (!pipe)
    {
        perror("[力传感器] 启动失败");
        return;
    }

    // 获取管道文件描述符
    int fd = fileno(pipe);
    if (fd < 0)
    {
        perror("[力传感器] 获取文件描述符失败");
        pclose(pipe);
        return;
    }

    char line[256];
    struct pollfd fds;
    fds.fd = fd;
    fds.events = POLLIN;

    while (!gStopping)
    {
        // 使用 poll 设置超时，每 100ms 检查一次退出标志
        int ret = poll(&fds, 1, 100);  // 100ms 超时
        
        if (ret < 0)
        {
            // poll 错误
            break;
        }
        else if (ret == 0)
        {
            // 超时，继续检查退出标志
            continue;
        }
        
        // 有数据可读
        if (fds.revents & POLLIN)
        {
            if (fgets(line, sizeof(line), pipe))
            {
                char *key = strstr(line, "十进制转换：");
                if (key)
                {
                    key += strlen("十进制转换：");
                    float fx, fy, fz, mx, my, mz;
                    if (sscanf(key, "%f,%f,%f,%f,%f,%f", &fx, &fy, &fz, &mx, &my, &mz) == 6)
                    {
                        std::lock_guard<std::mutex> l(g_force_mtx);
                        g_force[0] = fx;
                        g_force[1] = fy;
                        g_force[2] = fz;
                        g_force[3] = mx;
                        g_force[4] = my;
                        g_force[5] = mz;
                    }
                }
            }
            else
            {
                // 管道关闭或出错
                break;
            }
        }
    }

    // 主动关闭子进程
    int pclose_result = pclose(pipe);
    printf("[力传感器] 退出，pclose 返回值: %d\n", pclose_result);
}
// ====================================================================

// 线程安全的位姿缓冲区
struct PoseBuffer
{
    std::mutex mtx;
    double latest_pos[6] = {0};
    bool has_new_data = false;

    // 写入最新位姿
    void write(const double pos[6])
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < 6; ++i)
        {
            latest_pos[i] = pos[i];
        }
        has_new_data = true;
    }

    // 读取最新位姿（无新数据则返回false）
    bool read(double out_pos[6])
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!has_new_data)
            return false;
        for (int i = 0; i < 6; ++i)
        {
            out_pos[i] = latest_pos[i];
        }
        // 可选：是否重置标记（按需）
        // has_new_data = false;
        return true;
    }
} g_pose_buffer;

// ===================== 线程安全关节角缓冲区（SDK要求必须12长度）=====================
struct JointBuffer
{
    std::mutex mtx;
    double data[12] = {0};

    // 写入：永远覆盖最新数据
    void write(const double *joint)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < 12; ++i)
            data[i] = joint[i];
    }

    // 读取：永远返回数据（不判断是否更新）
    void read(double *out)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < 12; ++i)
            out[i] = data[i];
    }
} g_joint_buffer;

// 退出信号
void signal_handler(int sig)
{
    (void)sig;
    gStopping = true;
}

// 异步读取位姿线程
void pose_read_thread()
{
    auto start_time = std::chrono::steady_clock::now(); // 用于计算相对时间戳
    while (!gStopping)
    {
        double world_pos[6] = {0};
        // 读取位姿（耗时不固定，但不影响下发周期）
        if (g_robot->getCurrentPosition(Crp::CS_World, world_pos, 6))
        {
            g_pose_buffer.write(world_pos); // 写入缓冲区

            // printf("位姿更新: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     world_pos[0], world_pos[1], world_pos[2], world_pos[3], world_pos[4], world_pos[5]);
            //------------原代码，只写入位姿---------------
            // if (g_pose_log.is_open())
            // {
            //     auto now = std::chrono::steady_clock::now();
            //     auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            //     std::lock_guard<std::mutex> lock(g_log_mtx);
            //     g_pose_log << elapsed_ms << ","
            //                << std::fixed << std::setprecision(3)
            //                << world_pos[0] << "," << world_pos[1] << "," << world_pos[2] << ","
            //                << world_pos[3] << "," << world_pos[4] << "," << world_pos[5] << std::endl;
            // }
            // ----- 写入 CSV 文件 -----
            if (g_pose_log.is_open())
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

                // 1. 读取力数据（加锁复制）
                float force[6];
                {
                    // std::lock_guard<std::mutex> lock(g_force_mtx);
                    std::copy(g_force, g_force + 6, force);
                }

                // 2. 读取其他全局变量（直接取值，基本类型原子性足够）
                int flag = g_CFuncflagtest;
                double charge_finished = ChargeFinished;
                double move_distance = g_MoveDistance;

                // 3. 写入一行
                // std::lock_guard<std::mutex> lock(g_log_mtx);
                g_pose_log << elapsed_ms << ","
                           << std::fixed << std::setprecision(3)
                           << world_pos[0] << "," << world_pos[1] << "," << world_pos[2] << ","
                           << world_pos[3] << "," << world_pos[4] << "," << world_pos[5] << ","

                           << std::setprecision(3) // 力数据保留3位小数
                           << force[0] << "," << force[1] << "," << force[2] << ","
                           << force[3] << "," << force[4] << "," << force[5] << ","

                           << flag << ","
                           << std::setprecision(6) // 充电完成标志和移动距离保留6位
                           << charge_finished << "," << move_distance
                           << std::endl;
            }
        }
        // 读取线程的周期（可略小于10ms，保证数据新鲜）
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
}

// 放在 pose_read_thread 函数后面、main 函数前面
void joint_read_thread()
{
    double joint_pos[12] = {0};

    int consecutive_fail = 0;
    const int MAX_FAIL_PRINT = 10;
    printf("[关节角读取线程] 启动（安全模式）\n");

    while (!gStopping)
    {
        if (!g_robot->isConnected() || !g_robot->isServoOn())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 仅负责关节角读取
        bool joint_ok = g_robot->getCurrentPosition(Crp::CS_Joint, joint_pos, 12);
        if (joint_ok)
        {
            g_joint_buffer.write(joint_pos);
            consecutive_fail = 0;
        }
        else
        {
            consecutive_fail++;
            if (consecutive_fail % MAX_FAIL_PRINT == 0)
            {
                printf("[关节角读取线程] 关节角读取连续失败%d次\n", consecutive_fail);
            }
            g_joint_buffer.write(joint_pos);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    printf("[关节角读取线程] 退出\n");
}

const char *MovePathResultStr(Crp::EMovePathResult resa)
{
    switch (resa)
    {
    case Crp::EMovePathResult::Success:
        return "Success";
    case Crp::EMovePathResult::NotInit:
        return "NotInit";
    case Crp::EMovePathResult::BufEmpty:
        return "BufEmpty";
    case Crp::EMovePathResult::IsRunning:
        return "IsRunning";
    case Crp::EMovePathResult::ParamError:
        return "ParamError";
    default:
        return "Unknown";
    }
}



#ifdef __cplusplus
extern "C" {
#endif

int ForceControl_demo()
{
    // 确保 data 目录存在
    const char *data_dir = "data";
    struct stat st;
    if (::stat(data_dir, &st) != 0)
    {                            // 使用 ::stat 调用全局函数
        ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
    }
   
    const char *ip ="192.168.1.133";
    const char *eth = "eth1";

    // 初始化键盘（必须加）
    init_keyboard();
    atexit(close_keyboard); // 程序退出自动恢复终端

    // 2. 注册退出信号
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);

    if (!ensure_robot_session(ip, true))
    {
        return -1;
    }

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_info = std::localtime(&now_c);
    char filename[256];
    std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);

    // 打开位姿日志文件（文件名带时间戳）
    g_pose_log.open(filename, std::ios::out | std::ios::trunc);
    if (!g_pose_log.is_open())
    {
        printf("警告：无法创建位姿日志文件 filename\n");
    }
    else
    {
        // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
        //
        g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                   << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                   << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
        printf("位姿日志将保存至 %s\n", filename);
    }

    printf("跳过重复上下电，复用主流程机器人会话\n");

    // ===================== 核心顺滑参数 =====================
    const auto CONTROL_PERIOD = std::chrono::milliseconds(10); // 10ms 官方最优
    const double SMOOTH_KP = 0.2;                              // 柔顺系数（0.1~0.3 越大响应越快）
    const double MAX_STEP = 0.5;                               // 每10ms最大走 0.5mm（防止跳变）

    // 全局保存上一帧发送点（保证轨迹连续）
    Crp::SRobotPosition g_last_send_point;

    // 7. 设置速度与模式
    g_robot->setSpeedRatio(5);
    g_robot->setWorkMode(Crp::RM_Playing);
    // g_robot->FKine()
    if (!g_file->upload("data/guidancePos.pro", "guidancePos.pro"))
    {
        printf("Fail to upload guidancePos.pro\n");
        return -1;
    }

    if (!g_robot->startProgram("guidancePos.pro", 0))
    {
        printf("Fail to start program\n");
        return -1;
    }

    while (!g_motion->isReady(Crp::EMotionType::Path))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    printf("开始 10ms 实时位姿控制，按 Ctrl+C 退出\n");

    // ===================== 启动异步读取线程 =====================
    std::thread read_thread(pose_read_thread);
    std::thread force_therad(force_read_thread, eth);
    std::thread joint_thread(joint_read_thread); // 新增：关节角线程

    // ===================== 核心下发循环：严格10ms周期 =====================

    Crp::SRobotPosition target;
    Crp::SRobotPosition target_output;
    Crp::SInstMoveL movepoint;
    const int TOOL_NO = 10;
    const int USER_NO = 0;
    Crp::SRobotPosition sendpoint_last;

    double world_pos[6] = {0};
    // 基准时间（保证绝对周期，而非相对睡眠）
    auto base_time = std::chrono::steady_clock::now();
    int index = 0;
    constexpr int kJointBufSize = 12;
    double cur_pos_joint[kJointBufSize] = {0};

    while (!gStopping)
    {

        // g_motion->movePath(0);   // 或者 g_motion->movePath(0);
        //  ===================== A/S/D 切换模式 =====================
        if (kbhit())
        {
            char key = get_key();
            switch (key)
            {
            case 'A':
            case 'a':
                g_CFuncflagtest = 1;
                printf("A键按下：g_CFuncflag = 1\n");
                break;
            case 'S':
            case 's':
                g_CFuncflagtest = 2;
                printf("S键按下：g_CFuncflag = 2\n");
                break;
            case 'D':
            case 'd':
                g_CFuncflagtest = 3;
                printf("D键按下：g_CFuncflag = 3\n");
                break;
            case 'F':
            case 'f':
                g_CFuncflagtest = 0;
                printf("F键按下：g_CFuncflag = 0\n");
                break;
            case 'G':
            case 'g':
                g_CFuncflagtest = 4;
                printf("G键按下：g_CFuncflag = 4\n");
                break;
            case 'Q':
            case 'q':
                // 打印六维力
                // printf("原始读数 = %f ,%f ,%f ,%f ,%f ,%f   辨识值= %f ,%f ,%f,%f ,%f ,%f  \n", ForceAftFlt[0], ForceAftFlt[1], ForceAftFlt[2], ForceAftFlt[3], ForceAftFlt[4], ForceAftFlt[5],
                //        ForceAftCmp[0], ForceAftCmp[1], ForceAftCmp[2], ForceAftCmp[3], ForceAftCmp[4], ForceAftCmp[5]);
                break;
            }
        }
        // ==========================================================

        // 1) 严格对齐10ms周期（绝对时间基准）
        base_time += std::chrono::milliseconds(10);
        std::this_thread::sleep_until(base_time);

        // 2) 从缓冲区读取最新位姿（非阻塞）
        bool has_data = g_pose_buffer.read(world_pos);
        if (!has_data)
        {
            printf("警告：未读取到新位姿，使用上一帧数据\n");
            // 无新数据时可复用上次的world_pos，避免断流
        }
        if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
        {
            // printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
        }
        g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
        // 3) 构造目标位姿
        target.x = world_pos[0];
        target.y = world_pos[1];
        target.z = world_pos[2];
        target.Rx = world_pos[3];
        target.Ry = world_pos[4];
        target.Rz = world_pos[5];

        std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
        for (int i = 0; i < 6; i++)
        {
            jnt_pos_input[i] = cur_pos_joint[i];
        }
        // 读取力
        double Forcesensor_input[6];
        {
            // std::lock_guard<std::mutex> l(g_force_mtx);
            std::copy(g_force, g_force + 6, Forcesensor_input);
        }
        Vector6 target_pose;
        target_pose[0] = target.x;
        target_pose[1] = target.y;
        target_pose[2] = target.z;
        target_pose[3] = target.Rx;
        target_pose[4] = target.Ry;
        target_pose[5] = target.Rz;

        Vector6 init_joint;
        init_joint[0] = jnt_pos_input[0];
        init_joint[1] = jnt_pos_input[1];
        init_joint[2] = jnt_pos_input[2];
        init_joint[3] = jnt_pos_input[3];
        init_joint[4] = jnt_pos_input[4];
        init_joint[5] = jnt_pos_input[5];

        if (index > 2)
        {
            target = sendpoint_last;
        }

        if (index < 1)
        {
            //  //按vector进行输出
            Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
            jnt_vector_t joint_path_test;
            for (int i = 0; i < 2; i++)
            {
                Crp::SJointPosition point;
                point.body[0] = ik_result[0];
                point.body[1] = ik_result[1];
                point.body[2] = ik_result[2];
                point.body[3] = ik_result[3];
                point.body[4] = ik_result[4];
                point.body[5] = ik_result[5];
                joint_path_test.emplace_back(point);
                //  sendpoint_last = point;
            }
            g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
            Crp::EMovePathResult result = g_motion->movePath(5);
        }

        // 动态步长调整
        if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
        {
            // MAX_POS_DELTA = 0.2;
            // 基于行程线性增加步长
            MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
            if (MAX_POS_DELTA > 0.25f)
            {
                MAX_POS_DELTA = 0.25f;
            }
            if (MAX_POS_DELTA < 0.05f)
            {
                MAX_POS_DELTA = 0.05f;
            }
            MAX_ORI_DELTA = 0.005;
        }
        else if (g_CFuncflagtest == 4) // 零力模式
        {
            MAX_POS_DELTA = 0.15;
            MAX_ORI_DELTA = 0.02;
        }
        else
        {
            MAX_POS_DELTA = 0.1;
            MAX_ORI_DELTA = 0.005;
        }
        // printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //     Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
        // ==============================================================
        // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
        // 确保已经读取到了真实位姿
        if ((abs(target.x - 0.f) > 0.001))
        {
            // //力控输入
            double Force_input[6] = {0}; // 测试虚拟力数组
            Force_input[2] = 50.0;
            double pose_input[6] = {0}; // 创建力控输入位姿数组

            pose_input[0] = target.x / 1000.f;
            pose_input[1] = target.y / 1000.f;
            pose_input[2] = target.z / 1000.f;
            pose_input[3] = target.Rx;
            pose_input[4] = target.Ry;
            pose_input[5] = target.Rz;
            EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Forcesensor_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
            // 打印
            // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
            //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
            // printf("力控计算状态=%.2f \n ",ChargeFinished);

            // //复制力控运行结果  单位m转换为mm
            target_output.x = ForceControl_pos.X * 1000.f;
            target_output.y = ForceControl_pos.Y * 1000.f;
            target_output.z = ForceControl_pos.Z * 1000.f;
            target_output.Rx = ForceControl_pos.roll;
            target_output.Ry = ForceControl_pos.pitch;
            target_output.Rz = ForceControl_pos.yaw;

            // 起点：当前机器人实际位姿
            double start_x = target.x;
            double start_y = target.y;
            double start_z = target.z;
            double start_Rx = target.Rx;
            double start_Ry = target.Ry;
            double start_Rz = target.Rz;
            // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
            double raw_end_x = target_output.x;
            double raw_end_y = target_output.y;
            double raw_end_z = target_output.z;
            double raw_end_Rx = target_output.Rx;
            double raw_end_Ry = target_output.Ry;
            double raw_end_Rz = target_output.Rz;

            // 计算原始变化量
            double dx = raw_end_x - start_x;
            double dy = raw_end_y - start_y;
            double dz = raw_end_z - start_z;
            double drx = raw_end_Rx - start_Rx;
            double dry = raw_end_Ry - start_Ry;
            double drz = raw_end_Rz - start_Rz;

            bool is_limited = false;
            // 位置变化量：按比例限幅（保持方向）
            double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
            {
                double scale = MAX_POS_DELTA / pos_distance;
                dx = dx * scale;
                dy = dy * scale;
                dz = dz * scale;
                is_limited = true;
            }

            // 姿态变化量：按比例限幅（保持方向）
            double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
            if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
            {
                double scale = MAX_ORI_DELTA / ori_distance;
                drx = drx * scale;
                dry = dry * scale;
                drz = drz * scale;
                is_limited = true;
            }

            // 得到限幅后的最终终点
            double end_x = start_x + dx;
            double end_y = start_y + dy;
            double end_z = start_z + dz;
            double end_Rx = start_Rx + drx;
            double end_Ry = start_Ry + dry;
            double end_Rz = start_Rz + drz;

            // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
            pos_vector_t joint_path;
            jnt_vector_t jnt_path;
            for (int i = 0; i < 1; i++)
            {
                // 插值系数：5个点分4步，t从0到1，保证均匀间隔
                // double t = i / 9.0;
                double t = i / 1.0;
                Crp::SRobotPosition point;
                // 对6个位姿维度分别做线性插补
                point.x = start_x + (end_x - start_x);
                point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
                point.z = start_z + (end_z - start_z);
                point.Rx = start_Rx + (end_Rx - start_Rx);
                point.Ry = start_Ry + (end_Ry - start_Ry);
                point.Rz = start_Rz + (end_Rz - start_Rz);
                joint_path.emplace_back(point);

                Vector6 target_pose;
                target_pose[0] = point.x;
                target_pose[1] = point.y;
                target_pose[2] = point.z;
                target_pose[3] = point.Rx;
                target_pose[4] = point.Ry;
                target_pose[5] = point.Rz;
                Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
                Crp::SJointPosition jnt;
                jnt.body[0] = ik_result[0];
                jnt.body[1] = ik_result[1];
                jnt.body[2] = ik_result[2];
                jnt.body[3] = ik_result[3];
                jnt.body[4] = ik_result[4];
                jnt.body[5] = ik_result[5];
                jnt_path.emplace_back(jnt);

                sendpoint_last = point;
                // 打印
                // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
                //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
            }

            auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
            // 打印
            // printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);

            // 4) 单帧位姿下发
            // g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
            g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
            // g_motion->sendPath(&target, 1 , TOOL_NO, USER_NO);  //按vector发送
            Crp::EMovePathResult result = g_motion->movePath(5);
            index++;
            if (result == Crp::EMovePathResult::Success)
            {
                printf("start moving success...\n");
            }

            // 打印
            // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
            // printf("%.2f", ChargeFinished);
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            // printf("[%02d:%02d:%02d.%03lld]\n",
            //        (int)localtime(&t)->tm_hour,
            //        (int)localtime(&t)->tm_min,
            //        (int)localtime(&t)->tm_sec,
            //        (long long int)ms.count());
        }
    }
    g_motion->finalize(Crp::EMotionType::Instruction);

    // ===================== 退出清理 =======主循环中计算得到了力控插补路径 ==============
    printf("正在退出...\n");
    // 等待读取线程退出
    if (read_thread.joinable())
    {
        read_thread.join();
    }
    // 在等待 read_thread 之后
    if (g_pose_log.is_open())
    {
        g_pose_log.close();
        printf("位姿日志已保存至 %s\n", filename);
    }
    printf("保留机器人会话，不执行伺服下电或断连\n");
    printf("已退出\n");

    return 0;
}

int GetForcecontrolState(void)
{
    return ChargeFinished;
}

int PowerOn_PPmode() //PP模式启动
{
     // 确保 data 目录存在
     const char *data_dir = "data";
     struct stat st;
     if (::stat(data_dir, &st) != 0)
     {                            // 使用 ::stat 调用全局函数
         ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
     }
     
     const char *ip ="192.168.1.12";
     const char *eth = "eth2";
 
     // 2. 注册退出信号
     signal(SIGINT, signal_handler);
     signal(SIGTERM, signal_handler);
     signal(SIGABRT, signal_handler);
 
     if (!ensure_robot_session(ip, true))
     {
         return -1;
     }
 
     // 获取当前时间
     auto now = std::chrono::system_clock::now();
     std::time_t now_c = std::chrono::system_clock::to_time_t(now);
     std::tm *tm_info = std::localtime(&now_c);
     char filename[256];
     std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);
 
     // 打开位姿日志文件（文件名带时间戳）
     g_pose_log.open(filename, std::ios::out | std::ios::trunc);
     if (!g_pose_log.is_open())
     {
         printf("警告：无法创建位姿日志文件 filename\n");
     }
     else
     {
         // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
         //
         g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                    << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                    << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
         printf("位姿日志将保存至 %s\n", filename);
     }
 
     printf("跳过重复上下电，复用主流程机器人会话\n");
 
     // ===================== 核心顺滑参数 =====================
     const auto CONTROL_PERIOD = std::chrono::milliseconds(10); //10ms 官方最优
     const double SMOOTH_KP = 0.2;                              //柔顺系数（0.1~0.3 越大响应越快）
     const double MAX_STEP = 0.5;                               //每10ms最大走 0.5mm（防止跳变）
 
     // 全局保存上一帧发送点（保证轨迹连续）
     Crp::SRobotPosition g_last_send_point;
 
     // 7. 设置速度与模式
     g_robot->setSpeedRatio(5);
     g_robot->setWorkMode(Crp::RM_Playing);
     // g_robot->FKine()
    //  if (!g_file->upload("data/guidanceInst.pro", "guidanceInst.pro"))
    //  {
    //      printf("Fail to upload guidanceInst.pro\n");
    //      return -1;
    //  }
 
     if (!start_program_allow_reuse("guidanceInst.pro", 0, "PowerOn_PPmode"))
     {
         printf("Fail to start program\n");
         return -1;
     }

     printf("进入isready\n");
    //  while (!g_motion->isReady(Crp::EMotionType::Path))
    //  {
    //      std::this_thread::sleep_for(std::chrono::microseconds(100));
    //  }
 
    //  // ===================== 启动异步读取线程 =====================
    //  std::thread read_thread(pose_read_thread);
    //  std::thread force_therad(force_read_thread, eth);
    //  std::thread joint_thread(joint_read_thread); // 新增：关节角线程
 
    //  // ===================== 核心下发循环：严格10ms周期 =====================
 
    //  Crp::SRobotPosition target;
    //  Crp::SRobotPosition target_output;
    //  Crp::SInstMoveL movepoint;
    //  const int TOOL_NO = 10;
    //  const int USER_NO = 0;
    //  Crp::SRobotPosition sendpoint_last;
 
    //  double world_pos[6] = {0};
    //  // 基准时间（保证绝对周期，而非相对睡眠）
    //  auto base_time = std::chrono::steady_clock::now();
    //  int index = 0;
    //  constexpr int kJointBufSize = 12;
    //  double cur_pos_joint[kJointBufSize] = {0};
 
    //  g_CFuncflagtest = 1;
    //  gStopping = false;
 
    //  while (!gStopping)
    //  {
 
    //      // 1) 严格对齐10ms周期（绝对时间基准）
    //      base_time += std::chrono::milliseconds(10);
    //      std::this_thread::sleep_until(base_time);
 
    //      // 2) 从缓冲区读取最新位姿（非阻塞）
    //      bool has_data = g_pose_buffer.read(world_pos);
    //      if (!has_data)
    //      {
    //          printf("警告：未读取到新位姿，使用上一帧数据\n");
    //          // 无新数据时可复用上次的world_pos，避免断流
    //      }
    //      if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
    //      {
    //          printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
    //      }
    //      g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
    //      // 3) 构造目标位姿
    //      target.x = world_pos[0];
    //      target.y = world_pos[1];
    //      target.z = world_pos[2];
    //      target.Rx = world_pos[3];
    //      target.Ry = world_pos[4];
    //      target.Rz = world_pos[5];
 
    //      std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
    //      for (int i = 0; i < 6; i++)
    //      {
    //          jnt_pos_input[i] = cur_pos_joint[i];
    //      }
    //      // 读取力
    //      double Forcesensor_input[6];
    //      {
    //          std::lock_guard<std::mutex> l(g_force_mtx);
    //          std::copy(g_force, g_force + 6, Forcesensor_input);
    //      }
    //      Vector6 target_pose;
    //      target_pose[0] = target.x;
    //      target_pose[1] = target.y;
    //      target_pose[2] = target.z;
    //      target_pose[3] = target.Rx;
    //      target_pose[4] = target.Ry;
    //      target_pose[5] = target.Rz;
 
    //      Vector6 init_joint;
    //      init_joint[0] = jnt_pos_input[0];
    //      init_joint[1] = jnt_pos_input[1];
    //      init_joint[2] = jnt_pos_input[2];
    //      init_joint[3] = jnt_pos_input[3];
    //      init_joint[4] = jnt_pos_input[4];
    //      init_joint[5] = jnt_pos_input[5];
 
    //      if (index > 2)
    //      {
    //          target = sendpoint_last;
    //      }
 
    //      if (index < 1)
    //      {
    //          //  //按vector进行输出
    //          Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
    //          jnt_vector_t joint_path_test;
    //          for (int i = 0; i < 2; i++)
    //          {
    //              Crp::SJointPosition point;
    //              point.body[0] = ik_result[0];
    //              point.body[1] = ik_result[1];
    //              point.body[2] = ik_result[2];
    //              point.body[3] = ik_result[3];
    //              point.body[4] = ik_result[4];
    //              point.body[5] = ik_result[5];
    //              joint_path_test.emplace_back(point);
    //              //  sendpoint_last = point;
    //          }
    //          g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
    //          Crp::EMovePathResult result = g_motion->movePath(5);
    //      }
 
    //      // 动态步长调整
    //      if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
    //      {
    //          // MAX_POS_DELTA = 0.2;
    //          // 基于行程线性增加步长
    //          MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
    //          if (MAX_POS_DELTA > 0.25f)
    //          {
    //              MAX_POS_DELTA = 0.25f;
    //          }
    //          if (MAX_POS_DELTA < 0.05f)
    //          {
    //              MAX_POS_DELTA = 0.05f;
    //          }
    //          MAX_ORI_DELTA = 0.005;
    //      }
    //      else if (g_CFuncflagtest == 4) // 零力模式
    //      {
    //          MAX_POS_DELTA = 0.15;
    //          MAX_ORI_DELTA = 0.02;
    //      }
    //      else
    //      {
    //          MAX_POS_DELTA = 0.1;
    //          MAX_ORI_DELTA = 0.005;
    //      }
 
    //      printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
    //          Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
    //      // ==============================================================
 
    //      // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
    //      //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
    //      // 确保已经读取到了真实位姿
    //      if ((abs(target.x - 0.f) > 0.001))
    //      {
    //          // //力控输入
    //          double Force_input[6] = {0}; // 测试虚拟力数组
    //          Force_input[2] = 50.0;
    //          double pose_input[6] = {0}; // 创建力控输入位姿数组
 
    //          pose_input[0] = target.x / 1000.f;
    //          pose_input[1] = target.y / 1000.f;
    //          pose_input[2] = target.z / 1000.f;
    //          pose_input[3] = target.Rx;
    //          pose_input[4] = target.Ry;
    //          pose_input[5] = target.Rz;
    //          EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Force_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
    //          // 打印
    //          // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
    //          //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
    //          printf("力控计算状态=%.2f \n ",ChargeFinished);
 
    //          // //复制力控运行结果  单位m转换为mm
    //          target_output.x = ForceControl_pos.X * 1000.f;
    //          target_output.y = ForceControl_pos.Y * 1000.f;
    //          target_output.z = ForceControl_pos.Z * 1000.f;
    //          target_output.Rx = ForceControl_pos.roll;
    //          target_output.Ry = ForceControl_pos.pitch;
    //          target_output.Rz = ForceControl_pos.yaw;
 
    //          // 起点：当前机器人实际位姿
    //          double start_x = target.x;
    //          double start_y = target.y;
    //          double start_z = target.z;
    //          double start_Rx = target.Rx;
    //          double start_Ry = target.Ry;
    //          double start_Rz = target.Rz;
    //          // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
    //          double raw_end_x = target_output.x;
    //          double raw_end_y = target_output.y;
    //          double raw_end_z = target_output.z;
    //          double raw_end_Rx = target_output.Rx;
    //          double raw_end_Ry = target_output.Ry;
    //          double raw_end_Rz = target_output.Rz;
 
    //          // 计算原始变化量
    //          double dx = raw_end_x - start_x;
    //          double dy = raw_end_y - start_y;
    //          double dz = raw_end_z - start_z;
    //          double drx = raw_end_Rx - start_Rx;
    //          double dry = raw_end_Ry - start_Ry;
    //          double drz = raw_end_Rz - start_Rz;
 
    //          bool is_limited = false;
    //          // 位置变化量：按比例限幅（保持方向）
    //          double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
    //          if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
    //          {
    //              double scale = MAX_POS_DELTA / pos_distance;
    //              dx = dx * scale;
    //              dy = dy * scale;
    //              dz = dz * scale;
    //              is_limited = true;
    //          }
 
    //          // 姿态变化量：按比例限幅（保持方向）
    //          double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
    //          if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
    //          {
    //              double scale = MAX_ORI_DELTA / ori_distance;
    //              drx = drx * scale;
    //              dry = dry * scale;
    //              drz = drz * scale;
    //              is_limited = true;
    //          }
 
    //          // 得到限幅后的最终终点
    //          double end_x = start_x + dx;
    //          double end_y = start_y + dy;
    //          double end_z = start_z + dz;
    //          double end_Rx = start_Rx + drx;
    //          double end_Ry = start_Ry + dry;
    //          double end_Rz = start_Rz + drz;
 
    //          // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
    //          pos_vector_t joint_path;
    //          jnt_vector_t jnt_path;
    //          for (int i = 0; i < 1; i++)
    //          {
    //              // 插值系数：5个点分4步，t从0到1，保证均匀间隔
    //              // double t = i / 9.0;
    //              double t = i / 1.0;
    //              Crp::SRobotPosition point;
    //              // 对6个位姿维度分别做线性插补
    //              point.x = start_x + (end_x - start_x);
    //              point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
    //              point.z = start_z + (end_z - start_z);
    //              point.Rx = start_Rx + (end_Rx - start_Rx);
    //              point.Ry = start_Ry + (end_Ry - start_Ry);
    //              point.Rz = start_Rz + (end_Rz - start_Rz);
    //              joint_path.emplace_back(point);
 
    //              Vector6 target_pose;
    //              target_pose[0] = point.x;
    //              target_pose[1] = point.y;
    //              target_pose[2] = point.z;
    //              target_pose[3] = point.Rx;
    //              target_pose[4] = point.Ry;
    //              target_pose[5] = point.Rz;
    //              Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
    //              Crp::SJointPosition jnt;
    //              jnt.body[0] = ik_result[0];
    //              jnt.body[1] = ik_result[1];
    //              jnt.body[2] = ik_result[2];
    //              jnt.body[3] = ik_result[3];
    //              jnt.body[4] = ik_result[4];
    //              jnt.body[5] = ik_result[5];
    //              jnt_path.emplace_back(jnt);
 
    //              sendpoint_last = point;
    //              // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
    //              //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
    //          }
 
    //          auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
    //          // printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);
 
    //          // 4) 单帧位姿下发
    //          // g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
    //          g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
    //          // g_motion->sendPath(&target, 1 , TOOL_NO, USER_NO);  //按vector发送
    //          Crp::EMovePathResult result = g_motion->movePath(5);
    //          index++;
    //          if (result == Crp::EMovePathResult::Success)
    //          {
    //              printf("start moving success...\n");
    //          }
 
    //          // 打印
    //          // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
    //          //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
    //          // printf("%.2f", ChargeFinished);
    //          auto now = std::chrono::system_clock::now();
    //          auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    //          std::time_t t = std::chrono::system_clock::to_time_t(now);
    //          // printf("[%02d:%02d:%02d.%03lld]\n",
    //          //        (int)localtime(&t)->tm_hour,
    //          //        (int)localtime(&t)->tm_min,
    //          //        (int)localtime(&t)->tm_sec,
    //          //        (long long int)ms.count());
    //      }
    //      if(ChargeFinished == 1 || ChargeFinished == 2) //插枪完成
    //      {
    //          gStopping = true;
    //      }
 
    //  }
    //  if(ChargeFinished == 1)
    //  {
    //      printf("插枪完成\n");
    //  }
    //  if(ChargeFinished == 2)
    //  {
    //      printf("插枪失败\n");
    //  }
     
    g_motion->finalize(Crp::EMotionType::Instruction);
    
    const char* program_now = g_robot->getProgramPath();
    if (program_now != nullptr) {
        printf("当前程序路径: %s\n", program_now);
    } else {
        printf("未获取到程序路径\n");
    }

     printf("PP模式已启动\n");
 
     return 0;
}



int Forcecontrol_ChargeIn()
{
    // 确保 data 目录存在
    const char *data_dir = "data";
    struct stat st;
    if (::stat(data_dir, &st) != 0)
    {                            // 使用 ::stat 调用全局函数
        ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
    }
    
    const char *ip ="192.168.1.12";
    const char *eth = "eth2";

    // 2. 注册退出信号
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);

    // 3. 初始化SDK
    // if (!gLoader.initialize())
    // {
    //     printf("SDK初始化失败\n");
    //     return -1;
    // }

    // 4. 获取服务
    // g_robot = gLoader.getService<Crp::IRobotService>(ID_ROBOT_SERVICE);
    // g_motion = gLoader.getService<Crp::IMotionService>(ID_MOTION_SERVICE);
    // g_file = gLoader.getService<Crp::IFileService>(ID_FILE_SERVICE);
    // g_model = gLoader.getService<Crp::IModelService>(ID_MODEL_SERVICE);

    // if (!g_robot || !g_motion || !g_file)
    // {
    //     printf("获取服务失败\n");
    //     return -1;
    // }

    // // 5. 连接机器人
    // if (!g_robot->connect(ip))
    // {
    //     printf("连接机器人失败: %s\n", ip);
    //     return -1;
    // }
    // printf("已连接机器人: %s\n", ip);

    ResetForceControlRuntimeState();

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_info = std::localtime(&now_c);
    char filename[256];
    std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);

    // 打开位姿日志文件（文件名带时间戳）
    g_pose_log.open(filename, std::ios::out | std::ios::trunc);
    if (!g_pose_log.is_open())
    {
        printf("警告：无法创建位姿日志文件 filename\n");
    }
    else
    {
        // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
        //
        g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                   << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                   << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
        printf("位姿日志将保存至 %s\n", filename);
    }

    //  if (g_robot->isServoOn())
    //  {
    //      g_robot->servoPowerOff();
    //  }

    // // //6. 清错误 + 伺服上电
    // g_robot->clearError();
    // if (!g_robot->isServoOn())
    // {
    //     if (!g_robot->servoPowerOn())
    //     {
    //         printf("伺服上电失败\n");
    //         return -1;
    //     }
    //     // 等待伺服真正就绪
    //     for (int i = 0; i < 50; ++i)
    //     {
    //         if (g_robot->isServoOn())
    //             break;
    //         std::this_thread::sleep_for(std::chrono::milliseconds(20));
    //     }
    // }
    // printf("伺服已上电\n");

    // ===================== 核心顺滑参数 =====================
    const auto CONTROL_PERIOD = std::chrono::milliseconds(10); // 10ms 官方最优
    const double SMOOTH_KP = 0.2;                              // 柔顺系数（0.1~0.3 越大响应越快）
    const double MAX_STEP = 0.5;                               // 每10ms最大走 0.5mm（防止跳变）

    // 全局保存上一帧发送点（保证轨迹连续）
    Crp::SRobotPosition g_last_send_point;

    // 7. 设置速度与模式
    g_robot->setSpeedRatio(5);

    //**************在线模式切换部分*******************/
    //g_motion->finalize(Crp::EMotionType::Instruction);
    g_robot->stopProgram();//暂停程序
    g_robot->setWorkMode(Crp::RM_Manual); //切换teach模式
    g_robot->stopProgram();
    g_robot->setWorkMode(Crp::RM_Playing); //切换play模式
    
    const char* program_now = g_robot->getProgramPath();
    if (program_now != nullptr) {
        printf("当前程序路径: %s\n", program_now);
    } else {
        printf("未获取到程序路径\n");
    }

    //启动CSP模式
    if (!g_file->upload("data/guidancePos.pro", "guidancePos.pro"))
    {
        printf("Fail to upload guidancePos.pro\n");
        return -1;
    }
    if (!ensure_servo_ready_for_program_start("Forcecontrol_ChargeIn->guidancePos"))
    {
        return -1;
    }

    if (!g_robot->startProgram("guidancePos.pro", 0))
    {
        printf("Fail to start program\n");
        return -1;
    }

    
    const char* program_now2 = g_robot->getProgramPath();
    if (program_now2 != nullptr) {
        printf("当前程序路径: %s\n", program_now2);
    } else {
        printf("未获取到程序路径\n");
    }
   
    // int p_status = g_robot->getProgramStatus();
    // printf("current program status: %d\n",p_status);
    // if(p_status == Crp::EProgramStatus::PS_Stop){
    //     if (!g_robot->startProgram("guidancePos.pro", 0))
    //     {
    //         printf("Fail to start program\n");
    //         return -1;
    //     }
    // }else if(p_status == Crp::EProgramStatus::PS_Pause){
    //     if (!g_robot->resumeProgram("guidancePos.pro"))
    //     {
    //         printf("Fail to resume program\n");
    //         return -1;
    //     }
    // }else{
    //     printf("program has running\n");
    //     // if (!g_robot->stopProgram())
    //     // {
    //     //     printf("Fail to stop program\n");
    //     //     return -1;
    //     // }
    //     return 0;
    // }

    /****************已切换CSP模式***************** */
    while (!g_motion->isReady(Crp::EMotionType::Path))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    printf("开始 10ms 实时位姿控制，按 Ctrl+C 退出\n");
    // ===================== 启动异步读取线程 =====================
    std::thread read_thread(pose_read_thread);
    std::thread force_therad(force_read_thread, eth);
    std::thread joint_thread(joint_read_thread); // 新增：关节角线程
    // ===================== 核心下发循环：严格10ms周期 =====================

    Crp::SRobotPosition target;
    Crp::SRobotPosition target_output;
    Crp::SInstMoveL movepoint;
    const int TOOL_NO = 10;
    const int USER_NO = 0;
    Crp::SRobotPosition sendpoint_last;

    double world_pos[6] = {0};
    // 基准时间（保证绝对周期，而非相对睡眠）
    auto base_time = std::chrono::steady_clock::now();
    int index = 0;
    constexpr int kJointBufSize = 12;
    double cur_pos_joint[kJointBufSize] = {0};

    g_CFuncflagtest = 1;
    gStopping = false;

    while (!gStopping)
    {

        // 1) 严格对齐10ms周期（绝对时间基准）
        base_time += std::chrono::milliseconds(10);
        std::this_thread::sleep_until(base_time);

        // 2) 从缓冲区读取最新位姿（非阻塞）
        bool has_data = g_pose_buffer.read(world_pos);
        if (!has_data)
        {
            printf("警告：未读取到新位姿，使用上一帧数据\n");
            // 无新数据时可复用上次的world_pos，避免断流
        }
        if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
        {
            //printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
        }
        g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
        // 3) 构造目标位姿
        target.x = world_pos[0];
        target.y = world_pos[1];
        target.z = world_pos[2];
        target.Rx = world_pos[3];
        target.Ry = world_pos[4];
        target.Rz = world_pos[5];

        std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
        for (int i = 0; i < 6; i++)
        {
            jnt_pos_input[i] = cur_pos_joint[i];
        }
        // 读取力
        double Forcesensor_input[6];
        {
            // std::lock_guard<std::mutex> l(g_force_mtx);
            std::copy(g_force, g_force + 6, Forcesensor_input);
        }
        Vector6 target_pose;
        target_pose[0] = target.x;
        target_pose[1] = target.y;
        target_pose[2] = target.z;
        target_pose[3] = target.Rx;
        target_pose[4] = target.Ry;
        target_pose[5] = target.Rz;

        Vector6 init_joint;
        init_joint[0] = jnt_pos_input[0];
        init_joint[1] = jnt_pos_input[1];
        init_joint[2] = jnt_pos_input[2];
        init_joint[3] = jnt_pos_input[3];
        init_joint[4] = jnt_pos_input[4];
        init_joint[5] = jnt_pos_input[5];

        if (index > 2)
        {
            target = sendpoint_last;
        }

        if (index < 1)
        {
            //  //按vector进行输出
            Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
            jnt_vector_t joint_path_test;
            for (int i = 0; i < 2; i++)
            {
                Crp::SJointPosition point;
                point.body[0] = ik_result[0];
                point.body[1] = ik_result[1];
                point.body[2] = ik_result[2];
                point.body[3] = ik_result[3];
                point.body[4] = ik_result[4];
                point.body[5] = ik_result[5];
                joint_path_test.emplace_back(point);
                //  sendpoint_last = point;
            }
            g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
            Crp::EMovePathResult result = g_motion->movePath(5);
        }

        // 动态步长调整
        if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
        {
            // MAX_POS_DELTA = 0.2;
            // 基于行程线性增加步长
            MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
            if (MAX_POS_DELTA > 0.25f)
            {
                MAX_POS_DELTA = 0.25f;
            }
            if (MAX_POS_DELTA < 0.05f)
            {
                MAX_POS_DELTA = 0.05f;
            }
            MAX_ORI_DELTA = 0.005;
        }
        else if (g_CFuncflagtest == 4) // 零力模式
        {
            MAX_POS_DELTA = 0.15;
            MAX_ORI_DELTA = 0.02;
        }
        else
        {
            MAX_POS_DELTA = 0.1;
            MAX_ORI_DELTA = 0.005;
        }

        // printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //     Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
        // ==============================================================

        // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
        // 确保已经读取到了真实位姿
        if ((abs(target.x - 0.f) > 0.001))
        {
            // //力控输入
            double Force_input[6] = {0}; // 测试虚拟力数组
            Force_input[2] = 50.0;
            double pose_input[6] = {0}; // 创建力控输入位姿数组

            pose_input[0] = target.x / 1000.f;
            pose_input[1] = target.y / 1000.f;
            pose_input[2] = target.z / 1000.f;
            pose_input[3] = target.Rx;
            pose_input[4] = target.Ry;
            pose_input[5] = target.Rz;
            EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Forcesensor_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
            // 打印
            // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
            //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
            //printf("力控计算状态=%.2f \n ",ChargeFinished);

            // //复制力控运行结果  单位m转换为mm
            target_output.x = ForceControl_pos.X * 1000.f;
            target_output.y = ForceControl_pos.Y * 1000.f;
            target_output.z = ForceControl_pos.Z * 1000.f;
            target_output.Rx = ForceControl_pos.roll;
            target_output.Ry = ForceControl_pos.pitch;
            target_output.Rz = ForceControl_pos.yaw;

            // 起点：当前机器人实际位姿
            double start_x = target.x;
            double start_y = target.y;
            double start_z = target.z;
            double start_Rx = target.Rx;
            double start_Ry = target.Ry;
            double start_Rz = target.Rz;
            // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
            double raw_end_x = target_output.x;
            double raw_end_y = target_output.y;
            double raw_end_z = target_output.z;
            double raw_end_Rx = target_output.Rx;
            double raw_end_Ry = target_output.Ry;
            double raw_end_Rz = target_output.Rz;

            // 计算原始变化量
            double dx = raw_end_x - start_x;
            double dy = raw_end_y - start_y;
            double dz = raw_end_z - start_z;
            double drx = raw_end_Rx - start_Rx;
            double dry = raw_end_Ry - start_Ry;
            double drz = raw_end_Rz - start_Rz;

            bool is_limited = false;
            // 位置变化量：按比例限幅（保持方向）
            double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
            {
                double scale = MAX_POS_DELTA / pos_distance;
                dx = dx * scale;
                dy = dy * scale;
                dz = dz * scale;
                is_limited = true;
            }

            // 姿态变化量：按比例限幅（保持方向）
            double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
            if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
            {
                double scale = MAX_ORI_DELTA / ori_distance;
                drx = drx * scale;
                dry = dry * scale;
                drz = drz * scale;
                is_limited = true;
            }

            // 得到限幅后的最终终点
            double end_x = start_x + dx;
            double end_y = start_y + dy;
            double end_z = start_z + dz;
            double end_Rx = start_Rx + drx;
            double end_Ry = start_Ry + dry;
            double end_Rz = start_Rz + drz;

            // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
            pos_vector_t joint_path;
            jnt_vector_t jnt_path;
            for (int i = 0; i < 1; i++)
            {
                // 插值系数：5个点分4步，t从0到1，保证均匀间隔
                // double t = i / 9.0;
                double t = i / 1.0;
                Crp::SRobotPosition point;
                // 对6个位姿维度分别做线性插补
                point.x = start_x + (end_x - start_x);
                point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
                point.z = start_z + (end_z - start_z);
                point.Rx = start_Rx + (end_Rx - start_Rx);
                point.Ry = start_Ry + (end_Ry - start_Ry);
                point.Rz = start_Rz + (end_Rz - start_Rz);
                joint_path.emplace_back(point);

                Vector6 target_pose;
                target_pose[0] = point.x;
                target_pose[1] = point.y;
                target_pose[2] = point.z;
                target_pose[3] = point.Rx;
                target_pose[4] = point.Ry;
                target_pose[5] = point.Rz;
                Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
                Crp::SJointPosition jnt;
                jnt.body[0] = ik_result[0];
                jnt.body[1] = ik_result[1];
                jnt.body[2] = ik_result[2];
                jnt.body[3] = ik_result[3];
                jnt.body[4] = ik_result[4];
                jnt.body[5] = ik_result[5];
                jnt_path.emplace_back(jnt);

                sendpoint_last = point;
                // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
                //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
            }

            auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
            //printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);

            // 4) 单帧位姿下发
            //g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
            g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
            Crp::EMovePathResult result = g_motion->movePath(5);
            index++;
            if (result == Crp::EMovePathResult::Success)
            {
                printf("start moving success...\n");
            }

            // 打印
            // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
            // printf("%.2f", ChargeFinished);
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            // printf("[%02d:%02d:%02d.%03lld]\n",
            //        (int)localtime(&t)->tm_hour,
            //        (int)localtime(&t)->tm_min,
            //        (int)localtime(&t)->tm_sec,
            //        (long long int)ms.count());
        }
        if(ChargeFinished == 1 || ChargeFinished == 2) //插枪完成
        {
            gStopping = true;
        }
    }
    if(ChargeFinished == 1)
    {
        printf("插枪完成\n");
    }
    if(ChargeFinished == 2)
    {
        printf("插枪失败\n");
    }

    //**************在线模式切换部分****************** */
    g_motion->finalize(Crp::EMotionType::Path); //结束发送点位
    while (g_motion->getAvailPathBufferSize() < 2047)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    g_motion->finalize(Crp::EMotionType::Path); //结束发送点位
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    g_robot->stopProgram();//暂停程序
    g_robot->setWorkMode(Crp::RM_Manual); //切换teach模式
    g_robot->stopProgram();
    g_robot->setWorkMode(Crp::RM_Playing); //切换play模式
    
    //启动PP模式
    // int p_status = g_robot->getProgramStatus();
    // const char* program_now = g_robot->getProgramPath();
    // if (program_now != nullptr) {
    //     printf("当前程序路径: %s\n", program_now);
    // } else {
    //     printf("未获取到程序路径\n");
    // }

     if (!g_file->upload("data/guidanceInst.pro", "guidanceInst.pro"))
    {
        printf("Fail to upload guidanceInst.pro\n");
        return -1;
    }
    if (!ensure_servo_ready_for_program_start("Forcecontrol_ChargeIn->guidanceInst"))
    {
        return -1;
    }

    if (!g_robot->startProgram("guidanceInst.pro", 0))
    {
         printf("Fail to start program  PP mode\n");
         return -1;
    }

    // const char* program_now2 = g_robot->getProgramPath();
    // if (program_now2 != nullptr) {
    //     printf("当前程序路径: %s\n", program_now2);
    // } else {
    //     printf("未获取到程序路径\n");
    // }
    
     /****************已切换PP模式***************** */
    // ===================== 退出清理 =======主循环中计算得到了力控插补路径 ==============
    printf("正在退出...\n");
    // 等待读取线程退出
    if (read_thread.joinable())
    {
        read_thread.join();
    }
    printf("位姿读取已退出...\n");
    // 在 Forcecontrol_ChargeIn 函数最后，线程 join 之前添加
    // 设置退出标志
    gStopping = true;
    // 2. 等待一小段时间让线程自然退出
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // 3. 使用 sudo 强制终止 simple_test 进程
    printf("正在终止 simple_test 进程...\n");
    // 获取 simple_test 进程的 PID 并杀死
    system("ps aux | grep simple_test | grep -v grep | awk '{print $2}' | sudo xargs kill -9 2>/dev/null");
    // 或者直接使用 pkill with sudo
    system("sudo pkill -9 simple_test 2>/dev/null");

    // 4. 等待线程退出（带超时）
    auto wait_with_timeout = [](std::thread& t, const char* name, int timeout_ms) {
        if (t.joinable()) {
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start).count() < timeout_ms)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!t.joinable()) return true;
            }
            if (t.joinable()) {
                printf("警告: %s 线程超时，分离线程\n", name);
                t.detach();
                return false;
            }
        }
        return true;
    };
    printf("等待关节角读取线程退出...\n");
    wait_with_timeout(joint_thread, "joint_read", 1000);
    printf("等待力传感器线程退出...\n");
    wait_with_timeout(force_therad, "force_read", 2000);
    if (joint_thread.joinable())
    {
        joint_thread.join();
    }
    if (force_therad.joinable())
    {
        force_therad.join();
    }
    printf("线程已关闭\n");
    // 在等待 read_thread 之后
    if (g_pose_log.is_open())
    {
        g_pose_log.close();
        printf("位姿日志已保存至 %s\n", filename);
    }

    // if (g_robot->isServoOn())
    // {
    //     g_robot->servoPowerOff();
    // }
    // g_robot->disconnect();
    printf("已退出\n");

    return 0;
}

int ForceControl_Poseadjust()
{
    // 确保 data 目录存在
    const char *data_dir = "data";
    struct stat st;
    if (::stat(data_dir, &st) != 0)
    {                            // 使用 ::stat 调用全局函数
        ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
    }
    
    const char *ip ="192.168.1.12";
    const char *eth = "eth2";

    // 2. 注册退出信号
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_info = std::localtime(&now_c);
    char filename[256];
    std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);

    // 打开位姿日志文件（文件名带时间戳）
    g_pose_log.open(filename, std::ios::out | std::ios::trunc);
    if (!g_pose_log.is_open())
    {
        printf("警告：无法创建位姿日志文件 filename\n");
    }
    else
    {
        // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
        //
        g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                   << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                   << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
        printf("位姿日志将保存至 %s\n", filename);
    }

    // ===================== 核心顺滑参数 =====================
    const auto CONTROL_PERIOD = std::chrono::milliseconds(10); // 10ms 官方最优
    const double SMOOTH_KP = 0.2;                              // 柔顺系数（0.1~0.3 越大响应越快）
    const double MAX_STEP = 0.5;                               // 每10ms最大走 0.5mm（防止跳变）

    // 全局保存上一帧发送点（保证轨迹连续）
    Crp::SRobotPosition g_last_send_point;

    // 7. 设置速度与模式
    g_robot->setSpeedRatio(5);

    // //**************在线模式切换部分*******************/
    // g_motion->finalize(Crp::EMotionType::Instruction);
    // g_robot->stopProgram();//暂停程序
    // g_robot->setWorkMode(Crp::RM_Manual); //切换teach模式
    // g_robot->stopProgram();
    // g_robot->setWorkMode(Crp::RM_Playing); //切换play模式
    // //启动CSP模式
    // if (!g_file->upload("data/guidancePos.pro", "guidancePos.pro"))
    // {
    //     printf("Fail to upload guidancePos.pro\n");
    //     return -1;
    // }
    // if (!g_robot->startProgram("guidancePos.pro", 0))
    // {
    //     printf("Fail to start program\n");
    //     return -1;
    // }
    // /****************已切换CSP模式***************** */
    // while (!g_motion->isReady(Crp::EMotionType::Path))
    // {
    //     std::this_thread::sleep_for(std::chrono::microseconds(100));
    // }

    printf("开始 10ms 实时位姿控制，按 Ctrl+C 退出\n");
    // ===================== 启动异步读取线程 =====================
    std::thread read_thread(pose_read_thread);
    std::thread force_therad(force_read_thread, eth);
    std::thread joint_thread(joint_read_thread); // 新增：关节角线程
    // ===================== 核心下发循环：严格10ms周期 =====================

    Crp::SRobotPosition target;
    Crp::SRobotPosition target_output;
    Crp::SInstMoveL movepoint;
    const int TOOL_NO = 10;
    const int USER_NO = 0;
    Crp::SRobotPosition sendpoint_last;

    double world_pos[6] = {0};
    // 基准时间（保证绝对周期，而非相对睡眠）
    auto base_time = std::chrono::steady_clock::now();
    int index = 0;
    constexpr int kJointBufSize = 12;
    double cur_pos_joint[kJointBufSize] = {0};

    g_CFuncflagtest = 2;
    gStopping = false;

    while (!gStopping)
    {

        // 1) 严格对齐10ms周期（绝对时间基准）
        base_time += std::chrono::milliseconds(10);
        std::this_thread::sleep_until(base_time);

        // 2) 从缓冲区读取最新位姿（非阻塞）
        bool has_data = g_pose_buffer.read(world_pos);
        if (!has_data)
        {
            printf("警告：未读取到新位姿，使用上一帧数据\n");
            // 无新数据时可复用上次的world_pos，避免断流
        }
        if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
        {
            //printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
        }
        g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
        // 3) 构造目标位姿
        target.x = world_pos[0];
        target.y = world_pos[1];
        target.z = world_pos[2];
        target.Rx = world_pos[3];
        target.Ry = world_pos[4];
        target.Rz = world_pos[5];

        std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
        for (int i = 0; i < 6; i++)
        {
            jnt_pos_input[i] = cur_pos_joint[i];
        }
        // 读取力
        double Forcesensor_input[6];
        {
            // std::lock_guard<std::mutex> l(g_force_mtx);
            std::copy(g_force, g_force + 6, Forcesensor_input);
        }
        Vector6 target_pose;
        target_pose[0] = target.x;
        target_pose[1] = target.y;
        target_pose[2] = target.z;
        target_pose[3] = target.Rx;
        target_pose[4] = target.Ry;
        target_pose[5] = target.Rz;

        Vector6 init_joint;
        init_joint[0] = jnt_pos_input[0];
        init_joint[1] = jnt_pos_input[1];
        init_joint[2] = jnt_pos_input[2];
        init_joint[3] = jnt_pos_input[3];
        init_joint[4] = jnt_pos_input[4];
        init_joint[5] = jnt_pos_input[5];

        if (index > 2)
        {
            target = sendpoint_last;
        }

        if (index < 1)
        {
            //  //按vector进行输出
            Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
            jnt_vector_t joint_path_test;
            for (int i = 0; i < 2; i++)
            {
                Crp::SJointPosition point;
                point.body[0] = ik_result[0];
                point.body[1] = ik_result[1];
                point.body[2] = ik_result[2];
                point.body[3] = ik_result[3];
                point.body[4] = ik_result[4];
                point.body[5] = ik_result[5];
                joint_path_test.emplace_back(point);
                //  sendpoint_last = point;
            }
            // g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
            // Crp::EMovePathResult result = g_motion->movePath(5);
        }

        // 动态步长调整
        if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
        {
            // MAX_POS_DELTA = 0.2;
            // 基于行程线性增加步长
            MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
            if (MAX_POS_DELTA > 0.25f)
            {
                MAX_POS_DELTA = 0.25f;
            }
            if (MAX_POS_DELTA < 0.05f)
            {
                MAX_POS_DELTA = 0.05f;
            }
            MAX_ORI_DELTA = 0.005;
        }
        else if (g_CFuncflagtest == 4) // 零力模式
        {
            MAX_POS_DELTA = 0.15;
            MAX_ORI_DELTA = 0.02;
        }
        else
        {
            MAX_POS_DELTA = 0.1;
            MAX_ORI_DELTA = 0.005;
        }

        // printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //     Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
        // ==============================================================

        // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
        // 确保已经读取到了真实位姿
        if ((abs(target.x - 0.f) > 0.001))
        {
            // //力控输入
            double Force_input[6] = {0}; // 测试虚拟力数组
            Force_input[2] = 50.0;
            double pose_input[6] = {0}; // 创建力控输入位姿数组

            pose_input[0] = target.x / 1000.f;
            pose_input[1] = target.y / 1000.f;
            pose_input[2] = target.z / 1000.f;
            pose_input[3] = target.Rx;
            pose_input[4] = target.Ry;
            pose_input[5] = target.Rz;
            EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Force_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
            // 打印
            // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
            //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
            //printf("力控计算状态=%.2f \n ",ChargeFinished);

            // //复制力控运行结果  单位m转换为mm
            target_output.x = ForceControl_pos.X * 1000.f;
            target_output.y = ForceControl_pos.Y * 1000.f;
            target_output.z = ForceControl_pos.Z * 1000.f;
            target_output.Rx = ForceControl_pos.roll;
            target_output.Ry = ForceControl_pos.pitch;
            target_output.Rz = ForceControl_pos.yaw;

            // 起点：当前机器人实际位姿
            double start_x = target.x;
            double start_y = target.y;
            double start_z = target.z;
            double start_Rx = target.Rx;
            double start_Ry = target.Ry;
            double start_Rz = target.Rz;
            // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
            double raw_end_x = target_output.x;
            double raw_end_y = target_output.y;
            double raw_end_z = target_output.z;
            double raw_end_Rx = target_output.Rx;
            double raw_end_Ry = target_output.Ry;
            double raw_end_Rz = target_output.Rz;

            // 计算原始变化量
            double dx = raw_end_x - start_x;
            double dy = raw_end_y - start_y;
            double dz = raw_end_z - start_z;
            double drx = raw_end_Rx - start_Rx;
            double dry = raw_end_Ry - start_Ry;
            double drz = raw_end_Rz - start_Rz;

            bool is_limited = false;
            // 位置变化量：按比例限幅（保持方向）
            double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
            {
                double scale = MAX_POS_DELTA / pos_distance;
                dx = dx * scale;
                dy = dy * scale;
                dz = dz * scale;
                is_limited = true;
            }

            // 姿态变化量：按比例限幅（保持方向）
            double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
            if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
            {
                double scale = MAX_ORI_DELTA / ori_distance;
                drx = drx * scale;
                dry = dry * scale;
                drz = drz * scale;
                is_limited = true;
            }

            // 得到限幅后的最终终点
            double end_x = start_x + dx;
            double end_y = start_y + dy;
            double end_z = start_z + dz;
            double end_Rx = start_Rx + drx;
            double end_Ry = start_Ry + dry;
            double end_Rz = start_Rz + drz;

            // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
            pos_vector_t joint_path;
            jnt_vector_t jnt_path;
            for (int i = 0; i < 1; i++)
            {
                // 插值系数：5个点分4步，t从0到1，保证均匀间隔
                // double t = i / 9.0;
                double t = i / 1.0;
                Crp::SRobotPosition point;
                // 对6个位姿维度分别做线性插补
                point.x = start_x + (end_x - start_x);
                point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
                point.z = start_z + (end_z - start_z);
                point.Rx = start_Rx + (end_Rx - start_Rx);
                point.Ry = start_Ry + (end_Ry - start_Ry);
                point.Rz = start_Rz + (end_Rz - start_Rz);
                joint_path.emplace_back(point);

                Vector6 target_pose;
                target_pose[0] = point.x;
                target_pose[1] = point.y;
                target_pose[2] = point.z;
                target_pose[3] = point.Rx;
                target_pose[4] = point.Ry;
                target_pose[5] = point.Rz;
                Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
                Crp::SJointPosition jnt;
                jnt.body[0] = ik_result[0];
                jnt.body[1] = ik_result[1];
                jnt.body[2] = ik_result[2];
                jnt.body[3] = ik_result[3];
                jnt.body[4] = ik_result[4];
                jnt.body[5] = ik_result[5];
                jnt_path.emplace_back(jnt);

                sendpoint_last = point;
                // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
                //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
            }

            auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
            // printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);

            // 4) 单帧位姿下发
            //g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
            //g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
            // g_motion->sendPath(&target, 1 , TOOL_NO, USER_NO);  //按vector发送
            //Crp::EMovePathResult result = g_motion->movePath(5);
            index++;
            // if (result == Crp::EMovePathResult::Success)
            // {
            //     printf("start moving success...\n");
            // }

            // 打印
            // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
            // printf("%.2f", ChargeFinished);
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            // printf("[%02d:%02d:%02d.%03lld]\n",
            //        (int)localtime(&t)->tm_hour,
            //        (int)localtime(&t)->tm_min,
            //        (int)localtime(&t)->tm_sec,
            //        (long long int)ms.count());
        }
        if(ChargeFinished == 6) //插枪完成
        {
            gStopping = true;
        }
    }
    if(ChargeFinished == 6)
    {
        printf("调整完成\n");
    }
    
    //**************在线模式切换部分****************** */
    // g_motion->finalize(Crp::EMotionType::Path); //结束发送点位
    // while (g_motion->getAvailPathBufferSize() < 2047)
    // {
    //     std::this_thread::sleep_for(std::chrono::microseconds(100));
    // }
    // g_motion->finalize(Crp::EMotionType::Path); //结束发送点位
    // std::this_thread::sleep_for(std::chrono::microseconds(100));
    // g_robot->stopProgram();//暂停程序
    // g_robot->setWorkMode(Crp::RM_Manual); //切换teach模式
    // g_robot->stopProgram();
    // g_robot->setWorkMode(Crp::RM_Playing); //切换play模式
    
    //启动PP模式
    // int p_status = g_robot->getProgramStatus();
    // const char* program_now = g_robot->getProgramPath();
    // if (program_now != nullptr) {
    //     printf("当前程序路径: %s\n", program_now);
    // } else {
    //     printf("未获取到程序路径\n");
    // }

    //  if (!g_file->upload("data/guidanceInst.pro", "guidanceInst.pro"))
    // {
    //     printf("Fail to upload guidanceInst.pro\n");
    //     return -1;
    // }
    // if (!g_robot->startProgram("guidanceInst.pro", 0))
    // {
    //      printf("Fail to start program  PP mode\n");
    //      return -1;
    // }

    // const char* program_now2 = g_robot->getProgramPath();
    // if (program_now2 != nullptr) {
    //     printf("当前程序路径: %s\n", program_now2);
    // } else {
    //     printf("未获取到程序路径\n");
    // }
    
     /****************已切换PP模式***************** */
    // ===================== 退出清理 =======主循环中计算得到了力控插补路径 ==============
    printf("正在退出...\n");
    // 等待读取线程退出
    if (read_thread.joinable())
    {
        read_thread.join();
    }
    printf("位姿读取已退出...\n");
    // 在 Forcecontrol_ChargeIn 函数最后，线程 join 之前添加
    // 设置退出标志
    gStopping = true;
    // 2. 等待一小段时间让线程自然退出
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // 3. 使用 sudo 强制终止 simple_test 进程
    printf("正在终止 simple_test 进程...\n");
    // 获取 simple_test 进程的 PID 并杀死
    system("ps aux | grep simple_test | grep -v grep | awk '{print $2}' | sudo xargs kill -9 2>/dev/null");
    // 或者直接使用 pkill with sudo
    system("sudo pkill -9 simple_test 2>/dev/null");

    // 4. 等待线程退出（带超时）
    auto wait_with_timeout = [](std::thread& t, const char* name, int timeout_ms) {
        if (t.joinable()) {
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start).count() < timeout_ms)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!t.joinable()) return true;
            }
            if (t.joinable()) {
                printf("警告: %s 线程超时，分离线程\n", name);
                t.detach();
                return false;
            }
        }
        return true;
    };
    printf("等待关节角读取线程退出...\n");
    wait_with_timeout(joint_thread, "joint_read", 1000);
    printf("等待力传感器线程退出...\n");
    wait_with_timeout(force_therad, "force_read", 2000);
    if (joint_thread.joinable())
    {
        joint_thread.join();
    }
    if (force_therad.joinable())
    {
        force_therad.join();
    }
    printf("线程已关闭\n");
    // 在等待 read_thread 之后
    if (g_pose_log.is_open())
    {
        g_pose_log.close();
        printf("位姿日志已保存至 %s\n", filename);
    }

    printf("已退出\n");

    return 0;
}


int Forcecontrol_ChargeOut()
{
    // 确保 data 目录存在
    const char *data_dir = "data";
    struct stat st;
    if (::stat(data_dir, &st) != 0)
    {                            // 使用 ::stat 调用全局函数
        ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
    }
    
    const char *ip ="192.168.1.12";
    const char *eth = "eth2";

    // 2. 注册退出信号
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_info = std::localtime(&now_c);
    char filename[256];
    std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);

    // 打开位姿日志文件（文件名带时间戳）
    g_pose_log.open(filename, std::ios::out | std::ios::trunc);
    if (!g_pose_log.is_open())
    {
        printf("警告：无法创建位姿日志文件 filename\n");
    }
    else
    {
        // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
        //
        g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                   << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                   << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
        printf("位姿日志将保存至 %s\n", filename);
    }

    // ===================== 核心顺滑参数 =====================
    const auto CONTROL_PERIOD = std::chrono::milliseconds(10); // 10ms 官方最优
    const double SMOOTH_KP = 0.2;                              // 柔顺系数（0.1~0.3 越大响应越快）
    const double MAX_STEP = 0.5;                               // 每10ms最大走 0.5mm（防止跳变）

    // 全局保存上一帧发送点（保证轨迹连续）
    Crp::SRobotPosition g_last_send_point;

    // 7. 设置速度与模式
    g_robot->setSpeedRatio(5);

    //**************在线模式切换部分*******************/
    g_motion->finalize(Crp::EMotionType::Instruction);
    g_robot->stopProgram();//暂停程序
    g_robot->setWorkMode(Crp::RM_Manual); //切换teach模式
    g_robot->stopProgram();
    g_robot->setWorkMode(Crp::RM_Playing); //切换play模式
    //启动CSP模式
    if (!g_file->upload("data/guidancePos.pro", "guidancePos.pro"))
    {
        printf("Fail to upload guidancePos.pro\n");
        return -1;
    }
    if (!ensure_servo_ready_for_program_start("ForceControl_Poseadjust->guidancePos"))
    {
        return -1;
    }

    if (!g_robot->startProgram("guidancePos.pro", 0))
    {
        printf("Fail to start program\n");
        return -1;
    }
    /****************已切换CSP模式***************** */
    while (!g_motion->isReady(Crp::EMotionType::Path))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    printf("开始 10ms 实时位姿控制，按 Ctrl+C 退出\n");
    // ===================== 启动异步读取线程 =====================
    std::thread read_thread(pose_read_thread);
    std::thread force_therad(force_read_thread, eth);
    std::thread joint_thread(joint_read_thread); // 新增：关节角线程
    // ===================== 核心下发循环：严格10ms周期 =====================

    Crp::SRobotPosition target;
    Crp::SRobotPosition target_output;
    Crp::SInstMoveL movepoint;
    const int TOOL_NO = 10;
    const int USER_NO = 0;
    Crp::SRobotPosition sendpoint_last;

    double world_pos[6] = {0};
    // 基准时间（保证绝对周期，而非相对睡眠）
    auto base_time = std::chrono::steady_clock::now();
    int index = 0;
    constexpr int kJointBufSize = 12;
    double cur_pos_joint[kJointBufSize] = {0};

    g_CFuncflagtest = 3;
    gStopping = false;

    while (!gStopping)
    {

        // 1) 严格对齐10ms周期（绝对时间基准）
        base_time += std::chrono::milliseconds(8);
        std::this_thread::sleep_until(base_time);

        // 2) 从缓冲区读取最新位姿（非阻塞）
        bool has_data = g_pose_buffer.read(world_pos);
        if (!has_data)
        {
            printf("警告：未读取到新位姿，使用上一帧数据\n");
            // 无新数据时可复用上次的world_pos，避免断流
        }
        if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
        {
            //printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
        }
        g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
        // 3) 构造目标位姿
        target.x = world_pos[0];
        target.y = world_pos[1];
        target.z = world_pos[2];
        target.Rx = world_pos[3];
        target.Ry = world_pos[4];
        target.Rz = world_pos[5];

        std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
        for (int i = 0; i < 6; i++)
        {
            jnt_pos_input[i] = cur_pos_joint[i];
        }
        // 读取力
        double Forcesensor_input[6];
        {
            // std::lock_guard<std::mutex> l(g_force_mtx);
            std::copy(g_force, g_force + 6, Forcesensor_input);
        }
        Vector6 target_pose;
        target_pose[0] = target.x;
        target_pose[1] = target.y;
        target_pose[2] = target.z;
        target_pose[3] = target.Rx;
        target_pose[4] = target.Ry;
        target_pose[5] = target.Rz;

        Vector6 init_joint;
        init_joint[0] = jnt_pos_input[0];
        init_joint[1] = jnt_pos_input[1];
        init_joint[2] = jnt_pos_input[2];
        init_joint[3] = jnt_pos_input[3];
        init_joint[4] = jnt_pos_input[4];
        init_joint[5] = jnt_pos_input[5];

        if (index > 2)
        {
            target = sendpoint_last;
        }

        if (index < 1)
        {
            //  //按vector进行输出
            Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
            jnt_vector_t joint_path_test;
            for (int i = 0; i < 5; i++)
            {
                Crp::SJointPosition point;
                point.body[0] = ik_result[0];
                point.body[1] = ik_result[1];
                point.body[2] = ik_result[2];
                point.body[3] = ik_result[3];
                point.body[4] = ik_result[4];
                point.body[5] = ik_result[5];
                joint_path_test.emplace_back(point);
                //  sendpoint_last = point;
            }
            g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
            Crp::EMovePathResult result = g_motion->movePath(5);
        }

        // 动态步长调整
        if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
        {
            // MAX_POS_DELTA = 0.2;
            // 基于行程线性增加步长
            MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
            if (MAX_POS_DELTA > 0.25f)
            {
                MAX_POS_DELTA = 0.25f;
            }
            if (MAX_POS_DELTA < 0.05f)
            {
                MAX_POS_DELTA = 0.05f;
            }
            MAX_ORI_DELTA = 0.005;
        }
        else if (g_CFuncflagtest == 4) // 零力模式
        {
            MAX_POS_DELTA = 0.15;
            MAX_ORI_DELTA = 0.02;
        }
        else
        {
            MAX_POS_DELTA = 0.1;
            MAX_ORI_DELTA = 0.005;
        }

        // printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //     Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
        // ==============================================================

        // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
        // 确保已经读取到了真实位姿
        if ((abs(target.x - 0.f) > 0.001))
        {
            // //力控输入
            double Force_input[6] = {0}; // 测试虚拟力数组
            Force_input[2] = 50.0;
            double pose_input[6] = {0}; // 创建力控输入位姿数组

            pose_input[0] = target.x / 1000.f;
            pose_input[1] = target.y / 1000.f;
            pose_input[2] = target.z / 1000.f;
            pose_input[3] = target.Rx;
            pose_input[4] = target.Ry;
            pose_input[5] = target.Rz;
            EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Forcesensor_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
            // 打印
            // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
            //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
            //printf("力控计算状态=%.2f \n ",ChargeFinished);

            // //复制力控运行结果  单位m转换为mm
            target_output.x = ForceControl_pos.X * 1000.f;
            target_output.y = ForceControl_pos.Y * 1000.f;
            target_output.z = ForceControl_pos.Z * 1000.f;
            target_output.Rx = ForceControl_pos.roll;
            target_output.Ry = ForceControl_pos.pitch;
            target_output.Rz = ForceControl_pos.yaw;

            // 起点：当前机器人实际位姿
            double start_x = target.x;
            double start_y = target.y;
            double start_z = target.z;
            double start_Rx = target.Rx;
            double start_Ry = target.Ry;
            double start_Rz = target.Rz;
            // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
            double raw_end_x = target_output.x;
            double raw_end_y = target_output.y;
            double raw_end_z = target_output.z;
            double raw_end_Rx = target_output.Rx;
            double raw_end_Ry = target_output.Ry;
            double raw_end_Rz = target_output.Rz;

            // 计算原始变化量
            double dx = raw_end_x - start_x;
            double dy = raw_end_y - start_y;
            double dz = raw_end_z - start_z;
            double drx = raw_end_Rx - start_Rx;
            double dry = raw_end_Ry - start_Ry;
            double drz = raw_end_Rz - start_Rz;

            bool is_limited = false;
            // 位置变化量：按比例限幅（保持方向）
            double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
            {
                double scale = MAX_POS_DELTA / pos_distance;
                dx = dx * scale;
                dy = dy * scale;
                dz = dz * scale;
                is_limited = true;
            }

            // 姿态变化量：按比例限幅（保持方向）
            double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
            if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
            {
                double scale = MAX_ORI_DELTA / ori_distance;
                drx = drx * scale;
                dry = dry * scale;
                drz = drz * scale;
                is_limited = true;
            }

            // 得到限幅后的最终终点
            double end_x = start_x + dx;
            double end_y = start_y + dy;
            double end_z = start_z + dz;
            double end_Rx = start_Rx + drx;
            double end_Ry = start_Ry + dry;
            double end_Rz = start_Rz + drz;

            // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
            pos_vector_t joint_path;
            jnt_vector_t jnt_path;
            for (int i = 0; i < 1; i++)
            {
                // 插值系数：5个点分4步，t从0到1，保证均匀间隔
                // double t = i / 9.0;
                double t = i / 1.0;
                Crp::SRobotPosition point;
                // 对6个位姿维度分别做线性插补
                point.x = start_x + (end_x - start_x);
                point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
                point.z = start_z + (end_z - start_z);
                point.Rx = start_Rx + (end_Rx - start_Rx);
                point.Ry = start_Ry + (end_Ry - start_Ry);
                point.Rz = start_Rz + (end_Rz - start_Rz);
                joint_path.emplace_back(point);

                Vector6 target_pose;
                target_pose[0] = point.x;
                target_pose[1] = point.y;
                target_pose[2] = point.z;
                target_pose[3] = point.Rx;
                target_pose[4] = point.Ry;
                target_pose[5] = point.Rz;
                Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
                Crp::SJointPosition jnt;
                jnt.body[0] = ik_result[0];
                jnt.body[1] = ik_result[1];
                jnt.body[2] = ik_result[2];
                jnt.body[3] = ik_result[3];
                jnt.body[4] = ik_result[4];
                jnt.body[5] = ik_result[5];
                jnt_path.emplace_back(jnt);
                sendpoint_last = point;
                // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
                //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
            }

            auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
            // printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);
            // 4) 单帧位姿下发
            //g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
            g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
            Crp::EMovePathResult result = g_motion->movePath(5);
            index++;
            if (result == Crp::EMovePathResult::Success)
            {
                printf("start moving success...\n");
            }

            // 打印
            // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
            // printf("%.2f", ChargeFinished);
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            // printf("[%02d:%02d:%02d.%03lld]\n",
            //        (int)localtime(&t)->tm_hour,
            //        (int)localtime(&t)->tm_min,
            //        (int)localtime(&t)->tm_sec,
            //        (long long int)ms.count());
        }
        if(ChargeFinished == 4) //插枪完成
        {
            gStopping = true;
        }
    }
    if(ChargeFinished == 4)
    {
        printf("拔枪完成\n");
    }
    //状态刷新
    for(int j=0;j<15;j++)
    {
        g_CFuncflagtest = 0;
        double pose_input_res[6] = {0}; // 创建力控输入位姿数组
        double force_input_res[6] = {0};
        EulerAngles2 ForceControl_pos = ForceControlFunZYX2(force_input_res, pose_input_res, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
    }

    //**************在线模式切换部分****************** */
    g_motion->finalize(Crp::EMotionType::Path); //结束发送点位
    while (g_motion->getAvailPathBufferSize() < 2047)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    g_motion->finalize(Crp::EMotionType::Path); //结束发送点位
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    g_robot->stopProgram();//暂停程序
    g_robot->setWorkMode(Crp::RM_Manual); //切换teach模式
    g_robot->stopProgram();
    g_robot->setWorkMode(Crp::RM_Playing); //切换play模式
    
    //启动PP模式
    // int p_status = g_robot->getProgramStatus();
    // const char* program_now = g_robot->getProgramPath();
    // if (program_now != nullptr) {
    //     printf("当前程序路径: %s\n", program_now);
    // } else {
    //     printf("未获取到程序路径\n");
    // }

     if (!g_file->upload("data/guidanceInst.pro", "guidanceInst.pro"))
    {
        printf("Fail to upload guidanceInst.pro\n");
        return -1;
    }
    if (!ensure_servo_ready_for_program_start("Forcecontrol_ChargeOut->guidanceInst"))
    {
        return -1;
    }

    if (!g_robot->startProgram("guidanceInst.pro", 0))
    {
         printf("Fail to start program  PP mode\n");
         return -1;
    }

    // const char* program_now2 = g_robot->getProgramPath();
    // if (program_now2 != nullptr) {
    //     printf("当前程序路径: %s\n", program_now2);
    // } else {
    //     printf("未获取到程序路径\n");
    // }
    
     /****************已切换PP模式***************** */
    // ===================== 退出清理 =======主循环中计算得到了力控插补路径 ==============
    printf("正在退出...\n");
    // 等待读取线程退出
    if (read_thread.joinable())
    {
        read_thread.join();
    }
    printf("位姿读取已退出...\n");
    // 在 Forcecontrol_ChargeIn 函数最后，线程 join 之前添加
    // 设置退出标志
    gStopping = true;
    // 2. 等待一小段时间让线程自然退出
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // 3. 使用 sudo 强制终止 simple_test 进程
    printf("正在终止 simple_test 进程...\n");
    // 获取 simple_test 进程的 PID 并杀死
    system("ps aux | grep simple_test | grep -v grep | awk '{print $2}' | sudo xargs kill -9 2>/dev/null");
    // 或者直接使用 pkill with sudo
    system("sudo pkill -9 simple_test 2>/dev/null");

    // 4. 等待线程退出（带超时）
    auto wait_with_timeout = [](std::thread& t, const char* name, int timeout_ms) {
        if (t.joinable()) {
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start).count() < timeout_ms)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!t.joinable()) return true;
            }
            if (t.joinable()) {
                printf("警告: %s 线程超时，分离线程\n", name);
                t.detach();
                return false;
            }
        }
        return true;
    };
    printf("等待关节角读取线程退出...\n");
    wait_with_timeout(joint_thread, "joint_read", 1000);
    printf("等待力传感器线程退出...\n");
    wait_with_timeout(force_therad, "force_read", 2000);
    if (joint_thread.joinable())
    {
        joint_thread.join();
    }
    if (force_therad.joinable())
    {
        force_therad.join();
    }
    printf("线程已关闭\n");
    // 在等待 read_thread 之后
    if (g_pose_log.is_open())
    {
        g_pose_log.close();
        printf("位姿日志已保存至 %s\n", filename);
    }

    printf("已退出\n");

    return 0;
}


int ForceControl_OpenCover()
{
    // 确保 data 目录存在
    const char *data_dir = "data";
    struct stat st;
    if (::stat(data_dir, &st) != 0)
    {                            // 使用 ::stat 调用全局函数
        ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
    }
    
    const char *ip ="192.168.1.133";
    const char *eth = "eth1";

    // 2. 注册退出信号
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);

    if (!ensure_robot_session(ip, true))
    {
        return -1;
    }

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_info = std::localtime(&now_c);
    char filename[256];
    std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);

    // 打开位姿日志文件（文件名带时间戳）
    g_pose_log.open(filename, std::ios::out | std::ios::trunc);
    if (!g_pose_log.is_open())
    {
        printf("警告：无法创建位姿日志文件 filename\n");
    }
    else
    {
        // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
        //
        g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                   << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                   << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
        printf("位姿日志将保存至 %s\n", filename);
    }

    printf("跳过重复上下电，复用主流程机器人会话\n");

    // ===================== 核心顺滑参数 =====================
    const auto CONTROL_PERIOD = std::chrono::milliseconds(10); // 10ms 官方最优
    const double SMOOTH_KP = 0.2;                              // 柔顺系数（0.1~0.3 越大响应越快）
    const double MAX_STEP = 0.5;                               // 每10ms最大走 0.5mm（防止跳变）

    // 全局保存上一帧发送点（保证轨迹连续）
    Crp::SRobotPosition g_last_send_point;

    // 7. 设置速度与模式
    g_robot->setSpeedRatio(5);
    g_robot->setWorkMode(Crp::RM_Playing);
   
    if (!g_file->upload("data/guidancePos.pro", "guidancePos.pro"))
    {
        printf("Fail to upload guidancePos.pro\n");
        return -1;
    }

    if (!ensure_servo_ready_for_program_start("ForceControl_OpenCover->guidancePos"))
    {
        return -1;
    }

    if (!g_robot->startProgram("guidancePos.pro", 0))
    {
        printf("Fail to start program\n");
        return -1;
    }

    while (!g_motion->isReady(Crp::EMotionType::Path))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    printf("开始 10ms 实时位姿控制，按 Ctrl+C 退出\n");

    // ===================== 启动异步读取线程 =====================
    std::thread read_thread(pose_read_thread);
    std::thread force_therad(force_read_thread, eth);
    std::thread joint_thread(joint_read_thread); // 新增：关节角线程

    // ===================== 核心下发循环：严格10ms周期 =====================

    Crp::SRobotPosition target;
    Crp::SRobotPosition target_output;
    Crp::SInstMoveL movepoint;
    const int TOOL_NO = 10;
    const int USER_NO = 0;
    Crp::SRobotPosition sendpoint_last;

    double world_pos[6] = {0};
    // 基准时间（保证绝对周期，而非相对睡眠）
    auto base_time = std::chrono::steady_clock::now();
    int index = 0;
    constexpr int kJointBufSize = 12;
    double cur_pos_joint[kJointBufSize] = {0};

    g_CFuncflagtest = 5;
    gStopping = false;

    while (!gStopping)
    {

        // 1) 严格对齐10ms周期（绝对时间基准）
        base_time += std::chrono::milliseconds(10);
        std::this_thread::sleep_until(base_time);

        // 2) 从缓冲区读取最新位姿（非阻塞）
        bool has_data = g_pose_buffer.read(world_pos);
        if (!has_data)
        {
            printf("警告：未读取到新位姿，使用上一帧数据\n");
            // 无新数据时可复用上次的world_pos，避免断流
        }
        if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
        {
            // printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
        }
        g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
        // 3) 构造目标位姿
        target.x = world_pos[0];
        target.y = world_pos[1];
        target.z = world_pos[2];
        target.Rx = world_pos[3];
        target.Ry = world_pos[4];
        target.Rz = world_pos[5];

        std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
        for (int i = 0; i < 6; i++)
        {
            jnt_pos_input[i] = cur_pos_joint[i];
        }
        // 读取力
        double Forcesensor_input[6];
        {
            // std::lock_guard<std::mutex> l(g_force_mtx);
            std::copy(g_force, g_force + 6, Forcesensor_input);
        }
        Vector6 target_pose;
        target_pose[0] = target.x;
        target_pose[1] = target.y;
        target_pose[2] = target.z;
        target_pose[3] = target.Rx;
        target_pose[4] = target.Ry;
        target_pose[5] = target.Rz;

        Vector6 init_joint;
        init_joint[0] = jnt_pos_input[0];
        init_joint[1] = jnt_pos_input[1];
        init_joint[2] = jnt_pos_input[2];
        init_joint[3] = jnt_pos_input[3];
        init_joint[4] = jnt_pos_input[4];
        init_joint[5] = jnt_pos_input[5];

        if (index > 2)
        {
            target = sendpoint_last;
        }

        if (index < 1)
        {
            //  //按vector进行输出
            Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
            jnt_vector_t joint_path_test;
            for (int i = 0; i < 2; i++)
            {
                Crp::SJointPosition point;
                point.body[0] = ik_result[0];
                point.body[1] = ik_result[1];
                point.body[2] = ik_result[2];
                point.body[3] = ik_result[3];
                point.body[4] = ik_result[4];
                point.body[5] = ik_result[5];
                joint_path_test.emplace_back(point);
                //  sendpoint_last = point;
            }
            g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
            Crp::EMovePathResult result = g_motion->movePath(5);
        }

        // 动态步长调整
        if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
        {
            // MAX_POS_DELTA = 0.2;
            // 基于行程线性增加步长
            MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
            if (MAX_POS_DELTA > 0.25f)
            {
                MAX_POS_DELTA = 0.25f;
            }
            if (MAX_POS_DELTA < 0.05f)
            {
                MAX_POS_DELTA = 0.05f;
            }
            MAX_ORI_DELTA = 0.005;
        }
        else if (g_CFuncflagtest == 4) // 零力模式
        {
            MAX_POS_DELTA = 0.15;
            MAX_ORI_DELTA = 0.02;
        }
        else
        {
            MAX_POS_DELTA = 0.1;
            MAX_ORI_DELTA = 0.005;
        }

        // printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //     Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
        // ==============================================================

        // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
        // 确保已经读取到了真实位姿
        if ((abs(target.x - 0.f) > 0.001))
        {
            // //力控输入
            double Force_input[6] = {0}; // 测试虚拟力数组
            Force_input[2] = 50.0;
            double pose_input[6] = {0}; // 创建力控输入位姿数组

            pose_input[0] = target.x / 1000.f;
            pose_input[1] = target.y / 1000.f;
            pose_input[2] = target.z / 1000.f;
            pose_input[3] = target.Rx;
            pose_input[4] = target.Ry;
            pose_input[5] = target.Rz;
            EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Force_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
            // 打印
            // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
            //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
            // printf("力控计算状态=%.2f \n ",ChargeFinished);

            // //复制力控运行结果  单位m转换为mm
            target_output.x = ForceControl_pos.X * 1000.f;
            target_output.y = ForceControl_pos.Y * 1000.f;
            target_output.z = ForceControl_pos.Z * 1000.f;
            target_output.Rx = ForceControl_pos.roll;
            target_output.Ry = ForceControl_pos.pitch;
            target_output.Rz = ForceControl_pos.yaw;

            // 起点：当前机器人实际位姿
            double start_x = target.x;
            double start_y = target.y;
            double start_z = target.z;
            double start_Rx = target.Rx;
            double start_Ry = target.Ry;
            double start_Rz = target.Rz;
            // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
            double raw_end_x = target_output.x;
            double raw_end_y = target_output.y;
            double raw_end_z = target_output.z;
            double raw_end_Rx = target_output.Rx;
            double raw_end_Ry = target_output.Ry;
            double raw_end_Rz = target_output.Rz;

            // 计算原始变化量
            double dx = raw_end_x - start_x;
            double dy = raw_end_y - start_y;
            double dz = raw_end_z - start_z;
            double drx = raw_end_Rx - start_Rx;
            double dry = raw_end_Ry - start_Ry;
            double drz = raw_end_Rz - start_Rz;

            bool is_limited = false;
            // 位置变化量：按比例限幅（保持方向）
            double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
            {
                double scale = MAX_POS_DELTA / pos_distance;
                dx = dx * scale;
                dy = dy * scale;
                dz = dz * scale;
                is_limited = true;
            }

            // 姿态变化量：按比例限幅（保持方向）
            double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
            if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
            {
                double scale = MAX_ORI_DELTA / ori_distance;
                drx = drx * scale;
                dry = dry * scale;
                drz = drz * scale;
                is_limited = true;
            }

            // 得到限幅后的最终终点
            double end_x = start_x + dx;
            double end_y = start_y + dy;
            double end_z = start_z + dz;
            double end_Rx = start_Rx + drx;
            double end_Ry = start_Ry + dry;
            double end_Rz = start_Rz + drz;

            // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
            pos_vector_t joint_path;
            jnt_vector_t jnt_path;
            for (int i = 0; i < 1; i++)
            {
                // 插值系数：5个点分4步，t从0到1，保证均匀间隔
                // double t = i / 9.0;
                double t = i / 1.0;
                Crp::SRobotPosition point;
                // 对6个位姿维度分别做线性插补
                point.x = start_x + (end_x - start_x);
                point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
                point.z = start_z + (end_z - start_z);
                point.Rx = start_Rx + (end_Rx - start_Rx);
                point.Ry = start_Ry + (end_Ry - start_Ry);
                point.Rz = start_Rz + (end_Rz - start_Rz);
                joint_path.emplace_back(point);

                Vector6 target_pose;
                target_pose[0] = point.x;
                target_pose[1] = point.y;
                target_pose[2] = point.z;
                target_pose[3] = point.Rx;
                target_pose[4] = point.Ry;
                target_pose[5] = point.Rz;
                Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
                Crp::SJointPosition jnt;
                jnt.body[0] = ik_result[0];
                jnt.body[1] = ik_result[1];
                jnt.body[2] = ik_result[2];
                jnt.body[3] = ik_result[3];
                jnt.body[4] = ik_result[4];
                jnt.body[5] = ik_result[5];
                jnt_path.emplace_back(jnt);

                sendpoint_last = point;
                // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
                //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
            }

            auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
            // printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);

            // 4) 单帧位姿下发
            // g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
            g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
            // g_motion->sendPath(&target, 1 , TOOL_NO, USER_NO);  //按vector发送
            Crp::EMovePathResult result = g_motion->movePath(5);
            index++;
            if (result == Crp::EMovePathResult::Success)
            {
                printf("start moving success...\n");
            }

            // 打印
            // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
            // printf("%.2f", ChargeFinished);
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            // printf("[%02d:%02d:%02d.%03lld]\n",
            //        (int)localtime(&t)->tm_hour,
            //        (int)localtime(&t)->tm_min,
            //        (int)localtime(&t)->tm_sec,
            //        (long long int)ms.count());
        }
        if(ChargeFinished == 17) //插枪完成
        {
            gStopping = true;
        }

    }
    if(ChargeFinished == 17)
    {
        printf("开盖完成\n");
    }
    
    
    g_motion->finalize(Crp::EMotionType::Instruction);
    // ===================== 退出清理 =======主循环中计算得到了力控插补路径 ==============
    printf("正在退出...\n");
    // 等待读取线程退出
    if (read_thread.joinable())
    {
        read_thread.join();
    }
    if(joint_thread.joinable())
    {
        joint_thread.join();
    }
    if(force_therad.joinable())
    {
        force_therad.join();
    }
    // 在等待 read_thread 之后
    if (g_pose_log.is_open())
    {
        g_pose_log.close();
        printf("位姿日志已保存至 %s\n", filename);
    }
    printf("保留机器人会话，不执行伺服下电或断连\n");
    printf("已退出\n");

    return 0;
}

int return_the_gun()
{
    // 确保 data 目录存在
    const char *data_dir = "data";
    struct stat st;
    if (::stat(data_dir, &st) != 0)
    {                            // 使用 ::stat 调用全局函数
        ::mkdir(data_dir, 0777); // 使用 ::mkdir 创建目录
    }
    
    const char *ip ="192.168.1.133";
    const char *eth = "eth1";

    // 2. 注册退出信号
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, signal_handler);

    if (!ensure_robot_session(ip, true))
    {
        return -1;
    }

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_info = std::localtime(&now_c);
    char filename[256];
    std::strftime(filename, sizeof(filename), "data/robot_poses_%Y%m%d_%H%M%S.csv", tm_info);

    // 打开位姿日志文件（文件名带时间戳）
    g_pose_log.open(filename, std::ios::out | std::ios::trunc);
    if (!g_pose_log.is_open())
    {
        printf("警告：无法创建位姿日志文件 filename\n");
    }
    else
    {
        // g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg" << std::endl;   //原代码，只读取并记录位姿
        //
        g_pose_log << "timestamp_ms,x_mm,y_mm,z_mm,Rx_deg,Ry_deg,Rz_deg,"
                   << "fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm,"
                   << "CFuncflag,ChargeFinished,MoveDistance_m" << std::endl;
        printf("位姿日志将保存至 %s\n", filename);
    }

    printf("跳过重复上下电，复用主流程机器人会话\n");

    // ===================== 核心顺滑参数 =====================
    const auto CONTROL_PERIOD = std::chrono::milliseconds(10); // 10ms 官方最优
    const double SMOOTH_KP = 0.2;                              // 柔顺系数（0.1~0.3 越大响应越快）
    const double MAX_STEP = 0.5;                               // 每10ms最大走 0.5mm（防止跳变）

    // 全局保存上一帧发送点（保证轨迹连续）
    Crp::SRobotPosition g_last_send_point;

    // 7. 设置速度与模式
    g_robot->setSpeedRatio(5);
    g_robot->setWorkMode(Crp::RM_Playing);
    // g_robot->FKine()
    if (!g_file->upload("data/guidancePos.pro", "guidancePos.pro"))
    {
        printf("Fail to upload guidancePos.pro\n");
        return -1;
    }

    if (!ensure_servo_ready_for_program_start("return_the_gun->guidancePos"))
    {
        return -1;
    }

    if (!g_robot->startProgram("guidancePos.pro", 0))
    {
        printf("Fail to start program\n");
        return -1;
    }

    while (!g_motion->isReady(Crp::EMotionType::Path))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    printf("开始 10ms 实时位姿控制，按 Ctrl+C 退出\n");

    // ===================== 启动异步读取线程 =====================
    std::thread read_thread(pose_read_thread);
    std::thread force_therad(force_read_thread, eth);
    std::thread joint_thread(joint_read_thread); // 新增：关节角线程

    // ===================== 核心下发循环：严格10ms周期 =====================

    Crp::SRobotPosition target;
    Crp::SRobotPosition target_output;
    Crp::SInstMoveL movepoint;
    const int TOOL_NO = 10;
    const int USER_NO = 0;
    Crp::SRobotPosition sendpoint_last;

    double world_pos[6] = {0};
    // 基准时间（保证绝对周期，而非相对睡眠）
    auto base_time = std::chrono::steady_clock::now();
    int index = 0;
    constexpr int kJointBufSize = 12;
    double cur_pos_joint[kJointBufSize] = {0};

    g_CFuncflagtest = 8;
    gStopping = false;

    while (!gStopping)
    {

        // 1) 严格对齐10ms周期（绝对时间基准）
        base_time += std::chrono::milliseconds(10);
        std::this_thread::sleep_until(base_time);

        // 2) 从缓冲区读取最新位姿（非阻塞）
        bool has_data = g_pose_buffer.read(world_pos);
        if (!has_data)
        {
            printf("警告：未读取到新位姿，使用上一帧数据\n");
            // 无新数据时可复用上次的world_pos，避免断流
        }
        if (g_CFuncflagtest == 1 || g_CFuncflagtest == 3)
        {
            // printf("插枪/拔枪行程：%.4f\n", g_MoveDistance);
        }
        g_joint_buffer.read(cur_pos_joint); // 直接读，不判断返回值
        // 3) 构造目标位姿
        target.x = world_pos[0];
        target.y = world_pos[1];
        target.z = world_pos[2];
        target.Rx = world_pos[3];
        target.Ry = world_pos[4];
        target.Rz = world_pos[5];

        std::vector<double> jnt_pos_input(6); // 初始化vector大小为6
        for (int i = 0; i < 6; i++)
        {
            jnt_pos_input[i] = cur_pos_joint[i];
        }
        // 读取力
        double Forcesensor_input[6];
        {
            // std::lock_guard<std::mutex> l(g_force_mtx);
            std::copy(g_force, g_force + 6, Forcesensor_input);
        }
        Vector6 target_pose;
        target_pose[0] = target.x;
        target_pose[1] = target.y;
        target_pose[2] = target.z;
        target_pose[3] = target.Rx;
        target_pose[4] = target.Ry;
        target_pose[5] = target.Rz;

        Vector6 init_joint;
        init_joint[0] = jnt_pos_input[0];
        init_joint[1] = jnt_pos_input[1];
        init_joint[2] = jnt_pos_input[2];
        init_joint[3] = jnt_pos_input[3];
        init_joint[4] = jnt_pos_input[4];
        init_joint[5] = jnt_pos_input[5];

        if (index > 2)
        {
            target = sendpoint_last;
        }

        if (index < 1)
        {
            //  //按vector进行输出
            Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
            jnt_vector_t joint_path_test;
            for (int i = 0; i < 2; i++)
            {
                Crp::SJointPosition point;
                point.body[0] = ik_result[0];
                point.body[1] = ik_result[1];
                point.body[2] = ik_result[2];
                point.body[3] = ik_result[3];
                point.body[4] = ik_result[4];
                point.body[5] = ik_result[5];
                joint_path_test.emplace_back(point);
                //  sendpoint_last = point;
            }
            g_motion->sendPath(joint_path_test.data(), joint_path_test.size()); // 按vector
            Crp::EMovePathResult result = g_motion->movePath(5);
        }

        // 动态步长调整
        if (g_CFuncflagtest == 3 && g_MoveDistance > 0.04) // 拔枪后半段
        {
            // MAX_POS_DELTA = 0.2;
            // 基于行程线性增加步长
            MAX_POS_DELTA = 0.05f + 3.6364f * (g_MoveDistance - 0.04f);
            if (MAX_POS_DELTA > 0.25f)
            {
                MAX_POS_DELTA = 0.25f;
            }
            if (MAX_POS_DELTA < 0.05f)
            {
                MAX_POS_DELTA = 0.05f;
            }
            MAX_ORI_DELTA = 0.005;
        }
        else if (g_CFuncflagtest == 4) // 零力模式
        {
            MAX_POS_DELTA = 0.15;
            MAX_ORI_DELTA = 0.02;
        }
        else
        {
            MAX_POS_DELTA = 0.1;
            MAX_ORI_DELTA = 0.005;
        }

        // printf("六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //     Forcesensor_input[0], Forcesensor_input[1], Forcesensor_input[2], Forcesensor_input[3], Forcesensor_input[4], Forcesensor_input[5]);
        // ==============================================================

        // printf("位姿反馈: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
        //        target.x, target.y, target.z, target.Rx, target.Ry, target.Rz);
        // 确保已经读取到了真实位姿
        if ((abs(target.x - 0.f) > 0.001))
        {
            // //力控输入
            double Force_input[6] = {0}; // 测试虚拟力数组
            Force_input[2] = 50.0;
            double pose_input[6] = {0}; // 创建力控输入位姿数组

            pose_input[0] = target.x / 1000.f;
            pose_input[1] = target.y / 1000.f;
            pose_input[2] = target.z / 1000.f;
            pose_input[3] = target.Rx;
            pose_input[4] = target.Ry;
            pose_input[5] = target.Rz;
            EulerAngles2 ForceControl_pos = ForceControlFunZYX2(Force_input, pose_input, g_CFuncflagtest); // 力控函数 EulerAngles2为笛卡尔位姿结构体
            // 打印
            // printf("力控计算: X=%.2f Y=%.2f Z=%.2f roll=%.2f pitch=%.2f yaw=%.2f 状态=%.2f \n ",
            //        ForceControl_pos.X * 1000.f, ForceControl_pos.Y * 1000.f, ForceControl_pos.Z * 1000.f, ForceControl_pos.roll, ForceControl_pos.pitch, ForceControl_pos.yaw, ChargeFinished);
            // printf("力控计算状态=%.2f \n ",ChargeFinished);

            // //复制力控运行结果  单位m转换为mm
            target_output.x = ForceControl_pos.X * 1000.f;
            target_output.y = ForceControl_pos.Y * 1000.f;
            target_output.z = ForceControl_pos.Z * 1000.f;
            target_output.Rx = ForceControl_pos.roll;
            target_output.Ry = ForceControl_pos.pitch;
            target_output.Rz = ForceControl_pos.yaw;

            // 起点：当前机器人实际位姿
            double start_x = target.x;
            double start_y = target.y;
            double start_z = target.z;
            double start_Rx = target.Rx;
            double start_Ry = target.Ry;
            double start_Rz = target.Rz;
            // 终点：力控计算得到的目标位姿（单位转换：米转毫米）
            double raw_end_x = target_output.x;
            double raw_end_y = target_output.y;
            double raw_end_z = target_output.z;
            double raw_end_Rx = target_output.Rx;
            double raw_end_Ry = target_output.Ry;
            double raw_end_Rz = target_output.Rz;

            // 计算原始变化量
            double dx = raw_end_x - start_x;
            double dy = raw_end_y - start_y;
            double dz = raw_end_z - start_z;
            double drx = raw_end_Rx - start_Rx;
            double dry = raw_end_Ry - start_Ry;
            double drz = raw_end_Rz - start_Rz;

            bool is_limited = false;
            // 位置变化量：按比例限幅（保持方向）
            double pos_distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (pos_distance > MAX_POS_DELTA && pos_distance > 1e-6) // 避免除零
            {
                double scale = MAX_POS_DELTA / pos_distance;
                dx = dx * scale;
                dy = dy * scale;
                dz = dz * scale;
                is_limited = true;
            }

            // 姿态变化量：按比例限幅（保持方向）
            double ori_distance = sqrt(drx * drx + dry * dry + drz * drz);
            if (ori_distance > MAX_ORI_DELTA && ori_distance > 1e-6)
            {
                double scale = MAX_ORI_DELTA / ori_distance;
                drx = drx * scale;
                dry = dry * scale;
                drz = drz * scale;
                is_limited = true;
            }

            // 得到限幅后的最终终点
            double end_x = start_x + dx;
            double end_y = start_y + dy;
            double end_z = start_z + dz;
            double end_Rx = start_Rx + drx;
            double end_Ry = start_Ry + dry;
            double end_Rz = start_Rz + drz;

            // 生成5个均匀间隔的插补点，机器人将每10ms执行一个点，50ms完成过渡
            pos_vector_t joint_path;
            jnt_vector_t jnt_path;
            for (int i = 0; i < 1; i++)
            {
                // 插值系数：5个点分4步，t从0到1，保证均匀间隔
                // double t = i / 9.0;
                double t = i / 1.0;
                Crp::SRobotPosition point;
                // 对6个位姿维度分别做线性插补
                point.x = start_x + (end_x - start_x);
                point.y = start_y + (end_y - start_y); // 此时指令执行在 GuidancePosInit，机器人完成初始化，等待外部命令的数据和启动。
                point.z = start_z + (end_z - start_z);
                point.Rx = start_Rx + (end_Rx - start_Rx);
                point.Ry = start_Ry + (end_Ry - start_Ry);
                point.Rz = start_Rz + (end_Rz - start_Rz);
                joint_path.emplace_back(point);

                Vector6 target_pose;
                target_pose[0] = point.x;
                target_pose[1] = point.y;
                target_pose[2] = point.z;
                target_pose[3] = point.Rx;
                target_pose[4] = point.Ry;
                target_pose[5] = point.Rz;
                Vector6 ik_result = inverse_kinematics(target_pose, &init_joint, "euler");
                Crp::SJointPosition jnt;
                jnt.body[0] = ik_result[0];
                jnt.body[1] = ik_result[1];
                jnt.body[2] = ik_result[2];
                jnt.body[3] = ik_result[3];
                jnt.body[4] = ik_result[4];
                jnt.body[5] = ik_result[5];
                jnt_path.emplace_back(jnt);

                sendpoint_last = point;
                // printf("实时下发: J1=%.3f J2=%.3f J3=%.3f J4=%.3f J5=%.3f J6=%.3f\n",
                //     ik_result[0], ik_result[1], ik_result[2], ik_result[3], ik_result[4], ik_result[5]);
            }

            auto AvailPathBufferSize = g_motion->getAvailPathBufferSize();
            // printf("可用空间:=%.2f \n", (float)AvailPathBufferSize);

            // 4) 单帧位姿下发
            // g_motion->sendPath(joint_path.data(), joint_path.size(), TOOL_NO, USER_NO);  //按位姿控制
            g_motion->sendPath(jnt_path.data(), jnt_path.size()); // 按关节角控制
            // g_motion->sendPath(&target, 1 , TOOL_NO, USER_NO);  //按vector发送
            Crp::EMovePathResult result = g_motion->movePath(5);
            index++;
            if (result == Crp::EMovePathResult::Success)
            {
                printf("start moving success...\n");
            }

            // 打印
            // printf("实时下发: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
            //     target_output.x, target_output.y, target_output.z, target_output.Rx, target_output.Ry, target_output.Rz);
            // printf("%.2f", ChargeFinished);
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            // printf("[%02d:%02d:%02d.%03lld]\n",
            //        (int)localtime(&t)->tm_hour,
            //        (int)localtime(&t)->tm_min,
            //        (int)localtime(&t)->tm_sec,
            //        (long long int)ms.count());
        }
        if(ChargeFinished == 11 || ChargeFinished == 12) //插枪完成
        {
            gStopping = true;
        }

    }
    if(ChargeFinished == 1)
    {
        printf("归枪完成\n");
    }
    if(ChargeFinished == 2)
    {
        printf("归枪失败\n");
    }
    
    g_motion->finalize(Crp::EMotionType::Instruction);
    // ===================== 退出清理 =======主循环中计算得到了力控插补路径 ==============
    printf("正在退出...\n");
    // 等待读取线程退出
    if (read_thread.joinable())
    {
        read_thread.join();
    }
    if(force_therad.joinable())
    {
        force_therad.join();
    }
    if(joint_thread.joinable())
    {
        joint_thread.join();
    }
    // 在等待 read_thread 之后
    if (g_pose_log.is_open())
    {
        g_pose_log.close();
        printf("位姿日志已保存至 %s\n", filename);
    }
    printf("保留机器人会话，不执行伺服下电或断连\n");
    printf("已退出\n");

    return 0;
}

#ifdef __cplusplus
}
#endif
