
/******************************************************************************
* 名        称: SixSnsrComp.h
* 发布日期: 2026.1.29
* 创  建  人: 集成一科/尹君
* 版        本: V001
* 描        述:
* 备        注:
******************************************************************************/
#ifndef SIX_DOF_FORCE_CALIBRATION_COMMON_FUNCTIONS_H
#define SIX_DOF_FORCE_CALIBRATION_COMMON_FUNCTIONS_H
#include <stdbool.h>
#include <math.h>
#include<stdlib.h>
#include<stdio.h>
/*****************************************************************************/
/*TYPEDEFS AND STRUCTURES                                                    */
/*****************************************************************************/
// 定义PI常量
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



/*!< 3x3矩阵*/
typedef struct {
	double data[3][3];
} Matrix3x3;

/*!< 3x3矩阵*/
typedef struct {
	double data[4][4];
} Matrix4x4;

/*!< 3维向量 */
typedef struct {
	double x, y, z;
} Vector3D;
/*!<欧拉角 (ZYX顺序)*/
typedef struct {
	double yaw;     /*!<偏航角 (绕Z轴)*/
	double pitch;   /*!<俯仰角 (绕Y轴)*/
	double roll;    /*!<滚转角 (绕X轴)*/
} EulerAngles;

/*!< 位姿坐标 */
typedef struct {
	Vector3D XYZ;
	EulerAngles theta;
} PosData;


typedef struct {
	Vector3D F_meas;       // 测量力 (传感器坐标系)
	Vector3D T_meas;       // 测量力矩 (传感器坐标系)
	Matrix3x3 R_SB;    // 传感器 -> 基座旋转矩阵
	Vector3D XYZ;      // XYZ坐标
	double Pos[6]; 
} SampleData;


typedef struct {
	double m;          // 质量
	Vector3D r_T;          // 质心位置 (工具坐标系)
	Vector3D F_offset;     // 力偏置 (传感器坐标系)
	Vector3D T_offset;     // 力矩偏置 (传感器坐标系)
	Vector3D theta_ST;     // 传感器 -> 工具 欧拉角 (roll,pitch,yaw) ZYX顺序
	Matrix3x3 R_ST;    // 传感器 -> 工具 旋转矩阵 (由theta_ST计算得出)
} CalibParams;

typedef struct {
	double lambda;     // LM阻尼因子
	int max_iter;      // 最大迭代次数
	double ftol;       // 函数值变化容忍度
	double xtol;       // 参数变化容忍度
	double gtol;       // 梯度范数容忍度
	int verbose;       // 是否打印迭代信息
} LMOptions;

//// 数据集合
typedef struct {
	SampleData* data;      // 数据数组
	int count;                  // 数据点数量
	int capacity;               // 数组容量
} DataSet;




extern Vector3D vector3d_create(double x, double y, double z);
extern Vector3D vector3d_add(Vector3D a, Vector3D b);
// 向量减法
extern Vector3D vector3d_sub(Vector3D a, Vector3D b);
extern Vector3D vector3d_scale(Vector3D v, double s);
extern double vector3d_dot(Vector3D a, Vector3D b);

extern Vector3D vector3d_cross(Vector3D a, Vector3D b);
extern double vector3d_norm(Vector3D v);

extern Matrix3x3 matrix3x3_identity();
extern Matrix3x3 matrix3x3_multiply(Matrix3x3 a, Matrix3x3 b);

extern Matrix3x3 matrix3x3_transpose(Matrix3x3 m);

extern Vector3D matrix3x3_vector_multiply(Matrix3x3 m, Vector3D v);

extern Matrix3x3 euler_to_rotation_matrix(EulerAngles euler);


extern void PrintfPos(PosData Tar) ;

#endif // SIX_DOF_FORCE_CALIBRATION_COMMON_FUNCTIONS_H
