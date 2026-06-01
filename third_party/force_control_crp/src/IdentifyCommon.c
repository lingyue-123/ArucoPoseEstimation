

#include"IdentifyCommon.h"

// 创建3D向量
Vector3D vector3d_create(double x, double y, double z) {
	Vector3D v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

// 向量加法
Vector3D vector3d_add(Vector3D a, Vector3D b) {
	return vector3d_create(a.x + b.x, a.y + b.y, a.z + b.z);
}

// 向量减法
Vector3D vector3d_sub(Vector3D a, Vector3D b) {
	return vector3d_create(a.x - b.x, a.y - b.y, a.z - b.z);
}

// 向量数乘
Vector3D vector3d_scale(Vector3D v, double s) {
	return vector3d_create(v.x * s, v.y * s, v.z * s);
}

// 向量点积
double vector3d_dot(Vector3D a, Vector3D b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 向量叉积
Vector3D vector3d_cross(Vector3D a, Vector3D b) {
	Vector3D result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

// 向量范数
double vector3d_norm(Vector3D v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// 创建单位矩阵
Matrix3x3 matrix3x3_identity() {
	Matrix3x3 m;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			m.data[i][j] = (i == j) ? 1.0 : 0.0;
		}
	}
	return m;
}

// 矩阵乘法
Matrix3x3 matrix3x3_multiply(Matrix3x3 a, Matrix3x3 b) {
	Matrix3x3 result;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			result.data[i][j] = 0.0;
			for (int k = 0; k < 3; k++) {
				result.data[i][j] += a.data[i][k] * b.data[k][j];
			}
		}
	}
	return result;
}

// 矩阵转置
Matrix3x3 matrix3x3_transpose(Matrix3x3 m) {
	Matrix3x3 result;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			result.data[i][j] = m.data[j][i];
		}
	}
	return result;
}

// 矩阵向量乘法
Vector3D matrix3x3_vector_multiply(Matrix3x3 m, Vector3D v) {
	Vector3D result;
	result.x = m.data[0][0] * v.x + m.data[0][1] * v.y + m.data[0][2] * v.z;
	result.y = m.data[1][0] * v.x + m.data[1][1] * v.y + m.data[1][2] * v.z;
	result.z = m.data[2][0] * v.x + m.data[2][1] * v.y + m.data[2][2] * v.z;
	return result;
}
// 根据欧拉角创建旋转矩阵 (ZYX顺序)
Matrix3x3 euler_to_rotation_matrix(EulerAngles euler) {

	double cy = cos(euler.yaw);
	double sy = sin(euler.yaw);
	double cp = cos(euler.pitch);
	double sp = sin(euler.pitch);
	double cr = cos(euler.roll);
	double sr = sin(euler.roll);

	Matrix3x3 R;

	// ZYX顺序: R = Rz * Ry * Rx
	R.data[0][0] = cy * cp;
	R.data[0][1] = cy * sp * sr - sy * cr;
	R.data[0][2] = cy * sp * cr + sy * sr;

	R.data[1][0] = sy * cp;
	R.data[1][1] = sy * sp * sr + cy * cr;
	R.data[1][2] = sy * sp * cr - cy * sr;

	R.data[2][0] = -sp;
	R.data[2][1] = cp * sr;
	R.data[2][2] = cp * cr;

	return R;
}

void printfVec(Vector3D Ver)
{
	printf("%f , %f ,%f \n",Ver.x,Ver.y,Ver.z);

}
void printfTht(EulerAngles Agl){
	printf("%f , %f ,%f \n",Agl.roll,Agl.pitch,Agl.yaw);
}
void PrintfPos(PosData Tar) 
{
	printf("============位姿打印=============\n");
	printf("位置 X  Y  Z \n");
	printfVec(Tar.XYZ);
	printf("位置 roll  pitch  yaw \n");
	printfTht(Tar.theta);
}
