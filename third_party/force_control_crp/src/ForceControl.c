

#include "ForceControl.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

float g_test1 = 1.f, g_test2 = 2.f, g_test3 = 3.f, g_test4 = 4.f;

double velLast[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
double lenLast[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
double jntPos[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
double T_last[4][4] = { {0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f} };
double T_init[4][4] = { {0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f} };

double T_charged[4][4] = { {0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f} };

double T_opencover[4][4] = { {0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f} };
double T_opencover2[4][4] = { {0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f} };
int opevcover = 0;

double jntPos_now[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };

double T_error_debug[4][4] = { {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0} };


double Pos_input[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };

double Force[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
double Force_Dealed[6] = { 10.f,10.f,10.f,2.f,2.f,2.f };
double K = 100.f;
double Kx = 200.f; //200
double Ky = 200.f;
double Kz = 200.f;

double dt = 0.01f;
double T_now[4][4] = { {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0} };
double T_next[4][4] = { {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0} };
double T_next2[6] = { 0,0,0,0,0,0 };
int pos_init = 0;
int SinWavecount = 0;
int pos_adjust_init = 0;
double FinalVal = 0;
double ChargeFinished = 8.f;  // 0-插枪中  1-插枪成功 2-插枪失败  3-拔枪中  4-拔枪成功  5-拔枪失败 6-插枪保持  8-默认值
double g_MoveDistance = 0.f;

int pos_adjust_count = 0;
int vel_clearflag = 0;


double T_ForForceCmp[4][4] = { {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0} };
double T_ForForceCmpBfr[4][4] = { {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0} };
double JointForceCmp[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
uint32_t g_CFuncflag = 0;

double ChargeMaxMoveDistance = 0.1f;
double Forceinit_8 [6]={0,0,0,0,0,0};
double Forceinit_9 [6]={0,0,0,0,0,0};
EulerAngles2 result;


/* tansig激活函数：2/(1+exp(-2*x)) - 1 */
static double tansig(double x) {
	return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

/* 归一化：将原始x映射到[ymin, ymax] */
static double normalize(double x, double xmin, double xmax, double ymin, double ymax) {
	if (xmax - xmin < 1e-12) return ymin;
	return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin;
}

/* 反归一化：将归一化值y还原到原始范围 */
static double denormalize(double y, double xmin, double xmax, double ymin, double ymax) {
	if (ymax - ymin < 1e-12) return xmin;
	return (y - ymin) * (xmax - xmin) / (ymax - ymin) + xmin;
}

/**
* 神经网络前向计算
* @param input 输入数组[roll, pitch, yaw, x, y, z] (弧度, 米)
* @param output 输出数组[Fx, Fy, Fz, Tx, Ty, Tz] (N, Nm)
*/
void nn_forward(const double input[NN_INPUT_DIM], double output[NN_OUTPUT_DIM]) {
	// 1. 输入归一化
	double norm_in[NN_INPUT_DIM];
	for (int i = 0; i < NN_INPUT_DIM; i++) {
		norm_in[i] = normalize(input[i], XMIN_IN[i], XMAX_IN[i], YMIN_IN, YMAX_IN);
	}

	// 2. 第一隐藏层 (tansig)
	double hidden1[NN_HIDDEN1_DIM];
	for (int i = 0; i < NN_HIDDEN1_DIM; i++) {
		double sum = b1[i];
		for (int j = 0; j < NN_INPUT_DIM; j++) {
			sum += W1[i][j] * norm_in[j];
		}
		hidden1[i] = tansig(sum);
	}

	// 3. 第二隐藏层 (tansig)
	double hidden2[NN_HIDDEN2_DIM];
	for (int i = 0; i < NN_HIDDEN2_DIM; i++) {
		double sum = b2[i];
		for (int j = 0; j < NN_HIDDEN1_DIM; j++) {
			sum += W2[i][j] * hidden1[j];
		}
		hidden2[i] = tansig(sum);
	}

	// 4. 输出层 (purelin) 得到归一化输出
	double norm_out[NN_OUTPUT_DIM];
	for (int i = 0; i < NN_OUTPUT_DIM; i++) {
		double sum = b3[i];
		for (int j = 0; j < NN_HIDDEN2_DIM; j++) {
			sum += W3[i][j] * hidden2[j];
		}
		norm_out[i] = sum;  // purelin
	}

	// 5. 输出反归一化
	for (int i = 0; i < NN_OUTPUT_DIM; i++) {
		output[i] = denormalize(norm_out[i], XMIN_OUT[i], XMAX_OUT[i], YMIN_OUT, YMAX_OUT);
	}
}

/**
* 重力补偿函数：从原始测量中减去神经网络预测的重力影响
* @param raw_force 原始力测量 [Fx, Fy, Fz] (N)
* @param raw_torque 原始力矩测量 [Tx, Ty, Tz] (Nm)
* @param input 输入数组 [roll, pitch, yaw, x, y, z]
* @param comp_force 补偿后的力 (输出)
* @param comp_torque 补偿后的力矩 (输出)
*/
void Net_compensation(const double raw_force[3], const double raw_torque[3],
	const double input[NN_INPUT_DIM],
	double comp_force[3], double comp_torque[3]) {
	double gravity_effect[NN_OUTPUT_DIM];
	nn_forward(input, gravity_effect);

	for (int i = 0; i < 3; i++) {
		comp_force[i] = raw_force[i] - gravity_effect[i];
		comp_torque[i] = raw_torque[i] - gravity_effect[i + 3];
	}
}


/*
	4*4鐭╅樀鐩镐箻锛孉*B=C
*/
void matrixMultiply4x4_optimized(const double A[4][4], const double B[4][4], double C[4][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			// 灞曞紑鍐呭眰寰幆锛屽噺灏戝惊鐜紑閿�
			C[i][j] = A[i][0] * B[0][j] +
				A[i][1] * B[1][j] +
				A[i][2] * B[2][j] +
				A[i][3] * B[3][j];
		}

	}
}

void matrixMultiply4x4_optimized2(double A[4][4], const double B[4][4], double C[4][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			// 灞曞紑鍐呭眰寰幆锛屽噺灏戝惊鐜紑閿�
			C[i][j] = A[i][0] * B[0][j] +
				A[i][1] * B[1][j] +
				A[i][2] * B[2][j] +
				A[i][3] * B[3][j];
		}

	}
}

void TrasMatrix_SingleCal_SDH(double targetTM[4][4], double jntPos[6])
{
	double targetTMtemp[4][4][6] = { 0 };
	double targetTM_temp[4][4] = { 0 };


	double dh_par[6][4] = {
	{  0,     1.570796326794896 ,0.1635,0 },
	{ -0.6225,0                 ,0     ,0 },
	{ -0.5580,0                 ,0     ,0 },
	{  0     ,1.570796326794896 ,0.1645,0 },
	{  0,    -1.570796326794896 ,0.1195,0 },
	{  0,0 ,0.1175,0 }
	};

	//double dh_par[6][4] = {
	//	{ 0, 1.570796326794896, 0.1635, 0 },
	//	{ -0.6225, 0, 0, 0 },
	//	{ -0.5580, 0, 0, 0},
	//	{ 0, 1.570796326794896, -0.1645, 0},
	//	{ 0, 1.570796326794896, 0.1195, 0},
	//	{ 0, 0, 0.1175, 0 }
	//}; // a alpha  d theta


	/*jntPos[1] = jntPos[1] - 1.570796326794896;
	jntPos[3] = jntPos[3] - 1.570796326794896;*/
	//jntPos[2] = jntPos[2] ;
	/*jntPos[2] = jntPos[2] + 1.570796326794896;
	jntPos[3] = jntPos[3] + 1.570796326794896;*/


	for (int i = 0; i < 6; i++)
	{
		dh_par[i][3] = dh_par[i][3] + jntPos[i];
	}

	for (int i = 0; i < 6; i++)
	{
		targetTMtemp[0][0][i] = cos(dh_par[i][3]);
		targetTMtemp[0][1][i] = -sin(dh_par[i][3]) * cos(dh_par[i][1]);
		targetTMtemp[0][2][i] = sin(dh_par[i][3]) * sin(dh_par[i][1]);
		targetTMtemp[0][3][i] = cos(dh_par[i][3]) * dh_par[i][0];

		targetTMtemp[1][0][i] = sin(dh_par[i][3]);
		targetTMtemp[1][1][i] = cos(dh_par[i][3]) * cos(dh_par[i][1]);
		targetTMtemp[1][2][i] = -cos(dh_par[i][3]) * sin(dh_par[i][1]);
		targetTMtemp[1][3][i] = sin(dh_par[i][3]) * dh_par[i][0];

		targetTMtemp[2][0][i] = 0;
		targetTMtemp[2][1][i] = sin(dh_par[i][1]);
		targetTMtemp[2][2][i] = cos(dh_par[i][1]);
		targetTMtemp[2][3][i] = dh_par[i][2];

		targetTMtemp[3][0][i] = 0;
		targetTMtemp[3][1][i] = 0;
		targetTMtemp[3][2][i] = 0;
		targetTMtemp[3][3][i] = 1;
	}

	// 初始化 targetTM 为单位矩阵
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i == j)
			{
				targetTM_temp[i][j] = 1;
			}
			else
			{
				targetTM_temp[i][j] = 0;
			}
			//cout << targetTM[i][j] << "   ";
		}
		//cout << endl;
	}

	// 逐个相乘变换矩阵
	for (int i = 0; i < 6; i++)
	{
		double temp[4][4] = { 0 };

		for (int k = 0; k < 4; k++)
		{
			for (int l = 0; l < 4; l++)
			{
				temp[k][l] = targetTMtemp[k][l][i];
				//cout << temp[k][l] << "   ";
			}
			//cout << endl;
		}

		matrixMultiply4x4_optimized(targetTM_temp, temp, targetTM);

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				targetTM_temp[i][j] = targetTM[i][j];
			}
		}
	}
}


double max(double x, double y)
{
	if (x >= y)
		return x;
	else
		return y;
}
int abs(int x)
{
	if (x >= 0)
		return x;
	else
		return -x;
}
double abs2(double* x)
{
	double x1;
	x1 = *x;
	if (x1 >= 0)
		return x1;
	else
		return -x1;
}
/*
	几何法求逆解，T_goat,末端齐次变换矩阵，ithea逆解关节角，有8组搭配choose_theta使用
*/
void ikine_simplify(double T_goat[4][4], double itheta[8][6], double T06[4][4], double theta1[2], double* S5)
{
	double a2 = -0.6225;
	double a3 = -0.5580;
	double d1 = 0.1635;
	double d4 = 0.1645;
	double d5 = 0.1195;
	double d7 = 0.1175;

	/*double inv_T67[4][4] = {
		{1,0,0,0},
		{0,1,0,0},
		{0,0,1,-d7},
		{ 0,0,0,1}
	};*/
	double inv_T67[4][4] = {
		{1,0,0,0},
		{0,1,0,0},
		{0,0,1,-d7},
		{ 0,0,0,1}
	};

	//double T06[4][4] = { 0 };
	//T_goat近似为单位矩阵，相乘后齐次变换矩阵旋变部分不变，坐标部分变化
	//matrixMultiply4x4_optimized2(T_goat, inv_T67, T06);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			// 灞曞紑鍐呭眰寰幆锛屽噺灏戝惊鐜紑閿�
			T06[i][j] = T_goat[i][0] * inv_T67[0][j] +
				T_goat[i][1] * inv_T67[1][j] +
				T_goat[i][2] * inv_T67[2][j] +
				T_goat[i][3] * inv_T67[3][j];
		}

	}
	double nx = T06[0][0];
	double ox = T06[0][1];
	double ax = T06[0][2];
	double px = T06[0][3];

	double ny = T06[1][0];
	double oy = T06[1][1];
	double ay = T06[1][2];
	double py = T06[1][3];

	double nz = T06[2][0];
	double oz = T06[2][1];
	double az = T06[2][2];
	double pz = T06[2][3];

	int solution_count = 0;

	double D = px * px + py * py - d4 * d4;
	D = D > 0.F ? D : 0.F;
	//D = max(D, 0);

	// theta1的两个解
	//double theta1[2] = { 0 };
	double r = sqrt(D);
	theta1[0] = atan2(py, px) - atan2(-d4, r);
	theta1[1] = atan2(py, px) - atan2(-d4, -r);

	double s5_sign[2] = { 1.f, -1.f };
	double sign_sqrt[2] = { 1.f, -1.f };

	for (int i = 0; i < 2; i++)
	{
		double th1 = theta1[i];
		double c1 = cos(th1);
		double s1 = sin(th1);

		for (int j = 0; j < 2; j++)
		{
			// 计算 theta5
			*S5 = s5_sign[j] * sqrt((-s1 * nx + c1 * ny) * (-s1 * nx + c1 * ny) + (-s1 * ox + c1 * oy) * (-s1 * ox + c1 * oy));
			double th5 = atan2(*S5, s1 * ax - c1 * ay);

			if (abs2(S5) < 0.000001)
				continue;

			// 计算 theta6
			//double th6 = atan2((-s1 * ox + c1 * oy) / S5, (s1 * nx - c1 * ny) / S5);
			double th6 = atan2((-s1 * ox + c1 * oy), (s1 * nx - c1 * ny));

			// 计算 theta234
			double S234 = -az / *S5;
			double C234 = -(c1 * ax + s1 * ay) / *S5;

			double th234 = atan2(S234, C234);


			double B1 = c1 * px + s1 * py - d5 * S234;
			double B2 = pz - d1 + d5 * C234;

			double A = -2 * B2 * a2;
			double B = 2 * B1 * a2;
			double C = B1 * B1 + B2 * B2 + a2 * a2 - a3 * a3;

			double disc = A * A + B * B - C * C;
			if (disc < 0)
				continue;
			double sqrt_disc = sqrt(disc);

			for (int k = 0; k < 2; k++)
			{
				//计算theta2
				double th2 = atan2(B, A) - atan2(C, sign_sqrt[k] * sqrt_disc);
				double s2 = sin(th2);
				double c2 = cos(th2);

				double th23 = atan2((B2 - a2 * s2) / a3, (B1 - a2 * c2) / a3);
				//double th23 = atan2((B2 - a2 * s2) , (B1 - a2 * c2) );

				//计算theta3和theta4
				double th3 = th23 - th2;
				double th4 = th234 - th23;

				if (solution_count < 8)
				{
					itheta[solution_count][0] = th1;
					itheta[solution_count][1] = th2;
					itheta[solution_count][2] = th3;
					itheta[solution_count][3] = th4;
					itheta[solution_count][4] = th5;
					itheta[solution_count][5] = th6;
					solution_count = solution_count + 1;
				}
			}
		}


	}

	//for (int i = 0; i < 8; i++)
	//{
	//	itheta[i][1] += M_PI * 0.5F;
	//	itheta[i][3] += M_PI * 0.5F;
	//}

	// 角度归一化到[-π, π]  solution_count
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			itheta[i][j] = fmod(itheta[i][j], TWO_PI);
			if (itheta[i][j] > M_PI)
			{
				itheta[i][j] -= TWO_PI;
			}
			else if (itheta[i][j] < -M_PI)
			{
				itheta[i][j] += TWO_PI;
			}
		}
	}

}

/*
	从8组解中选取一组正确的，itheta_last上一时刻参考值，itheta_now当前解
*/
void choose_theta(double itheta[8][6], double itheta_last[6], double itheta_now[6])
{
	double delta[6] = { 0 };
	double norm_val = 0;
	double thetaNorms[8] = { 0 };
	int minIndex = 0;
	double minNorm = 0;

	for (int i = 0; i < 8; i++)
	{
		// 计算角度差值（考虑周期性）
		for (int j = 0; j < 6; j++)
		{
			double diff = itheta[i][j] - itheta_last[j];
			// 将差值映射到[-π, π)区间
			delta[j] = fmod(diff + M_PI, 2 * M_PI) - M_PI;
			//delta[j] = itheta[i][j] - itheta_last[j];
			// 计算欧几里得范数
			norm_val = norm_val + delta[j] * delta[j];

		}
		thetaNorms[i] = sqrt(norm_val);
		norm_val = 0;
	}

	// 找到最小范数及其索引
	minNorm = thetaNorms[0];
	for (int i = 1; i < 8; i++)
	{
		if (thetaNorms[i] < minNorm)
		{
			minNorm = thetaNorms[i];
			minIndex = i;
		}
	}
	// 复制选择的解
	for (int j = 0; j < 6; j++)
	{
		itheta_now[j] = itheta[minIndex][j];
	}

	// 应用特定调整
	//itheta_now[1] += PI / 2.0;  // 第2关节
	//itheta_now[3] += PI / 2.0;  // 第4关节
	//itheta_now[2] = -itheta_now[2];  // 第3关节取反
}

void matrix_multiply_3x3_unrolled(const double A[3][3], const double B[3][3], double C[3][3])
{
	// 璁＄畻绗竴琛�
	C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0];
	C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1];
	C[0][2] = A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] * B[2][2];

	// 璁＄畻绗簩琛�
	C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0];
	C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] * B[2][1];
	C[1][2] = A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] * B[2][2];

	// 璁＄畻绗笁琛�
	C[2][0] = A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] * B[2][0];
	C[2][1] = A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] * B[2][1];
	C[2][2] = A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] * B[2][2];
}

void matrix_mult_6x6_6x1(const double A[6][6], const double b[6], double result[6])
{
	for (int i = 0; i < 6; i++)
	{
		result[i] = 0.0;
		for (int j = 0; j < 6; j++)
		{
			result[i] += A[i][j] * b[j];
		}
	}
}

void eye_4x4(double matrix[4][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (i == j)
				matrix[i][j] = 1;
			else
				matrix[i][j] = 0;
		}
	}
}

void expm_rodrigues(const double w[3], double R[3][3])
{
	// 浠庡弽瀵圭О鐭╅樀鎻愬彇鏃嬭浆鍚戦噺
	//double w[3];
	//w[0] = -skew[1][2];
	//w[1] = skew[0][2];
	//w[2] = -skew[0][1];

	double theta_sq = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
	double theta = sqrt(theta_sq);

	if (theta < 1e-10) {
		// 灏忚搴﹁繎浼�
		R[0][0] = 1.0;       R[0][1] = -w[2];   R[0][2] = w[1];
		R[1][0] = w[2];      R[1][1] = 1.0;     R[1][2] = -w[0];
		R[2][0] = -w[1];     R[2][1] = w[0];    R[2][2] = 1.0;
		return;
	}

	double wx = w[0];
	double wy = w[1];
	double wz = w[2];

	double sin_theta = sin(theta);
	double cos_theta = cos(theta);
	double one_minus_cos = 1.0 - cos_theta;

	// 棰勮绠�
	double k1 = sin_theta / theta;
	double k2 = one_minus_cos / theta_sq;
	double k3 = (one_minus_cos / (theta_sq * theta));  // 鐢ㄤ簬鍙変箻鐭╅樀骞虫柟鐨勭郴鏁�

	// 鐩存帴璁＄畻鍏紡鐨勫厓绱�
	// R = I + k1 * [w]脳 + k2 * [w]脳虏
	// 娉ㄦ剰锛氳繖閲屾垜浠洿鎺ュ睍寮�鍏紡锛岄伩鍏嶇煩闃佃繍绠�

	double wx2 = wx * wx;
	double wy2 = wy * wy;
	double wz2 = wz * wz;
	double wxwy = wx * wy;
	double wxwz = wx * wz;
	double wywz = wy * wz;

	// 瀵硅绾垮厓绱�
	R[0][0] = 1.0 - k2 * (wy2 + wz2);
	R[1][1] = 1.0 - k2 * (wx2 + wz2);
	R[2][2] = 1.0 - k2 * (wx2 + wy2);

	// 闈炲瑙掔嚎鍏冪礌
	R[0][1] = k2 * wxwy - k1 * wz;
	R[0][2] = k2 * wxwz + k1 * wy;
	R[1][0] = k2 * wxwy + k1 * wz;
	R[1][2] = k2 * wywz - k1 * wx;
	R[2][0] = k2 * wxwz - k1 * wy;
	R[2][1] = k2 * wywz + k1 * wx;
}

void transPot(double A[3][3], double B[3][3])
{
	A[0][0] = B[0][0];
	A[1][1] = B[1][1];
	A[2][2] = B[2][2];

	A[0][1] = B[1][0];
	A[0][2] = B[2][0];
	A[1][0] = B[0][1];
	A[1][2] = B[2][1];
	A[2][0] = B[0][2];
	A[2][1] = B[1][2];
}

void Matrix2AxisAngle(double R[3][3], double axisAngle[3])
{
	double trace = R[0][0] + R[1][1] + R[2][2];
	double cosTheta = 0.5 * (trace - 1.0);

	// 澶勭悊鏁板�艰宸�
	if (cosTheta > 1.0)
		cosTheta = 1.0;
	if (cosTheta < -1.0)
		cosTheta = -1.0;

	double theta = acos(cosTheta);

	if (theta < 0.001)
	{
		axisAngle[0] = 0.0;
		axisAngle[1] = 0.0;
		axisAngle[2] = 0.0;
		return;
	}

	double sinTheta = sin(theta);
	if (fabs(sinTheta) < 1e-10)
	{
		axisAngle[0] = 0.0;
		axisAngle[1] = 0.0;
		axisAngle[2] = 0.0;

	}

	double factor = 0.5 / sinTheta;
	axisAngle[0] = factor * (R[2][1] - R[1][2]) * theta;
	axisAngle[1] = factor * (R[0][2] - R[2][0]) * theta;
	axisAngle[2] = factor * (R[1][0] - R[0][1]) * theta;
}


void ForceControl2(double Force[6], double Kx, double Ky, double Kz, double K, double dt, double velLast[6], double lenLast[6], double T_init[4][4], double T_last[4][4], double T_now[4][4], double T_next[4][4])
{
	double M = 200;
	double C = 100;
	// ===================== 关键修改：临界阻尼 =====================
	double Cx = 100;  // X阻尼 (2*sqrt(200*260))  400
	double Cy = 100;  // Y阻尼
	double Cz = 100;  // Z阻尼 (2*sqrt(200*200)) → 你最关心
	double Cr = 80;  // 旋转阻尼
	// ==============================================================
	double RotMartx[6][6] = { 0 };
	double T_error[4][4] = { 0 };
	double R_error[3][3] = { 0 };
	double R_temp[3][3] = { 0 };
	double R_last[3][3] = { 0 };
	double R_init[3][3] = { 0 };
	double R_next[3][3] = { 0 };
	double Force_temp[6] = { 0 };
	double w[3] = { 0 };
	double acc[6] = { 0 };
	double vel[6] = { 0 };
	double len[6] = { 0 };

	
	// for (int i = 0; i < 3; i++)
	// {
	// 	for (int j = 0; j < 3; j++)
	// 	{
	// 		RotMartx[i][j] = T_now[i][j];
	// 		RotMartx[i + 3][j + 3] = T_now[i][j];
	// 	}
	// }

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			RotMartx[i][j] = T_init[i][j];
			RotMartx[i + 3][j + 3] = T_init[i][j];
		}
	}

	matrix_mult_6x6_6x1(RotMartx, Force, Force_temp);

	// printf("基坐标系六维力传感器: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f\n",
	// 	Force_temp[0], Force_temp[1], Force_temp[2], Force_temp[3], Force_temp[4], Force_temp[5]);
	
	
	//	for (int i = 0; i < 6; i++)
	//	{
	//		Force_temp[i] = Force[i];
	//	}
	// if(Force_temp[3]<0.01f)
	// {
	// 	Force_temp[3] = 0.f;
	// }
	// if(Force_temp[4]<0.01f)
	// {
	// 	Force_temp[4] = 0.f;
	// }
	if(abs(Force_temp[5])<0.01f)
	{
		Force_temp[5] = 0.f;
	}
	

	int temp = 1;

	acc[0] = (Force_temp[0] - Cx * velLast[0] - Kx * lenLast[0]) / (M*temp);
	vel[0] = velLast[0] + acc[0] * dt;
	len[0] = vel[0] * dt;
	velLast[0] = vel[0];

	acc[1] = (Force_temp[1] - Cy * velLast[1] - Ky * lenLast[1]) / (M*temp);
	vel[1] = velLast[1] + acc[1] * dt;
	len[1] = vel[1] * dt;
	velLast[1] = vel[1];

	acc[2] = (Force_temp[2] - Cz * velLast[2] - Kz * lenLast[2]) / (M*temp);
	vel[2] = velLast[2] + acc[2] * dt;
	len[2] = vel[2] * dt;
	velLast[2] = vel[2];

	for (int i = 3; i < 6; i++)
	{
		acc[i] = (Force_temp[i] - Cr * velLast[i] - K * lenLast[i]) / M;
		vel[i] = velLast[i] + acc[i] * dt;
		// 单步旋转速度软限幅
		// if(fabs(vel[i]) > 0.001)
		// 	vel[i] = vel[i] > 0 ? 0.001 : -0.001;

		len[i] = vel[i] * dt;
		velLast[i] = vel[i];
	}
	

	eye_4x4(T_next);
	eye_4x4(T_error);
	for (int i = 0; i < 3; i++)
	{
		w[i] = len[i + 3];
	}
	expm_rodrigues(w, R_error);
	for (int i = 0; i < 3; i++)
	{
		T_error[i][3] = len[i];
		for (int j = 0; j < 3; j++)
		{
			T_error[i][j] = R_error[i][j];
			R_last[i][j] = T_last[i][j];
		}
	}

	matrix_multiply_3x3_unrolled(R_error, R_last, R_temp);
	//鏇存柊濮挎��
	for (int i = 0; i < 3; i++)
	{
		/*
		 * 杩欓噷鐨凾_next[i][3],0/1/2鍒嗗埆瀵瑰簲x/y/z,濡傛灉涓嶉渶瑕佷笉鏇存柊鍗冲彲
		 * 渚嬪鍙渶瑕亁杞达細T_next[0][3] = T_error[0][3]+ T_last[0][3];
		 */
		T_next[i][3] = T_error[i][3] + T_last[i][3];
		for (int j = 0; j < 3; j++)
		{
			T_next[i][j] = R_temp[i][j];
		}
	}


	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			T_last[i][j] = T_next[i][j];//原版
			//T_last[i][j] = T_now[i][j];//2026.02.24swg修改
		}
	}

	//鏇存柊lenLast
	for (int i = 0; i < 3; i++)
	{
		lenLast[i] = T_next[i][3] - T_init[i][3];
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R_next[i][j] = T_next[i][j];
			R_init[i][j] = T_init[i][j];
		}
	}
	double R_init_temp[3][3] = { 0 };
	transPot(R_init_temp, R_init);
	double R_axisAngle[3][3] = { 0 };
	matrix_multiply_3x3_unrolled(R_next, R_init_temp, R_axisAngle);
	double axisAngle[3] = { 0 };
	Matrix2AxisAngle(R_axisAngle, axisAngle);
	for (int i = 0; i < 3; i++)
	{
		lenLast[i + 3] = axisAngle[i];
	}


}

// ======================== 【重构版】力控阻抗控制 ========================
// 功能：6自由度纯正阻抗控制，自动根据力/力矩调整位姿减小受力，无自发漂移
// 输入：仅Z向力，其余力/力矩=0 → 姿态保持不动；有外力矩 → 柔顺微调
// ======================================================================
void ForceControl3(
    double Force[6],
    double Kx, double Ky, double Kz,
    double Kr,
    double dt,
    double velLast[6],
    double lenLast[6],
    double T_init[4][4],
    double T_last[4][4],
    double T_now[4][4],
    double T_next[4][4])
{
    const double M  = 200.0;
    const double Mr = 240.0;
    const double Cx = 100.0; //100
    const double Cy = 100.0;
    const double Cz = 100.0;
    const double Cr = 80.0;

	const double DRIFT_COMPENSATION = 0.3;   // 漂移自动回正刚度

    double W_transform[6][6] = {0};
    double F_world[6] = {0};
    double R_now[3][3]   = {0};
    double p_now[3]      = {0};
    double R_err[3][3]   = {0};
    double R_temp[3][3]  = {0};
    double w_vec[3]      = {0};
    double acc[6]        = {0};
    double vel[6]        = {0};
    double d_pos[6]      = {0};
    double err[6]        = {0};

    // 提取 T_now 的旋转矩阵 R_now[3][3]
    // for(int i=0;i<3;i++){
    //     for(int j=0;j<3;j++)
    //         R_now[i][j] = T_now[i][j];
    // }
    // p_now[0] = T_now[0][3];
    // p_now[1] = T_now[1][3];
    // p_now[2] = T_now[2][3];

	// 提取 T_init 的旋转矩阵 R_now[3][3]
	for(int i=0;i<3;i++){
        for(int j=0;j<3;j++)
            R_now[i][j] = T_init[i][j];
    }
    p_now[0] = T_init[0][3];
    p_now[1] = T_init[1][3];
    p_now[2] = T_init[2][3];


    // 构建正确的力旋量变换矩阵
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            W_transform[i][j] = R_now[i][j];
            W_transform[i+3][j+3] = R_now[i][j];
        }
    }

    // 叉乘矩阵 S(p) * R
    double skew[3][3] = {
        {0, -p_now[2], p_now[1]},
        {p_now[2], 0, -p_now[0]},
        {-p_now[1], p_now[0], 0}
    };
    double skewR[3][3] = {0};
    matrix_multiply_3x3_unrolled(skew, R_now, skewR);
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            W_transform[i+3][j] = skewR[i][j];

    // 力坐标变换
    matrix_mult_6x6_6x1(W_transform, Force, F_world);

	
	
	// printf("基坐标系六维力传感器: X=%.4f Y=%.4f Z=%.4f Rx=%.2f Ry=%.2f Rz=%.2f\n",
	// 	F_world[0], F_world[1], F_world[2], F_world[3], F_world[4], F_world[5]);
	// printf("法向量: X=%.4f Y=%.4f Z=%.4f \n",
	// 	T_init[0][2], T_init[1][2], T_init[2][2]);

	// ====================== 【稳定核心：力矩死区】 ======================
    for(int i=3;i<6;i++){
        if(fabs(F_world[i]) < 0.1)
            F_world[i] = 0.0;
    }
    // ===================== 计算真实姿态偏差 =====================
    for(int i=0;i<3;i++)
        err[i] = T_now[i][3] - T_init[i][3];

	// printf("位移: X=%.6f Y=%.6f Z=%.6f \n",
	// 		err[0], err[1], err[2]);

    // 提取 T_init 旋转矩阵
    double R_init[3][3] = {0};
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            R_init[i][j] = T_init[i][j];

    double R_init_T[3][3] = {0};
    transPot(R_init_T, R_init);  // 现在类型完全匹配

    double R_rel[3][3] = {0};
    matrix_multiply_3x3_unrolled(R_now, R_init_T, R_rel);

    double angErr[3] = {0};
    Matrix2AxisAngle(R_rel, angErr);
    for(int i=0;i<3;i++)
        err[i+3] = angErr[i];
	// ====================== 【根治漂移：无外力时自动回正】 ======================
    double effective_Kr[3] = {0};
    for(int i=0;i<3;i++){
        if(fabs(F_world[i+3]) < 1e-3){
            effective_Kr[i] = Kr + DRIFT_COMPENSATION;  // 无外力 → 加回正力
        }else{
            effective_Kr[i] = Kr;                       // 有力矩 → 柔顺
        }
    }

    // ===================== 标准阻抗控制 =====================
	acc[0] = (F_world[0] - Cx*velLast[0] - Kx*err[0]) / M;
    acc[1] = (F_world[1] - Cy*velLast[1] - Ky*err[1]) / M;
    acc[2] = (F_world[2] - Cz*velLast[2] - Kz*err[2]) / M;
    acc[3] = (F_world[3] - Cr*velLast[3] - effective_Kr[0]*err[3]) / Mr;
    acc[4] = (F_world[4] - Cr*velLast[4] - effective_Kr[1]*err[4]) / Mr;
    acc[5] = (F_world[5] - Cr*velLast[5] - effective_Kr[2]*err[5]) / Mr;

    for(int i=0;i<6;i++){
        vel[i]     = velLast[i] + acc[i] * dt;
        d_pos[i]   = vel[i] * dt;
        velLast[i] = vel[i];
    }

	

    // 旋转量
    eye_4x4(T_next);
    w_vec[0] = d_pos[3];
    w_vec[1] = d_pos[4];
    w_vec[2] = d_pos[5];
    expm_rodrigues(w_vec, R_err);

    // 用初始姿态作为基准，保证不飘
    matrix_multiply_3x3_unrolled(R_err, R_init, R_temp);

    // 输出新位姿
    for(int i=0;i<3;i++){
        T_next[i][3] = T_now[i][3] + d_pos[i];
        for(int j=0;j<3;j++)
            T_next[i][j] = R_temp[i][j];
    }
    T_next[3][0] = T_next[3][1] = T_next[3][2] = 0.0;
    T_next[3][3] = 1.0;

    // 更新历史位姿
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)
            T_last[i][j] = T_next[i][j];
}

void ForceControlfor5U(
	double Force[6],
	double Kx, double Ky, double Kz,
	double Kr,
	double dt,
	double velLast[6],
	double lenLast[6],
	double T_init[4][4],
	double T_last[4][4],
	double T_now[4][4],
	double T_next[4][4])
{
	const double M = 25.0;
	const double Mr = 240.0;
	const double Cx = 100.0;
	const double Cy = 100.0;
	const double Cz = 2000.0;
	const double Cr = 80.0;

	const double DRIFT_COMPENSATION = 0.3; // 漂移自动回正刚度

	double W_transform[6][6] = {0};
	double F_world[6] = {0};
	double R_now[3][3] = {0};
	double p_now[3] = {0};
	double R_err[3][3] = {0};
	double R_temp[3][3] = {0};
	double w_vec[3] = {0};
	double acc[6] = {0};
	double vel[6] = {0};
	double d_pos[6] = {0};
	double err[6] = {0};

	// 提取 T_now 的旋转矩阵 R_now[3][3]
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			R_now[i][j] = T_now[i][j];
	}
	p_now[0] = T_now[0][3];
	p_now[1] = T_now[1][3];
	p_now[2] = T_now[2][3];

	// 构建正确的力旋量变换矩阵
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			W_transform[i][j] = R_now[i][j];
			W_transform[i + 3][j + 3] = R_now[i][j];
		}
	}

	// 叉乘矩阵 S(p) * R
	double skew[3][3] = {
		{0, -p_now[2], p_now[1]},
		{p_now[2], 0, -p_now[0]},
		{-p_now[1], p_now[0], 0}};
	double skewR[3][3] = {0};
	matrix_multiply_3x3_unrolled(skew, R_now, skewR);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			W_transform[i + 3][j] = skewR[i][j];

	// 力坐标变换
	matrix_mult_6x6_6x1(W_transform, Force, F_world);

	// ====================== 【稳定核心：力矩死区】 ======================
	for (int i = 3; i < 6; i++)
	{
		if (fabs(F_world[i]) < 0.1)
			F_world[i] = 0.0;
	}
	// ===================== 计算真实姿态偏差 =====================
	for (int i = 0; i < 3; i++)
		err[i] = T_now[i][3] - T_init[i][3];

	// 提取 T_init 旋转矩阵
	double R_init[3][3] = {0};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R_init[i][j] = T_init[i][j];

	double R_init_T[3][3] = {0};
	transPot(R_init_T, R_init); // 现在类型完全匹配

	double R_rel[3][3] = {0};
	matrix_multiply_3x3_unrolled(R_now, R_init_T, R_rel);

	double angErr[3] = {0};
	Matrix2AxisAngle(R_rel, angErr);
	for (int i = 0; i < 3; i++)
		err[i + 3] = angErr[i];
	// ====================== 【根治漂移：无外力时自动回正】 ======================
	double effective_Kr[3] = {0};
	for (int i = 0; i < 3; i++)
	{
		if (fabs(F_world[i + 3]) < 1e-3)
		{
			effective_Kr[i] = Kr + DRIFT_COMPENSATION; // 无外力 → 加回正力
		}
		else
		{
			effective_Kr[i] = Kr; // 有力矩 → 柔顺
		}
	}

	// ===================== 标准阻抗控制 =====================
	acc[0] = (F_world[0] - Cx * velLast[0] - Kx * err[0]) / M;
	acc[1] = (F_world[1] - Cy * velLast[1] - Ky * err[1]) / M;
	acc[2] = (F_world[2] - Cz * velLast[2] - Kz * err[2]) / M;
	acc[3] = (F_world[3] - Cr * velLast[3] - effective_Kr[0] * err[3]) / Mr;
	acc[4] = (F_world[4] - Cr * velLast[4] - effective_Kr[1] * err[4]) / Mr;
	acc[5] = (F_world[5] - Cr * velLast[5] - effective_Kr[2] * err[5]) / Mr;

	for (int i = 0; i < 6; i++)
	{
		vel[i] = velLast[i] + acc[i] * dt;
		d_pos[i] = vel[i] * dt;
		velLast[i] = vel[i];
	}

	// 旋转量
	eye_4x4(T_next);
	w_vec[0] = d_pos[3];
	w_vec[1] = d_pos[4];
	w_vec[2] = d_pos[5];
	expm_rodrigues(w_vec, R_err);

	// 用初始姿态作为基准，保证不飘
	matrix_multiply_3x3_unrolled(R_err, R_init, R_temp);

	// 输出新位姿
	for (int i = 0; i < 3; i++)
	{
		T_next[i][3] = T_now[i][3] + d_pos[i];
		for (int j = 0; j < 3; j++)
			T_next[i][j] = R_temp[i][j];
	}
	T_next[3][0] = T_next[3][1] = T_next[3][2] = 0.0;
	T_next[3][3] = 1.0;

	// 更新历史位姿
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			T_last[i][j] = T_next[i][j];
}

void ForceControlfor9U(
	double Force[6],
	double Kx, double Ky, double Kz,
	double Kr,
	double dt,
	double velLast[6],
	double lenLast[6],
	double T_init[4][4],
	double T_last[4][4],
	double T_now[4][4],
	double T_next[4][4],
	double Fzworld)
{
	const double M = 200.0;
	const double Mr = 240.0;
	const double Cx = 100.0;
	const double Cy = 100.0;
	const double Cz = 100.0;
	const double Cr = 80.0;

	const double DRIFT_COMPENSATION = 0.3; // 漂移自动回正刚度

	double W_transform[6][6] = {0};
	double F_world[6] = {0};
	double R_now[3][3] = {0};
	double p_now[3] = {0};
	double R_err[3][3] = {0};
	double R_temp[3][3] = {0};
	double w_vec[3] = {0};
	double acc[6] = {0};
	double vel[6] = {0};
	double d_pos[6] = {0};
	double err[6] = {0};

	// 提取 T_now 的旋转矩阵 R_now[3][3]
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			R_now[i][j] = T_now[i][j];
	}
	p_now[0] = T_now[0][3];
	p_now[1] = T_now[1][3];
	p_now[2] = T_now[2][3];

	// 构建正确的力旋量变换矩阵
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			W_transform[i][j] = R_now[i][j];
			W_transform[i + 3][j + 3] = R_now[i][j];
		}
	}

	// 叉乘矩阵 S(p) * R
	double skew[3][3] = {
		{0, -p_now[2], p_now[1]},
		{p_now[2], 0, -p_now[0]},
		{-p_now[1], p_now[0], 0}};
	double skewR[3][3] = {0};
	matrix_multiply_3x3_unrolled(skew, R_now, skewR);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			W_transform[i + 3][j] = skewR[i][j];

	// 力坐标变换
	matrix_mult_6x6_6x1(W_transform, Force, F_world);

	// ====================== 【稳定核心：力矩死区】 ======================
	for (int i = 3; i < 6; i++)
	{
		if (fabs(F_world[i]) < 0.1)
			F_world[i] = 0.0;
	}
	// ===================== 计算真实姿态偏差 =====================
	for (int i = 0; i < 3; i++)
		err[i] = T_now[i][3] - T_init[i][3];

	// 提取 T_init 旋转矩阵
	double R_init[3][3] = {0};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R_init[i][j] = T_init[i][j];

	double R_init_T[3][3] = {0};
	transPot(R_init_T, R_init); // 现在类型完全匹配

	double R_rel[3][3] = {0};
	matrix_multiply_3x3_unrolled(R_now, R_init_T, R_rel);

	double angErr[3] = {0};
	Matrix2AxisAngle(R_rel, angErr);
	for (int i = 0; i < 3; i++)
		err[i + 3] = angErr[i];
	// ====================== 【根治漂移：无外力时自动回正】 ======================
	double effective_Kr[3] = {0};
	for (int i = 0; i < 3; i++)
	{
		if (fabs(F_world[i + 3]) < 1e-3)
		{
			effective_Kr[i] = Kr + DRIFT_COMPENSATION; // 无外力 → 加回正力
		}
		else
		{
			effective_Kr[i] = Kr; // 有力矩 → 柔顺
		}
	}

	// ===================== 标准阻抗控制 =====================
	acc[0] = (F_world[0] - Cx * velLast[0] - Kx * err[0]) / M;
	acc[1] = (F_world[1] - Cy * velLast[1] - Ky * err[1]) / M;
	acc[2] = (F_world[2]+Fzworld - Cz * velLast[2] - Kz * err[2]) / M;
	acc[3] = (F_world[3] - Cr * velLast[3] - effective_Kr[0] * err[3]) / Mr;
	acc[4] = (F_world[4] - Cr * velLast[4] - effective_Kr[1] * err[4]) / Mr;
	acc[5] = (F_world[5] - Cr * velLast[5] - effective_Kr[2] * err[5]) / Mr;

	for (int i = 0; i < 6; i++)
	{
		vel[i] = velLast[i] + acc[i] * dt;
		d_pos[i] = vel[i] * dt;
		velLast[i] = vel[i];
	}

	// 旋转量
	eye_4x4(T_next);
	w_vec[0] = d_pos[3];
	w_vec[1] = d_pos[4];
	w_vec[2] = d_pos[5];
	expm_rodrigues(w_vec, R_err);

	// 用初始姿态作为基准，保证不飘
	matrix_multiply_3x3_unrolled(R_err, R_init, R_temp);

	// 输出新位姿
	for (int i = 0; i < 3; i++)
	{
		T_next[i][3] = T_now[i][3] + d_pos[i];
		for (int j = 0; j < 3; j++)
			T_next[i][j] = R_temp[i][j];
	}
	T_next[3][0] = T_next[3][1] = T_next[3][2] = 0.0;
	T_next[3][3] = 1.0;

	// 更新历史位姿
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			T_last[i][j] = T_next[i][j];
}

// 将旋转矩阵转换为XYZ欧拉角 (Roll-Pitch-Yaw)
// 顺序：先绕Z轴旋转(yaw)，再绕Y轴旋转(pitch)，最后绕X轴旋转(roll)
void rotationMatrixToEulerXYZ(double T_next[4][4], double OutPosArr[6]) {
	EulerAngles angles;

	double pitch, yaw, roll;

	// 提取旋转矩阵元素（行优先）
	double r11 = T_next[0][0];
	double r12 = T_next[0][1];
	double r13 = T_next[0][2];
	double r21 = T_next[1][0];
	double r22 = T_next[1][1];
	double r23 = T_next[1][2];
	double r31 = T_next[2][0];
	double r32 = T_next[2][1];
	double r33 = T_next[2][2];

	// 计算pitch（绕Y轴转角）
	double sin_pitch = -r31;
	double cos_pitch_sq = 1.0 - sin_pitch * sin_pitch; // 等价于 r11^2 + r21^2
	// 为避免数值误差，使用 sqrt(r11^2 + r21^2) 更稳定
	double cos_pitch = sqrt(r11 * r11 + r21 * r21);

	// 判断奇异：当cos_pitch接近0时，pitch接近±90°
	const double singularity_threshold = 1e-6;
	if (cos_pitch > singularity_threshold) {
		// 正常情况  beta a gama
		pitch = atan2(sin_pitch, cos_pitch);
		yaw = atan2(r21, r11);
		roll = atan2(r32, r33);
	}
	else {
		// 奇异情况：pitch 接近 ±90°
		pitch = atan2(sin_pitch, 0.0); // 结果为 ±π/2
		// 通常将 yaw 置 0，然后根据 pitch 符号求解 roll
		yaw = 0.0;
		if (pitch > 0) {
			// pitch = +90°
			roll = atan2(r12, r13);
		}
		else {
			// pitch = -90°
			roll = atan2(-r12, -r13);
		}
	}

	angles.yaw = yaw * 180 / M_PI;
	angles.pitch = pitch * 180 / M_PI;
	angles.roll = roll * 180 / M_PI;

	/*angles.roll = roll * 180 / M_PI;
	angles.yaw = yaw * 180 / M_PI;
	angles.pitch = pitch * 180 / M_PI;*/

	//	angles.X = T_next[0][3];
	//	angles.Y = T_next[1][3];
	//	angles.Z = T_next[2][3];

	OutPosArr[0] = yaw;
	OutPosArr[1] = pitch;
	OutPosArr[2] = roll;
	OutPosArr[3] = T_next[0][3];
	OutPosArr[4] = T_next[1][3];
	OutPosArr[5] = T_next[2][3];


	return;
}



/*！<滑动窗口均值滤波相关定义*/
#define MOV_AVERAGE_LEN 10
signed int s16MovArrIndex;
signed int  s16MovMeanCnts = MOV_AVERAGE_LEN;
double MovArrFx[MOV_AVERAGE_LEN] = { 0 };
double MovArrFy[MOV_AVERAGE_LEN] = { 0 };
double MovArrFz[MOV_AVERAGE_LEN] = { 0 };
double MovArrMx[MOV_AVERAGE_LEN] = { 0 };
double MovArrMy[MOV_AVERAGE_LEN] = { 0 };
double MovArrMz[MOV_AVERAGE_LEN] = { 0 };
double ForceAftFlt[6] = { 0 };
double ForceAftCmp[6] = { 0 };
double MovMeanSumFx, MovMeanSumFy, MovMeanSumFz, MovMeanSumMx, MovMeanSumMy, MovMeanSumMz = 0.f;

void MovMean(double dataNew[6])
{
	/*！<低转速时默认*/
	s16MovArrIndex = s16MovArrIndex % s16MovMeanCnts;

	MovMeanSumFx -= MovArrFx[s16MovArrIndex];
	MovMeanSumFx += dataNew[0];
	MovArrFx[s16MovArrIndex] = dataNew[0];
	ForceAftFlt[0] = MovMeanSumFx / (double)s16MovMeanCnts;


	MovMeanSumFy -= MovArrFy[s16MovArrIndex];
	MovMeanSumFy += dataNew[1];
	MovArrFy[s16MovArrIndex] = dataNew[1];
	ForceAftFlt[1] = MovMeanSumFy / (double)s16MovMeanCnts;

	MovMeanSumFz -= MovArrFz[s16MovArrIndex];
	MovMeanSumFz += dataNew[2];
	MovArrFz[s16MovArrIndex] = dataNew[2];
	ForceAftFlt[2] = MovMeanSumFz / (double)s16MovMeanCnts;

	MovMeanSumMx -= MovArrMx[s16MovArrIndex];
	MovMeanSumMx += dataNew[3];
	MovArrMx[s16MovArrIndex] = dataNew[3];
	ForceAftFlt[3] = MovMeanSumMx / (double)s16MovMeanCnts;

	MovMeanSumMy -= MovArrMy[s16MovArrIndex];
	MovMeanSumMy += dataNew[4];
	MovArrMy[s16MovArrIndex] = dataNew[4];
	ForceAftFlt[4] = MovMeanSumMy / (double)s16MovMeanCnts;

	MovMeanSumMz -= MovArrMz[s16MovArrIndex];
	MovMeanSumMz += dataNew[5];
	MovArrMz[s16MovArrIndex] = dataNew[5];
	ForceAftFlt[5] = MovMeanSumMz / (double)s16MovMeanCnts;

	s16MovArrIndex++;

	return;
}

void ForceCmpByX2(double distence)
{

	double x_cmp = -0.00691589f * distence * distence + 0.44322267 * distence - 9.72355541;
	double y_cmp = 0.00067085f * distence * distence - 0.04058499 * distence - 20.14191197;
	double z_cmp = 0.00542369f * distence * distence - 0.38667792 * distence - 43.79896389;
	//	ForceAftCmp[0]=ForceAftFlt[0]-x_cmp;
	//	ForceAftCmp[1]=ForceAftFlt[1]-y_cmp;
	//	ForceAftCmp[2]=ForceAftFlt[2]-z_cmp;

		//喷漆机械臂，将X,Y互换
	ForceAftCmp[0] = ForceAftFlt[1] - y_cmp;
	ForceAftCmp[1] = ForceAftFlt[0] - x_cmp;
	ForceAftCmp[2] = ForceAftFlt[2] - z_cmp;

	return;

}

//机器人ZYX 笛卡尔位姿转换齐次变换矩阵  输入角度为弧度制
void pose_to_transform(double x, double y, double z,
	double roll, double pitch, double yaw,
	double T[4][4])
{
	

	double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);

    // 正确 ZYX 旋转矩阵
    T[0][0] = cy * cp;
    T[0][1] = cy*sp*sr - sy*cr;
    T[0][2] = cy*sp*cr + sy*sr;
    T[1][0] = sy * cp;
    T[1][1] = sy*sp*sr + cy*cr;
    T[1][2] = sy*sp*cr - cy*sr;
    T[2][0] = -sp;
    T[2][1] = cp * sr;
    T[2][2] = cp * cr;

    // 平移
    T[0][3] = x;
    T[1][3] = y;
    T[2][3] = z;

    // 齐次行
    T[3][0] = 0.0; T[3][1] = 0.0; T[3][2] = 0.0; T[3][3] = 1.0;
}

// 将旋转矩阵转换为XYZ欧拉角 (Roll-Pitch-Yaw)   输出为角度值
// 顺序：先绕Z轴旋转(yaw)，再绕Y轴旋转(pitch)，最后绕X轴旋转(roll)
EulerAngles2 rotationMatrixToEulerXYZ2(double T_next[4][4]) 
{
	

	EulerAngles2 angles;

	double r11 = T_next[0][0];
    double r12 = T_next[0][1];
    double r13 = T_next[0][2];
    double r21 = T_next[1][0];
    double r22 = T_next[1][1];
    double r23 = T_next[1][2];
    double r31 = T_next[2][0];
    double r32 = T_next[2][1];
    double r33 = T_next[2][2];

    double yaw, pitch, roll;
    const double eps = 1e-6;

    // ================================
    // 【唯一正确】ZYX 欧拉角解算公式
    // ================================
    pitch = atan2(-r31, sqrt(r11*r11 + r21*r21));  // 绕Y轴

    if (fabs(r31 + 1.0) < eps) {
        // 奇异点: pitch = +90°
        yaw = atan2(-r12, r22);
        roll = 0.0;
    }
    else if (fabs(r31 - 1.0) < eps) {
        // 奇异点: pitch = -90°
        yaw = atan2(r12, r22);
        roll = 0.0;
    }
    else {
        // 正常情况（99%场景都走这里）
        yaw = atan2(r21, r11);     // 绕Z轴
        roll = atan2(r32, r33);    // 绕X轴
    }

    // 输出角度（必须严格对应！）
    angles.roll  = roll  * 180.0 / M_PI;  // Rx
    angles.pitch = pitch * 180.0 / M_PI; // Ry
    angles.yaw   = yaw   * 180.0 / M_PI;  // Rz

    // 平移
    angles.X = T_next[0][3];
    angles.Y = T_next[1][3];
    angles.Z = T_next[2][3];

    return angles;

}
/*!< ==================力偏执减0策略============*/
double ZeroCmpBaisArr[6] = {0.f};
int ZeroInFlg = 0;
void ForceZeroBaisCmp(int InputFlg,double InPutArr[6])
{
	//插枪状态
	if(InputFlg==1)
	{
		if(ZeroInFlg==0)
		{
			for(int i = 0;i<6;i++)
			{
				ZeroCmpBaisArr[i] =InPutArr[i];
			}
			ZeroInFlg = 1;
		}
		else
		{
			;
		}
	}
	//默认移动状态
	else if(InputFlg==0)
	{
		ZeroInFlg = 0;
		for(int i = 0;i<6;i++)
		{
			ZeroCmpBaisArr[i] =0.f;
		}
	}
	//其余状态保持不变
	else
	{
		;
	}

	for(int i = 0;i<6;i++)
	{
		InPutArr[i] -=ZeroCmpBaisArr[i];
	}
	return;

}

int OnlineFitRestCnts = 0;

void ForceControl_Init()
{
	for(int i=0;i<6;i++)
	{
		velLast[i] = 0.f;
		lenLast[i] = 0.f;
	}
}

void ResetForceControlRuntimeState(void)
{
	pos_init = 0;
	SinWavecount = 0;
	pos_adjust_init = 0;
	FinalVal = 0.f;
	ChargeFinished = 8.f;
	g_MoveDistance = 0.f;
	pos_adjust_count = 0;
	vel_clearflag = 0;
	g_CFuncflag = 0;
	OnlineFitRestCnts = 0;

	memset(velLast, 0, sizeof(velLast));
	memset(lenLast, 0, sizeof(lenLast));
	memset(jntPos, 0, sizeof(jntPos));
	memset(T_last, 0, sizeof(T_last));
	memset(T_init, 0, sizeof(T_init));
	memset(T_charged, 0, sizeof(T_charged));
	memset(jntPos_now, 0, sizeof(jntPos_now));
	memset(T_error_debug, 0, sizeof(T_error_debug));
	memset(Pos_input, 0, sizeof(Pos_input));
	memset(Force, 0, sizeof(Force));
	memset(Force_Dealed, 0, sizeof(Force_Dealed));
	memset(T_now, 0, sizeof(T_now));
	memset(T_next, 0, sizeof(T_next));
	memset(T_next2, 0, sizeof(T_next2));
	memset(T_ForForceCmp, 0, sizeof(T_ForForceCmp));
	memset(T_ForForceCmpBfr, 0, sizeof(T_ForForceCmpBfr));
	memset(JointForceCmp, 0, sizeof(JointForceCmp));
	memset(Forceinit_8, 0, sizeof(Forceinit_8));

	ForceControl_Init();
	ResetOnlineFitAll();
}
//输入：六维力数据  关节角数据（角度值）  力控指令
EulerAngles2 ForceControlFunZYX2(double Force[6], double pose[6], int g_CFuncflag)
{

	
	EulerAngles2 PoseControl;
	SinWavecount++;
	if (SinWavecount > 100000)
	{
		SinWavecount = 0;
	}

	double ForceArr[6] = { 0.f };
	for (int cnts = 0; cnts < 6; cnts++)
	{
		ForceArr[cnts] = Force[cnts]; //六维力传感器数据
	}

	MovMean(ForceArr);
	double pose_input_stable[6]={0.f};
	for(int i=0;i<6;i++)
	{
		pose_input_stable[i] = pose[i];
	}

	//角度制转换弧度制
	pose[3] = pose[3] * M_PI / 180.F;
	pose[4] = pose[4] * M_PI / 180.F;
	pose[5] = pose[5] * M_PI / 180.F;
	//将当前笛卡尔位姿转换为齐次变换矩阵
	pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
	
	pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_ForForceCmp);



#if 1
	/*!< ==================以下定义为Z轴在线移动算法============*/
	/*!< ==================以下定义为最小二乘拟合方法============*/
	//4.29添加在线标定
	//5.3添加在线标定
	//未标定时，进入在线标定
	if(FitDoneFlg==0)
	{
		if (1U == g_CFuncflag)
		{
			//進行標定
			OnlineCubicFitCalc(ForceAftFlt , T_now, pose, ForceAftCmp);
			//点位规划完成后，进行点位移动
			if(OnlineFitStateFlg==OnlineFitPointActState)
			{
				//如果第一个点位为0，则发送当前点位
				if(TargetPos.XYZ.x==0 && TargetPos.XYZ.y==0 && TargetPos.XYZ.z==0)
				{
					pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
					PoseControl = rotationMatrixToEulerXYZ2(T_now);
				}
				else
				{
					PoseControl.X   =TargetPos.XYZ.x;
					PoseControl.Y   =TargetPos.XYZ.y;
					PoseControl.Z   =TargetPos.XYZ.z;
					PoseControl.yaw = TargetPos.theta.yaw;
					PoseControl.pitch = TargetPos.theta.pitch;
					PoseControl.roll = TargetPos.theta.roll;
				}

			}
			else
			{
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
				PoseControl = rotationMatrixToEulerXYZ2(T_now);

			}

	
		}
		else if(0U == g_CFuncflag)
		{

			pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
			PoseControl = rotationMatrixToEulerXYZ2(T_now);
			ResetOnlineFitAll();
		}
		else
		{

			pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
			PoseControl = rotationMatrixToEulerXYZ2(T_now);

		}
	}
	else if(FitDoneFlg ==1)
	{
		//标定完成，执行运动控制，则说明插枪完成，在线辨识直接复位
		if(0U == g_CFuncflag)
		{
			OnlineFitRestCnts++;
			if(OnlineFitRestCnts>10)
			{
				OnlineFitRestCnts = 0;
				FitDoneFlg = 0;
				ResetOnlineFitAll();
			}
		}
		else
		{
			ThreePointCmp(ForceAftFlt,pose, ForceAftCmp);
			pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
			PoseControl = rotationMatrixToEulerXYZ2(T_now);
		}

		// ThreePointCmp(ForceAftFlt,pose, ForceAftCmp);
		// pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
		// PoseControl = rotationMatrixToEulerXYZ2(T_now);

	}
	else
	{
		printf("===辨识精度不达标====\n");
		// printf("当前辨识值=%f ,%f ,%f,%f,%f,%f\n",ForceAftCmp[0],ForceAftCmp[1],ForceAftCmp[2],ForceAftCmp[3],ForceAftCmp[4],ForceAftCmp[5]);
	}

	/*!< ==================以上定义为最小二乘拟合方法============*/

#else
	/*!< ==================以下定义为神经网络拟合方法============*/
		//3.23测试精度对于插枪的影响

	// printf("%f  %f  %f \n",CurPosData[0],CurPosData[1],CurPosData[2]);
 
	// printf("输入Pose %f , %f ,%f ,%f ,%f ,%f\n",pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]);
	double raw_force[3] = { 0.f }; double raw_torque[3] = { 0.f };
	double comp_force[3] = { 0.f }; double comp_torque[3] = { 0.f };
	raw_force[0] = ForceAftFlt[0];	raw_force[1] = ForceAftFlt[1];	raw_force[2] = ForceAftFlt[2];
	raw_torque[0] = ForceAftFlt[3];	raw_torque[1] = ForceAftFlt[4];	raw_torque[2] = ForceAftFlt[5];
	Net_compensation(raw_force, raw_torque, pose, comp_force, comp_torque);
	ForceAftCmp[0] = comp_force[0];
	ForceAftCmp[1] = comp_force[1];
	ForceAftCmp[2] = comp_force[2];
	ForceAftCmp[3] = comp_torque[0];
	ForceAftCmp[4] = comp_torque[1];
	ForceAftCmp[5] = comp_torque[2];
	OnlineFitDoneFlg = 1;
	//偏执是减0策略
	ForceZeroBaisCmp(g_CFuncflag,ForceAftCmp);


	/*!< ==================以上定义为神经网络拟合方法============*/
#endif

	//接入六维力数据
	for (int i = 0; i < 6; i++)
	{
		//Force[i] = ForceAftFlt[i];
		Force[i] = ForceAftCmp[i];
	}

	for (int i = 0; i < 3; i++)
	{
		if (Force[i] > -10.f && Force[i] < 10.f)
		{
			Force[i] = 0.f;
		}
	}
	for (int i = 3; i < 6; i++)
	{
		if (Force[i] > -3 && Force[i] < 3)
		{
			Force[i] = 0.f;
		}
	}
	ForceAftCmp[5] = 0.f; //Z轴力矩屏蔽
	// printf("当前辨识值=%f ,%f ,%f,%f,%f,%f\n",Force[0],Force[1],Force[2],Force[3],Force[4],Force[5]);
	
	if(FitDoneFlg==1)
	//if(1)
	{
				//开始插枪力控运行标志位
		if (1U == g_CFuncflag)
		{
			//位姿初始化
			if (0U == pos_init )
			{
				ChargeFinished = 0.f; //开始插枪标志位
				//进力控时刻，力控函数初始化
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pos_init++;
				//第一次不进入力控
				ForceControl_Init();
				g_MoveDistance = 0.f;//末端空间位移初始化
				pos_adjust_count = 0.f;
				vel_clearflag = 0;//速度清零标志位初始化
				PoseControl = rotationMatrixToEulerXYZ2(T_init);
			}
			else
			{
				
				double Fz = 20.f;
				//添加虚拟力
				//当前所受力矩较小时，正常按照行程给定虚拟力
				if (abs(Force[3]) < 6.f && abs(Force[4]) < 6.f)
				{
					// pos_adjust_count = 0.f;
					Fz = 60.f-(35.f/0.043f)*g_MoveDistance;
					if(Fz > 60.f)
					{
						Fz = 60.f;
					}
					if(Fz < 40.f)
					{
						Fz = 40.f;
					}
					
				}
				//当前所受力矩较大时，限制Z轴虚拟力大小，以调整力矩为主
				else
				{
					pos_adjust_count++;
					if (pos_adjust_count < 2.f) //超过阈值100ms以内力和力矩数据均为0
					{
						Force[0] = 0.f;
						Force[1] = 0.f;
						//Force[2] = 0.f;
						Force[3] = 0.f;
						Force[4] = 0.f;
						Force[5] = 0.f;
						for (int i = 0; i < 6; i++)
						{
							velLast[i] = 0.f;
						}
					}
				}
				//记录真实的Z轴受力用来判断插枪到为
				double RealForceZ = Force[2];
				Force[2] = Fz;
				//添加虚拟力矩-摇摆控制
				pos_init++;
				double max = 100.f / 2.f; //分母为频率
				if (pos_init > max)
				{
					pos_init = 1;
				}
				double MyControl = 100 * sin(2 * M_PI * pos_init / max);

				if (SinWavecount % 20 == 0)
				{
					FinalVal = MyControl;
				}
				//Force[4] += FinalVal; //摇摆插枪暂时未启用
				//六维力 力矩效果权重
				for (int i = 3; i < 6; i++)
				{
					Force[i] = Force[i] * 1.5f;//50.5
				}
				//六维力  力效果权重
				Force[0] = Force[0] * 0.f;
				Force[1] = Force[1] * 0.f;

				//力控部分  Z轴虚拟力
				//ForceControl(Force, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				ForceControl3(Force, Kx, Ky, Kz, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);

				//判断是否插枪到位 LenLast为下一个时刻位姿相对于初始位姿的坐标变化可以计算出位移量
				double MovedistanceX = T_now[0][3] - T_init[0][3];
				double MovedistanceY = T_now[1][3] - T_init[1][3];
				double MovedistanceZ = T_now[2][3] - T_init[2][3];
				double Movedistance = sqrt(MovedistanceX * MovedistanceX + MovedistanceY * MovedistanceY + MovedistanceZ * MovedistanceZ);
				
				g_MoveDistance = Movedistance;

				//速度清零策略
				if (Movedistance > 0.020 && vel_clearflag == 0)
				{
					for (int i = 0; i < 3; i++)
					{
						//velLast[i] = 0.f;
					}
					vel_clearflag = 1;
				}
				if (Movedistance > 0.035 && vel_clearflag == 1)
				{
					for (int i = 0; i < 3; i++)
					{
						//velLast[i] = 0.f;
					}
					vel_clearflag = 2;
				}
				if (Movedistance > 0.038 && vel_clearflag == 2)
				{
					for (int i = 0; i < 3; i++)
					{
						//velLast[i] = 0.f;
					}
					vel_clearflag = 3;
				}

				//插枪成功判断 若行程大于阈值2，且受到Z向阻力大于100 x y方向力小于20，则插枪成功，直接输出输入的位姿，力控计算结果不在输出  &&(Force[0] < 20.f)&&(Force[1] < 20.f)
				//if ((Movedistance > 0.043f) && (RealForceZ < -70.f))     
				if ((Movedistance > 0.04f)&& (RealForceZ < -70.f))   
				{
					ChargeFinished = 1.f;
					if(vel_clearflag == 3)
					{
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								T_charged[i][j] = T_now[i][j];
				
							}
						}
						
						vel_clearflag = 4;
					}
				}
				//插枪失败判断 若行程小于阈值1，且受到Z向阻力，则插枪失败，需搜索枪口
				if ((Movedistance < 0.01f) && (RealForceZ < -80.f))
				{
					ChargeFinished = 2.f;
				}

				//插枪中，直接输出力控计算的关节角
				if (ChargeFinished == 0.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
				}
				//插枪成功，直接输入输出的关节角，保持该位姿
				else if (ChargeFinished == 1.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_charged);
				}
				//插枪失败，直接输入输出的关节角，保持该位姿  后续修改为枪口搜索
				else if (ChargeFinished == 2.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_now);
				}
				else
				{

				}
			}
		}
		//开始插枪保持
		else if (2U == g_CFuncflag)
		{
			pos_init = 0U;
			ChargeFinished = 6.f; //插枪保持状态
			//位姿初始化
			if (0U == pos_adjust_init)
			{
				//进力控时刻，力控函数初始化
				//关节角转换齐次变换矩阵
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pos_adjust_init++;
				ForceControl_Init();
				PoseControl = rotationMatrixToEulerXYZ2(T_init);
				//第一次不进入力控
			}
			else if (pos_adjust_init < 600.f)//延时等待
			{
				pos_adjust_init++;
				PoseControl = rotationMatrixToEulerXYZ2(T_init);
			}
			else
			{
				pos_adjust_init++;
				if (pos_adjust_init > 2000.f)
				{
					pos_adjust_init = 2000.f;
				}
				//屏蔽三维力
				Force[0] = 0.f;
				Force[1] = 0.f;
				Force[2] = 0.f;
				Force[5] = 0.f;
				//力矩效果权重
				Force[3] = Force[3] * 0.f;
				Force[4] = Force[4] * 0.f;

				//力控部分  Z轴10Nm虚拟力
				//ForceControl(Force, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				ForceControl3(Force, Kx, Ky, Kz, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				//插枪中，直接输出力控计算的关节角
				if (pos_adjust_init < 600.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
				}
				//插枪成功，直接输入输出的关节角，保持该位姿
				else
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_now);
				}
			}
		}
		//开始拔枪力控运行标志位
		else if (3U == g_CFuncflag)
		{
			//位姿初始化
			if (pos_init == 0U && ChargeFinished == 6.f)
			{
				ChargeFinished = 3.f; //开始拔枪标志位
				//进力控时刻，力控函数初始化
				//关节角转换齐次变换矩阵
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pos_init++;
				//第一次不进入力控
				ForceControl_Init();
				pos_adjust_count = 0.f;
				vel_clearflag = 0;
				PoseControl = rotationMatrixToEulerXYZ2(T_now);
				g_MoveDistance = 0;
			}
			else
			{
				double Fz = 0;
				//添加虚拟力
				if(g_MoveDistance<0.03)
				{
					Fz = -30.f;
				}
				else
				{
					Fz = -50.f;
				}
				if (abs(Force[3]) < 8.f && abs(Force[4]) < 8.f)
				{
					//Fz = -80.f;
					pos_adjust_count = 0.f;
				}
				else
				{
					pos_adjust_count++;
					if (pos_adjust_count < 3.f) //超过阈值100ms以内力和力矩数据均为0
					{
						Force[0] = 0.f;
						Force[1] = 0.f;
						//Force[2] = 0.f;
						Force[3] = 0.f;
						Force[4] = 0.f;
						Force[5] = 0.f;
						for (int i = 0; i < 6; i++)
						{
							velLast[i] = 0.f;
						}
					}
				}
				Force[2] = Fz;
				//六维力 力矩效果放大
				//添加虚拟力矩-摇摆控制
				double max = 100.f / 2.f; //分母为频率
				if (pos_init > max)
				{
					pos_init = 1;
				}
				double MyControl = 50 * sin(2 * M_PI * pos_init / max);

				if (SinWavecount % 20 == 0)
				{
					FinalVal = MyControl;
				}
				Force[4] += FinalVal; //摇摆插枪暂时未启用


				for (int i = 3; i < 6; i++)
				{
					Force[i] = Force[i] * 3.5f;
				}
				Force[0] = 0.f;
				Force[1] = 0.f;
				Force[5] = 0.f;
				//力控部分  Z轴10Nm虚拟力
				//ForceControl(Force, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				ForceControl3(Force, Kx, Ky, Kz, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);

				pos_init++;
				if (pos_init > 1000)
				{
					pos_init = 1000;
				}

				//计算机械臂末端空间位移量 单位为m
				double MovedistanceX = T_now[0][3] - T_init[0][3];
				double MovedistanceY = T_now[1][3] - T_init[1][3];
				double MovedistanceZ = T_now[2][3] - T_init[2][3];
				double Movedistance = sqrt(MovedistanceX * MovedistanceX + MovedistanceY * MovedistanceY + MovedistanceZ * MovedistanceZ);
				
				g_MoveDistance = Movedistance;
				//拔枪成功判断 若行程大于阈值判断拔枪成功
				if (Movedistance > ChargeMaxMoveDistance)
				{
					ChargeFinished = 4.f;
					if(vel_clearflag == 0)
					{
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								T_charged[i][j] = T_now[i][j];
				
							}
						}
						
						vel_clearflag = 1;
					}
				}

				//拔枪中，直接输出力控计算的关节角
				if (ChargeFinished == 3.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
				}
				//拔枪成功，直接输入输出的关节角，保持该位姿
				else if (ChargeFinished == 4.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_charged);
				}
				//拔枪失败
				else
				{

				}
			}
		}
		else if(4U == g_CFuncflag)
		{
			if(0U == pos_init)
			{
				//进力控时刻，力控函数初始化
				//关节角转换齐次变换矩阵
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pos_init++;
				ForceControl_Init();
				PoseControl = rotationMatrixToEulerXYZ2(T_init);
			}
			else
			{
				Kx = 50;
				Ky = 50;
				Kz = 50;
				K = 0;
				Force[0] = 0.f;
				Force[1] = 0.f;
				Force[3] = 0.f;
				Force[4] = 0.f;
				Force[5] = 0.f;

				ForceControl3(Force, Kx, Ky, Kz, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				pos_init++;
				if (pos_init > 1000)
				{
					pos_init = 1000;
				}
				PoseControl = rotationMatrixToEulerXYZ2(T_next);
			}
			// **********************1. 输入当前机械臂起点XYZ
			// double x0 = pose[0];
			// double y0 = pose[1];
			// double z0 = pose[2];
			// double step = 0.001f;    // 每次向外移动 1mm
			// double len = sqrt(x0*x0 + y0*y0 + z0*z0);
		
			// // 单位方向向量
			// double ex = x0 / len;
			// double ey = y0 / len;
			// double ez = z0 / len;

			// double x = x0 + ex * step ;
			// double y = y0 + ey * step ;
			// double z = z0 + ez * step ;

			// PoseControl.X = x;
			// PoseControl.Y = y;
			// PoseControl.Z = z;
			// PoseControl.roll = pose_input_stable[3];
			// PoseControl.pitch = pose_input_stable[4];
			// PoseControl.yaw = pose_input_stable[5];
			//******************************************************** */
			pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);

			double DirX = T_init[0][2];
			double DirY = T_init[1][2];
			double DirZ = T_init[2][2];

			double step_mm = 0.0001;
			double xx = pose[0] + DirX * step_mm;
			double yy = pose[1] + DirY * step_mm;
			double zz = pose[2] + DirZ * step_mm;

			PoseControl.X = xx;
			PoseControl.Y = yy;
			PoseControl.Z = zz;
			PoseControl.roll = pose_input_stable[3];
			PoseControl.pitch = pose_input_stable[4];
			PoseControl.yaw = pose_input_stable[5];

			printf("法向量测试 \n ");


		}
		else if(5U == g_CFuncflag)
		{
			// 静态局部变量，确保多次调用保持状态
			static uint32_t zero_cnt = 0;
			static float openCover_Flag = 0.f;   // 1:正向力控 2:等待回退 3:反向回退 4:完成
			static double Force_init[6];
			static uint32_t pos_init = 0;
			static uint8_t task_completed = 0;   // 任务完成标志

			if(0U == pos_init)
			{
				// 进入力控：记录初始位姿、力零点、初始化力控
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
				pos_init++;
				ForceControl_Init();
				PoseControl = rotationMatrixToEulerXYZ2(T_init);
				openCover_Flag = 1.f;
				ChargeFinished = 15.f;
				zero_cnt = 0;
				opevcover = 0;
				task_completed = 0;
				for (int i = 0; i < 6; i++)
					Force_init[i] = Force[i];
				
			}
			else if(task_completed)
			{
				// 任务已完成，保持初始位姿，不再执行力控
				PoseControl = rotationMatrixToEulerXYZ2(T_init);
				// 如需重新触发，可在外部复位 pos_init 和 task_completed
			}
			else
			{
				// 刚度设置（保持原逻辑）
				Kx = 0; Ky = 0; Kz = 0; K = 0;

				double Fz = 10.0;   // 降低开盖基准力（原30过大）
				// 力零点偏移
				for (int i = 0; i < 6; i++)
				{
					Force[i] = Force[i] - Force_init[i];
					if (i !=  2)
						Force[i] = 0.0;

					//Force_overall[i]=Force[i];
				}

				if (Force[2]>100||Force[2]<-100)
				{
					ChargeFinished = 17.f;
				}

				// 正向开盖力控
				if (openCover_Flag == 1.f)
				{
					double forceCmd = Fz + Force[2];
					// 限幅，防止冲击过大
					if (forceCmd > 40.0) forceCmd = 40.0;
					if (forceCmd < -40.0) forceCmd = -40.0;
					Force[2] = forceCmd;
				}
				// 反向回退力控
				else if (openCover_Flag == 3.f)
				{
					double forceCmd = -Fz;
					if (forceCmd > 40.0) forceCmd = 40.0;
					if (forceCmd < -40.0) forceCmd = -40.0;
					Force[2] = forceCmd;
				}

				// 执行力控
				ForceControlfor5U(Force, Kx, Ky, Kz, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);

				pos_init++;
				if (pos_init > 1000) pos_init = 1000;

				// 计算当前位置相对于初始位置的距离
				double MovedistanceX = T_now[0][3] - T_init[0][3];
				double MovedistanceY = T_now[1][3] - T_init[1][3];
				double MovedistanceZ = T_now[2][3] - T_init[2][3];
				double Movedistance = sqrt(MovedistanceX*MovedistanceX + MovedistanceY*MovedistanceY + MovedistanceZ*MovedistanceZ);

				// 正向开盖到位检测（速度接近零且移动超过1.5cm）
				if (openCover_Flag == 1.f)
				{
					if (fabs(velLast[2]) < 0.001f && Movedistance > 0.015f)
					{
						zero_cnt++;
						if (zero_cnt > 3)
						{
							openCover_Flag = 2.f;   // 进入等待力卸载状态
							zero_cnt = 0;
						}
					}
					else
						zero_cnt = 0;
				}

				// 等待力卸载完成
				if (openCover_Flag == 2.f)
				{
					if (fabs(Force[2]) < 10.0f)
					{
						if(opevcover == 0)
						{
							for (int i = 0; i < 4; i++)
							{
								for (int j = 0; j < 4; j++)
								{
									T_opencover[i][j] = T_now[i][j];
								}
							}
							opevcover = 1;
						}
						openCover_Flag = 3.f;   // 开始反向回退
					}
						
				}

				// 计算当前位置相对于初始位置的距离
				double MovedistanceX2 = T_now[0][3] - T_opencover[0][3];
				double MovedistanceY2 = T_now[1][3] - T_opencover[1][3];
				double MovedistanceZ2 = T_now[2][3] - T_opencover[2][3];
				double Movedistance2 = sqrt(MovedistanceX2*MovedistanceX2 + MovedistanceY2*MovedistanceY2 + MovedistanceZ2*MovedistanceZ2);

				// 回退到位检测
				uint8_t skip_pose = 0;
				if (openCover_Flag == 3.f)
				{
					// 放宽阈值至2mm，提高稳定性  Movedistance < 0.002f
					if (Movedistance2 > 0.029f)
					{
						if(opevcover == 1)
						{
							for (int i = 0; i < 4; i++)
							{
								for (int j = 0; j < 4; j++)
								{
									T_opencover2[i][j] = T_now[i][j];
								}
							}
							opevcover = 2;
						}
						task_completed = 1;
						openCover_Flag = 4.f;   // 标记完成
						ChargeFinished = 16.f;
						PoseControl = rotationMatrixToEulerXYZ2(T_opencover2);
						skip_pose = 1;          // 跳过后续的T_next赋值
					}
				}

				if (!skip_pose)
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
			}
		}
		else if(6U == g_CFuncflag)   // 闭盖力控（优化版）
		{
			// 静态局部变量，保持状态
			static uint32_t close_state = 0;      // 0:初始化, 1:正向闭盖, 2:完成
			static uint32_t stable_cnt = 0;
			static double   close_force = 30.0;   // 闭合施加的力 (N)
			static double   Force_init[6];        // 初始力零点
			static uint32_t pos_init = 0;         // 初始位姿已记录标志
			static uint8_t  task_done = 0;        // 任务完成标志（保持位姿）

			if(0U == pos_init)
			{
				// 1. 记录初始位姿（力控基准）
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);

				// 2. 记录初始力传感器值（力零点）
				for (int i = 0; i < 6; i++)
					Force_init[i] = Force[i];

				// 3. 初始化力控相关
				ForceControl_Init();
				PoseControl = rotationMatrixToEulerXYZ2(T_init);

				close_state = 1;      // 进入正向闭盖阶段
				stable_cnt = 0;
				task_done = 0;
				pos_init = 1;
			}
			else if(task_done)
			{
				// 任务已完成：保持当前位置，不再执行力控
				PoseControl = rotationMatrixToEulerXYZ2(T_now);
				// 可选：如果上层需要自动退出，可在此复位 pos_init 和 close_state
				// 但建议由外部逻辑将 g_CFuncflag 置为其他值
			}
			else
			{
				// 刚度设置：仅保留 Z 向柔顺，其他方向刚度为0
				Kx = 0; Ky = 0; Kz = 0;
				K  = 0;

				// 力零点补偿（减去初始力，得到实际外力）
				double actual_force[6];
				for (int i = 0; i < 6; i++)
				{
					actual_force[i] = Force[i] - Force_init[i];
					if (fabs(actual_force[i]) < 1.0)
						actual_force[i] = 0.0;
				}

				// ---------- 阶段1：正向闭盖（施加向下力）----------
				if (close_state == 1)
				{
					// 期望力误差：向下力 = close_force - 实际Z向力
					// 假设 Force[2] 正方向向上，则误差为正表示需要向下加力
					double fz_cmd = close_force + actual_force[2];
					// 力命令限幅，防止冲击（最大50N）
					if (fz_cmd > 50.0) fz_cmd = 50.0;
					if (fz_cmd < -50.0) fz_cmd = -50.0;
					Force[2] = fz_cmd;

					// 执行力控（内部使用了阻尼C、质量m等参数，需提前设好C=1500左右）
					ForceControlfor5U(Force, Kx, Ky, Kz, K, dt, velLast, lenLast,
								T_init, T_last, T_now, T_next);

					// 计算从初始位姿移动的总距离（欧氏距离）
					double dx = T_now[0][3] - T_init[0][3];
					double dy = T_now[1][3] - T_init[1][3];
					double dz = T_now[2][3] - T_init[2][3];
					double moved = sqrt(dx*dx + dy*dy + dz*dz);

					// 到位条件：移动距离超过闭合行程（0.04m） 或 Z向受力超过30N
					if (moved > 0.04 || fabs(actual_force[2]) < -30.0)
					{
						stable_cnt++;
						if (stable_cnt > 3)   // 连续3帧满足，确认为到位
						{
							close_state = 2;   // 进入完成阶段
							stable_cnt = 0;
						}
					}
					else
					{
						stable_cnt = 0;
					}

					// 输出当前控制位姿
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
				}

				// ---------- 阶段2：闭盖完成，保持当前位姿并退出力控 ----------
				if (close_state == 2)
				{
					// 不再施加虚拟力，直接输出当前实际位姿（或初始位姿+移动距离）
					PoseControl = rotationMatrixToEulerXYZ2(T_now);
					task_done = 1;          // 标记任务完成，后续周期保持
				}
			}
		}
		else if (8U == g_CFuncflag) // 力控归枪  ChargeFinished:10正在归枪，11，归枪成功，12归枪失败
		{
			if (0U == pos_init)
			{
				// 进力控时刻，力控函数初始化
				// 关节角转换齐次变换矩阵
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pos_init++;
				ForceControl_Init();

				g_MoveDistance = 0.f; // 末端空间位移初始化
				pos_adjust_count = 0.f;
				vel_clearflag = 0; // 速度清零标志位初始化
				PoseControl = rotationMatrixToEulerXYZ2(T_init);

				ChargeFinished=0+10;

				for(int i=0;i<6;i++)
					Forceinit_8[i]=Force[i];
			}
			else
			{
				
				double Fz = 0.f;
				// 添加虚拟力
				// 当前所受力矩较小时，正常按照行程给定虚拟力
				double Forcereal[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
				for(int i=0;i<6;i++)
				{
					Forcereal[i]=Force[i]-Forceinit_8[i];
					//Force_overall[i]=Forcereal[i];
					//Force_overall[i]=Forceinit_8[i];
				}
				if (abs(Forcereal[3]) < 20.f && abs(Forcereal[4]) < 20.f)
				{
					// pos_adjust_count = 0.f;
					Fz = 30.f;
					
				}
				// 当前所受力矩较大时，限制Z轴虚拟力大小，以调整力矩为主
				// else
				// {
				// 	pos_adjust_count++;
				// 	if (pos_adjust_count < 2.f) // 超过阈值100ms以内力和力矩数据均为0
				// 	{
				// 		Forcereal[0] = 0.f;
				// 		Forcereal[1] = 0.f;
						
				// 		Forcereal[3] = 0.f;
				// 		Forcereal[4] = 0.f;
				// 		Forcereal[5] = 0.f;
					
				// 	}
				// }
				// 记录真实的Z轴受力用来判断插枪到为
				double RealForceZ = Forcereal[2];
				Forcereal[2] = Fz;

				// 添加虚拟力矩-摇摆控制
				pos_init++;
				
				// 六维力 力矩效果权重
				for (int i = 3; i < 6; i++)
				{
					Forcereal[i] = Forcereal[i] * 1.f;
				}
				// 六维力  力效果权重
				Forcereal[0] = Forcereal[0];
				Forcereal[1] = Forcereal[1];

				ForceControl3(Forcereal, 0.2*Kx, 0.2*Ky, 0.2*Kz,0.2*K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				
				// 判断是否插枪到位 LenLast为下一个时刻位姿相对于初始位姿的坐标变化可以计算出位移量
				double MovedistanceX = T_now[0][3] - T_init[0][3];
				double MovedistanceY = T_now[1][3] - T_init[1][3];
				double MovedistanceZ = T_now[2][3] - T_init[2][3];
				double Movedistance = sqrt(MovedistanceX * MovedistanceX + MovedistanceY * MovedistanceY + MovedistanceZ * MovedistanceZ);
				
				g_MoveDistance = Movedistance;
				//movedist_overall=Movedistance;
				// 插枪成功判断 若行程大于阈值2，且受到Z向阻力大于100 x y方向力小于20，则插枪成功，直接输出输入的位姿，力控计算结果不在输出  &&(Force[0] < 20.f)&&(Force[1] < 20.f)
				// if ((Movedistance > 0.043f) && (RealForceZ < -70.f))
				if ((Movedistance > 0.04f)&&RealForceZ<-60)
				{
					ChargeFinished = 1.f+10;
				
					for (int i = 0; i < 4; i++)
					{
						for (int j = 0; j < 4; j++)
						{
							T_charged[i][j] = T_now[i][j];
							
						}
					}

				}
				else if(Movedistance > 0.1f)
				{
					ChargeFinished = 1.f+10;
					for (int i = 0; i < 4; i++)
					{
						for (int j = 0; j < 4; j++)
						{
							T_charged[i][j] = T_now[i][j];
						}
					}
				}
				else if(fabs(Forcereal[2])>150||fabs(Forcereal[3]>20||fabs(Forcereal[4])>20||fabs(Forcereal[5])>20)||fabs(Forcereal[0])>50||fabs(Forcereal[1]>50))//归枪失败
				{
					ChargeFinished = 2.f+10;
				}
				
		
				if ((Movedistance < 0.01f) && (RealForceZ < -80.f))//插枪失败
				{
					ChargeFinished = 2.f+10;
				}
				if (ChargeFinished == 10.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
				}
					else if (ChargeFinished == 11.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_charged);
				}
					else if (ChargeFinished == 12.f)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_now);
				}
				else{}
			
				printf("8U状态:=%.2f\n",
					ChargeFinished);

			}
		}
		else if(9U == g_CFuncflag)//力控取枪
		{
			if (pos_init == 0U)
			{
				ChargeFinished = 3.f+10; // 开始拔枪标志位
				// 进力控时刻，力控函数初始化
				// 关节角转换齐次变换矩阵
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_init);
				pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_last);
				pos_init++;
				// 第一次不进入力控
				ForceControl_Init();
				pos_adjust_count = 0.f;
				vel_clearflag = 0;
				PoseControl = rotationMatrixToEulerXYZ2(T_now);
				g_MoveDistance = 0;

				for(int i=0;i<6;i++)
				Forceinit_9[i]=Force[i];
			}
			else
			{
				for(int i=0;i<6;i++)
				{
					Force[i]=Force[i]-Forceinit_9[i];
				}

				double Fz = 0;
				double Fzworld=0;
				// 添加虚拟力
				if (g_MoveDistance < 0.03)
				{
					Fz = -30.f;
				}
				else
				{
					Fz = -50.f;
				}

				if(g_MoveDistance<0.01)
				{
					Fzworld=20;
				}

				Force[2] = Fz;
			

				for (int i = 3; i < 6; i++)
				{
					Force[i] = Force[i] * 50.3f;
				}
		
				// 力控部分  Z轴10Nm虚拟力
				// ForceControl(Force, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next);
				ForceControlfor9U(Force, Kx, Ky, Kz, K, dt, velLast, lenLast, T_init, T_last, T_now, T_next,Fzworld);

				pos_init++;
				if (pos_init > 1000)
				{
					pos_init = 1000;
				}

				// 计算机械臂末端空间位移量 单位为m
				double MovedistanceX = T_now[0][3] - T_init[0][3];
				double MovedistanceY = T_now[1][3] - T_init[1][3];
				double MovedistanceZ = T_now[2][3] - T_init[2][3];
				double Movedistance = sqrt(MovedistanceX * MovedistanceX + MovedistanceY * MovedistanceY + MovedistanceZ * MovedistanceZ);
				//movedist_overall=Movedistance;
				g_MoveDistance = Movedistance;
				// 拔枪成功判断 若行程大于阈值判断拔枪成功
				if (Movedistance > 0.05)
				{
					ChargeFinished = 4.f+10;
					if (vel_clearflag == 0)
					{
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								T_charged[i][j] = T_now[i][j];
							}
						}

						vel_clearflag = 1;
					}
				}
				else if(Movedistance<0.01&&Force[2]>100)
				{
					ChargeFinished=5.f+10;
				}
				else if(fabs(Force[2])>150||fabs(Force[3]>20||fabs(Force[4])>20||fabs(Force[5])>20)||fabs(Force[0])>50||fabs(Force[1]>50))//归枪失败
				{
					ChargeFinished = 5.f+10;
				}

				// 拔枪中，直接输出力控计算的关节角
				if (ChargeFinished == 3.f+10)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_next);
				}
				// 拔枪成功，直接输入输出的关节角，保持该位姿
				else if (ChargeFinished == 4.f+10)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_charged);
				}
				// 拔枪失败
				else if(ChargeFinished==5.f+10)
				{
					PoseControl = rotationMatrixToEulerXYZ2(T_now);
				}
				else{}

				printf("9U状态:=%.2f\n",
					ChargeFinished);
			}



		}
		else
		{
			pos_init = 0U;
			pos_adjust_init = 0U;
			ChargeFinished = 8.f;
			pose_to_transform(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], T_now);
			PoseControl = rotationMatrixToEulerXYZ2(T_now);
			
		}
	}
	else
	{
		;
	}


	return PoseControl;

}
