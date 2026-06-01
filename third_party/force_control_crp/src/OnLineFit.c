
#include"OnLineFit.h"

//g++ -std=c++11 -I ../cpp/include -L . it_robotposTEST4.cpp ForceControl.c  IdentifyCommon.c OnLineFit.c -o it_robotposTEST4 -lRobotService -lpthread -ldl
//./it_robotposTEST4 192.168.1.133 eth1
// sudo ip link set eth0 up
// sudo ip link set eth1 up 
/*！<==============全局变量=====================*/
#define f32G  9.80665f/*!< 重力加速度 */
int AutoFitErrCode = 0;/*!< 在线辨识故障标志位 */
int DelayCnts = 0;/*!< 系统延迟计数器 */

/*!< 状态机0::初始化上电状态，1：点位生成状态，2：机械臂点位执行状态，3：数据采集状态，4：辨识，5：补偿*/
int OnlineFitStateFlg = 0;

/*!< 辨识完成标志位0:默认状态，未辨识  1：辨识完成  -1：辨识失败或误差过大*/
int FitDoneFlg = 0;
/*!< 辨识完成计数器*/
int FitDoneComfirmCnts = 0;
/*！<==============在线参数辨识（点位规划）=====================*/
PosData PlanPosTbl[PLAN_POINTS_NUMS];
double CmpStartMat[4][4]; /*!< 起始变化矩阵 */

/*！<==============在线参数辨识（点位执行）=====================*/
int PlanPosActIndex = 0;/*!< 当前执行下标 */

/*！<==============在线参数辨识（执行到位判断）=====================*/
PosData TargetPos;/*!<目标位姿 */
PosData CurrentPos;/*!<当前位姿 */
int MovConfmCnts = 0;/*!<判断到位标志位 */
PosData PlanStartPos;/*!<起点点位位姿 */
/*！<==============在线参数辨识（数据采集）=====================*/
DataSet DataCollectionSet;  

/*!<数据滤波 */
double DataCollectionForceFltArr[6] = { 0, 0, 0, 0, 0, 0 };
double DataCollectionCurPosArr[6] = { 0, 0, 0, 0, 0, 0 };
double DataCollectionMatrixFltArr[4][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
int DataCollectionFltCnts = 0;


/*！<==============在线参数辨识（参数辨识）=====================*/
double MovDist;/*!<当前点位Z轴位移*/
/*!<一次项拟合系数*/
double B_Tbl[6] = { 0.f };
double K_Tbl[6] = { 0.f };



/*！<==============在线参数辨识（通用函数-延迟函数）=====================*/
int OSdelayCnts(int* OSCnts, int InputDelayCnts)
{
	*OSCnts = *OSCnts + 1;
	if (*OSCnts >= InputDelayCnts)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/*！<==============在线参数辨识（Z轴点位规划）=====================*/
void ZaxisPointPlan(double BaisPosMat[4][4])
{
	double NormDirX = BaisPosMat[0][2];
	double NormDirY = BaisPosMat[1][2];
	double NormDirZ = BaisPosMat[2][2];

	double Z_MoveDist = Z_DIR_MOVE_STEP_MM/1000.f;

	for(int i = 0;i<PLAN_POINTS_NUMS;i++)
	{
		PosData CurPlan;
		if(i==0)
		{
			CurPlan.theta = PlanStartPos.theta;
			CurPlan.XYZ = PlanStartPos.XYZ;
		}
		else
		{
			CurPlan.theta = PlanStartPos.theta;
			CurPlan.XYZ.x = PlanStartPos.XYZ.x + Z_MoveDist*i*NormDirX;
			CurPlan.XYZ.y = PlanStartPos.XYZ.y + Z_MoveDist*i*NormDirY;
			CurPlan.XYZ.z = PlanStartPos.XYZ.z + Z_MoveDist*i*NormDirZ;
		}
		/*!< 依据姿态生成齐次变化矩阵 */
		PlanPosTbl[i] = CurPlan;
	}

	return;
}
/*！<==============在线参数辨识（三点式拟合算法）=====================*/
void OnlineThreePointFitFunc(double StartVal, double MidVal, double EndVal, double ZDistence, int Index)
{

	double Diff_Start = MidVal - StartVal;
	double Diff_End = EndVal - MidVal;
	double AbsDiff_Start = fabs(Diff_Start);
	double AbsDiff_End = fabs(Diff_End);

	double SmallOne = AbsDiff_Start > AbsDiff_End ? AbsDiff_End : AbsDiff_Start;
	double BigOne = AbsDiff_Start > AbsDiff_End ? AbsDiff_Start : AbsDiff_End;

	double SimilarRate = SmallOne / BigOne;


	if (SimilarRate <= 0.8f)
	{
		// printf("Diff_End= %f,ZDistence = %f\n",Diff_End,ZDistence);
		//斜率相似率低于80%，以后半段为准
		K_Tbl[Index] = (Diff_End) / ZDistence;
		B_Tbl[Index] = MidVal- K_Tbl[Index] * ZDistence;
	}
	else
	{
		B_Tbl[Index] = StartVal;
		K_Tbl[Index] = (EndVal - StartVal) / ZDistence;
	}

}
/*！<==============计算Z轴移动距离函数=====================*/
double Cal_points_distance(PosData Cur_pos,PosData Start_pos){
	double MovDist = sqrtf(	
		(Cur_pos.XYZ.x - Start_pos.XYZ.x)*(Cur_pos.XYZ.x - Start_pos.XYZ.x)+
		(Cur_pos.XYZ.y- Start_pos.XYZ.y)*(Cur_pos.XYZ.y- Start_pos.XYZ.y)+
		(Cur_pos.XYZ.z - Start_pos.XYZ.z)*(Cur_pos.XYZ.z - Start_pos.XYZ.z));
		return MovDist;
}
/*！<==============系数拟合（三点式拟合算法）=====================*/
void ThreePointFit(DataSet InputSet)
{
	if(InputSet.count<2)
	{
		printf("====数据存储数量小于3个，推出拟合===\n");
		return;
	}
	OnlineThreePointFitFunc(InputSet.data[0].F_meas.x,InputSet.data[1].F_meas.x,InputSet.data[2].F_meas.x,Z_DIR_MOVE_STEP_MM,0);
	OnlineThreePointFitFunc(InputSet.data[0].F_meas.y,InputSet.data[1].F_meas.y,InputSet.data[2].F_meas.y,Z_DIR_MOVE_STEP_MM,1);
	OnlineThreePointFitFunc(InputSet.data[0].F_meas.z,InputSet.data[1].F_meas.z,InputSet.data[2].F_meas.z,Z_DIR_MOVE_STEP_MM,2);
	OnlineThreePointFitFunc(InputSet.data[0].T_meas.x,InputSet.data[1].T_meas.x,InputSet.data[2].T_meas.x,Z_DIR_MOVE_STEP_MM,3);
	OnlineThreePointFitFunc(InputSet.data[0].T_meas.y,InputSet.data[1].T_meas.y,InputSet.data[2].T_meas.y,Z_DIR_MOVE_STEP_MM,4);
	OnlineThreePointFitFunc(InputSet.data[0].T_meas.z,InputSet.data[1].T_meas.z,InputSet.data[2].T_meas.z,Z_DIR_MOVE_STEP_MM,5);
}
/*！<==============补偿函数（三点式拟合算法）=====================*/
void ThreePointCmp(double InputForce[6],double InputPos[6], double OutputForece[6])
{
	 MovDist = sqrtf(	
	(InputPos[0] - PlanStartPos.XYZ.x)*(InputPos[0] - PlanStartPos.XYZ.x)+
	(InputPos[1] - PlanStartPos.XYZ.y)*(InputPos[1] - PlanStartPos.XYZ.y)+
	(InputPos[2] - PlanStartPos.XYZ.z)*(InputPos[2] - PlanStartPos.XYZ.z))*1000.f;
	OutputForece[2] = InputForce[2]-(MovDist*K_Tbl[2]+B_Tbl[2]);
	// printf("原始：%f   计算%f\n",InputForce[2],(MovDist*K_Tbl[2]+B_Tbl[2]));
	for(int i=0;i<6;i++)
	{
		OutputForece[i] = InputForce[i]-(MovDist*K_Tbl[i]+B_Tbl[i]);
	}
}


/*！<==============在线参数辨识（执行到位判断）=====================*/
/*!< 判断是否下发到位*/
int JudgeActDone(PosData InputTargetPos, PosData CurPos)
{
	/*!< 偏差精度(角度和XYZ分开统计)*/
	double AngleAccuracy = 0.01f;
	double XYZAccuracy = 0.001f;

	double ErrArr[6] = { fabs(InputTargetPos.theta.pitch - CurPos.theta.pitch),
						 fabs(InputTargetPos.theta.roll - CurPos.theta.roll),
						 fabs(InputTargetPos.theta.yaw - CurPos.theta.yaw),
						 fabs(InputTargetPos.XYZ.x - CurPos.XYZ.x),
						 fabs(InputTargetPos.XYZ.y - CurPos.XYZ.y),
						 fabs(InputTargetPos.XYZ.z - CurPos.XYZ.z) };

	if ((ErrArr[0] <= AngleAccuracy) &&
		(ErrArr[1] <= AngleAccuracy) &&
		(ErrArr[2] <= AngleAccuracy) &&
		(ErrArr[3] <= XYZAccuracy) &&
		(ErrArr[4] <= XYZAccuracy) &&
		(ErrArr[5] <= XYZAccuracy)
		)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
/*！<==============在线参数辨识（数据采集）=====================*/
// 初始化数据集
void dataset_init(int initial_capacity, DataSet *dataset) {

	dataset->data = (SampleData*)malloc(initial_capacity * sizeof(SampleData));
	dataset->count = 0;
	dataset->capacity = initial_capacity;
	return;
}

// 向数据集添加数据点
void dataset_add_point(DataSet* dataset, SampleData point) {
	if (dataset->count >= dataset->capacity) {
		// 扩容
		int new_capacity = dataset->capacity * 2;
		SampleData* new_data = (SampleData*)realloc(
			dataset->data, new_capacity * sizeof(SampleData));

		if (new_data == NULL) {
			//fprintf(stderr, "Failed to reallocate memory for dataset\n");
			AutoFitErrCode = 1;
			return;
		}

		dataset->data = new_data;
		dataset->capacity = new_capacity;
	}

	dataset->data[dataset->count] = point;
	dataset->count++;
}

/*!< 用于参数辨识的数据采集 */
int  FitDataCollect(double ForceArr[6], double T_CurMatrix[4][4],double CurPosArr[6])
{
	/*!< 数据采集满足条件后取均值 */
	if (DataCollectionFltCnts < DATA_COLLECTION_FILTER_COUNTS)
	{
		DataCollectionFltCnts++;
		for (int i = 0; i < 6; i++)
		{
			DataCollectionForceFltArr[i] += ForceArr[i];
			DataCollectionCurPosArr[i]+= CurPosArr[i];
		}
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				DataCollectionMatrixFltArr[i][j] += T_CurMatrix[i][j];
			}
		}
		return 0;
	}
	else
	{

		SampleData DataPoint;
		/*!< 满足条件，取六维力传感器的均值，并清零*/
		DataPoint.F_meas.x = DataCollectionForceFltArr[0] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.F_meas.y = DataCollectionForceFltArr[1] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.F_meas.z = DataCollectionForceFltArr[2] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.T_meas.x = DataCollectionForceFltArr[3] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.T_meas.y = DataCollectionForceFltArr[4] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.T_meas.z = DataCollectionForceFltArr[5] / (double)DATA_COLLECTION_FILTER_COUNTS;

		DataPoint.Pos[0] = DataCollectionCurPosArr[0] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.Pos[1] = DataCollectionCurPosArr[1] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.Pos[2] = DataCollectionCurPosArr[2] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.Pos[3] = DataCollectionCurPosArr[3] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.Pos[4] = DataCollectionCurPosArr[4] / (double)DATA_COLLECTION_FILTER_COUNTS;
		DataPoint.Pos[5] = DataCollectionCurPosArr[5] / (double)DATA_COLLECTION_FILTER_COUNTS;

		for (int i = 0; i < 6; i++)
		{
			DataCollectionForceFltArr[i] = 0.f;
			DataCollectionCurPosArr[i] = 0.f;
		}
		/*!< 满足条件，取齐次变化矩阵的数据均值，并清零*/
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				DataCollectionMatrixFltArr[i][j] = DataCollectionMatrixFltArr[i][j] / (double)DATA_COLLECTION_FILTER_COUNTS;
				DataPoint.R_SB.data[i][j] = DataCollectionMatrixFltArr[i][j];
				DataCollectionMatrixFltArr[i][j] = 0.f;
			}
		}

		/*!< 向数据集添加数据*/
		dataset_add_point(&DataCollectionSet, DataPoint);
		printf("数据添加完毕，共%d个数据\n",DataCollectionSet.count);
		printf("存储位姿数据： %f,%f,%f,%f,%f,%f\n",DataCollectionSet.data[DataCollectionSet.count-1].Pos[0],DataCollectionSet.data[DataCollectionSet.count-1].Pos[1],
			DataCollectionSet.data[DataCollectionSet.count-1].Pos[2],DataCollectionSet.data[DataCollectionSet.count-1].Pos[3],DataCollectionSet.data[DataCollectionSet.count-1].Pos[4],
			DataCollectionSet.data[DataCollectionSet.count-1].Pos[5]);
		DataCollectionFltCnts = 0;

		return 1;
	}
}

/*！<==============在线参数辨识（复位）=====================*/
void ResetOnlineFitAll()
{
	/*!< 完全清除，复位*/
	OnlineFitStateFlg = 0;
	/*!< 清零辨识起点点位*/
	memset(CmpStartMat, 0, sizeof(CmpStartMat));
	/*!< 清零点位规划表格*/
	memset(PlanPosTbl, 0, sizeof(PosData) * PLAN_POINTS_NUMS);

	/*!< 清零点位执行下标*/
	PlanPosActIndex = 0;

	/*!< 清除数据采集set*/
	DataCollectionSet.count = 0;

	/*!<清除当前点位Z轴位移*/
	MovDist = 0;

	/*!< 清除完成辨识标志位*/
	FitDoneFlg = 0;


	return;
}


/*！<==============在线参数辨识（状态机）=====================*/
void OnlineCubicFitCalc(double InputSixForce[6],double CurExgMat[4][4] ,double InputPos[6], double OutputCmpzForece[6])
{
	// printf("==========进入状态机============\n");
	/*!< 状态跳转*/
	switch (OnlineFitStateFlg)
	{
	case OnlineFitInitState:
		if (1 == OSdelayCnts(&DelayCnts, 20))
		{
			DelayCnts = 0;
			// /*!< 初始上电，进行清零操作，所有变量函数清理*/
			// /*!< 记录初始点位，后续用作距离计算*/
			memcpy(CmpStartMat, CurExgMat, sizeof(CmpStartMat)); /*!< 记录当前点位位姿（0~2PI）*/

			dataset_init(PLAN_POINTS_NUMS, &DataCollectionSet); /*!< 初始化数据采集结构体*/

			PlanStartPos.XYZ.x = InputPos[0];	PlanStartPos.XYZ.y = InputPos[1];	PlanStartPos.XYZ.z= InputPos[2];
			PlanStartPos.theta.roll= InputPos[3]*180.f/M_PI;	PlanStartPos.theta.pitch= InputPos[4]*180.f/M_PI;PlanStartPos.theta.yaw= InputPos[5]*180.f/M_PI;
	
			OnlineFitStateFlg = OnlineFitPointPlanState;
		}
		else
		{
			;
		}
		break;
	case OnlineFitPointPlanState:
		if (1 == OSdelayCnts(&DelayCnts, 20))
		{
			DelayCnts = 0;

			ZaxisPointPlan(CmpStartMat);
			for(int i = 0;i<PLAN_POINTS_NUMS;i++)
			{
				printf("%f ,%f ,%f ,%f ,%f ,%f\n",PlanPosTbl[i].XYZ.x,PlanPosTbl[i].XYZ.y,PlanPosTbl[i].XYZ.z,PlanPosTbl[i].theta.roll,PlanPosTbl[i].theta.pitch,PlanPosTbl[i].theta.yaw);
			}
			printf("========点位规划成功======\n");
			OnlineFitStateFlg = OnlineFitPointActState;
		}
		else
		{
			;
		}
		break;
	case OnlineFitPointActState:
		
		/*!< 点位执行（放主函数执行）*/
		TargetPos = PlanPosTbl[PlanPosActIndex];

		/*!< 对比当前位姿*/
		CurrentPos.XYZ.x = InputPos[0];	CurrentPos.XYZ.y = InputPos[1];	CurrentPos.XYZ.z= InputPos[2];
		CurrentPos.theta.roll= InputPos[3]*180.f/M_PI;	CurrentPos.theta.pitch= InputPos[4]*180.f/M_PI;CurrentPos.theta.yaw= InputPos[5]*180.f/M_PI;
		/*!< 判断是否执行到位（对比下发角度和当前角度是否在范围内），
		* 如果执行到位，进入数据采集状态，反之则继续保持在点位执行状态*/
		if (1 == JudgeActDone(TargetPos, CurrentPos))
		{
			/*!< 连续30次认定执行完毕，则下周期进入数据采集中*/
			MovConfmCnts++;
			if (MovConfmCnts >= 50)
			{
				printf("点位%d 执行完毕\n",PlanPosActIndex);
				MovConfmCnts = 0;
				OnlineFitStateFlg =OnlineFitDateCollectionState;
			}
			else
			{
				;
			}

		}
		else
		{
			;
		}

		break;
	case OnlineFitDateCollectionState:
		//延时
		if (1 == OSdelayCnts(&DelayCnts, 3))
		{
			DelayCnts = 0;

			//数据采集
			if (1 == FitDataCollect(InputSixForce, CurExgMat,InputPos))
			{
				printf("距离初始点Z轴距离： %f\n",Cal_points_distance(CurrentPos,PlanStartPos));
				OnlineFitStateFlg = OnlineFitPointActState;
				PlanPosActIndex++;
			}
			else
			{
				;/*!< 当前数据点位采集未完成，下一周期继续执行点位采集*/
			}

			//状态跳转
			if(PlanPosActIndex>PLAN_POINTS_NUMS-1)
			{
				printf("采集的力的数据：\n");
				for(int i = 0;i<PLAN_POINTS_NUMS;i++)
				{
					printf("%f ,%f ,%f ,%f ,%f ,%f\n",DataCollectionSet.data[i].F_meas.x,
						DataCollectionSet.data[i].F_meas.y,DataCollectionSet.data[i].F_meas.z,
						DataCollectionSet.data[i].T_meas.x,DataCollectionSet.data[i].T_meas.y,DataCollectionSet.data[i].T_meas.z);
				}

				OnlineFitStateFlg = OnlineFitCalcState;
				printf("数据采集完毕\n");
			}

		}
		else
		{
			;
		}

		break;
	case OnlineFitCalcState:
		/*!< 判断斜率并拟合*/
		ThreePointFit(DataCollectionSet);
		printf("===数据拟合完毕==\n");

		/*!< 跳转到补偿函数*/
		OnlineFitStateFlg = OnlineFitCompensateSate;
		break;
	case OnlineFitCompensateSate:

		/*!< 精度判断*/
		if(FitDoneFlg==0 || FitDoneFlg==-1)
		{
			ThreePointCmp(InputSixForce,InputPos, OutputCmpzForece);

			double F_ErrMax = 13.f;
			double T_ErrMax = 3.5f;
			if(OutputCmpzForece[0]>F_ErrMax||OutputCmpzForece[0]<-F_ErrMax 
			|| OutputCmpzForece[1]>F_ErrMax||OutputCmpzForece[1]<-F_ErrMax
			|| OutputCmpzForece[2]>F_ErrMax||OutputCmpzForece[2]<-F_ErrMax
			||OutputCmpzForece[3]>F_ErrMax||OutputCmpzForece[3]<-F_ErrMax 
			|| OutputCmpzForece[4]>F_ErrMax||OutputCmpzForece[4]<-F_ErrMax
			|| OutputCmpzForece[5]>F_ErrMax||OutputCmpzForece[5]<-F_ErrMax  )
			{
				FitDoneComfirmCnts--;
				if(FitDoneComfirmCnts<-FIT_DONE_CONFORM_COUNTS)
				{
					FitDoneFlg=-1;
				}

			}
			else
			{
				FitDoneComfirmCnts++;
				if(FitDoneComfirmCnts>FIT_DONE_CONFORM_COUNTS)
				{
					FitDoneFlg=1;
				}
			}
		}


		break;
	case 10:
		ResetOnlineFitAll();

	default:
		break;
	}

}