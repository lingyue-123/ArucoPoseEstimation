/******************************************************************************
* 名        称: OnLineFit.h
* 发布日期: 2026.4.29
* 创  建  人: 集成一科/尹君
* 版        本: V001
* 描        述:
* 备        注:
******************************************************************************/
#ifndef ONLINE_LEAST_SQUARES_FIT_H
#define ONLINE_LEAST_SQUARES_FIT_H
#include"IdentifyCommon.h"
#include"string.h"
/*!< 状态机0::初始化上电状态，1：点位生成状态，2：机械臂点位执行状态，3：数据采集状态，4：辨识，5：补偿*/
extern int OnlineFitStateFlg;
/*!< 辨识完成标志位0:默认状态，未辨识  1：辨识完成  -1：辨识失败或误差过大*/
extern int FitDoneFlg;
/*!< 当前参数*/
extern CalibParams strOupt_SysParm;
/*!< 外发当前目标位姿*/
extern PosData TargetPos;
/*!< 当前规划点位下标 */
extern int PlanPosActIndex ;
/*!< 采集的数据集 */
extern DataSet DataCollectionSet;  
/*!< Z移动距离 */
extern double MovDist;


/*！<==============在线参数辨识（点位规划）=====================*/
#define PLAN_POINTS_NUMS 3/*!<点位规划个数 */
/*！<==============在线参数辨识（数据采集）=====================*/
#define DATA_COLLECTION_FILTER_COUNTS 15   /*!<均值滤波个数 */
/*！<==============在线参数辨识（Z轴单步位移距离mm）=====================*/
#define Z_DIR_MOVE_STEP_MM   10.f


#define FIT_DONE_CONFORM_COUNTS 5 /*!<精度确认计数值*/

typedef enum 
{
	OnlineFitInitState=0,
	OnlineFitPointPlanState = 1,
	OnlineFitPointActState = 2,
	OnlineFitDateCollectionState = 3,
	OnlineFitCalcState = 4,
	OnlineFitCompensateSate = 5
} OnlineFitState;




/*！<==============在线参数辨识（状态机）=====================*/
extern void OnlineCubicFitCalc( double InputSixForce[6], double CurExgMat[4][4], double InputPos[6], double OutputCmpzForece[6]);
/*！<==============在线参数辨识（复位函数）=====================*/
extern void ResetOnlineFitAll();
/*！<==============在线参数辨识（在线补偿函数）=====================*/
extern void ThreePointCmp(double InputForce[6],double InputPos[6], double OutputForece[6]);

#endif