#include "TimerGPU.h"


CTimerGPU::CTimerGPU(void)
{
	cudaEventCreate(&m_start);
	cudaEventCreate(&m_current);
}


CTimerGPU::~CTimerGPU(void)
{
	cudaEventDestroy(m_start);
	cudaEventDestroy(m_current);
}


void CTimerGPU::Start(void)
{
	m_bStarted = true;

	cudaEventRecord(m_start, 0);
}


float CTimerGPU::GetElapsedTimeInMilliseconds(void)
{
	if (m_bStarted)
	{
		cudaEventRecord(m_current, 0);
		cudaEventSynchronize(m_current);

		float fElapsedTime;
		cudaEventElapsedTime(&fElapsedTime, m_start, m_current);

		return fElapsedTime;
	}

	return 0.0f;
}
