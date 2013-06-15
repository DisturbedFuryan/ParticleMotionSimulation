#include "TimerCPU.h"


void CTimerCPU::Start(void)
{
	m_bStarted = true;

	// Get ticks per second.
	QueryPerformanceFrequency(&m_frequency);

	QueryPerformanceCounter(&m_start);
}


float CTimerCPU::GetElapsedTimeInMilliseconds(void)
{
	if (m_bStarted)
	{
		QueryPerformanceCounter(&m_current);

		return ((m_current.QuadPart - m_start.QuadPart) * 1000.0f / m_frequency.QuadPart);
	}

	return 0.0f;
}