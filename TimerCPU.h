#pragma once

#include <Windows.h>
#include "Timer.h"


class CTimerCPU :public ITimer {
public:
	void Start(void);

	float GetElapsedTimeInMilliseconds(void);

private:
	// Ticks per second.
	LARGE_INTEGER m_frequency;

	LARGE_INTEGER m_start;
	LARGE_INTEGER m_current;
};