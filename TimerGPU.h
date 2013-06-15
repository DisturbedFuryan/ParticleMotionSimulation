#pragma once

#include <cuda_runtime.h>
#include "Timer.h"


class CTimerGPU : public ITimer {
public:
	CTimerGPU(void);
	~CTimerGPU(void);

	void Start(void);

	float GetElapsedTimeInMilliseconds(void);

private:
	cudaEvent_t m_start;
	cudaEvent_t m_current;
};

