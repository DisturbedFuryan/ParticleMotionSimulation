#pragma once


class ITimer
{
public:
	ITimer(void) : m_bStarted(false) {}

	virtual void Start(void) = 0;
	void Stop(void);

	virtual float GetElapsedTimeInMilliseconds(void) = 0;

	bool IsStarted(void) const { return m_bStarted; }

protected:
	bool m_bStarted;
};

