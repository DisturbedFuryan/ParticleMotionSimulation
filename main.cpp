#include <sstream>
#include "App.h"
#include "TimerCPU.h"
#include "Vector.h"
#include "singletons.h"


int main(int argc, char** argv)
{
	CApp app;

	app.AddFewParticles(g_parameters.GetNumParticles());

	// Keep track of the frame count.
	unsigned int uFrame = 0;

	// Time spent on updates and draw.
	CVector vUpdatesTime;  // x - CPU, y - GPU
	float fDrawTime = 0.0f;

	// Keeps track of time since last rendering.
	CTimerCPU deltaTmrCPU;
	float fDeltaTime;

	// Keeps track of duration of the program.
	CTimerCPU durationTmrCPU;
	float fDurationTime;

	// Start timers.
	deltaTmrCPU.Start();
	durationTmrCPU.Start();

	while (app.IsRunning())
	{
		app.HandleEvents();
		
		// Get delta time and restart timer.
		fDeltaTime = deltaTmrCPU.GetElapsedTimeInMilliseconds();
		deltaTmrCPU.Start();

		// Update.
		if (g_parameters.IsGPUUsing())
		{
			vUpdatesTime += app.UpdateGPU(fDeltaTime, uFrame);
		}
		else
		{
			vUpdatesTime.x += app.UpdateCPU(fDeltaTime, uFrame);
		}

		fDrawTime += app.Draw();

		uFrame++;

		// Keep track of duration.
		fDurationTime = durationTmrCPU.GetElapsedTimeInMilliseconds();
		if ((g_parameters.GetDuration() != 0) && ((fDurationTime / 1000.0f) > g_parameters.GetDuration()))
		{
			app.Quit();
		}
	}

	// Send the average time spent on single update, on draw and duration of the program to file.

	std::stringstream ss;

	if (g_parameters.IsGPUUsing())
	{
		ss << "Average time spent on single update on GPU (only running time of kernel): " << (vUpdatesTime.y / uFrame) << " ms\n";
		ss << "Average time spent on single update on CPU: " << (vUpdatesTime.x / uFrame) << " ms\n\n";
	}
	else
	{
		ss << "Average time spent on single update on CPU: " << (vUpdatesTime.x / uFrame) << " ms\n\n";
	}

	ss << "Average time spent on single draw: " << (fDrawTime / uFrame) << " ms\n\n";
	ss << "Duration of the program: " << (fDurationTime / 1000.0f) << " s\n\n";

	g_writer.ToFile(ss.str().c_str());

	return 0;
}