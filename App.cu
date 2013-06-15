#include "App.h"

#include <cuda_runtime.h>
#include "kernels.h"
#include "TimerGPU.h"
#include "singletons.h"


CVector CApp::UpdateGPU(float fDeltaTime, unsigned int uFrame)
{
	CTimerCPU tmrCPU;
	tmrCPU.Start();

	//--------------------------------------
	// Prepare particles for update on GPU.
	//--------------------------------------

	// Get number of particles.
	unsigned int uNumParticles = m_particles.size();

	// Pack vectors from particles to arrays.
	std::vector<CParticle*>::iterator it;
	unsigned int u;
	for (it = m_particles.begin(), u = 0; it != m_particles.end(); ++it, u++)
	{
		// Get position.
		m_pafPos[u] = (*it)->GetPos().x;
		m_pafPos[u + uNumParticles] = (*it)->GetPos().y;

		// Get velocity.
		m_pafVel[u] = (*it)->GetVel().x;
		m_pafVel[u + uNumParticles] = (*it)->GetVel().y;
	}

	//------------------------
	// Prepare device memory.
	//------------------------

	// Compute size of Pos, Vel and Out.
	size_t size = (uNumParticles * 2 * sizeof(float));

	// Load Pos to device memory.
	cudaMemcpy(m_pafDevPos, m_pafPos, size, cudaMemcpyHostToDevice);

	// Load Vel to device memory.
	cudaMemcpy(m_pafDevVel, m_pafVel, size, cudaMemcpyHostToDevice);

	//-----------------------------------------
	// Get execution configuration paramaters.
	//-----------------------------------------

	int iTileWidth = m_parameters.GetTileWidth();

	dim3 d3Grid;
	d3Grid.x = ((uNumParticles % iTileWidth) > 0) ? ((uNumParticles / iTileWidth) + 1) : (uNumParticles / iTileWidth);

	dim3 d3Threads;
	d3Threads.x = iTileWidth;

	float fParticleSpriteWidth = static_cast<float>(g_app.GetParticleSprite()->w);
	float fParticleSpriteHeight = static_cast<float>(g_app.GetParticleSprite()->h);

	float fScreenWidth = static_cast<float>(m_parameters.GetScreenWidth());
	float fScreenHeight = static_cast<float>(m_parameters.GetScreenHeight());

	//-------------------
	// Update particles.
	//-------------------

	float fElapsedTimeGPU;
	CTimerGPU tmrGPU;
	tmrGPU.Start();

	// Run kernel.
	UpdateParticlesCUDA<<< d3Grid, d3Threads >>>(m_pafDevOut, m_pafDevPos, m_pafDevVel, uNumParticles, fDeltaTime, 
		fParticleSpriteWidth, fParticleSpriteHeight, fScreenWidth, fScreenHeight, iTileWidth);
	cudaError_t error = cudaThreadSynchronize();

	fElapsedTimeGPU = tmrGPU.GetElapsedTimeInMilliseconds();

	std::stringstream ss;
	if (uFrame == 0)
	{
		ss << "Run kernel for the first time: " << cudaGetErrorString(error) << "\n";
	}

	//-------------------------------------------------------------
	// Get Out from device memory and associate it with particles.
	//-------------------------------------------------------------
	
	error = cudaMemcpy(m_pafOut, m_pafDevOut, size, cudaMemcpyDeviceToHost);

	if (uFrame == 0)
	{
		ss << "Get Out from device for the first time: " << cudaGetErrorString(error) << "\n\n";
	}

	for (it = m_particles.begin(), u = 0; it != m_particles.end(); ++it, u++)
	{
		(*it)->SetPos(m_pafOut[u], m_pafOut[u + uNumParticles]);
	}

	//---------
	// Timing.
	//---------

	float fElapsedTimeCPU = tmrCPU.GetElapsedTimeInMilliseconds();

	if (uFrame == 0)
	{
		ss << "Elapsed time on GPU in first update (only running time of kernel): " << fElapsedTimeGPU << " ms\n";
		ss << "Elapsed time on CPU in first update: " << fElapsedTimeCPU << " ms\n\n";
		g_writer.ToFile(ss.str().c_str());
	}

	return CVector(fElapsedTimeCPU, fElapsedTimeGPU);
}


void CApp::AllocateDeviceMemoryCUDA(void)
{
	// Get number of particles.
	unsigned int uNumParticles = g_parameters.GetNumParticles();

	// Compute size of Pos, Vel and Out.
	size_t size = (uNumParticles * 2 * sizeof(float));

	// Allocate Pos in device memory.
	cudaError_t error = cudaMalloc((void**)&m_pafDevPos, size);

	std::stringstream ss;
	ss << "CUDA malloc Pos: " << cudaGetErrorString(error) << "\n";

	// Allocate Vel in device memory.
	error = cudaMalloc((void**)&m_pafDevVel, size);

	ss << "CUDA malloc Vel: " << cudaGetErrorString(error) << "\n";

	// Allocate Out in device memory.
	error = cudaMalloc((void**)&m_pafDevOut, size);

	ss << "CUDA malloc Out: " << cudaGetErrorString(error) << "\n\n";

	m_writer.ToFile(ss.str().c_str());
}


void CApp::FreeDeviceMemoryCUDA(void)
{
	cudaFree(m_pafDevPos);
	cudaFree(m_pafDevVel);
	cudaFree(m_pafDevOut);
}