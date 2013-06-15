#pragma once

#include <SDL.h>
#include <vector>
#include <time.h>
#include <cassert>
#include <sstream>
#include "Drawer.h"
#include "Singleton.h"
#include "Parameters.h"
#include "Writer.h"
#include "Particle.h"
#include "TimerCPU.h"
#include "Vector.h"
#include "utilities.h"


class CApp : public ISingleton<CApp>
{
public:
	// Components.
	CParameters m_parameters;
	CWriter m_writer;

	CApp(void);
	~CApp(void);

	void HandleEvents(void);

	// Updates and returns the amount of time spent on this (in milliseconds).
	float UpdateCPU(float fDeltaTime, unsigned int uFrame);
	CVector UpdateGPU(float fDeltaTime, unsigned int uFrame);

	// Draws and returns the amount of time spent on this (in milliseconds).
	float Draw(void);

	// Particles production.
	void AddFewParticles(unsigned int uNum);
	CParticle* AddParticle(void);

	// Accessor methods.
	SDL_Surface* GetScreen(void) const { return m_pScreen; }
	SDL_Surface* GetParticleSprite(void) const { return m_pParticleSprite; }
	bool IsRunning(void) const { return m_bRunning; }

	void Quit(void) { m_bRunning = false; }

private:
	// SDL surface for screen.
	SDL_Surface* m_pScreen;

	// Graphical representation of particle.
	SDL_Surface* m_pParticleSprite;

	// Container of particles.
	std::vector<CParticle*> m_particles;

	// Arrays used in GPU computing to hold vectors of particles on host.
	float* m_pafPos;
	float* m_pafVel;
	float* m_pafOut;

	// Arrays used in GPU computing to hold vectors of particles on device.
	float* m_pafDevPos;
	float* m_pafDevVel;
	float* m_pafDevOut;

	bool m_bRunning;

	void InitWindow(void);

	void AllocateDeviceMemoryCUDA(void);
	void FreeDeviceMemoryCUDA(void);

	bool SendParametersInfoToFile(void);
};