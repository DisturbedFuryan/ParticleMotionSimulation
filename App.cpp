#include "App.h"

#include "singletons.h"


CApp::CApp(void) : m_bRunning(true)
{
	srand(static_cast<unsigned int>(time(NULL)));

	// Initialize SDL.
	SDL_Init(SDL_INIT_EVERYTHING);

	InitWindow();

	std::string strOutputFile = m_parameters.GetOutputFileName();

	if (m_parameters.IsGPUUsing())
	{
		strOutputFile += "_gpu.txt";
	}
	else
	{
		strOutputFile += "_cpu.txt";
	}

	m_writer.OpenFile(strOutputFile.c_str());

	SendParametersInfoToFile();

	m_pParticleSprite = IDrawer::Load(m_parameters.GetParticleGraphFileName());

	if (m_parameters.IsGPUUsing())
	{
		m_pafPos = new float[m_parameters.GetNumParticles() * 2];
		m_pafVel = new float[m_parameters.GetNumParticles() * 2];
		m_pafOut = new float[m_parameters.GetNumParticles() * 2];

		AllocateDeviceMemoryCUDA();
	}
}


CApp::~CApp(void)
{
	if (m_parameters.IsGPUUsing())
	{
		FreeDeviceMemoryCUDA();

		delete[] m_pafPos;
		delete[] m_pafVel;
		delete[] m_pafOut;
	}

	// Remove particles.
	std::vector<CParticle*>::iterator it;
	for (it = m_particles.begin(); it != m_particles.end(); )
	{
			delete *it;
			it = m_particles.erase(it);
	}

	SDL_FreeSurface(m_pParticleSprite);

	// Shutdown SDL.
	SDL_Quit();
}


void CApp::HandleEvents(void)
{
	SDL_Event event;

	if (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
			case SDL_QUIT:		Quit();
								break;

			case SDL_KEYDOWN:	switch (event.key.keysym.sym)
								{
									case SDLK_ESCAPE:	Quit();
														break;
								}
								break;
		}
	}
}


float CApp::UpdateCPU(float fDeltaTime, unsigned int uFrame)
{
	float fElapsedTime;
	CTimerCPU tmrCPU;
	tmrCPU.Start();

	// Update particles.
	std::vector<CParticle*>::iterator it;
	for (it = m_particles.begin(); it != m_particles.end(); ++it)
	{
		(*it)->UpdateCPU(fDeltaTime);
	}

	fElapsedTime = tmrCPU.GetElapsedTimeInMilliseconds();

	if (uFrame == 0)
	{
		std::stringstream ss;
		ss << "Elapsed time on CPU in first update: " << fElapsedTime << " ms\n\n";
		g_writer.ToFile(ss.str().c_str());
	}

	return fElapsedTime;
}


float CApp::Draw(void)
{
	CTimerCPU tmrCPU;
	tmrCPU.Start();

	// Make screen black.
	SDL_FillRect(m_pScreen, &m_pScreen->clip_rect, SDL_MapRGB(m_pScreen->format, 0x00, 0x00, 0x00));

	// Draw particles.
	std::vector<CParticle*>::iterator it;
	for (it = m_particles.begin(); it != m_particles.end(); ++it)
	{
		(*it)->Draw();
	}

	// Update the screen.
	SDL_Flip(m_pScreen);

	return tmrCPU.GetElapsedTimeInMilliseconds();
}


void CApp::AddFewParticles(unsigned int uNum)
{
	for (unsigned int u = 0; u < uNum; u++)
	{
		AddParticle();
	}
}


CParticle* CApp::AddParticle(void)
{
	CParticle* pNewParticle = new CParticle(m_pParticleSprite);

	m_particles.push_back(pNewParticle);

	return pNewParticle;
}


void CApp::InitWindow(void)
{
	Uint32 iFlags = 0;

	// Set the title bar text.
	SDL_WM_SetCaption(g_strApplicationName.c_str(), NULL);

	if (m_parameters.IsFullscreen())
	{
		iFlags = SDL_FULLSCREEN;
	}

	// Create the screen surface.
	m_pScreen = SDL_SetVideoMode(m_parameters.GetScreenWidth(), m_parameters.GetScreenHeight(), g_iBPP, iFlags);
}


bool CApp::SendParametersInfoToFile(void)
{
	if (g_writer.IsFileOpened())
	{
		std::stringstream ss;
		ss << "Runtime parameters:\n";

		if (g_parameters.IsGPUUsing())
		{
			ss << "  Using GPU\n";
			ss << "  Tile width: " << g_parameters.GetTileWidth() << "\n";
		}
		else
		{
			ss << "  Using CPU only\n";
		}

		ss << "  Number of particles: " << g_parameters.GetNumParticles() << "\n";
		ss << "  Max velocity of particle: " << g_parameters.GetMaxParticleVel() << "\n\n";

		g_writer.ToFile(ss.str().c_str());

		return true;
	}

	return false;
}
