#include "Particle.h"

#include "singletons.h"


CParticle::CParticle(SDL_Surface* pSprite) : m_pSprite(pSprite)
{
	// Set position vector.
	m_vPos.x = (Rand() * g_app.GetScreen()->w);
	m_vPos.y = (Rand() * g_app.GetScreen()->h);

	// Set velocity vector.

	m_vVel.x = Rand();
	m_vVel.y = Rand();

	if (Rand() > 0.5f)
	{
		m_vVel.x *= -1;
	}

	if (Rand() > 0.5f)
	{
		m_vVel.y *= -1;
	}

	m_vVel.SetMagnitude(g_parameters.GetMaxParticleVel() * Rand());
}


void CParticle::UpdateCPU(float fDeltaTime)
{
	m_vPos += (m_vVel * (fDeltaTime / 1000.0f));

	WorldBoundCPU();
}


void CParticle::Draw(void)
{
	IDrawer::Draw(m_pSprite, g_app.GetScreen(), static_cast<int>(m_vPos.x), static_cast<int>(m_vPos.y));
}


void CParticle::WorldBoundCPU(void)
{
	float fScreenWidth = static_cast<float>(g_parameters.GetScreenWidth());
	float fScreenHeight = static_cast<float>(g_parameters.GetScreenHeight());

	float fSpriteWidth = static_cast<float>(m_pSprite->w);
	float fSpriteHeight = static_cast<float>(m_pSprite->h);

	if ((m_vPos.x + fSpriteWidth) < 0.0f)
	{
		m_vPos.x = fScreenWidth;
	}
	else if (m_vPos.x > fScreenWidth)
	{
		m_vPos.x = -fSpriteWidth;
	}

	if ((m_vPos.y + fSpriteHeight) < 0.0f)
	{
		m_vPos.y = fScreenHeight;
	}
	else if (m_vPos.y > fScreenHeight)
	{
		m_vPos.y = -fSpriteHeight;
	}
}
