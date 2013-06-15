#pragma once

#include <SDL.h>
#include "Drawer.h"
#include "Vector.h"
#include "utilities.h"


class CParticle
{
public:
	CParticle(SDL_Surface* pSprite);

	void UpdateCPU(float fDeltaTime);

	void Draw(void);

	// Accessor methods.
	CVector GetPos(void) const { return m_vPos; }
	CVector GetVel(void) const { return m_vVel; }
	void SetPos(float fX, float fY) { m_vPos.x = fX; m_vPos.y = fY; }

private:
	CVector m_vPos;
	CVector m_vVel;

	// Graphical representation.
	SDL_Surface* m_pSprite;

	void WorldBoundCPU(void);
};

