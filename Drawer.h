#pragma once

#include <SDL.h>
#include <SDL_image.h>


class IDrawer
{
public:
	static SDL_Surface* Load(const char* pcFileName);

	static bool Draw(SDL_Surface* pSource, SDL_Surface* pDest, int iX, int iY);
	static bool Draw(SDL_Surface* pSource, SDL_Surface* pDest, int iDestX, int iDestY, int iSourceX, int iSourceY, int iSourceWidth, int iSourceHeight);
};

