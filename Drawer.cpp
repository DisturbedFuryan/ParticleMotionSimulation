#include "Drawer.h"


SDL_Surface* IDrawer::Load(const char* pcFileName)
{
	SDL_Surface* pLoadedImage = NULL;
	SDL_Surface* pOptimizedImage = NULL;

	// Load the image using SDL_image.
	pLoadedImage = IMG_Load(pcFileName);

	if (pLoadedImage != NULL)
	{
		// Create an optimized image.
		pOptimizedImage = SDL_DisplayFormatAlpha(pLoadedImage);

		// Free the old image.
		SDL_FreeSurface(pLoadedImage);
	}

	return pOptimizedImage;
}


bool IDrawer::Draw(SDL_Surface* pSource, SDL_Surface* pDest, int iX, int iY)
{
	if ((pSource == NULL) || (pDest == NULL))
	{
		return false;
	}

	SDL_Rect destRect;

	destRect.x = iX;
	destRect.y = iY;

	SDL_BlitSurface(pSource, NULL, pDest, &destRect);

	return true;
}


bool IDrawer::Draw(SDL_Surface* pSource, SDL_Surface* pDest, int iDestX, int iDestY, int iSourceX, int iSourceY, int iSourceWidth, int iSourceHeight)
{
	if ((pSource == NULL) || (pDest == NULL))
	{
		return false;
	}

	SDL_Rect destRect;

	destRect.x = iDestX;
	destRect.y = iDestY;

	SDL_Rect sourceRect;

	sourceRect.x = iSourceX;
	sourceRect.y = iSourceY;
	sourceRect.w = iSourceWidth;
	sourceRect.h = iSourceHeight;

	SDL_BlitSurface(pSource, &sourceRect, pDest, &destRect);

	return true;
}
