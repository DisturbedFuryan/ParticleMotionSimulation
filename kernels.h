#pragma once

#include <cuda_runtime.h>


__global__ void UpdateParticlesCUDA(float* pafOut, const float* pafPos, const float* pafVel, unsigned int uNumParticles, float fDeltaTime, 
									float fParticleSpriteWidth, float fParticleSpriteHeight, float fScreenWidth, float fScreenHeight, int iTileWidth)
{
	//------------------
	// Get coordinates.
	//------------------

	// Get block and thread x coordinate.
	unsigned int uBlockX = blockIdx.x;
	unsigned int uThreadX = threadIdx.x;

	// Compute position of first coordinate of particle position in the array.
	unsigned int uX = ((uBlockX * iTileWidth) + uThreadX);

	if (uX < uNumParticles)
	{
		//------------------------------------
		// Compute new position for particle.
		//------------------------------------

		float fNewPosX = (pafPos[uX] + (pafVel[uX] * (fDeltaTime / 1000.0f)));
		float fNewPosY = (pafPos[uX + uNumParticles] + (pafVel[uX + uNumParticles] * (fDeltaTime / 1000.0f)));

		// World bound.

		if ((fNewPosX + fParticleSpriteWidth) < 0.0f)
		{
			fNewPosX = fScreenWidth;
		}
		else if (fNewPosX > fScreenWidth)
		{
			fNewPosX = -fParticleSpriteWidth;
		}

		if ((fNewPosY + fParticleSpriteHeight) < 0.0f)
		{
			fNewPosY = fScreenHeight;
		}
		else if (fNewPosY > fScreenHeight)
		{
			fNewPosY = -fParticleSpriteHeight;
		}

		//--------------------
		// Save new position.
		//--------------------

		pafOut[uX] = fNewPosX;
		pafOut[uX + uNumParticles] = fNewPosY;
	}
}