#pragma once

#include <fstream>
#include <string>
#include "Singleton.h"
#include "utilities.h"


class CParameters : public ISingleton<CParameters>
{
public:
	CParameters(void);

	// Accessor methods.
	const char* GetOutputFileName(void) const { return m_strOutputFile.c_str(); }
	const char* GetParticleGraphFileName(void) const { return m_strParticleGraphFile.c_str(); }
	unsigned int GetNumParticles(void) const { return m_uNumParticles; }
	int GetMaxParticleVel(void) const { return m_iMaxParticleVel; }
	int GetScreenWidth(void) const { return m_iScreenWidth; }
	int GetScreenHeight(void) const { return m_iScreenHeight; }
	int GetTileWidth(void) const { return m_iTileWidth; }
	int GetDuration(void) const { return m_iDuration; }
	bool IsFullscreen(void) const { return m_bFullscreen; }
	bool IsGPUUsing(void) const { return m_bUsingGPU; }

private:
	// Parameters.
	std::string m_strOutputFile;
	std::string m_strParticleGraphFile;
	unsigned int m_uNumParticles;
	int m_iMaxParticleVel;
	int m_iScreenWidth;
	int m_iScreenHeight;
	int m_iTileWidth;
	int m_iDuration;
	bool m_bFullscreen;
	bool m_bUsingGPU;

	bool LoadParameters(void);

	bool Load(const char* pcParameterDescription, std::string* pstrParameter);
	bool Load(const char* pcParameterDescription, int* piParameter);
	bool Load(const char* pcParameterDescription, unsigned int* piParameter);
	bool Load(const char* pcParameterDescription, bool* pbParameter);
};

