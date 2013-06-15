#include "Parameters.h"


CParameters::CParameters(void)
{
	if (!LoadParameters())
	{
		exit(EXIT_FAILURE);
	}
}


bool CParameters::LoadParameters(void)
{
	if (!Load("output_file_name_(without_extension)", &m_strOutputFile))
	{
		return false;
	}

	if (!Load("particle_graphic_file_name", &m_strParticleGraphFile))
	{
		return false;
	}

	if (!Load("number_of_particles", &m_uNumParticles))
	{
		return false;
	}

	if (!Load("max_particle_velocity", &m_iMaxParticleVel))
	{
		return false;
	}

	if (!Load("screen_width", &m_iScreenWidth))
	{
		return false;
	}

	if (!Load("screen_height", &m_iScreenHeight))
	{
		return false;
	}

	if (!Load("tile_width", &m_iTileWidth))
	{
		return false;
	}

	if (!Load("duration_(0_=_infinity)", &m_iDuration))
	{
		return false;
	}

	if (!Load("fullscreen", &m_bFullscreen))
	{
		return false;
	}

	if (!Load("using_GPU", &m_bUsingGPU))
	{
		return false;
	}

	return true;
}


bool CParameters::Load(const char* pcParameterDescription, std::string* pstrParameter)
{
	std::string strSearched = pcParameterDescription;
	strSearched += ":";

	std::ifstream in;

	in.open(g_strConfigFile.c_str());

	if (!in.is_open())
	{
		return false;
	}

	std::string str;

	do
	{
		in >> str;
	} while ((str != strSearched) && (in.eof() == false));

	if(in.eof())
	{
		return false;
	}

	in >> *pstrParameter;

	in.close();
	return true;
}


bool CParameters::Load(const char* pcParameterDescription, int* piParameter)
{
	std::string strSearched = pcParameterDescription;
	strSearched += ":";

	std::ifstream in;

	in.open(g_strConfigFile.c_str());

	if (!in.is_open())
	{
		return false;
	}

	std::string str;

	do
	{
		in >> str;
	} while ((str != strSearched) && (in.eof() == false));

	if(in.eof())
	{
		return false;
	}

	in >> *piParameter;

	in.close();
	return true;
}


bool CParameters::Load(const char* pcParameterDescription, unsigned int* puParameter)
{
	std::string strSearched = pcParameterDescription;
	strSearched += ":";

	std::ifstream in;

	in.open(g_strConfigFile.c_str());

	if (!in.is_open())
	{
		return false;
	}

	std::string str;

	do
	{
		in >> str;
	} while ((str != strSearched) && (in.eof() == false));

	if(in.eof())
	{
		return false;
	}

	in >> *puParameter;

	in.close();
	return true;
}


bool CParameters::Load(const char* pcParameterDescription, bool* pbParameter)
{
	std::string strSearched = pcParameterDescription;
	strSearched += ":";

	std::ifstream in;

	in.open(g_strConfigFile.c_str());

	if (!in.is_open())
	{
		return false;
	}

	std::string str;

	do
	{
		in >> str;
	} while ((str != strSearched) && (in.eof() == false));

	if(in.eof())
	{
		return false;
	}

	in >> *pbParameter;

	in.close();
	return true;
}
