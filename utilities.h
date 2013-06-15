#pragma once

#include <cstdlib>
#include <string>


const std::string g_strApplicationName = "Particle Motion Simulation by Marcin Rainka";
const std::string g_strConfigFile = "config.txt";
const int g_iBPP = 32;


inline float Rand(void)
{
	return static_cast<float>(rand() / (RAND_MAX * 1.0));
}