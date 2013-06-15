#pragma once

#include "Parameters.h"
#include "App.h"
#include "Writer.h"

#define g_parameters CParameters::GetSingleton()
#define g_app CApp::GetSingleton()
#define g_writer CWriter::GetSingleton()