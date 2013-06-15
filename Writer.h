#pragma once

#include <fstream>
#include "Singleton.h"


class CWriter : public ISingleton<CWriter>
{
public:
	CWriter(void);
	~CWriter(void);

	bool OpenFile(const char* pcName);
	bool CloseFile(void);

	bool ToFile(const char* pcText);

	// Accessor methods.
	bool IsFileOpened(void) const { return m_bFileOpened; }
	const char* GetFileName(void) const { return m_pcFile; }

private:
	std::ofstream m_out;

	const char* m_pcFile;

	bool m_bFileOpened;
};

