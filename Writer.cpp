#include "Writer.h"


CWriter::CWriter(void) : m_bFileOpened(false)
{
}


CWriter::~CWriter(void)
{
	CloseFile();
}


bool CWriter::OpenFile(const char* pcName)
{
	if (!m_bFileOpened)
	{
		m_out.open(pcName);

		m_pcFile = pcName;
		m_bFileOpened = true;
		return true;
	}
	
	return false;
}


bool CWriter::CloseFile(void)
{
	if (m_bFileOpened)
	{
		m_out.close();

		m_bFileOpened = false;
		return true;
	}

	return false;
}


bool CWriter::ToFile(const char* pcText)
{
	if (m_bFileOpened)
	{
		m_out << pcText;

		return true;
	}

	return false;
}
