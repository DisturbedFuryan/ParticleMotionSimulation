#pragma once

#include <cassert>


template<typename T> class ISingleton
{
public:
	ISingleton(void) { assert(!ms_pSingleton); int iOffset = ((int)(T*)1 - (int)(ISingleton<T>*)(T*)1); ms_pSingleton = ((T*)((int)this + iOffset)); }
	~ISingleton(void) { assert(ms_pSingleton); ms_pSingleton = NULL; }

	static T& GetSingleton(void) { assert(ms_pSingleton); return *ms_pSingleton; }
	static T* GetSingletonPtr(void) { return ms_pSingleton; }

private:
	static T* ms_pSingleton;
};


template<typename T> T* ISingleton<T>::ms_pSingleton = NULL;