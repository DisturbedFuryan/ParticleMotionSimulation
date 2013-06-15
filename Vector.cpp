#include "Vector.h"


bool operator==(const CVector& vA, const CVector& vB)
{
	return ((vA.x == vB.x) && (vA.y == vB.y));
}


bool operator!=(const CVector& vA, const CVector& vB)
{
	return ((vA.x != vB.x) && (vA.y != vB.y));
}


CVector operator+(const CVector& vA, const CVector& vB)
{
	CVector vSum(vA);
	vSum += vB;

	return vSum;
}


CVector operator-(const CVector& vA, const CVector& vB)
{
	CVector vSum(vA);
	vSum -= vB;

	return vSum;
}


CVector operator-(const CVector& v)
{
	return CVector(-v.x, -v.y);
}


CVector operator*(const CVector& v, float fVal)
{
	return CVector((v.x * fVal), (v.y * fVal));
}


CVector operator*(float fVal, const CVector& v)
{
	return CVector((v.x * fVal), (v.y * fVal));
}


CVector operator/(const CVector& v, float fVal)
{
	return CVector((v.x / fVal), (v.y / fVal));
}


CVector& CVector::Normalize(void)
{
	float fM = Length();

	if (fM > 0.0f)
	{
		fM = (1.0f / fM);
	}
	else
	{
		fM = 0.0f;
	}

	x *= fM;
	y *= fM;

	return *this;
}


void CVector::SetMagnitude(float fVel)
{
	this->Normalize();

	x *= fVel;
	y *= fVel;
}