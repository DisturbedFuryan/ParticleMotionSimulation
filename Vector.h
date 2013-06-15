#pragma once

#include <cmath>


class CVector
{
public:
	float x, y;

	CVector(void) : x(0.0f), y(0.0f) {}
	CVector(float fX, float fY) : x(fX), y(fY) {}
	CVector(const CVector& v) : x(v.x), y(v.y) {}

	// Operators.
	CVector& operator=(const CVector& v) { x = v.x; y = v.y; return *this; }
	CVector& operator+=(const CVector& v) { x += v.x; y += v.y; return *this; }
	CVector& operator-=(const CVector& v) { x -= v.x; y -= v.y; return *this; }
	CVector& operator*=(float fVal) { x *= fVal; y *= fVal; return *this; }
	CVector& operator/=(float fVal) { x /= fVal; y /= fVal; return *this; }
	friend bool operator==(const CVector& vA, const CVector& vB);
	friend bool operator!=(const CVector& vA, const CVector& vB);
	friend CVector operator+(const CVector& vA, const CVector& vB);
	friend CVector operator-(const CVector& vA, const CVector& vB);
	friend CVector operator-(const CVector& v);
	friend CVector operator*(const CVector& v, float fVal);
	friend CVector operator*(float fVal, const CVector& v);
	friend CVector operator/(const CVector& v, float fVal);

	float Length(void) const { return sqrt((x * x) + (y * y)); }
	CVector& Normalize(void);
	void SetMagnitude(float fVel);
};


