//Copyright ETH Zurich, IWF

//This file is part of iwf_mfree_gpu_3d.

//iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#include "eigen_solver.cuh"
__device__ __inline__
float3_t Cross(const float3_t &U, const float3_t &V) {
	return make_float3_t(
			U.y * V.z - U.z * V.y,
			U.z * V.x - U.x * V.z,
			U.x * V.y - U.y * V.x);
}

__device__ __inline__
float3_t Multiply(float_t s, float3_t const& U) {
    float3_t product = make_float3_t( s * U.x, s * U.y, s * U.z );
    return product;
}

__device__ __inline__
float3_t Subtract(float3_t const& U, float3_t const& V) {
    float3_t difference = make_float3_t( U.x - V.x, U.y - V.y, U.z - V.z );
    return difference;
}

__device__ __inline__
float3_t Divide(float3_t const& U, float_t s) {
    float_t invS = (float_t)1 / s;
    float3_t division = make_float3_t( U.x * invS, U.y * invS, U.z * invS );
    return division;
}

__device__ __inline__
float_t Dot(float3_t const& U, float3_t const& V) {
    float_t dot = U.x * V.x + U.y * V.y + U.z * V.z;
    return dot;
}
__device__ __inline__
void ComputeOrthogonalComplement(float3_t const& W, float3_t& U, float3_t& V) {
    // Robustly compute a right-handed orthonormal set { U, V, W }.  The
    // vector W is guaranteed to be unit-length, in which case there is no
    // need to worry about a division by zero when computing invLength.
    float_t invLength;
    if (fabs(W.x) > fabs(W.y)) {
        // The component of maximum absolute value is either W.x or W.z.
        invLength = (float_t)1 / sqrt(W.x * W.x + W.z * W.z);
        U = make_float3_t( -W.z * invLength, (float_t)0, +W.x * invLength );
    } else {
        // The component of maximum absolute value is either W.y or W.z.
        invLength = (float_t)1 / sqrt(W.y * W.y + W.z * W.z);
        U = make_float3_t( (float_t)0, +W.z * invLength, -W.y * invLength );
    }
    V = Cross(W, U);
}

__device__
void ComputeEigenvector0(float_t a00, float_t a01,
    float_t a02, float_t a11, float_t a12, float_t a22, float_t eval0, float3_t& evec0)  {
    // Compute a unit-length eigenvector for eigenvalue[i0].  The matrix is
    // rank 2, so two of the rows are linearly independent.  For a robust
    // computation of the eigenvector, select the two rows whose cross product
    // has largest length of all pairs of rows.
    float3_t row0 = make_float3_t( a00 - eval0, a01, a02 );
    float3_t row1 = make_float3_t( a01, a11 - eval0, a12 );
    float3_t row2 = make_float3_t( a02, a12, a22 - eval0 );
    float3_t  r0xr1 = Cross(row0, row1);
    float3_t  r0xr2 = Cross(row0, row2);
    float3_t  r1xr2 = Cross(row1, row2);
    float_t d0 = Dot(r0xr1, r0xr1);
    float_t d1 = Dot(r0xr2, r0xr2);
    float_t d2 = Dot(r1xr2, r1xr2);

    float_t dmax = d0;
    int imax = 0;
    if (d1 > dmax)
    {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax)
    {
        imax = 2;
    }

    if (imax == 0)
    {
        evec0 = Divide(r0xr1, sqrt(d0));
    }
    else if (imax == 1)
    {
        evec0 = Divide(r0xr2, sqrt(d1));
    }
    else
    {
        evec0 = Divide(r1xr2, sqrt(d2));
    }
}

__device__
void ComputeEigenvector1(float_t a00, float_t a01,
    float_t a02, float_t a11, float_t a12, float_t a22, float3_t const& evec0,
    float_t eval1, float3_t& evec1) {
    // Robustly compute a right-handed orthonormal set { U, V, evec0 }.
    float3_t U, V;
    ComputeOrthogonalComplement(evec0, U, V);

    // Let e be eval1 and let E be a corresponding eigenvector which is a
    // solution to the linear system (A - e*I)*E = 0.  The matrix (A - e*I)
    // is 3x3, not invertible (so infinitely many solutions), and has rank 2
    // when eval1 and eval are different.  It has rank 1 when eval1 and eval2
    // are equal.  Numerically, it is difficult to compute robustly the rank
    // of a matrix.  Instead, the 3x3 linear system is reduced to a 2x2 system
    // as follows.  Define the 3x2 matrix J = [U V] whose columns are the U
    // and V computed previously.  Define the 2x1 vector X = J*E.  The 2x2
    // system is 0 = M * X = (J^T * (A - e*I) * J) * X where J^T is the
    // transpose of J and M = J^T * (A - e*I) * J is a 2x2 matrix.  The system
    // may be written as
    //     +-                        -++-  -+       +-  -+
    //     | U^T*A*U - e  U^T*A*V     || x0 | = e * | x0 |
    //     | V^T*A*U      V^T*A*V - e || x1 |       | x1 |
    //     +-                        -++   -+       +-  -+
    // where X has row entries x0 and x1.

    float3_t AU = make_float3_t(
        a00 * U.x + a01 * U.y + a02 * U.z,
        a01 * U.x + a11 * U.y + a12 * U.z,
        a02 * U.x + a12 * U.y + a22 * U.z
    );

    float3_t AV = make_float3_t(
        a00 * V.x + a01 * V.y + a02 * V.z,
        a01 * V.x + a11 * V.y + a12 * V.z,
        a02 * V.x + a12 * V.y + a22 * V.z
    );

    float_t m00 = U.x * AU.x + U.y * AU.y + U.z * AU.z - eval1;
    float_t m01 = U.x * AV.x + U.y * AV.y + U.z * AV.z;
    float_t m11 = V.x * AV.x + V.y * AV.y + V.z * AV.z - eval1;

    // For robustness, choose the largest-length row of M to compute the
    // eigenvector.  The 2-tuple of coefficients of U and V in the
    // assignments to eigenvector.y lies on a circle, and U and V are
    // unit length and perpendicular, so eigenvector.y is unit length
    // (within numerical tolerance).
    float_t absM00 = fabs(m00);
    float_t absM01 = fabs(m01);
    float_t absM11 = fabs(m11);
    float_t maxAbsComp;
    if (absM00 >= absM11)
    {
        maxAbsComp = fmax(absM00, absM01);
        if (maxAbsComp > (float_t)0)
        {
            if (absM00 >= absM01)
            {
                m01 /= m00;
                m00 = (float_t)1 / sqrt((float_t)1 + m01 * m01);
                m01 *= m00;
            }
            else
            {
                m00 /= m01;
                m01 = (float_t)1 / sqrt((float_t)1 + m00 * m00);
                m00 *= m01;
            }
            evec1 = Subtract(Multiply(m01, U), Multiply(m00, V));
        }
        else
        {
            evec1 = U;
        }
    }
    else
    {
        maxAbsComp = fmax(absM11, absM01);
        if (maxAbsComp > (float_t)0)
        {
            if (absM11 >= absM01)
            {
                m01 /= m11;
                m11 = (float_t)1 / sqrt((float_t)1 + m01 * m01);
                m01 *= m11;
            }
            else
            {
                m11 /= m01;
                m01 = (float_t)1 / sqrt((float_t)1 + m11 * m11);
                m11 *= m01;
            }
            evec1 = Subtract(Multiply(m11, U), Multiply(m01, V));
        }
        else
        {
            evec1 = U;
        }
    }
}

__device__
void solve_eigen(float_t a00, float_t a01, float_t a02, float_t a11, float_t a12, float_t a22,
		float3_t &eval, float3_t &e1, float3_t &e2, float3_t &e3) {

	float_t max0 = fmax(fabs(a00), fabs(a01));
	float_t max1 = fmax(fabs(a02), fabs(a11));
	float_t max2 = fmax(fabs(a12), fabs(a22));
	float_t maxAbsElement = fmax(fmax(max0, max1), max2);
	if (maxAbsElement < 1e-8) {
		eval = make_float3_t(0., 0., 0.);

		e1 = make_float3_t(1., 0., 0.);
		e2 = make_float3_t(0., 1., 0.);
		e3 = make_float3_t(0., 0., 1.);
		return;
	}

	float_t invMaxAbsElement = (float_t) 1 / maxAbsElement;
	a00 *= invMaxAbsElement;
	a01 *= invMaxAbsElement;
	a02 *= invMaxAbsElement;
	a11 *= invMaxAbsElement;
	a12 *= invMaxAbsElement;
	a22 *= invMaxAbsElement;

	float_t norm = a01 * a01 + a02 * a02 + a12 * a12;
	if (norm > (float_t)0)
	{
		// Compute the eigenvalues of A.  The acos(z) function requires |z| <= 1,
		// but will fail silently and return NaN if the input is larger than 1 in
		// magnitude.  To avoid this condition due to rounding errors, the halfDet
		// value is clamped to [-1,1].
		float_t traceDiv3 = (a00 + a11 + a22) / (float_t)3;
		float_t b00 = a00 - traceDiv3;
		float_t b11 = a11 - traceDiv3;
		float_t b22 = a22 - traceDiv3;
		float_t denom = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * (float_t)2) / (float_t)6);
		float_t c00 = b11 * b22 - a12 * a12;
		float_t c01 = a01 * b22 - a12 * a02;
		float_t c02 = a01 * a12 - b11 * a02;
		float_t det = (b00 * c00 - a01 * c01 + a02 * c02) / (denom * denom * denom);
		float_t halfDet = det * (float_t)0.5;
		halfDet = fmin(fmax(halfDet, (float_t)-1), (float_t)1);

		// The eigenvalues of B are ordered as beta0 <= beta1 <= beta2.  The
		// number of digits in twoThirdsPi is chosen so that, whether float or
		// double, the floating-point number is the closest to theoretical 2*pi/3.
		float_t angle = acos(halfDet) / (float_t)3;
		float_t const twoThirdsPi = (float_t)2.09439510239319549;
		float_t beta2 = cos(angle) * (float_t)2;
		float_t beta0 = cos(angle + twoThirdsPi) * (float_t)2;
		float_t beta1 = -(beta0 + beta2);

		// The eigenvalues of A are ordered as alpha0 <= alpha1 <= alpha2.
		eval.x = traceDiv3 + denom * beta0;
		eval.y = traceDiv3 + denom * beta1;
		eval.z = traceDiv3 + denom * beta2;

		// Compute the eigenvectors so that the set {e1, e2, e3}
		// is right handed and orthonormal.
		if (halfDet >= (float_t)0)
		{
			ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval.z, e3);
			ComputeEigenvector1(a00, a01, a02, a11, a12, a22, e3, eval.y, e2);
			e1 = Cross(e2, e3);
		}
		else
		{
			ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval.x, e1);
			ComputeEigenvector1(a00, a01, a02, a11, a12, a22, e1, eval.y, e2);
			e3 = Cross(e1, e2);
		}
	}
	else
	{
		// The matrix is diagonal.
		eval.x = a00;
		eval.y = a11;
		eval.z = a22;
		e1 = make_float3_t( (float_t)1, (float_t)0, (float_t)0 );
		e2 = make_float3_t( (float_t)0, (float_t)1, (float_t)0 );
		e3 = make_float3_t( (float_t)0, (float_t)0, (float_t)1 );
	}

	// The preconditioning scaled the matrix A, which scales the eigenvalues.
	// Revert the scaling.
	eval.x *= maxAbsElement;
	eval.y *= maxAbsElement;
	eval.z *= maxAbsElement;

}
