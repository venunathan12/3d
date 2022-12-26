#ifndef MathUtils_h
#define MathUtils_h

#include "MathStructures.cu"

__host__ __device__ double PI()
{
    return 4 * atan(1.0);
}

__host__ __device__ Matrix GetIdentityMatrix()
{
    Matrix Result;
    Result.Data[0][0] = 1; Result.Data[1][1] = 1; Result.Data[2][2] = 1;
    return Result;
}

__host__ __device__ Matrix GetRotationMatrix(double X, double Y, double Z)
{
    Matrix Result, Step;

    Step = Matrix();
    Step.Data[0][0] = 1;
    Step.Data[1][1] = Step.Data[2][2] = cos(X);
    Step.Data[2][1] = sin(X); Step.Data[1][2] = - sin(X);
    Result = Step;

    Step = Matrix();
    Step.Data[1][1] = 1;
    Step.Data[0][0] = Step.Data[2][2] = cos(Y);
    Step.Data[0][2] = sin(Y); Step.Data[2][0] = - sin(Y);
    Result = Step * Result;

    Step = Matrix();
    Step.Data[2][2] = 1;
    Step.Data[0][0] = Step.Data[1][1] = cos(Z);
    Step.Data[1][0] = sin(Z); Step.Data[0][1] = - sin(Z);
    Result = Step * Result;

    return Result;
}

__host__ __device__ Matrix GetRotationMatrix(Vector Rotation)
{
    return GetRotationMatrix(Rotation.X, Rotation.Y, Rotation.Z);
}

__host__ __device__ Matrix GetScaleMatrix(double X, double Y, double Z)
{
    Matrix Result;
    Result.Data[0][0] = X; Result.Data[1][1] = Y; Result.Data[2][2] = Z;
    return Result;
}

__host__ __device__ Matrix GetScaleMatrix(double S)
{
    return GetScaleMatrix(S, S, S);
}

__host__ __device__ Matrix GetScaleMatrix(Vector Scale)
{
    return GetScaleMatrix(Scale.X, Scale.Y, Scale.Z);
}

__host__ __device__ Matrix GetTransformMultiplier(Vector Rotation, Vector Scale)
{
    return GetRotationMatrix(Rotation) * GetScaleMatrix(Scale);
}

#endif