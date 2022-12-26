#ifndef Kernel_h
#define Kernel_h

#define BLOCKSIZE 64

#define STACKSIZE 8192

#include "Render.cu"
#include "Camera.cu"
#include "Lighting.cu"

__global__ void KernelGetImage(int Ym, int Xm, bool *WarningPin, unsigned char (*Img)[3], unsigned char *Objects, int ObjectsSize, LightSource *Lights, int LightsNum, WorldProperties *Properties, Camera *Cam)
{
    int ID = threadIdx.x + blockIdx.x * BLOCKSIZE;
    if (ID < Ym*Xm)
    {
        int X = ID % Xm, Y = ID / Xm;

        Colour C = GetPixelColour(Cam -> GenerateRay(Y, X), Objects, ObjectsSize, Lights, LightsNum, Properties, 0);

        if (C.R > 1 || C.G > 1 || C.B > 1)
            *WarningPin = true;

        Img[X + Y * Xm][0] = (int) 255.99 * sqrt(C.R);
        Img[X + Y * Xm][1] = (int) 255.99 * sqrt(C.G);
        Img[X + Y * Xm][2] = (int) 255.99 * sqrt(C.B);
    }    
}

#endif