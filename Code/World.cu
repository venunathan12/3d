#ifndef World_h
#define World_h

#include "MathStructures.cu"

class WorldProperties
{
public:
    double Background[3];
    double GlobalIllumination[3];
    double MaxGlobalIlluminationFraction;

    WorldProperties()
    {
        for (int i = 0; i < 3; i++)
            Background[i] = GlobalIllumination[i] = 0;
        MaxGlobalIlluminationFraction = 0.25;
    }
    WorldProperties(Vector Back, Vector Global, double Fraction)
    {
        Background[0] = Back.X; Background[1] = Back.Y; Background[2] = Back.Z;
        GlobalIllumination[0] = Global.X; GlobalIllumination[1] = Global.Y; GlobalIllumination[2] = Global.Z;
        MaxGlobalIlluminationFraction = Fraction;
    }
};

#include "Render.cu"
#include "Camera.cu"
#include "Lighting.cu"

#include "Kernel.cu"

class World
{
public:

    WorldProperties *Properties;
    unsigned char *Objects; int ObjectsSize;
    LightSource *Lights; int LightsNum;
    Camera *Cam;

    World()
    {
        Properties = NULL;
        Objects = NULL; ObjectsSize = 0;
        Lights = NULL; LightsNum = 0;
        Cam = NULL;
    }
    World(WorldProperties *Pr, unsigned char *Obj, int ObjSz, LightSource *Lht, int LhtN, Camera *C)
    {
        Properties = Pr;
        Objects = Obj; ObjectsSize = ObjSz;
        Lights = Lht; LightsNum = LhtN;
        Cam = C;
    }
    ~World()
    {
        delete Properties;
        delete Objects; delete Lights;
        delete Cam;
    }

    bool RenderGPU(unsigned char (*Img)[3], int NRows, int NCols)
    {
        Cam -> Setup(NRows, NCols);

        cudaDeviceSetLimit(cudaLimitStackSize, STACKSIZE);

        unsigned char *GPU_Img, *GPU_Objects;
        cudaMalloc(&GPU_Img, NRows * NCols * 3 * sizeof(char));
        cudaMalloc(&GPU_Objects, ObjectsSize * sizeof(char));
        LightSource *GPU_Lights;
        cudaMalloc(&GPU_Lights, LightsNum * sizeof(LightSource));
        WorldProperties *GPU_Properties;
        cudaMalloc(&GPU_Properties, sizeof(WorldProperties));
        Camera *GPU_Cam;
        cudaMalloc(&GPU_Cam, sizeof(Camera));

        cudaMemcpy(GPU_Objects, Objects, ObjectsSize * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_Lights, Lights, LightsNum * sizeof(LightSource), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_Properties, Properties, sizeof(WorldProperties), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_Cam, Cam, sizeof(Camera), cudaMemcpyHostToDevice);

        bool *WarningPin;
        cudaHostAlloc(&WarningPin, sizeof(int), 0);
        
        KernelGetImage <<< (NRows * NCols - 1) / BLOCKSIZE + 1, BLOCKSIZE >>> (NRows, NCols, WarningPin, (unsigned char (*)[3]) GPU_Img, GPU_Objects, ObjectsSize, GPU_Lights, LightsNum, GPU_Properties, GPU_Cam);

        cudaDeviceSynchronize();
        cudaError_t E = cudaGetLastError();
        printf("GPU Last Error : %s\n", cudaGetErrorString(E));
        
        cudaMemcpy(Img, GPU_Img, NRows * NCols * 3 * sizeof(char), cudaMemcpyDeviceToHost);

        cudaFree(GPU_Img);
        cudaFree(GPU_Objects);
        cudaFree(GPU_Lights);
        cudaFree(GPU_Properties);
        cudaFree(GPU_Cam);

        bool Return = *WarningPin;
        cudaFree(WarningPin);        

        return Return;
    }

    bool RenderCPU(unsigned char (*Img)[3], int NRows, int NCols)
    {
        bool Warning = false;
        Cam -> Setup(NRows, NCols);

        for (int y = 0; y < NRows; y++)
            for (int x = 0; x < NCols; x++)
            {
                Colour C = GetPixelColour(Cam -> GenerateRay(y, x), Objects, ObjectsSize, Lights, LightsNum, Properties, 0);
                if (C.R >= 1 || C.G >= 1 || C.B >= 1)
                    Warning = true;

                Img[x + y*NCols][0] = (int) 255.99 * sqrt(C.R);
                Img[x + y*NCols][1] = (int) 255.99 * sqrt(C.G);
                Img[x + y*NCols][2] = (int) 255.99 * sqrt(C.B);
            }
        return Warning;
    }
};

#endif