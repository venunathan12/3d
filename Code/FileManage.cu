#ifndef FileManage_h
#define FileManage_h

#include <stdio.h>

#include <iostream>
#include <fstream>
using namespace std;

#define OUT_IMG_XSZ 2048
#define OUT_IMG_YSZ 1024

#define InputPath "SceneDescription.txt"
#define OutputPath "Image.ppm"

#include "World.cu"
#include "WorldObjects.cu"
#include "Camera.cu"
#include "Lighting.cu"

World LoadWorldFromFile()
{
    ifstream IN; IN.open(InputPath);
    string Tag = "";

    int LightsNum = 0, ObjectsSize = 0;

    while (Tag != "END")
    {
        if (Tag == "OBJ")
        {
            string Type;
            IN >> Type;
            if (Type == "PLANE")
                ObjectsSize += sizeof(Plane);
            else if (Type == "SPHERE")
                ObjectsSize += sizeof(Sphere);
            else if (Type == "CUBOID")
                ObjectsSize += sizeof(Cuboid);
            else if (Type == "TETRAHEDRON")
                ObjectsSize += sizeof(Tetrahedron);
        }
        else if (Tag == "LIGHT")
            LightsNum ++;

        IN >> Tag;
    }

    WorldProperties *Properties = new WorldProperties;
    unsigned char *Objects = new unsigned char[ObjectsSize]; int ObjStart = 0;
    LightSource *Lights = new LightSource[LightsNum]; int LightsStart = 0;
    Camera *Cam = new Camera;
    Material TempMat;

    IN.close();
    IN.open(InputPath);

    Tag = "";
    while (Tag != "END")
    {
        if (Tag == "SET_BG")
            IN >> Properties -> Background[0] >> Properties -> Background[1] >> Properties -> Background[2];
        else if (Tag == "SET_GI")
            IN >> Properties -> GlobalIllumination[0] >> Properties -> GlobalIllumination[1] >> Properties -> GlobalIllumination[2] >> Properties -> MaxGlobalIlluminationFraction;
        
        else if (Tag == "SET_MAT")
        {
            IN >> TempMat.Polish;
            IN >> TempMat.Reflectivity[0] >> TempMat.Reflectivity[1] >> TempMat.Reflectivity[2];
            IN >> TempMat.Transmitivity[0] >> TempMat.Transmitivity[1] >> TempMat.Transmitivity[2];
            IN >> TempMat.Absorptivity[0] >> TempMat.Absorptivity[1] >> TempMat.Absorptivity[2];
            IN >> TempMat.RefractiveIndex;
        }

        else if (Tag == "LIGHT")
        {
            Vector Origin; Colour Power;
            IN >> Origin.X >> Origin.Y >> Origin.Z;
            IN >> Power.R >> Power.G >> Power.B;

            Lights[LightsStart] = LightSource(Origin, Power);
            LightsStart ++;
        }
        else if (Tag == "OBJ")
        {
            string Type;
            IN >> Type;

            Vector Translation, Rotation, Scale;
            IN >> Translation.X >> Translation.Y >> Translation.Z;
            IN >> Rotation.X >> Rotation.Y >> Rotation.Z;
            IN >> Scale.X >> Scale.Y >> Scale.Z;

            if (Type == "PLANE")
            {
                Plane *Target = (Plane *) (Objects + ObjStart);
                (*Target) = Plane();
                Target -> Mat = TempMat;
                Target -> Transform(Translation, Rotation * (PI() / 180), Scale);
                ObjStart += sizeof(Plane);
            }
            else if (Type == "SPHERE")
            {
                Sphere *Target = (Sphere *) (Objects + ObjStart);
                *Target = Sphere();
                Target -> Mat = TempMat;
                Target -> Transform(Translation, Rotation * (PI() / 180), Scale);
                ObjStart += sizeof(Sphere);
            }
            else if (Type == "CUBOID")
            {
                Cuboid *Target = (Cuboid *) (Objects + ObjStart);
                *Target = Cuboid();
                Target -> Mat = TempMat;
                Target -> Transform(Translation, Rotation * (PI() / 180), Scale);
                ObjStart += sizeof(Cuboid);
            }
            else if (Type == "TETRAHEDRON")
            {
                Tetrahedron *Target = (Tetrahedron *) (Objects + ObjStart);
                *Target = Tetrahedron();
                Target -> Mat = TempMat;
                Target -> Transform(Translation, Rotation * (PI() / 180), Scale);
                ObjStart += sizeof(Tetrahedron);
            }
        }
        else if (Tag == "SET_CAM")
        {
            Vector Translation, Rotation, Scale;
            IN >> Translation.X >> Translation.Y >> Translation.Z;
            IN >> Rotation.X >> Rotation.Y >> Rotation.Z;
            IN >> Scale.X >> Scale.Y >> Scale.Z;

            Cam -> Transform(Translation, Rotation * (PI() / 180), Scale);
        }

        IN >> Tag;
    }
    return World(Properties, Objects, ObjectsSize, Lights, LightsNum, Cam);
}

void WriteImageToFile(unsigned char (*Img)[3])
{
    FILE *Out = fopen(OutputPath, "w");
    fprintf(Out, "P3\n%d %d\n255\n", OUT_IMG_XSZ, OUT_IMG_YSZ);
    for (int y = 0; y < OUT_IMG_YSZ; y++)
        for (int x = 0; x < OUT_IMG_XSZ; x++)
            fprintf(Out, "%d %d %d\n", Img[x + y * OUT_IMG_XSZ][0], Img[x + y * OUT_IMG_XSZ][1], Img[x + y * OUT_IMG_XSZ][2]);
}

void GenerateWorldSummary(World *W)
{
    cout << "Summary of World : " << W << endl;

    cout << "\t" << "Properties :" << endl;
    cout << "\t\t" << "BG : " << W -> Properties -> Background[0] << ' ' << W -> Properties->Background[1] << ' ' << W -> Properties -> Background[2] << endl;
    cout << "\t\t" << "GI : " << W -> Properties -> GlobalIllumination[0] << ' ' << W -> Properties -> GlobalIllumination[1] << ' ' << W -> Properties -> GlobalIllumination[2] << endl;

    cout << "\t" << "Camera :" << endl;
    cout << "\t\t" << "Focus : " << W -> Cam -> L_Focus << endl;
    cout << "\t\t" << "Center : " << W -> Cam -> G_Center << endl;

    cout << "\t" << "Lighting :" << endl;
    for (int n = 0; n < W -> LightsNum; n++)
        cout << "\t\t" << "Source at : " << W -> Lights[n].G_Center << "; Power : " << W -> Lights[n].P_Power << endl;
    
    cout << '\t' << "Objects : (Total Size = " << W -> ObjectsSize << " Bytes)" << endl;

    int ObjIdx = 0;
    while (ObjIdx < W -> ObjectsSize)
    {
        cout << "\t\t" << "At Idx = " << ObjIdx << endl;
        
        EmptyMaterial *TargMat = (EmptyMaterial *) (W -> Objects + ObjIdx);
        cout << "\t\t\t" << "ID : " << TargMat -> ID << endl;
        
        cout << "\t\t\t" << "Polish : " << TargMat -> Mat.Polish << endl;
        cout << "\t\t\t" << "Ref : ";
        for (int i = 0; i < 3; i++)
            cout << TargMat -> Mat.Reflectivity[i] << ' ';
        cout << endl;
        cout << "\t\t\t" << "Trn : ";
        for (int i = 0; i < 3; i++)
            cout << TargMat -> Mat.Transmitivity[i] << ' ';
        cout << endl;
        cout << "\t\t\t" << "Abs : ";
        for (int i = 0; i < 3; i++)
            cout << TargMat -> Mat.Absorptivity[i] << ' ';
        cout << endl;
        cout << "\t\t\t" << "RI : " << TargMat -> Mat.RefractiveIndex << endl;

        if (TargMat -> ID[0] == 'P')
        {
            Plane *Target = (Plane *) TargMat;
            cout << "\t\t\t\t" << "Center " << Target -> G_Center <<  endl;
            ObjIdx += sizeof(Plane);
        }
        else if (TargMat -> ID[0] == 'S')
        {
            Sphere *Target = (Sphere *) TargMat;
            cout << "\t\t\t\t" << "Center " << Target -> G_Center << endl;
            cout << "\t\t\t\t" << "Radius " << Target -> P_Radius << endl;
            ObjIdx += sizeof(Sphere);
        }
        else if (TargMat -> ID[0] == 'C')
        {
            Cuboid *Target = (Cuboid *) TargMat;
            cout << "\t\t\t\t" << "Center " << Target -> G_Center << endl;
            ObjIdx += sizeof(Cuboid);
        }
        else if (TargMat -> ID[0] == 'T')
        {
            Tetrahedron *Target = (Tetrahedron *) TargMat;
            cout << "\t\t\t\t" << "Center " << Target -> G_Center << endl;
            ObjIdx += sizeof(Tetrahedron);
        }
    }
}

#endif