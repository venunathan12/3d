#ifndef PhysicsMaterial_h
#define PhysicsMaterial_h

class Material
{
public:
    double Polish;
    double Reflectivity[3];
    double Transmitivity[3];
    double Absorptivity[3];
    double RefractiveIndex;

    Material()
    {
        Polish = RefractiveIndex = 0;
        for (int i = 0; i < 3; i++)
            Reflectivity[i] = Transmitivity[i] = Absorptivity[i] = 0;
    }
};

#endif