#ifndef Lighting_h
#define Lighting_h

#include "MathStructures.cu"

class LightSource
{
public:
    Vector G_Center;
    Colour P_Power;

    LightSource()
    {
        G_Center = Vector();
        P_Power = Colour();
    }

    LightSource(Vector Center, Colour Power)
    {
        G_Center = Center;
        P_Power = Power;
    }
};

#endif