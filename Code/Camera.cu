#ifndef Camera_h
#define Camera_h

#include "MathStructures.cu"
#include "MathUtils.cu"

#include "Physics.cu"

class Camera
{
public:
    Vector G_Center;
    Vector L_TLCorner, L_TRCorner, L_BLCorner, L_Focus;
    Vector P_TRStep, P_BLStep;

    Camera()
    {
        G_Center = Vector();
        L_TLCorner = Vector(-1, 1, 0); L_TRCorner = Vector(1, 1, 0); L_BLCorner = Vector(-1, -1, 0);
        L_Focus = Vector(0, 0, 1);
        P_TRStep = Vector(); P_BLStep = Vector();
    }

    void Transform(Vector Translation, Vector Rotation, Vector Scale)
    {
        Matrix Multiplier = GetTransformMultiplier(Rotation, Scale);
        L_TLCorner = Multiplier * L_TLCorner; L_TRCorner = Multiplier * L_TRCorner; L_BLCorner = Multiplier * L_BLCorner;
        L_Focus = Multiplier * L_Focus;
        G_Center = G_Center + Translation;
    }

    void Setup(int NRows, int NCols)
    {
        P_TRStep = (L_TRCorner - L_TLCorner) / (NCols - 1);
        P_BLStep = (L_BLCorner - L_TLCorner) / (NRows - 1);
    }

    __host__ __device__ Ray GenerateRay(int Y, int X)
    {
        return Ray(G_Center + L_Focus, ~(L_TLCorner + X*P_TRStep + Y*P_BLStep - L_Focus));
    }
};

#endif