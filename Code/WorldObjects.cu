#ifndef WorldObjects_h
#define WorldObjects_h

#include "MathStructures.cu"
#include "MathUtils.cu"

#include "PhysicsMaterial.cu"

class EmptyName
{
public:
    char ID[8];

    EmptyName() = delete;
};

class EmptyMaterial
{
public:
    char ID[8];
    Material Mat;

    EmptyMaterial() = delete;
};

class Plane
{
public:
    char ID[8];
    Material Mat;
    
    Vector G_Center;
    Vector L_AxisX, L_AxisY;
    Vector P_Normal;

    Plane()
    {
        ID[0] = 'P'; ID[1] = 'L'; ID[2] = 'N'; ID[3] = '\0';
        Mat = Material();

        G_Center = Vector();
        L_AxisX = Vector(1, 0, 0); L_AxisY = Vector(0, 1, 0);
        P_Normal = Vector(0, 0, 1);
    }

    void Transform(Vector Translation, Vector Rotation, Vector Scale)
    {
        Matrix Multiplier = GetTransformMultiplier(Rotation, Scale);
        L_AxisX = Multiplier * L_AxisX; L_AxisY = Multiplier * L_AxisY;
        G_Center = G_Center + Translation;
        P_Normal = ~ (L_AxisX * L_AxisY);
    }
};

class Sphere
{
public:
    char ID[8];
    Material Mat;

    Vector G_Center;
    double P_Radius;

    Sphere()
    {
        ID[0] = 'S'; ID[1] = 'P'; ID[2] = 'H'; ID[3] = '\0';
        Mat = Material();

        G_Center = Vector();
        P_Radius = 1;
    }

    void Transform(Vector Translation, Vector Rotation, Vector Scale)
    {
        G_Center = G_Center + Translation;
        if (Scale.X == Scale.Y && Scale.Y == Scale.Z)
            P_Radius *= Scale.X;
    }
};

class Tetrahedron
{
public:
    char ID[8];
    Material Mat;

    Vector G_Center;
    Vector L_Corners[4];

    Tetrahedron()
    {
        ID[0] = 'T'; ID[1] = 'E'; ID[2] = 'T'; ID[3] = '\0';
        Mat = Material();

        G_Center = Vector();
        L_Corners[0] = Vector(0, 0, sqrt(2));
        L_Corners[1] = Vector(1, 0, 0);
        L_Corners[2] = Vector(-0.5, sqrt(3)/2, 0);
        L_Corners[3] = Vector(-0.5, -sqrt(3)/2, 0);
    }

    void Transform(Vector Translation, Vector Rotation, Vector Scale)
    {
        Matrix Multiplier = GetTransformMultiplier(Rotation, Scale);
        for (int c = 0; c < 4; c ++)
            L_Corners[c] = Multiplier * L_Corners[c];
        G_Center = G_Center + Translation;
    }
};

class Cuboid
{
public:
    char ID[8];
    Material Mat;

    Vector G_Center;
    Vector L_PrimaryCorners[2];
    Vector L_SecondaryCorners[2][3];

    Cuboid()
    {
        ID[0] = 'C'; ID[1] = 'B'; ID[2] = 'D'; ID[3] = '\0';
        Mat = Material();

        L_PrimaryCorners[0] = Vector(-1, -1, -1);
        L_SecondaryCorners[0][0] = Vector(1, -1, -1);
        L_SecondaryCorners[0][1] = Vector(-1, -1, 1);
        L_SecondaryCorners[0][2] = Vector(-1, 1, -1);
        L_PrimaryCorners[1] = Vector(1, 1, 1);
        L_SecondaryCorners[1][0] = Vector(-1, 1, 1);
        L_SecondaryCorners[1][1] = Vector(1, -1, 1);
        L_SecondaryCorners[1][2] = Vector(1, 1, -1);
    }

    void Transform(Vector Translation, Vector Rotation, Vector Scale)
    {
        Matrix Multiplier = GetTransformMultiplier(Rotation, Scale);
        for (int p = 0; p < 2; p++)
        {
            L_PrimaryCorners[p] = Multiplier * L_PrimaryCorners[p];
            for (int s = 0; s < 3; s++)
                L_SecondaryCorners[p][s] = Multiplier * L_SecondaryCorners[p][s];
        }
        G_Center = G_Center + Translation;
    }
};

#endif