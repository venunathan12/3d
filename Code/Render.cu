#ifndef Render_h
#define Render_h

#include "MathStructures.cu"

#include "WorldObjects.cu"
#include "Lighting.cu"
#include "Physics.cu"

#define MaxDepth 4

__host__ __device__ Colour GetPhongColour(Vector Point, Vector SurfaceNormal, double CastDistance, unsigned char *Objects, int ObjectsSize, LightSource *Lights, int LightsNum, WorldProperties *Properties)
{
    Colour Result = Vector(Properties -> GlobalIllumination) / (4*PI()*CastDistance*CastDistance);
    double MaxGlobalIlluminationFraction = Properties -> MaxGlobalIlluminationFraction;
    double MaxInResult = -1;
    if (Result.R > MaxInResult) MaxInResult = Result.R;
    if (Result.G > MaxInResult) MaxInResult = Result.G;
    if (Result.B > MaxInResult) MaxInResult = Result.B;
    if (MaxInResult > MaxGlobalIlluminationFraction)
        Result = Result * (MaxGlobalIlluminationFraction / MaxInResult);

    for (int SourceIdx = 0; SourceIdx < LightsNum; SourceIdx ++)
    {
        Vector ToSource = Lights[SourceIdx].G_Center - Point;
        double PresentDistance = ToSource.Magnitude();
        Vector DirectionToSource = ToSource / PresentDistance;
        Ray RayToSource = Ray(Point, DirectionToSource);

        Colour Transmission = Vector(1, 1, 1);

        int ObjIdx = 0;
        while (ObjIdx < ObjectsSize)
        {
            EmptyName* Name = (EmptyName *) (Objects + ObjIdx);

            if (Name -> ID[0] == 'P')
            {
                Plane ObjectCopy = *((Plane *) (Objects + ObjIdx));

                RayCastResult Rx = RayCast(RayToSource, ObjectCopy);
                if (Rx.Distance > 0 && Rx.Distance < PresentDistance)
                    Transmission = Transmission * Colour(ObjectCopy.Mat.Transmitivity) * ObjectCopy.Mat.Polish;                

                ObjIdx += sizeof(Plane);
            }
            else if (Name -> ID[0] == 'S')
            {
                Sphere ObjectCopy = *((Sphere *) (Objects + ObjIdx));

                RayCastResult Rx = RayCast(RayToSource, ObjectCopy);
                if (Rx.Distance > 0 && Rx.Distance < PresentDistance)
                    Transmission = Transmission * Colour(ObjectCopy.Mat.Transmitivity) * ObjectCopy.Mat.Polish;

                ObjIdx += sizeof(Sphere);
            }
            else if (Name -> ID[0] == 'C')
            {
                Cuboid ObjectCopy = *((Cuboid *) (Objects + ObjIdx));

                RayCastResult Rx = RayCast(RayToSource, ObjectCopy);
                if (Rx.Distance > 0 && Rx.Distance < PresentDistance)
                    Transmission = Transmission * Colour(ObjectCopy.Mat.Transmitivity) * ObjectCopy.Mat.Polish;

                ObjIdx += sizeof(Cuboid);
            }
            else if (Name -> ID[0] == 'T')
            {
                Tetrahedron ObjectCopy = *((Tetrahedron *) (Objects + ObjIdx));

                RayCastResult Rx = RayCast(RayToSource, ObjectCopy);
                if (Rx.Distance > 0 && Rx.Distance < PresentDistance)
                    Transmission = Transmission * Colour(ObjectCopy.Mat.Transmitivity) * ObjectCopy.Mat.Polish;

                ObjIdx += sizeof(Tetrahedron);
            }
        }

        Colour LightEmission = Lights[SourceIdx].P_Power;
        double CosineFactor = SurfaceNormal % DirectionToSource;
        if (CosineFactor < 0)
            CosineFactor *= -1;
        Result = Result + CosineFactor * (Transmission * LightEmission) / (4*PI()*PresentDistance*PresentDistance);
    }

    return Result;
}

__host__ __device__ Colour GetPixelColour(Ray R, unsigned char *Objects, int ObjectsSize, LightSource *Lights, int LightsNum, WorldProperties *Properties, int Depth)
{
    RayCastResult Result;
    Result.Distance = Outer_CLIP_DISTANCE;

    int ObjIdx = 0;
    while (ObjIdx < ObjectsSize)
    {
        EmptyName* Name = (EmptyName *) (Objects + ObjIdx);
        
        if (Name -> ID[0] == 'P')
        {
            RayCastResult Rx = RayCast(R, (Plane *) (Objects + ObjIdx));
            if (Rx.Distance > 0 && Rx.Distance < Result.Distance)
            {
                Result = Rx;
                Result.ID = ObjIdx;
            }
            ObjIdx += sizeof(Plane);
        }
        else if (Name -> ID[0] == 'S')
        {
            RayCastResult Rx = RayCast(R, (Sphere *) (Objects + ObjIdx));
            if (Rx.Distance > 0 && Rx.Distance < Result.Distance)
            {
                Result = Rx;
                Result.ID = ObjIdx;
            }
            ObjIdx += sizeof(Sphere);
        }
        else if (Name -> ID[0] == 'C')
        {
            RayCastResult Rx = RayCast(R, *((Cuboid *) (Objects + ObjIdx)));
            if (Rx.Distance > 0 && Rx.Distance < Result.Distance)
            {
                Result = Rx;
                Result.ID = ObjIdx;
            }
            ObjIdx += sizeof(Cuboid);
        }
        else if (Name -> ID[0] == 'T')
        {
            RayCastResult Rx = RayCast(R, *((Tetrahedron *) (Objects + ObjIdx)));
            if (Rx.Distance > 0 && Rx.Distance < Result.Distance)
            {
                Result = Rx;
                Result.ID = ObjIdx;
            }
            ObjIdx += sizeof(Tetrahedron);
        }
    }

    if (Result.ID >= 0)
    {
        EmptyMaterial *Target = (EmptyMaterial *) (Objects + Result.ID);
        Vector ContactPoint = R.Origin + R.Direction * Result.Distance;
        
        Colour ContactPhong = GetPhongColour(ContactPoint + 2 * CLIP_DISTANCE * Result.HitOutwardNormal, Result.HitOutwardNormal, Result.Distance, Objects, ObjectsSize, Lights, LightsNum, Properties);
        Colour DiffuseMultiplier = (1 - Colour(Target -> Mat.Absorptivity));

        ContactPhong = ContactPhong * DiffuseMultiplier;

        if (Target -> Mat.Polish == 0 || Depth >= MaxDepth)
            return ContactPhong;
        
        Colour Reflection = GetPixelColour(Ray(ContactPoint, R.Direction - 2 * (R.Direction % Result.HitOutwardNormal) * Result.HitOutwardNormal), Objects, ObjectsSize, Lights, LightsNum, Properties, Depth + 1);
        
        double RefPerpCompLength = - R.Direction % Result.HitOutwardNormal;
        double RefParlCompLength = sqrt(1 - RefPerpCompLength*RefPerpCompLength);
        Vector RefParlCompVector = R.Direction + RefPerpCompLength * Result.HitOutwardNormal;

        Vector RefrDirn;
        double SineMultiplier; double RI = Target -> Mat.RefractiveIndex;
        if (Result.HitOutwardNormal % Result.SurfaceOutwardNormal > 0)
            SineMultiplier = 1 / RI;
        else
            SineMultiplier = RI;
        double SineAfterInterface = SineMultiplier * RefParlCompLength;
        if (SineAfterInterface <= 1)
        {
            double LengthAfterInterface = RefParlCompLength / SineAfterInterface;
            double LengthAfterInterfacePerp = sqrt(LengthAfterInterface*LengthAfterInterface - RefParlCompLength*RefParlCompLength);
            RefrDirn = ~ (RefParlCompVector - LengthAfterInterfacePerp * Result.HitOutwardNormal);
        }
        else
            RefrDirn = R.Direction + 2 * RefPerpCompLength * Result.HitOutwardNormal;

        Colour Refraction = GetPixelColour(Ray(ContactPoint, RefrDirn), Objects, ObjectsSize, Lights, LightsNum, Properties, Depth + 1);

        Colour Reflectivity = Colour(Target -> Mat.Reflectivity); Colour Transmitivity = Colour(Target -> Mat.Transmitivity);
        double Polish = Target -> Mat.Polish;

        Colour CombinedColour = (1 - Polish) * ContactPhong + Polish * (Reflection * Reflectivity + Refraction * Transmitivity);
        return CombinedColour;
    }
    else
    {
        return Colour(Properties -> Background);
    }
}

#endif