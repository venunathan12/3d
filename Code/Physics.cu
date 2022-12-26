#ifndef Physics_h
#define Physics_h

#include "MathStructures.cu"

#include "WorldObjects.cu"

#define CLIP_DISTANCE 1e-6
#define Outer_CLIP_DISTANCE 1e6

class Ray
{
public:
    Vector Origin, Direction;

    __host__ __device__ Ray()
    {
        Origin = Vector(); Direction = Vector();
    }
    __host__ __device__ Ray(Vector O, Vector D)
    {
        Origin = O; Direction = D;
    }
};

class RayCastResult
{
public:
    double Distance;
    int ID;
    Vector SurfaceOutwardNormal;
    Vector HitOutwardNormal;

    __host__ __device__ RayCastResult()
    {
        Distance = -1; ID = -1;
        SurfaceOutwardNormal = Vector(); HitOutwardNormal = Vector();
    }
    __host__ __device__ RayCastResult(double Dist, int Id, Vector SN, Vector HN)
    {
        Distance = Dist; ID = Id;
        SurfaceOutwardNormal = SN; HitOutwardNormal = HN;
    }
};

__host__ __device__ RayCastResult RayCast(Ray R, Plane PL)
{
    R.Origin = R.Origin - PL.G_Center;
    double HitDistance = ((- R.Origin) % (PL.P_Normal)) / (R.Direction % PL.P_Normal);
    Vector HitPoint = R.Origin + HitDistance * R.Direction;

    double XPos = HitPoint % (~ PL.L_AxisX) / PL.L_AxisX.Magnitude();
    double YPos = HitPoint % (~ PL.L_AxisY) / PL.L_AxisY.Magnitude();
    if (XPos < -1 || XPos > 1 || YPos < -1 || YPos > 1 || HitDistance < CLIP_DISTANCE)
        return RayCastResult();
    else
    {
        Vector HitNormal;
        if (R.Direction % PL.P_Normal >= 0)
            HitNormal = - PL.P_Normal;
        else
            HitNormal = PL.P_Normal;
        
        return RayCastResult(HitDistance, 0, PL.P_Normal, HitNormal);
    }
}

__host__ __device__ RayCastResult RayCast(Ray R, Plane *PL)
{
    R.Origin = R.Origin - PL -> G_Center;
    double HitDistance = ((- R.Origin) % (PL -> P_Normal)) / (R.Direction % PL -> P_Normal);
    Vector HitPoint = R.Origin + HitDistance * R.Direction;

    double XPos = HitPoint % (~ PL -> L_AxisX) / PL -> L_AxisX.Magnitude();
    double YPos = HitPoint % (~ PL -> L_AxisY) / PL -> L_AxisY.Magnitude();
    if (XPos < -1 || XPos > 1 || YPos < -1 || YPos > 1 || HitDistance < CLIP_DISTANCE)
        return RayCastResult();
    else
    {
        Vector HitNormal;
        if (R.Direction % PL -> P_Normal >= 0)
            HitNormal = - PL -> P_Normal;
        else
            HitNormal = PL -> P_Normal;
        
        return RayCastResult(HitDistance, 0, PL -> P_Normal, HitNormal);
    }
}

__host__ __device__ RayCastResult RayCast(Ray R, Sphere SP)
{
    R.Origin = R.Origin - SP.G_Center;

    double LegLength = - (R.Origin % R.Direction);
    Vector PerpFromCenter = R.Origin + LegLength * R.Direction;
    double PerpDist = PerpFromCenter.Magnitude();

    if (PerpDist <= SP.P_Radius)
    {
        double Cuts = sqrt(SP.P_Radius * SP.P_Radius - PerpDist * PerpDist);
        if (LegLength - Cuts > CLIP_DISTANCE)
            {
                Vector ContactPoint = R.Origin + (LegLength - Cuts) * R.Direction;
                ContactPoint = ContactPoint / ContactPoint.Magnitude();
                return RayCastResult(LegLength - Cuts, 0, ContactPoint, ContactPoint);
            }
        else if (LegLength + Cuts > CLIP_DISTANCE)
            {
                Vector ContactPoint = R.Origin + (LegLength + Cuts) * R.Direction;
                ContactPoint = ContactPoint / ContactPoint.Magnitude();
                return RayCastResult(LegLength + Cuts, 0, ContactPoint, - ContactPoint);
            }
        else
            return RayCastResult();
    }
    else
        return RayCastResult();
}

__host__ __device__ RayCastResult RayCast(Ray R, Sphere *SP)
{
    R.Origin = R.Origin - SP -> G_Center;

    double LegLength = - (R.Origin % R.Direction);
    Vector PerpFromCenter = R.Origin + LegLength * R.Direction;
    double PerpDist = PerpFromCenter.Magnitude();

    if (PerpDist <= SP -> P_Radius)
    {
        double Cuts = sqrt(SP -> P_Radius * SP -> P_Radius - PerpDist * PerpDist);
        if (LegLength - Cuts > CLIP_DISTANCE)
            {
                Vector ContactPoint = R.Origin + (LegLength - Cuts) * R.Direction;
                ContactPoint = ContactPoint / ContactPoint.Magnitude();
                return RayCastResult(LegLength - Cuts, 0, ContactPoint, ContactPoint);
            }
        else if (LegLength + Cuts > CLIP_DISTANCE)
            {
                Vector ContactPoint = R.Origin + (LegLength + Cuts) * R.Direction;
                ContactPoint = ContactPoint / ContactPoint.Magnitude();
                return RayCastResult(LegLength + Cuts, 0, ContactPoint, - ContactPoint);
            }
        else
            return RayCastResult();
    }
    else
        return RayCastResult();
}

__host__ __device__ RayCastResult RayCast(Ray R, Cuboid CD)
{
    R.Origin = R.Origin - CD.G_Center;

    RayCastResult Result;
    Result.Distance = Outer_CLIP_DISTANCE;

    for (int p = 0; p < 2; p++)
    {
        Vector PC = CD.L_PrimaryCorners[p];
        Vector Origin = R.Origin - PC;

        for (int s = 0; s < 3; s++)
        {
            Vector S0 = CD.L_SecondaryCorners[p][s] - PC;
            Vector S1 = CD.L_SecondaryCorners[p][(s + 1) % 3] - PC;

            Vector Normal = ~(S0 * S1);

            double HitDistance = - (Origin % Normal) / (R.Direction % Normal);
            if (HitDistance > CLIP_DISTANCE && HitDistance < Result.Distance)
            {
                Vector HitPoint = Origin + HitDistance * R.Direction;

                double CompS0 = HitPoint % (~ S0) / S0.Magnitude();
                double CompS1 = HitPoint % (~ S1) / S1.Magnitude();

                if (CompS0 >= 0 && CompS0 <= 1 && CompS1 >= 0 && CompS1 <= 1)
                {
                    Result.Distance = HitDistance;
                    Result.ID = 3*p + s;
                    Result.SurfaceOutwardNormal = Normal;
                    if (Normal % R.Direction > 0)
                        Result.HitOutwardNormal = - Normal;
                    else
                        Result.HitOutwardNormal = Normal;
                }
            }
        }
    }

    return Result;
}

__host__ __device__ RayCastResult RayCast(Ray R, Tetrahedron TT)
{
    R.Origin = R.Origin - TT.G_Center;

    RayCastResult Result;
    Result.Distance = Outer_CLIP_DISTANCE;

    Vector Corners[3];
    Corners[0] = Vector();
    Vector Origin = R.Origin - TT.L_Corners[0];
    for (int s = 0; s < 3; s++)
    {
        Corners[1] = TT.L_Corners[1 + s] - TT.L_Corners[0];
        Corners[2] = TT.L_Corners[1 + (s + 1) % 3] - TT.L_Corners[0];

        Vector Normal = ~(Corners[1] * Corners[2]);

        double HitDistance = - (Origin % Normal) / (R.Direction % Normal);
        if (HitDistance > CLIP_DISTANCE && HitDistance < Result.Distance)
        {
            Vector HitPoint = Origin + HitDistance * R.Direction;

            bool Inside = true;
            for (int i = 0; i < 3 && Inside; i++)
            {
                Vector HitPointRel = HitPoint - Corners[i];
                Vector L1 = Corners[(i + 1) % 3] - Corners[i];
                Vector L2 = Corners[(i + 2) % 3] - Corners[i];

                double Dot = (HitPointRel * L1) % (HitPointRel * L2);
                if (Dot > 0)
                    Inside = false;
            }
            if (Inside)
            {
                Result.Distance = HitDistance;
                Result.ID = s;
                Result.SurfaceOutwardNormal = Normal;
                if (Normal % R.Direction > 0)
                        Result.HitOutwardNormal = - Normal;
                    else
                        Result.HitOutwardNormal = Normal;
            }
        }
    }

    Corners[0] = Vector(); Corners[1] = TT.L_Corners[3] - TT.L_Corners[1]; Corners[2] = TT.L_Corners[2] - TT.L_Corners[1];
    Origin = R.Origin - TT.L_Corners[1]; Vector Normal = ~(Corners[1] * Corners[2]);

    double HitDistance = - (Origin % Normal) / (R.Direction % Normal);
    if (HitDistance > CLIP_DISTANCE && HitDistance < Result.Distance)
    {
        Vector HitPoint = Origin + HitDistance * R.Direction;

        bool Inside = true;
        for (int i = 0; i < 3 && Inside; i++)
        {
            Vector HitPointRel = HitPoint - Corners[i];
            Vector L1 = Corners[(i + 1) % 3] - Corners[i];
            Vector L2 = Corners[(i + 2) % 3] - Corners[i];

            double Dot = (HitPointRel * L1) % (HitPointRel * L2);
            if (Dot > 0)
                Inside = false;
        }
        if (Inside)
        {
            Result.Distance = HitDistance;
            Result.ID = 3;
            Result.SurfaceOutwardNormal = Normal;
            if (Normal % R.Direction > 0)
                    Result.HitOutwardNormal = - Normal;
                else
                    Result.HitOutwardNormal = Normal;
        }
    }

    return Result;
}

#endif