#ifndef MathStructures_h
#define MathStructures_h

class Vector
{
public:
    double X, Y, Z;
    
    __host__ __device__ Vector()
    {
        X = 0; Y = 0; Z = 0;
    }
    __host__ __device__ Vector(double x, double y, double z)
    {
        X = x; Y = y; Z = z;
    }
    __host__ __device__ Vector(double *Start)
    {
        X = Start[0]; Y = Start[1]; Z = Start[2];
    }
    __host__ __device__ Vector(double I)
    {
        X = Y = Z = I;
    }

    __host__ __device__ double Magnitude()
    {
        return sqrt(X*X + Y*Y + Z*Z);
    }

    __host__ __device__ void operator += (Vector R)
    {
        X += R.X; Y += R.Y; Z += R.Z;
    }
    __host__ __device__ void operator -= (Vector R)
    {
        X -= R.X; Y -= R.Y; Z -= R.Z;
    }
    __host__ __device__ void operator *= (double S)
    {
        X *= S; Y *= S; Z *= S;
    }
    __host__ __device__ void operator /= (double S)
    {
        X /= S; Y /= S; Z /= S;
    }
};

__host__ __device__ Vector operator + (Vector L, Vector R)
{
    return Vector(L.X + R.X, L.Y + R.Y, L.Z + R.Z);
}
__host__ __device__ Vector operator - (Vector L, Vector R)
{
    return Vector(L.X - R.X, L.Y - R.Y, L.Z - R.Z);
}
__host__ __device__ Vector operator - (Vector L)
{
    return Vector(-L.X, -L.Y, -L.Z);
}
__host__ __device__ Vector operator * (Vector L, double S)
{
    return Vector(L.X * S, L.Y * S, L.Z * S);
}
__host__ __device__ Vector operator * (double S, Vector L)
{
    return Vector(L.X * S, L.Y * S, L.Z * S);
}
__host__ __device__ Vector operator / (Vector L, double S)
{
    return Vector(L.X / S, L.Y / S, L.Z / S);
}

__host__ __device__ double operator % (Vector L, Vector R)
{
    return L.X*R.X + L.Y*R.Y + L.Z*R.Z;
}
__host__ __device__ Vector operator * (Vector L, Vector R)
{
    Vector Result;
    Result.X = L.Y * R.Z - R.Y * L.Z;
    Result.Y = R.X * L.Z - L.X * R.Z;
    Result.Z = L.X * R.Y - R.X * L.Y;
    return Result;
}
__host__ __device__ double operator | (Vector L, Vector R)
{
    return (L % R)/L.Magnitude()/R.Magnitude();
}
__host__ __device__ Vector operator ~ (Vector L)
{
    return L / L.Magnitude();
}

__host__ __device__ bool operator == (Vector L, Vector R)
{
    return L.X == R.X && L.Y == R.Y && L.Z == R.Z;
}


#ifdef _GLIBCXX_IOSTREAM
ostream& operator << (ostream& Stream, Vector L)
{
    Stream << "[ " << L.X << " , " << L.Y << " , " << L.Z << " ]";
    return Stream;
}
#endif

class Matrix
{
public:
    double Data[3][3];

    __host__ __device__ Matrix()
    {
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Data[y][x] = 0;
    }
    __host__ __device__ Matrix(double *Start)
    {
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Data[y][x] = Start[x + 3*y];
    }
    __host__ __device__ Matrix(double (*Start)[3])
    {
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Data[y][x] = Start[y][x];
    }
};

__host__ __device__ Matrix operator + (Matrix L, Matrix R)
{
    Matrix Result;
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            Result.Data[y][x] = L.Data[y][x] + R.Data[y][x];
    return Result;
}
__host__ __device__ Matrix operator - (Matrix L, Matrix R)
{
    Matrix Result;
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            Result.Data[y][x] = L.Data[y][x] - R.Data[y][x];
    return Result;
}
__host__ __device__ Matrix operator - (Matrix L)
{
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            L.Data[y][x] = -L.Data[y][x];
    return L;
}
__host__ __device__ Matrix operator * (Matrix L, double S)
{
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            L.Data[y][x] *= S;
    return L;
}
__host__ __device__ Matrix operator * (double S, Matrix L)
{
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            L.Data[y][x] *= S;
    return L;
}
__host__ __device__ Matrix operator / (Matrix L, double S)
{
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            L.Data[y][x] /= S;
    return L;
}

__host__ __device__ Matrix operator * (Matrix L, Matrix R)
{
    Matrix Result;
    for (int y = 0; y < 3; y++)
        for (int i = 0; i < 3; i++)
            for (int x = 0; x < 3; x++)            
                Result.Data[y][x] += L.Data[y][i] * R.Data[i][x];
    return Result;
}
__host__ __device__ Vector operator * (Matrix L, Vector R)
{
    Vector Result;
    Result.X = L.Data[0][0] * R.X + L.Data[0][1] * R.Y + L.Data[0][2] * R.Z;
    Result.Y = L.Data[1][0] * R.X + L.Data[1][1] * R.Y + L.Data[1][2] * R.Z;
    Result.Z = L.Data[2][0] * R.X + L.Data[2][1] * R.Y + L.Data[2][2] * R.Z;
    return Result;
}

__host__ __device__ bool operator == (Matrix L, Matrix R)
{
    bool Ans = true;
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
            Ans = Ans && L.Data[y][x] == R.Data[y][x];
    return Ans;
}

#ifdef _GLIBCXX_IOSTREAM
ostream& operator << (ostream& Stream, Matrix L)
{
    for (int y = 0; y < 3; y++)
    {
        for (int x = 0; x < 3; x++)
            Stream << L.Data[y][x] << " ";
        Stream << endl;
    }
    return Stream;
}
#endif

class Colour
{
public:
    double R, G, B;

    __host__ __device__ Colour()
    {
        R = G = B = 0;
    }
    __host__ __device__ Colour(double r, double g, double b)
    {
        R = r; G = g; B = b;
    }
    __host__ __device__ Colour(double *Start)
    {
        R = Start[0]; G = Start[1]; B = Start[2];
    }
    __host__ __device__ Colour(Vector V)
    {
        R = V.X; G = V.Y; B = V.Z;
    }

    __host__ __device__ operator Vector()
    {
        return Vector(R, G, B);
    }
};

__host__ __device__ Colour operator * (Colour L, Colour R)
{
    return Colour(L.R * R.R, L.G * R.G, L.B * R.B);
}

#endif