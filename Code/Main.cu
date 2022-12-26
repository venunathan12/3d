#include <iostream>
using namespace std;

#include "MathStructures.cu"
#include "MathUtils.cu"

#include "World.cu"
#include "WorldObjects.cu"
#include "Physics.cu"
#include "Camera.cu"

#include "FileManage.cu"

int main()
{
    unsigned char (*Img)[3] = (unsigned char (*)[3]) malloc(OUT_IMG_XSZ * OUT_IMG_YSZ * 3 * sizeof(char));

    World W = LoadWorldFromFile();
    bool Warning = W.RenderGPU(Img, OUT_IMG_YSZ, OUT_IMG_XSZ);
    if (Warning)
        cout << "Warning : 8 bit Overflow" << endl;
    WriteImageToFile(Img);

    free(Img);

    return 0;
}