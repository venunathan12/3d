# 3d

Note:
There is no License attached to this repository.<br>
Hence, as stated in [GitHub documentation about Licensing](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository#choosing-the-right-license), default copyright laws apply.

A mesh generation algorithm which allows the user to define the maximum cell size at every point in the domain.<br>
The mesh starts as a collection of uniformly sized cells. Each cell is divided into smaller cells depending on the maximum cell size stated by the user.<br>
The process of dividing the cell is implemented as a Quad Tree and parallelized using Open MP.

This is a course project for CS6023: GPU Programming.

Please check 'Demo Presentation.pdf', 'README.pdf' and 'Usage Instructions.pdf' for full details on the project.<br>
The folder named 'Code' contains the source code for the project.
