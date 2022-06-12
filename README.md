# Surface Fluid Simulation via Mesh Parameterization
Novel approach to mesh surface fluid simulation that I came up with building off of previous research in mesh parameterization. Algorithm geared toward application in real-time games/visual effects. Code features a CPU-based proof-of-concept.

## Included in Repository

- Python Code
- Assets for Testing
- Research Paper
- Figures

## Demo Video

https://youtu.be/Jbz7SobDzok

## How to Run

After installing dependencies, open the terminal/command prompt in the src directory and run the following command: 

python render.py

## Dependencies

Most of the following dependencies are python packages that can be installed using pip, with the exception of GLUT. Instructions for installing GLUT coming soon...

- Python 2.7
- GLUT
- PyOpenGL
- OpenGLContext
- PyVRML97
- chart_studio
- PIL
- numpy
- scipy
- trimesh
- pyglet
- matplotlib

## Next Steps

Porting code to C++ to take advantage of DirectX 12 and/or Cuda for GPU acceleration and to provide a more realistic visualization. Looking into using libigl for mesh parameterization and Nvidia libraries for fluid simulation.
