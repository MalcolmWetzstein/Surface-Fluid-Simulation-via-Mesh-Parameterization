# -*- coding: utf-8 -*-
"""
Created on Sat May 27 02:57:50 2017

@author: Malcolm
"""

from __future__ import print_function
from OpenGLContext import testingcontext
BaseContext = testingcontext.getInteractive()
from OpenGLContext import texture
from OpenGL.GL import *
from OpenGL.GL.ARB.multitexture import *
import sys
from OpenGLContext.events.timer import Timer
from OpenGL.extensions import alternate

import PIL
import hybrid
import rasterizer
import FluidSim
import random

SIMULATE = True
USE_TEXTURE = False

SIM_SIZE = 256
SUBDIVIDE_FACTOR = 1
UP = [0.0, -1.0, 0.0]
X_DIR = [0.0, 0.0, -1.0]
Y_DIR = [1.0, 0.0, 0.0]
MAX_PARTICLES = SIM_SIZE*SIM_SIZE*100
CELL_SIZE = 1.0/SIM_SIZE
DENSITY = 1000.0
GRAVITY = rasterizer.np.array([0.0, 10.0, 0.0])
ATM_PRESSURE = 0.0

TIME_STEP = 0.03
INJECT_POINTS = []
for i in range(24):
    INJECT_POINTS.append((int(random.random()*SIM_SIZE/8-SIM_SIZE/16)+SIM_SIZE/2, int(random.random()*SIM_SIZE/8-SIM_SIZE/16)+SIM_SIZE/2))
PARTICLE_INJECT_COUNT = 15
RENDER_MAX = int(20/SUBDIVIDE_FACTOR)
XV_INJECT = 0.0
YV_INJECT = 0.0
P_INJECT = 0.0
INJECT_RADIUS = 3

def loadMesh(path, parameterize = False, L = 0, cholesky = False):
    V, U, N, I, J, K = hybrid.Utl.load_obj(path)
    T = hybrid.Utl.np.column_stack((I,J,K))
    if parameterize:
        U = hybrid.hybrid_parameterize(V, T, U, L, None, None, 100, cholesky)
    
    vBuffer = V
    tBuffer = U
    nBuffer = N
    cBuffer = []
    for i in range(len(V)):
        cBuffer.append([1.0, 1.0, 1.0])
    cBuffer = hybrid.Utl.np.array(cBuffer)
    iBuffer = []
    for i in range(len(T)):
        iBuffer.append(T[i][0])
        iBuffer.append(T[i][1])
        iBuffer.append(T[i][2])
        
    # Calculate Tangents and Bitangents:
    tan1 = []
    for i in range(len(V)):
        tan1.append([0.0, 0.0, 0.0])
    tan2 = []
    for i in range(len(V)):
        tan2.append([0.0, 0.0, 0.0])
    handedness = []
    for triangle in T:
        i1 = triangle[0]
        i2 = triangle[1]
        i3 = triangle[2]
        
        v1 = vBuffer[i1]
        v2 = vBuffer[i2]
        v3 = vBuffer[i3]
        
        w1 = tBuffer[i1]
        w2 = tBuffer[i2]
        w3 = tBuffer[i3]
        
        x1 = v2[0] - v1[0]
        x2 = v3[0] - v1[0]
        y1 = v2[1] - v1[1]
        y2 = v3[1] - v1[1]
        z1 = v2[2] - v1[2]
        z2 = v3[2] - v1[2]
        
        s1 = w2[0] - w1[0]
        s2 = w3[0] - w1[0]
        t1 = w2[1] - w1[1]
        t2 = w3[1] - w1[1]
        
        r = 1.0 / (s1 * t2 - s2 * t1)
        sdir = [(t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r]
        tdir = [(s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r]
        
        tan1[i1][0] += sdir[0]
        tan1[i2][0] += sdir[0]
        tan1[i3][0] += sdir[0]
        tan1[i1][1] += sdir[1]
        tan1[i2][1] += sdir[1]
        tan1[i3][1] += sdir[1]
        tan1[i1][2] += sdir[2]
        tan1[i2][2] += sdir[2]
        tan1[i3][2] += sdir[2]
            
        tan2[i1][0] += tdir[0]
        tan2[i2][0] += tdir[0]
        tan2[i3][0] += tdir[0]
        tan2[i1][1] += tdir[1]
        tan2[i2][1] += tdir[1]
        tan2[i3][1] += tdir[1]
        tan2[i1][2] += tdir[2]
        tan2[i2][2] += tdir[2]
        tan2[i3][2] += tdir[2]
            
    tanBuffer = []
    for i in range(len(V)):
        n = rasterizer.np.array(nBuffer[i])
        t = rasterizer.np.array(tan1[i])
        
        tangent = (t - n * rasterizer.np.dot(n, t))
        tangent = tangent / rasterizer.np.linalg.norm(tangent)
        tanBuffer.append(tangent.tolist())
                 
        handy = rasterizer.np.dot(rasterizer.np.cross(n, t), rasterizer.np.array(tan2[i]))
        if handy < 0.0:
            handedness.append(-1)
        else:
            handedness.append(1)
        
    bitanBuffer = []
    for i in range(len(V)):
        n = rasterizer.np.array(nBuffer[i])
        t = rasterizer.np.array(tanBuffer[i])
        
        bitangent = rasterizer.np.cross(n, t)*handedness[i]
        bitanBuffer.append(bitangent.tolist())

    return vBuffer, tBuffer, nBuffer, tanBuffer, bitanBuffer, cBuffer, iBuffer, T
    

'''We set up alternate objects that will use whichever function 
is available at run-time.'''
glMultiTexCoord2f = alternate(
    glMultiTexCoord2f,
    glMultiTexCoord2fARB 
)
glActiveTexture = alternate(
    glActiveTexture,
    glActiveTextureARB,
)

class TestContext( BaseContext ):
    """Multi-texturing demo
    """
    initialPosition = (0,0,0)
    rotation =  0
    theta = 360-45
    phi = 30
    
    def OnInit( self ):
        """Do all of our setup functions..."""
        if not glMultiTexCoord2f:
            print('Multitexture not supported!')
            sys.exit(1)
            
        self.update_sim = False
        vBuffer, uvBuffer, nBuffer, tBuffer, bBuffer, cBuffer, iBuffer, triangles = \
            loadMesh("TestMesh.obj", True, 0.001)
        
        self.indices = iBuffer
        self.vertices = vBuffer
        self.tex_coords = uvBuffer
        self.normals = nBuffer
        self.tangents = tBuffer
        self.biTangents = bBuffer
        self.colors = cBuffer
        
        normal_array = rasterizer.raster_vector_attrib(self.normals, self.tex_coords,\
                    triangles, SIM_SIZE, SIM_SIZE, UP)
        tangent_array = rasterizer.raster_vector_attrib(self.tangents, self.tex_coords,\
                    triangles, SIM_SIZE, SIM_SIZE, X_DIR)
        biTangent_array = rasterizer.raster_vector_attrib(self.biTangents, self.tex_coords,\
                    triangles, SIM_SIZE, SIM_SIZE, Y_DIR)
        
        self.fluid_sim = FluidSim.Fluid_Sim_2D(SIM_SIZE, SIM_SIZE, CELL_SIZE, DENSITY, GRAVITY,\
                    ATM_PRESSURE)
        for r in range(SIM_SIZE):
            for c in range(SIM_SIZE):
                self.fluid_sim.setNormal(r, c, normal_array[r][c])
                self.fluid_sim.setXTangent(r, c, tangent_array[r][c])
                self.fluid_sim.setYTangent(r, c, biTangent_array[r][c])
                
        sources = []
        for rc in INJECT_POINTS:
            for dr in range(-INJECT_RADIUS/2, INJECT_RADIUS/2+1):
                for dc in range(-INJECT_RADIUS/2, INJECT_RADIUS/2+1):
                    if dr*dr + dc*dc <= INJECT_RADIUS*INJECT_RADIUS:
                        sources.append((rc[0]+dr, rc[1]+dc))
        self.fluid_sim.markSources(sources)

        self.addEventHandler( "keypress", name="r", function = self.OnReverse)
        self.addEventHandler( "keypress", name="s", function = self.OnSlower)
        self.addEventHandler( "keypress", name="f", function = self.OnFaster)
        self.addEventHandler( "keypress", name="w", function = self.incPhi)
        self.addEventHandler( "keypress", name="s", function = self.decPhi)
        self.addEventHandler( "keypress", name="d", function = self.incTheta)
        self.addEventHandler( "keypress", name="a", function = self.decTheta)
        self.addEventHandler( "keypress", name="u", function = self.promptUpdate)
        print('r -- reverse time\ns -- slow time\nf -- speed time')
        self.time = Timer( duration = 8.0, repeating = 1 )
        self.time.addEventHandler( "fraction", self.OnTimerFraction )
        self.time.register (self)
        self.time.start ()
        '''Load both of our textures.'''
        self.Load()
        
        self.frameCount = 0

    ### Timer callback
    def OnTimerFraction( self, event ):
        self.rotation = event.fraction()* -360
    '''Keyboard callbacks, to allow for manipulating timer'''
    def OnReverse( self, event ):
        self.time.internal.multiplier = -self.time.internal.multiplier
        print("reverse",self.time.internal.multiplier)
    def OnSlower( self, event ):
        self.time.internal.multiplier = self.time.internal.multiplier /2.0
        print("slower",self.time.internal.multiplier)
    def OnFaster( self, event ):
        self.time.internal.multiplier = self.time.internal.multiplier * 2.0
        print("faster",self.time.internal.multiplier)
    def incTheta( self, event ):
        self.theta += 0.5
    def decTheta( self, event ):
        self.theta -= 0.5
    def incPhi( self, event ):
        self.phi += 0.5
    def decPhi( self, event ):
        self.phi -= 0.5
    def promptUpdate( self, event ):
        self.update_sim = True
    def Load( self ):
        self.image = self.loadImage ("Checker_Sparse.png")
        self.watermap = self.loadWaterMap()
        
    def Render( self, mode):
        """Render scene geometry"""
        BaseContext.Render( self, mode )
        if mode.visible:
            glDisable(GL_CULL_FACE)
            glClearColor(0.5, 0.5, 0.5, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glEnable(GL_LIGHTING)
            
            glTranslatef(0.0,8.0,-40.0);
            glRotated( self.phi+180.0, 1,0,0)
            glRotated( self.theta, 0,1,0)
            
            
            '''We set up each texture in turn, the only difference 
            between them being their application model.  We want texture
            0 applied as a simple decal, while we want the light-map 
            to modulate the colour in the base texture.'''
            if USE_TEXTURE:
                glActiveTexture(GL_TEXTURE0); 
                glTexParameterf(
                    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST
                )
                glTexParameterf(
                    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST
                )
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
                '''Enable the image (with the current texture unit)'''
                self.image()
                
            if SIMULATE:
                glActiveTexture(GL_TEXTURE1);
                glTexParameterf(
                    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST
                )
                glTexParameterf(
                    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST
                )
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                '''Enable the image (with the current texture unit)'''
                
                if self.frameCount % 5 == 0:
                    self.update_sim = True
                if self.update_sim:
                    self.watermap = self.loadWaterMap()
                    for rc in INJECT_POINTS:
                        for dr in range(-INJECT_RADIUS/2, INJECT_RADIUS/2+1):
                            for dc in range(-INJECT_RADIUS/2, INJECT_RADIUS/2+1):
                                if dr*dr + dc*dc <= INJECT_RADIUS*INJECT_RADIUS:
                                    if (self.fluid_sim.getParticleCount() < MAX_PARTICLES):
                                        self.fluid_sim.inject(rc[0]+dr, rc[1]+dc, PARTICLE_INJECT_COUNT)
                                    if (XV_INJECT or YV_INJECT):
                                        print("NO!")
                                        self.fluid_sim.setXV(rc[0]+dr, rc[1]+dc, XV_INJECT)
                                        self.fluid_sim.setXV(rc[0]+dr, rc[1]+dc, XV_INJECT)
                                    if (P_INJECT):
                                        print("NAH!")
                                        self.fluid_sim.setP(rc[0]+dr, rc[1]+dc, P_INJECT)
                    self.fluid_sim.simFrame(TIME_STEP)
                    self.update_sim = False
                self.watermap()
                
            self.drawSurface()
            self.frameCount += 1

    def loadImage( self, imageName = "nehe_wall.bmp" ):
        """Load an image from a file using PIL."""
        try:
            from PIL.Image import open
        except ImportError:
            from Image import open
        glActiveTexture(GL_TEXTURE0_ARB);
        return texture.Texture( open(imageName) )
    def loadWaterMap( self ):
        origSources = self.fluid_sim.sources
        sources = set()
        for source in origSources:
            oldr = source[0]
            oldc = source[1]
            sources.add((oldr*SUBDIVIDE_FACTOR, oldc*SUBDIVIDE_FACTOR))
            
        fluid = []
        for sdr in range(SIM_SIZE*SUBDIVIDE_FACTOR):
            fluid.append([])
            for sdc in range(SIM_SIZE*SUBDIVIDE_FACTOR):
                fluid[sdr].append([])
        for r in range(SIM_SIZE):
            for c in range(SIM_SIZE):
                subdivCell = self.fluid_sim.subdivParticles(r, c, SUBDIVIDE_FACTOR)
                for cr in range(SUBDIVIDE_FACTOR):
                    for cc in range(SUBDIVIDE_FACTOR):
                        for particle in subdivCell[cr][cc]:
                            fluid[r*SUBDIVIDE_FACTOR+cr][c*SUBDIVIDE_FACTOR+cc].append(particle)
            
        minV = rasterizer.np.array([255, 255, 255, 255])
        maxV = rasterizer.np.array([0, 0, 255, 255])
        sourceAlt = rasterizer.np.array([127, 127, 0, 0])
        data = rasterizer.fluid_to_texture(fluid, SIM_SIZE*SUBDIVIDE_FACTOR, SIM_SIZE*SUBDIVIDE_FACTOR, \
                float(RENDER_MAX), minV, maxV, sources, sourceAlt)
        glActiveTexture(GL_TEXTURE1_ARB)
        return texture.Texture( PIL.Image.frombuffer("RGBA", (SIM_SIZE*SUBDIVIDE_FACTOR, SIM_SIZE*SUBDIVIDE_FACTOR), data)  )
        
    def drawSurface( self ):
        """Draw a cube with texture coordinates"""
        glBegin(GL_TRIANGLES)
        for i in range(len(self.indices)):
            i0 = self.indices[i]
            mTexture(self.tex_coords[i0][0], self.tex_coords[i0][1])
            glVertex3f(self.vertices[i0][0], self.vertices[i0][1], self.vertices[i0][2])
            glNormal3f(self.normals[i0][0], self.normals[i0][1], self.normals[i0][2])
        glEnd()
        
    def OnIdle( self, ):
        """Request refresh of the context whenever idle"""
        self.triggerRedraw(1)
        return 1

'''This is a trivial indirection point setting both texture 
coordinates to the same value.'''
def mTexture( a,b ):
    glMultiTexCoord2f(GL_TEXTURE0, a,b)
    glMultiTexCoord2f(GL_TEXTURE1, a,b) 

if __name__ == "__main__":
    TestContext.ContextMainLoop()