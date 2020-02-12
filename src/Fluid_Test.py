# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:41:37 2017

@author: Malcolm
"""

import FluidSim
import Utilities
import time

size = 50

fluid_sim = FluidSim.Fluid_Sim_2D(size, size, 0.01, 1000.0, Utilities.np.array([0.0, 0.0, -10.0]), 0.0)

for r in range(size):
    for c in range(size):
        x_norm = 2.0*(c-size/2)/size
        y_norm = 2.0*(size/2-r)/size
        normal = Utilities.np.array([x_norm, y_norm, 1.0])
        normal = normal/Utilities.np.linalg.norm(normal)
        
        tangent = Utilities.np.array([1.0, 0.0, -x_norm])
        tangent = tangent/Utilities.np.linalg.norm(tangent)
        
        bi_tangent = Utilities.np.array([1.0, 0.0, -y_norm])
        bi_tangent = bi_tangent/Utilities.np.linalg.norm(bi_tangent)
        
        tangent = Utilities.np.cross(bi_tangent, normal)
        tangent = tangent/Utilities.np.linalg.norm(tangent)
        
        bi_tangent = Utilities.np.cross(normal, tangent)
        bi_tangent = bi_tangent/Utilities.np.linalg.norm(bi_tangent)
        
        fluid_sim.setNormal(r, c, normal)
        fluid_sim.setXTangent(r, c, tangent)
        fluid_sim.setYTangent(r, c, bi_tangent)
        
def drawSim(fluidSim):
    for i in range(2):
        print "\n"
    for r in range(size):
        row = ""
        for c in range(size):
            if fluidSim.isLiquid(r, c):
                row = row + "*"
            else:
                row = row + "o"
        print row

#fluid_sim.createStaticVelocityField(1.0)

counter = 0
lastTime = time.time()
totalTime = 0.001
while(totalTime < 120.0):
    currTime = time.time()
    deltaT = currTime-lastTime
    lastTime = currTime
    totalTime += deltaT
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            fluid_sim.inject(size/2+dr, size/2+dc, 10)
    if deltaT == 0.0:
        deltaT = 0.001
    fluid_sim.simFrame(deltaT)
    if counter % 1 == 0:
        drawSim(fluid_sim)
    counter += 1
    