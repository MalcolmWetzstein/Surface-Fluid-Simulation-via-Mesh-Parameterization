# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:53:48 2017

@author: Malcolm
"""

import numpy
import scipy
import math
import random

def fRange(start, stop, inc):
    rangeList = []
    current = start
    while current < stop:
        rangeList.append(current)
        current += inc
        
    return rangeList

# Bottom left of cell is 0, 0
class Particle(object):
    def __init__(self, cellSize):
        self.x = cellSize*random.random()
        self.y = cellSize*random.random()
        
    def getCellXY(self):
        return self.x, self.y
        
    def moveParticle(self, xVel, yVel, t, cellSize):
        # TODO: Decide if we should Runge-Kutta this.
        newX = self.x + xVel*t
        newY = self.y + yVel*t
        self.x = newX % cellSize
        self.y = newY % cellSize
        
        rAdj = 0
        cAdj = 0
        
        if newY < 0.0:
            rAdj = 1*(1+abs(int(newY/cellSize)))
        elif newY >= cellSize:
            rAdj = -1*abs(int(newY/cellSize))
        
        if newX < 0.0:
            cAdj = -1*(1+abs(int(newX/cellSize)))
        elif newX >= cellSize:
            cAdj = 1*abs(int(newX/cellSize))
            
        return rAdj, cAdj
        
# Positive Y velocity travels toward row zero, positive X velocity travels toward column n.
class Fluid_Sim_2D(object):
    def __init__(self, xSize, ySize, cellSize, density, g, atm_p, solid_boundary = False):
        self.solid_bounds = solid_boundary # True if boundary of domain is solid walls. 
        # in grid cells.
        self.width = xSize 
        self.height = ySize
        self.cell_measure = cellSize # Width of a cell in millimeters.
        self.max_vel = 1.0 # used for determining time steps. 
        self.g = g # gravitational constant as vector.
        self.fluid_density = density # In kg per millimeter cubed.
        self.air_pressure = atm_p # surrounding atmospheric pressure.
        self.particle_count = 0 # Number of particles in the system.
        self.sources = set()
        
        # In newtons per millimeter squared.
        self.pressure_grid = []
        self.pressure_update = []
        for i in range(self.height):
            self.pressure_grid.append([0]*self.width)
            self.pressure_update.append([0]*self.width)
            
        # Velocities in millimeters per a second.
        self.x_vel_grid = []
        self.x_vel_update = []
        for i in range(self.height):
            self.x_vel_grid.append([0]*(self.width+1))
            self.x_vel_update.append([0]*(self.width+1))
        self.y_vel_grid = []
        self.y_vel_update = []
        for i in range(self.height+1):
            self.y_vel_grid.append([0]*self.width)
            self.y_vel_update.append([0]*self.width)
            
        self.normal_grid = []
        for i in range(self.height):
            self.normal_grid.append([])
            for j in range(self.width):
                self.normal_grid[i].append(numpy.array([0.0, 0.0, 1.0]))
                
        self.x_tangent_grid = []
        for i in range(self.height):
            self.x_tangent_grid.append([])
            for j in range(self.width):
                self.x_tangent_grid[i].append(numpy.array([1.0, 0.0, 0.0]))
                
        self.y_tangent_grid = []
        for i in range(self.height):
            self.y_tangent_grid.append([])
            for j in range(self.width):
                self.y_tangent_grid[i].append(numpy.array([0.0, 1.0, 0.0]))
                
        self.particle_grid = []
        self.particle_update = []
        for i in range(self.height):
            self.particle_grid.append([])
            self.particle_update.append([])
            for j in range(self.width):
                self.particle_grid[i].append([])
                self.particle_update[i].append([])
                
        self.pressure_solver = self.create_pressure_matrix()
        self.divergence_vector = numpy.zeros(self.width*self.height)
        
    def create_pressure_matrix(self):
        data = []
        rows = []
        columns = []
        num_cells = self.width*self.height
        for i in range(num_cells):
            # Left side column
            if i % self.width == 0:
                # Top left
                if i == 0:
                    data.append(-2.0)
                    rows.append(i)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i+self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i)
                    columns.append(i+1)
                    
                # Bottom left
                elif i == (self.width-1)*self.height:
                    data.append(-2.0)
                    rows.append(i)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i-self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i)
                    columns.append(i+1)
                    
                # Everything else along left side
                else:
                    data.append(-3.0)
                    rows.append(i)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i+self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i-self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i)
                    columns.append(i+1)
            
            # Right side column
            elif i % self.width == self.width-1:
                # Top right
                if i == self.width-1:
                    data.append(-2.0)
                    rows.append(i)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i+self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i)
                    columns.append(i-1)
                    
                # Bottom right
                elif i == (self.width*self.height)-1:
                    data.append(-2.0)
                    rows.append(i)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i-self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i)
                    columns.append(i-1)
                    
                # Everything else along right side
                else:
                    data.append(-3.0)
                    rows.append(i)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i+self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i-self.width)
                    columns.append(i)
                    data.append(1.0)
                    rows.append(i)
                    columns.append(i-1)
                
            # Top row
            elif i//self.width == 0:
                data.append(-3.0)
                rows.append(i)
                columns.append(i)
                data.append(1.0)
                rows.append(i+self.width)
                columns.append(i)
                data.append(1.0)
                rows.append(i)
                columns.append(i+1)
                data.append(1.0)
                rows.append(i)
                columns.append(i-1)
                
            # Bottom row
            elif i//self.width == self.height-1:
                data.append(-3.0)
                rows.append(i)
                columns.append(i)
                data.append(1.0)
                rows.append(i-self.width)
                columns.append(i)
                data.append(1.0)
                rows.append(i)
                columns.append(i+1)
                data.append(1.0)
                rows.append(i)
                columns.append(i-1)
                
            # Inner cells
            else:
                data.append(-4.0)
                rows.append(i)
                columns.append(i)
                data.append(1.0)
                rows.append(i+self.width)
                columns.append(i)
                data.append(1.0)
                rows.append(i-self.width)
                columns.append(i)
                data.append(1.0)
                rows.append(i)
                columns.append(i+1)
                data.append(1.0)
                rows.append(i)
                columns.append(i-1)
        
        mat = scipy.sparse.csr_matrix((data, (rows, columns)))
        return scipy.sparse.linalg.factorized(mat)
    
    def markSources(self, sources):
        sourceSet = set()
        for source in sources:
            sourceSet.add(source)
            
        self.sources = sourceSet
      
    def getNormal(self, r, c):
        r = min(max(r, 0), self.height-1)
        c = min(max(c, 0), self.width-1)
            
        return self.normal_grid[r][c]
    
    def setNormal(self, r, c, normal):
        self.normal_grid[r][c] = normal
        
    def getXTangent(self, r, c):
        r = min(max(r, 0), self.height-1)
        c = min(max(c, 0), self.width-1)
            
        return self.x_tangent_grid[r][c]
    
    def setXTangent(self, r, c, tangent):
        self.x_tangent_grid[r][c] = tangent
        
    def getYTangent(self, r, c):
        r = min(max(r, 0), self.height-1)
        c = min(max(c, 0), self.width-1)
            
        return self.y_tangent_grid[r][c]
    
    def setYTangent(self, r, c, tangent):
        self.y_tangent_grid[r][c] = tangent
        
    def inject(self, r, c, particles):
        for i in range(particles):
            self.particle_grid[r][c].append(Particle(self.cell_measure))
        self.particle_count += particles
        
    def getParticleCount(self):
        return self.particle_count
    
    def timeStep(self, k=1.0):
        if self.max_vel == 0.0:
            return 1.0
        return k*self.cell_measure/self.max_vel
    
    def getP(self, r, c):
        r = min(max(r, 0), self.height-1)
        c = min(max(c, 0), self.width-1)
        return self.pressure_grid[r][c]
    
    def setP(self, r, c, p):
        self.pressure_grid[r][c] = p
                          
    def setPUpdate(self, r, c, p):
        self.pressure_update[r][c] = p
                         
    def addP(self, r, c, p):
        self.pressure_grid[r][c] += p
    
    def getXV(self, r, c):
        r = min(max(r, 0), self.height-1)
        c = min(max(c, -0.5), self.width-0.5)
        
        return self.x_vel_grid[r][int(c+0.5)]
    
    def setXV(self, r, c, x):
        self.x_vel_grid[r][int(c+0.5)] = x
                    
    def setXVUpdate(self, r, c, x):
        self.x_vel_update[r][int(c+0.5)] = x
                         
    def addXV(self, r, c, x):
        self.x_vel_grid[r][int(c+0.5)] += x
    
    def getYV(self, r, c):
        r = min(max(r, -0.5), self.height-0.5)
        c = min(max(c, 0), self.width-1)
        
        return self.y_vel_grid[int(r+0.5)][c]
    
    def setYV(self, r, c, y):
        self.y_vel_grid[int(r+0.5)][c] = y
                        
    def setYVUpdate(self, r, c, y):
        self.y_vel_update[int(r+0.5)][c] = y
                         
    def addYV(self, r, c, y):
        self.y_vel_grid[int(r+0.5)][c] += y
    
    def isLiquid(self, r, c):
        if (r, c) in self.sources:
            return True
        if r < 0 or r > self.height-1 or c < 0 or c > self.width-1:
            return False
        if len(self.particle_grid[r][c]) > 0:
            return True
        return False
    
    def rcToPos(self, r, c):
        return c*self.cell_measure, -(r-self.height+1)*self.cell_measure
    
    def posToRC(self, x, y):
        return -((y/self.cell_measure)-self.height+1), x/self.cell_measure
    
    def interpXV(self, pr, pc):
        pr = min(max(pr, 0.0), self.height-1.0)
        pc = min(max(pc, 0.0), self.width-1.0)
        
        rLerp, r = math.modf(pr)
        cLerp, c = math.modf(pc+0.5)
        r = int(r)
        c = int(c)
        
        top = (1.0-cLerp)*self.x_vel_grid[r][c] + cLerp*self.x_vel_grid[r][c+1]
        bottom = top
        if rLerp > 0.0:
            bottom = (1.0-cLerp)*self.x_vel_grid[r+1][c] + cLerp*self.x_vel_grid[r+1][c+1]
        
        return (1.0-rLerp)*top + rLerp*bottom
    
    def interpYV(self, pr, pc):
        pr = min(max(pr, 0.0), self.height-1.0)
        pc = min(max(pc, 0.0), self.width-1.0)
        
        rLerp, r = math.modf(pr+0.5)
        cLerp, c = math.modf(pc)
        r = int(r)
        c = int(c)
        
        left = (1.0-rLerp)*self.y_vel_grid[r][c] + rLerp*self.y_vel_grid[r+1][c]
        right = left
        if cLerp > 0.0:
            right = (1.0-rLerp)*self.y_vel_grid[r][c+1] + rLerp*self.y_vel_grid[r+1][c+1]
            
        return (1.0-cLerp)*left + cLerp*right
#        top = cLerp*self.y_vel_grid[r][c] + (1.0-cLerp)*self.y_vel_grid[r][c+1]
#        bottom = cLerp*self.y_vel_grid[r+1][c] + (1.0-cLerp)*self.y_vel_grid[r+1][c+1]
#        
#        return rLerp*top + (1.0-rLerp)*bottom
    
    def simFrame(self, deltaT):
        self.advectVelocity(deltaT)
        self.addForcesAndClear(deltaT)
        self.solvePressure(deltaT)
        self.projectVelocity(deltaT)
        self.advectParticles(deltaT)
        
    def simFrameSimple(self, deltaT):
        self.advectParticles(deltaT)
            
    def addForcesAndClear(self, t):
        for r in range(self.height):
            for c in fRange(-0.5, self.width+0.5, 1.0):
                if self.isLiquid(r, int(c-0.5)) or self.isLiquid(r, int(c+0.5)):
                    # Force due to gravity
                    normal = 0.5*self.getNormal(r, int(c-0.5)) + 0.5*self.getNormal(r, int(c+0.5))
                    normal = normal/numpy.linalg.norm(normal)
                    cellG = self.g - (numpy.dot(self.g, normal))*normal               
                    x_tan = 0.5*self.getXTangent(r, int(c-0.5)) + 0.5*self.getXTangent(r, int(c+0.5))
                    x_tan = x_tan/numpy.linalg.norm(x_tan)
                    
                    self.addXV(r, c, numpy.dot(x_tan, cellG)*t)
                else:
                    self.setXV(r, c, 0.0)
                self.setXVUpdate(r, c, 0.0)
                    
        for r in fRange(-0.5, self.height+0.5, 1.0):
            for c in range(self.width):
                if self.isLiquid(int(r-0.5), c) or self.isLiquid(int(r+0.5), c):
                    # Force due to gravity
                    normal = 0.5*self.getNormal(int(r-0.5), c) + 0.5*self.getNormal(int(r+0.5), c)
                    normal = normal/numpy.linalg.norm(normal)
                    cellG = self.g - (numpy.dot(self.g, normal))*normal
                    y_tan = 0.5*self.getYTangent(int(r-0.5), c) + 0.5*self.getYTangent(int(r+0.5), c)
                    y_tan = y_tan/numpy.linalg.norm(y_tan)
                    
                    self.addYV(r, c, numpy.dot(y_tan, cellG)*t)
                else:
                    self.setYV(r, c, 0.0)
                self.setYVUpdate(r, c, 0.0)
                
    def advectVelocity(self, t):
        for r in range(self.height):
            for c in fRange(-0.5, self.width+0.5, 1.0):
                if self.isLiquid(r, int(c-0.5)) or self.isLiquid(r, int(c+0.5)):
                    # A suggested integrator similar to runge-kutta, O(h^3) accurate.
                    x_0, y_0 = self.rcToPos(r, c)
                    
                    xVel_1 = self.interpXV(r, c)
                    yVel_1 = self.interpYV(r, c)
                    
                    x_1 = x_0 - xVel_1*t*0.5
                    y_1 = y_0 - yVel_1*t*0.5
                    r1, c1 = self.posToRC(x_1, y_1)
                    xVel_2 = self.interpXV(r1, c1)
                    yVel_2 = self.interpYV(r1, c1)
                    
                    x_2 = x_0 - xVel_2*t*0.75
                    y_2 = y_0 - yVel_2*t*0.75
                    r2, c2 = self.posToRC(x_2, y_2)
                    xVel_3 = self.interpXV(r2, c2)
                    yVel_3 = self.interpYV(r2, c2)
                    
                    x_advect = x_0 - xVel_1*t*(2.0/9.0) - xVel_2*t*(3.0/9.0) - xVel_3*t*(4.0/9.0)
                    y_advect = y_0 - yVel_1*t*(2.0/9.0) - yVel_2*t*(3.0/9.0) - yVel_3*t*(4.0/9.0)
                    
                    r_advect, c_advect = self.posToRC(x_advect, y_advect)
                    xVel_advect = self.interpXV(r_advect, c_advect)
                    self.setXVUpdate(r, c, xVel_advect)
                    
        temp = self.x_vel_grid
        self.x_vel_grid = self.x_vel_update
        self.x_vel_update = temp
            
        for r in fRange(-0.5, self.height+0.5, 1.0):
            for c in range(self.width):
                if self.isLiquid(int(r-0.5), c) or self.isLiquid(int(r+0.5), c):
                    # A suggested integrator similar to runge-kutta, O(h^3) accurate.
                    x_0, y_0 = self.rcToPos(r, c)
                    
                    xVel_1 = self.interpXV(r, c)
                    yVel_1 = self.interpYV(r, c)
                    
                    x_1 = x_0 + xVel_1*t*0.5
                    y_1 = y_0 + yVel_1*t*0.5
                    r1, c1 = self.posToRC(x_1, y_1)
                    xVel_2 = self.interpXV(r1, c1)
                    yVel_2 = self.interpYV(r1, c1)
                    
                    x_2 = x_0 + xVel_2*t*0.75
                    y_2 = y_0 + yVel_2*t*0.75
                    r2, c2 = self.posToRC(x_2, y_2)
                    xVel_3 = self.interpXV(r2, c2)
                    yVel_3 = self.interpYV(r2, c2)
                    
                    x_advect = x_0 + xVel_1*t*(2.0/9.0) + xVel_2*t*(3.0/9.0) + xVel_3*t*(4.0/9.0)
                    y_advect = y_0 + yVel_1*t*(2.0/9.0) + yVel_2*t*(3.0/9.0) + yVel_3*t*(4.0/9.0)
                    
                    r_advect, c_advect = self.posToRC(x_advect, y_advect)
                    yVel_advect = self.interpYV(r_advect, c_advect)
                    self.setYVUpdate(r, c, yVel_advect)
                    
        temp = self.y_vel_grid
        self.y_vel_grid = self.y_vel_update
        self.y_vel_update = temp
        
    def solvePressure(self, t, iterations=0):
        if iterations == 0:
            # Solve system of linear equations for laplace(pressure) = (rho*dh/dt)*div(velocity)
            num_cells = self.width*self.height
            for i in range(num_cells):
                r = i/self.width
                c = i%self.width
                xvGrad = self.getXV(r, c+0.5) - self.getXV(r, c-0.5)
                yvGrad = self.getYV(r+0.5, c) - self.getYV(r-0.5, c)
                self.divergence_vector[i] = (xvGrad + yvGrad)*self.fluid_density*self.cell_measure/t
                                      
            p_soln = self.pressure_solver(self.divergence_vector)
            
            for i in range(num_cells):
                r = i/self.width
                c = i%self.width
                self.setP(r, c, p_soln[i])
            
        else:
            # Relaxation solver for laplace(pressure) = (rho*dh/dt)*div(velocity)
            wetCells = []
            for r in range(self.height):
                for c in range(self.width):
                    if self.isLiquid(r, c):
                        wetCells.append((r,c))
                    else:
                        self.setP(r, c, self.air_pressure)
                    self.setPUpdate(r, c, self.air_pressure)
            for i in range(iterations):
                for cell in wetCells:
                    newP = 0.0
                    if cell[0] > 0:
                        newP += self.getP(cell[0]-1, cell[1])
                    if cell[0] < self.height-1:
                        newP += self.getP(cell[0]+1, cell[1])
                    if cell[1] > 0:
                        newP += self.getP(cell[0], cell[1]-1)
                    if cell[1] < self.width-1:
                        newP += self.getP(cell[0], cell[1]+1)
                    xvGrad = self.getXV(cell[0], cell[1]+0.5) - self.getXV(cell[0], cell[1]-0.5)
                    yvGrad = self.getYV(cell[0]+0.5, cell[1]) - self.getYV(cell[0]-0.5, cell[1])
                    newP -= (xvGrad + yvGrad)*self.cell_measure*self.fluid_density/t
                    newP /= 4.0
                    self.setPUpdate(cell[0], cell[1], newP)
                    
                temp = self.pressure_grid
                self.pressure_grid = self.pressure_update
                self.pressure_update = temp
    
    def projectVelocity(self, t):
        # Need to calculate max velocity in this final step.
        self.max_vel = 0.0
        
        for r in range(self.height):
            for c in fRange(-0.5, self.width+0.5, 1.0):
                if self.isLiquid(r, int(c-0.5)) or self.isLiquid(r, int(c+0.5)):
                    lP = self.getP(r, int(c-0.5))
                    rP = self.getP(r, int(c+0.5))
                    pForce = -t*(rP-lP)/self.fluid_density/self.cell_measure
                    self.addXV(r, c, pForce)
                    
                    currVel = self.getXV(r, c)
                    if abs(currVel) > self.max_vel:
                        self.max_vel = abs(currVel)
                        
        for r in fRange(-0.5, self.height+0.5, 1.0):
            for c in range(self.width):
                if self.isLiquid(int(r-0.5), c) or self.isLiquid(int(r+0.5), c):
                    tP = self.getP(int(r-0.5), c)
                    bP = self.getP(int(r+0.5), c)
                    pForce = -t*(tP-bP)/self.fluid_density/self.cell_measure
                                
                    self.addYV(r, c, pForce)
                    
                    currVel = self.getYV(r, c)
                    if abs(currVel) > self.max_vel:
                        self.max_vel = abs(currVel)
    
    def advectParticles(self, t):
        for r in range(self.height):
            for c in range(self.width):
                while len(self.particle_grid[r][c]) > 0:
                    particle = self.particle_grid[r][c].pop()
                    x, y = self.rcToPos(r, c)
                    cellX, cellY = particle.getCellXY()
                    x = x + cellX - 0.5*self.cell_measure
                    y = y + cellY - 0.5*self.cell_measure
                    pr, pc = self.posToRC(x, y)
                    xVel = self.interpXV(pr, pc)
                    yVel = self.interpYV(pr, pc)
                    # Runge-Kutta?
                    rAdj, cAdj = particle.moveParticle(xVel, yVel, t, self.cell_measure)
                    # If particle goes out of grid, we remove it from the simulation.
                    if r+rAdj >= 0 and r+rAdj < self.height and c+cAdj >= 0 and c+cAdj < self.width:
                        self.particle_update[r+rAdj][c+cAdj].append(particle)
                    else:
                        self.particle_count -= 1
        
        temp = self.particle_grid
        self.particle_grid = self.particle_update
        self.particle_update = temp
        
    def createStaticVelocityField(self, factor):
        for r in range(self.height):
            for c in fRange(-0.5, self.width+0.5, 1.0):
                normal = 0.5*self.getNormal(r, int(c-0.5)) + 0.5*self.getNormal(r, int(c+0.5))
                normal = normal/numpy.linalg.norm(normal)
                cellG = self.g - (numpy.dot(self.g, normal))*normal               
                x_tan = 0.5*self.getXTangent(r, int(c-0.5)) + 0.5*self.getXTangent(r, int(c+0.5))
                x_tan = x_tan/numpy.linalg.norm(x_tan)
                self.setXV(r, c, numpy.dot(x_tan, cellG)*factor)
                
        for r in fRange(-0.5, self.height+0.5, 1.0):
            for c in range(self.width):
                normal = 0.5*self.getNormal(int(r-0.5), c) + 0.5*self.getNormal(int(r+0.5), c)
                normal = normal/numpy.linalg.norm(normal)
                cellG = self.g - (numpy.dot(self.g, normal))*normal
                y_tan = 0.5*self.getYTangent(int(r-0.5), c) + 0.5*self.getYTangent(int(r+0.5), c)
                y_tan = y_tan/numpy.linalg.norm(y_tan)
                self.setYV(r, c, numpy.dot(y_tan, cellG)*factor)
                
    def subdivParticles(self, r, c, subdiv):
        subdivList = []
        for sdr in range(subdiv):
            subdivList.append([])
            for sdc in range(subdiv):
                subdivList[sdr].append([])
        
        for particle in self.particle_grid[r][c]:
            px, py = particle.getCellXY()
            py = 1.0-py
            
            sdr = int(py*subdiv)
            sdc = int(px*subdiv)
            
            subdivList[sdr][sdc].append(particle)
                      
        return subdivList