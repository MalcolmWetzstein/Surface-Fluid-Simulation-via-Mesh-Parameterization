# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:47:23 2017

@author: Malcolm
"""

import numpy as np

def bary_coords(point, a, b, c):
    point = np.array(point)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    v0 = b - a
    v1 = c - a
    v2 = point - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return u, v, w

def img_to_uv(r, c, rowSize, colSize):
    return c/float(colSize-1), -(r-rowSize+1)/float(rowSize-1)

def uv_to_img(u, v, rowSize, colSize):
    return -((v*(rowSize-1))-rowSize+1), u*(colSize-1)

def raster_vector_attrib(vec_attrib, tex_coords, triangles, num_rows, num_cols, clear_value, normalize = True):
    attrib_raster = []
    for r in range(num_rows):
        attrib_raster.append([])
        for c in range(num_cols):
            attrib_raster[r].append(np.array(clear_value))
        
    for triangle in triangles:
        v0 = np.array([ tex_coords[triangle[0]][0], tex_coords[triangle[0]][1] ])
        v1 = np.array([ tex_coords[triangle[1]][0], tex_coords[triangle[1]][1] ])
        v2 = np.array([ tex_coords[triangle[2]][0], tex_coords[triangle[2]][1] ])
        
        r0, c0 = uv_to_img(v0[0], v0[1], num_rows, num_cols)
        r1, c1 = uv_to_img(v1[0], v1[1], num_rows, num_cols)
        r2, c2 = uv_to_img(v2[0], v2[1], num_rows, num_cols)
        
        rMin = int(min(r0, r1, r2)-1)
        rMax = int(max(r0, r1, r2)+1)
        cMin = int(min(c0, c1, c2)-1)
        cMax = int(max(c0, c1, c2)+1)
        
        for r in range(rMin, rMax):
            for c in range(cMin, cMax):
                u, v = img_to_uv(r, c, num_rows, num_cols)
                p = np.array([u,v])
                
                alpha, beta, gamma = bary_coords(p, v0, v1, v2)
                
                if alpha >= 0.0 and beta >= 0.0 and gamma >= 0.0:
                    value = alpha*np.array(vec_attrib[triangle[0]]) +\
                                beta*np.array(vec_attrib[triangle[1]]) + gamma*np.array(vec_attrib[triangle[2]])
                    if normalize:
                        value = value/np.linalg.norm(value)
                    attrib_raster[r][c] = value 
    
    return attrib_raster
                                 
def fluid_to_texture(fluid_array, uSize, vSize, maxPerCell, minValue, maxValue, sources, sourceAlt):
    if minValue.shape[0] != 4 or maxValue.shape[0] != 4:
        raise ValueError()
        
    texture = np.ones(uSize*vSize*4, dtype=np.uint8)
    maxColor = np.array([255, 255, 255, 255])
    minColor = np.array([0, 0, 0, 0])
    
    for r in range(vSize):
        for c in range(uSize):
            color = (float(len(fluid_array[r][c]))/maxPerCell)*maxValue + (1.0-float(len(fluid_array[r][c]))/maxPerCell)*minValue
            if (r, c) in sources:
                color = color - sourceAlt
            color = np.maximum(np.minimum(color, maxColor), minColor)
            color = color.astype(np.uint8)
            texture[(r*uSize+c)*4] = color[0]
            texture[(r*uSize+c)*4+1] = color[1]
            texture[(r*uSize+c)*4+2] = color[2]
            texture[(r*uSize+c)*4+3] = color[3]
    return texture