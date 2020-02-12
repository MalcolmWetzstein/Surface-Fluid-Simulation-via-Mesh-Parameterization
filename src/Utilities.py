# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:51:06 2017

@author: Malcolm
"""

import numpy as np
import numpy.linalg as la
import trimesh
import plotly
import chart_studio.plotly as py
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla
import scipy.linalg as scila
import pyglet
import math

def load_obj(filename) :
    V = [] #vertex
    T = [] #texcoords
    N = [] #normals
    I = []
    J = []
    K = []
    
    fh = open(filename)
    for line in fh:
        if line[0] == '#': 
            continue
 
        line = line.strip().split(' ')
        if line[0] == 'v':
            V.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'vt':
            T.append([float(line[1]), float(line[2])])
        elif line[0] == 'vn':
            N.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'f':
            #print line
            I.append(int(line[1].split('/')[0])-1)
            #print int(line[1].split('/')[0])-1
            J.append(int(line[2].split('/')[0])-1)
            #print int(line[2].split('/')[0])-1
            K.append(int(line[3].split('/')[0])-1)
            #print int(line[3].split('/')[0])-1
                   
    return V, T, N, I, J, K

def map_z2color(zval, colormap, vmin, vmax):
    t=(zval-vmin)/float((vmax-vmin)); R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+','+'{:d}'.format(int(B*255+0.5))+')'   

def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trisurf(x, y, z, colors, simplices, colormap=cm.RdBu, plot_edges=False):
    
    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)
    
    ncolors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
    
    I,J,K=tri_indices(simplices)
    triangles=Mesh3d(x=x,y=y,z=z,
                     intensity=ncolors,
                     colorscale='Viridis',
                     i=I,j=J,k=K,name='',
                     showscale=True,
                     colorbar=ColorBar(tickmode='array', 
                                       tickvals=[np.min(z), np.max(z)], 
                                       ticktext=['{:.3f}'.format(np.min(colors)), 
                                                 '{:.3f}'.format(np.max(colors))]))
    
    if plot_edges is False: # the triangle sides are not plotted
        return Data([triangles])
    else:
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
        lines=Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=Line(color='rgb(50,50,50)', width=1.5))
        return Data([triangles, lines])

def textured_mesh(mesh, per_vertex_signal, filename):
    x = mesh.vertices.transpose()[0]; y = mesh.vertices.transpose()[1]; z = mesh.vertices.transpose()[2];
    data1 = plotly_trisurf(x, y, z, per_vertex_signal, mesh.faces, colormap=cm.RdBu,plot_edges=True)
    axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
    layout = Layout(width=800, height=800,scene=Scene(xaxis=XAxis(axis),yaxis=YAxis(axis),zaxis=ZAxis(axis),aspectratio=dict(x=1,y=1,z=1)))
    fig1 = Figure(data=data1, layout=layout)
    iplot(fig1, filename=filename)
    
    
def triangle_area(vertices, indices):
    e1 = np.subtract(vertices[indices[1]], vertices[indices[0]])
    e2 = np.subtract(vertices[indices[2]], vertices[indices[0]])
    crossed = np.cross(e1, e2)
    return 0.5*np.linalg.norm(crossed)

def angle_between(vertices, indices, head):
    tail1 = 0 
    tail2 = 2
    if head == 0:
        tail1 = 1
    if head == 2:
        tail2 = 1
        
    e1 = np.subtract(vertices[indices[tail1]], vertices[indices[head]])
    e2 = np.subtract(vertices[indices[tail2]], vertices[indices[head]])
    return np.arccos(np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))

def dot_sides(vertices, indices, head):
    tail1 = 0 
    tail2 = 2
    if head == 0:
        tail1 = 1
    if head == 2:
        tail2 = 1
        
    e1 = np.subtract(vertices[indices[tail1]], vertices[indices[head]])
    e2 = np.subtract(vertices[indices[tail2]], vertices[indices[head]])
    return np.dot(e1, e2)

def cross_sides(vertices, indices, head, flip = False):
    tail1 = 0 
    tail2 = 2
    if head == 0:
        tail1 = 1
    if head == 2:
        tail2 = 1
        
    e1 = np.subtract(vertices[indices[tail1]], vertices[indices[head]])
    e2 = np.subtract(vertices[indices[tail2]], vertices[indices[head]])
    if flip:
        return np.cross(e2, e1)
    else:
        return np.cross(e1, e2)
    
def cot_sides(vertices, indices, head):
    tail1 = 0 
    tail2 = 2
    if head == 0:
        tail1 = 1
    if head == 2:
        tail2 = 1
        
    e1 = np.subtract(vertices[indices[tail1]], vertices[indices[head]])
    e2 = np.subtract(vertices[indices[tail2]], vertices[indices[head]])
    
    crossed = np.cross(e1, e2)
    return np.dot(e1, e2) / np.linalg.norm(crossed)

def distance(index1, index2, vertices):
    return la.norm(vertices[index1] - vertices[index2])
  
def dict_add(dictionary, key, value):
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value
                  
def tri_to_2d(vertices, triangles, flip = False):
    sign = 1
    if flip:
        sign = -1
    
    triangles_2d = []
    
    for triangle in triangles:
        l01 = distance(triangle[0], triangle[1], vertices)
        l02 = distance(triangle[0], triangle[2], vertices)
        
        x0 = (0, 0)
        x1 = (sign*l01, 0)
        crossNorm = la.norm(cross_sides(vertices, triangle, 0))
        x2 = (sign*((1-(crossNorm/l01/l02)**2)**0.5)*l02, crossNorm/l01)
        
        triangles_2d.append((x0, x1, x2))
        
    return triangles_2d

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
                  
#MOOMOO = trimesh.load_mesh('moomoo.off')
#TETRA = trimesh.load_mesh('tetra.off')