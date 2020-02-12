# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:30:49 2017

@author: Malcolm
"""

import hybrid

def export_mesh(verts, norms, tex_c, triangles, filename):
    with open(filename, 'w') as f:
        f.write("# OBJ file\n")
        for vert in verts:
            f.write("v " + str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\n")
        for tex in tex_c:
            f.write("vt " + str(tex[0]) + " " + str(tex[1]) + "\n")
        for norm in norms:
            f.write("vn " + str(norm[0]) + " " + str(norm[1]) + " " + str(norm[2]) + "\n")
        for p in triangles:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")

def test_run():
    mesh = hybrid.Utl.trimesh.load_mesh('TestMesh.obj')
    V, T, N, I, J, K = hybrid.Utl.load_obj('TestMesh.obj')
    
    Tri = hybrid.Utl.np.column_stack((I, J, K))
    
    T_0 = hybrid.hybrid_parameterize(V, Tri, T, 0, None, None, 100)
    T_1 = hybrid.hybrid_parameterize(V, Tri, T, 0.5, None, None, 100)
    
    export_mesh(V, N, T_0, Tri, "lambda0_tex.obj")
    export_mesh(V, N, T_1, Tri, "lambda05_tex.obj")
    
    
test_run()
         