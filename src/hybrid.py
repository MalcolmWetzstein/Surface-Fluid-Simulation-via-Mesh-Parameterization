# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:12:09 2017

@author: Malcolm
"""

import Utilities as Utl

# Check your cots
def hybrid_parameterize(V, T, P_0, L, pinTop, pinBottom, iterations, cholesky=False):
    V = Utl.np.array(V)
    T = Utl.np.array(T)
    P_0 = Utl.np.array(P_0)
    uv_vec = Utl.np.zeros(V.shape[0]*2) # Stored (u0, v0, u1, v1, u2, v2, ...), per vertex
    ab_vec = Utl.np.zeros(T.shape[0]*2) # Stored (a0, b0,  a1, b1, a2, b2, ...), per triangle
    cot_vec = Utl.np.zeros(T.shape[0]*3) # Stored with cots for a single triangle in sequence.
    dx_vec = Utl.np.zeros(T.shape[0]*3) # Stored with dx's for a single triangle in sequence.
    dy_vec = Utl.np.zeros(T.shape[0]*3) # Stored with dy's for a single triangle in sequence.
    
    for i in range(P_0.shape[0]):
        if pinTop == None:
            if P_0[i][0] == 0 and P_0[i][1] == 0:
                pinTop = i
        if pinBottom == None:
            if P_0[i][0] == 1 and P_0[i][1] == 1:
                pinBottom = i
        
        uv_vec[i*2] = P_0[i][0]
        uv_vec[i*2+1] = P_0[i][1]
        
    T_2d = Utl.tri_to_2d(V, T) # Not sure tri_to_2d is correct.
    for i in range(T.shape[0]):
        dx_vec[i*3] = T_2d[i][0][0] - T_2d[i][1][0]
        dx_vec[i*3+1] = T_2d[i][1][0] - T_2d[i][2][0]
        dx_vec[i*3+2] = T_2d[i][2][0] - T_2d[i][0][0]
        
        dy_vec[i*3] = T_2d[i][0][1] - T_2d[i][1][1]
        dy_vec[i*3+1] = T_2d[i][1][1] - T_2d[i][2][1]
        dy_vec[i*3+2] = T_2d[i][2][1] - T_2d[i][0][1]
        
        cot_vec[i*3] = Utl.cot_sides(V, T[i], 2)
        cot_vec[i*3+1] = Utl.cot_sides(V, T[i], 0)
        cot_vec[i*3+2] = Utl.cot_sides(V, T[i], 1)
        
    def local_phase(t, L):
        i0 = T[t][0]
        i1 = T[t][1]
        i2 = T[t][2]
        du0 = uv_vec[i0*2]-uv_vec[i1*2]
        dv0 = uv_vec[i0*2+1]-uv_vec[i1*2+1]
        du1 = uv_vec[i1*2]-uv_vec[i2*2]
        dv1 = uv_vec[i1*2+1]-uv_vec[i2*2+1]
        du2 = uv_vec[i2*2]-uv_vec[i0*2]
        dv2 = uv_vec[i2*2+1]-uv_vec[i0*2+1]
        
        N1 = cot_vec[t*3]*(dx_vec[t*3]**2+dy_vec[t*3]**2) + \
                    cot_vec[t*3+1]*(dx_vec[t*3+1]**2 + dy_vec[t*3+1]**2) + \
                           cot_vec[t*3+2]*(dx_vec[t*3+2]**2 + dy_vec[t*3+2]**2)
        N2 = cot_vec[t*3]*(dx_vec[t*3]*du0 + dy_vec[t*3]*dv0) + \
                    cot_vec[t*3+1]*(dx_vec[t*3+1]*du1 + dy_vec[t*3+1]*dv1) + \
                           cot_vec[t*3+2]*(dx_vec[t*3+2]*du2 + dy_vec[t*3+2]*dv2)
        N3 = cot_vec[t*3]*(dy_vec[t*3]*du0 - dx_vec[t*3]*dv0) + \
                    cot_vec[t*3+1]*(dy_vec[t*3+1]*du1 - dx_vec[t*3+1]*dv1) + \
                           cot_vec[t*3+2]*(dy_vec[t*3+2]*du2 - dx_vec[t*3+2]*dv2)
                           
        polyEq = Utl.np.array([2.0*L*(N2**2 + N3**2)/N2**2, 0, N1-2.0*L, -N2])
        soln = Utl.np.roots(polyEq)
        
        # Which root do we use?
        reals = []
        for sol in soln:
            if Utl.np.isreal(sol):
                reals.append(sol)
        a = reals[0]
        b = N3/N2*a
        
        ab_vec[t*2] = a
        ab_vec[t*2+1] = b
        
    
    def global_vec():
        vec = Utl.np.zeros(V.shape[0]*2)
        
        for t in range(T.shape[0]):
            i0 = T[t][0]
            i1 = T[t][1]
            i2 = T[t][2]
            
            a = ab_vec[t*2]
            b = ab_vec[t*2+1]
            
            #Check sign direction for d's
            vec[i0*2] += cot_vec[t*3]*(a*dx_vec[t*3] + b*dy_vec[t*3]) + \
               cot_vec[t*3+2]*(a*(-dx_vec[t*3+2]) + b*(-dy_vec[t*3+2]))
            vec[i0*2+1] += cot_vec[t*3]*(-b*dx_vec[t*3] + a*dy_vec[t*3]) + \
               cot_vec[t*3+2]*(-b*(-dx_vec[t*3+2]) + a*(-dy_vec[t*3+2]))
            
            vec[i1*2] += cot_vec[t*3+1]*(a*dx_vec[t*3+1] + b*dy_vec[t*3+1]) + \
               cot_vec[t*3]*(a*(-dx_vec[t*3]) + b*(-dy_vec[t*3]))
            vec[i1*2+1] += cot_vec[t*3+1]*(-b*dx_vec[t*3+1] + a*dy_vec[t*3+1]) + \
               cot_vec[t*3]*(-b*(-dx_vec[t*3]) + a*(-dy_vec[t*3]))
               
            vec[i2*2] += cot_vec[t*3+2]*(a*dx_vec[t*3+2] + b*dy_vec[t*3+2]) + \
               cot_vec[t*3+1]*(a*(-dx_vec[t*3+1]) + b*(-dy_vec[t*3+1]))
            vec[i2*2+1] += cot_vec[t*3+2]*(-b*dx_vec[t*3+2] + a*dy_vec[t*3+2]) + \
               cot_vec[t*3+1]*(-b*(-dx_vec[t*3+1]) + a*(-dy_vec[t*3+1]))
        
        return vec
    
    def global_mat():
        # Change to compute cholesky
        mat = Utl.np.zeros((V.shape[0]*2, V.shape[0]*2))
        
        for t in range(T.shape[0]):
            i0 = T[t][0]
            i1 = T[t][1]
            i2 = T[t][2]
            
            mat[i0*2][i0*2] += cot_vec[t*3] + cot_vec[t*3+2]
            mat[i0*2][i1*2] += -cot_vec[t*3]
            mat[i0*2][i2*2] += -cot_vec[t*3+2]
            mat[i0*2+1][i0*2+1] += cot_vec[t*3] + cot_vec[t*3+2]
            mat[i0*2+1][i1*2+1] += -cot_vec[t*3]
            mat[i0*2+1][i2*2+1] += -cot_vec[t*3+2]
            
            mat[i1*2][i1*2] += cot_vec[t*3+1] + cot_vec[t*3]
            mat[i1*2][i2*2] += -cot_vec[t*3+1]
            mat[i1*2][i0*2] += -cot_vec[t*3]
            mat[i1*2+1][i1*2+1] += cot_vec[t*3+1] + cot_vec[t*3]
            mat[i1*2+1][i2*2+1] += -cot_vec[t*3+1]
            mat[i1*2+1][i0*2+1] += -cot_vec[t*3]
            
            mat[i2*2][i2*2] += cot_vec[t*3+1] + cot_vec[t*3+2]
            mat[i2*2][i1*2] += -cot_vec[t*3+1]
            mat[i2*2][i0*2] += -cot_vec[t*3+2]
            mat[i2*2+1][i2*2+1] += cot_vec[t*3+1] + cot_vec[t*3+2]
            mat[i2*2+1][i1*2+1] += -cot_vec[t*3+1]
            mat[i2*2+1][i0*2+1] += -cot_vec[t*3+2]
        
        if cholesky:
            return Utl.scila.cho_factor(mat)
        return Utl.spla.factorized(Utl.csr_matrix(mat))
        
    gp_mat = global_mat()
        
    for i in range(iterations):
        for t in range(T.shape[0]):
            local_phase(t, L)
        if cholesky:
            Utl.scila.cho_solve(gp_mat, global_vec())
        else:
            uv_vec = gp_mat(global_vec())
    
    UV = Utl.np.zeros(P_0.shape)
    lowestU = float('+inf')
    lowestV = float('+inf')
    for i in range(P_0.shape[0]):
        UV[i][0] = uv_vec[i*2]
        UV[i][1] = uv_vec[i*2+1]
        
        if uv_vec[i*2] < lowestU:
            lowestU = uv_vec[i*2]
        if uv_vec[i*2+1] < lowestV:
            lowestV = uv_vec[i*2+1]
    
    highestU = float('-inf')
    highestV = float('-inf')
    for i in range(UV.shape[0]):
        UV[i][0] -= lowestU
        UV[i][1] -= lowestV
        
        if UV[i][0] > highestU:
            highestU = UV[i][0]
        if UV[i][1] > highestV:
            highestV = UV[i][1]
    
    scale = max([highestU, highestV])
    for i in range(UV.shape[0]):
        UV[i][0] /= scale
        UV[i][1] /= scale
    
    return UV
