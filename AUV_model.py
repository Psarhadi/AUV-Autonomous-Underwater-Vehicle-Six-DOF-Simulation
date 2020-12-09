"""
AUV_model model
The model adapted from:
T. Prestero, "Verification of a six degree of freedom simulation model for the REMUS autonomous under water vehicle", MIT, 2001. 
Code by: Pouria Sarhadi
"""

import math
import numpy as np


# Vehicle Parameters
rho = +1.03e+003
Af = +2.85e-002
B = +3.08e+002
W = +2.99e+002
Ix = +1.77e-001
Iy = +3.45e+000
Iz = +3.45e+000
m = +W/9.8
xB = +0.00e+000
yB = +0.00e+000
zB = +0.00e+000
xG = -0.00e+000
yG = +0.00e+000
zG = +1.96e-002
X_du = -9.30e-001
X_wq = -3.55e+001
X_qq = -1.93e+000
X_vr = +3.55e+001
X_rr = -1.93e+000
Y_dv = -3.55e+001
Y_dr = +1.93e+000
Yvav = -1.31e+003
Yrar = +6.32e-001
Y_wp = +3.55e+001
Y_pq = +1.93e+000
Y_uv = -2.86e+001
Y_ur = +5.22e+000
Z_dw = -3.55e+001
Z_dq = -1.93e+000
Zwaw = -1.31e+002
Zqaq = -6.32e-001
Z_vp = -3.55e+001
Z_rp = +1.93e+000
Z_uw = -2.86e+001
Z_uq = -5.22e+000
K_dp = -7.04e-002
Kpap = -1.30e-001*10
M_dw = -1.93e+000
M_dq = -4.88e+000
Mwaw = +3.18e+000
Mqaq = -1.88e+002 
M_vp = -1.93e+000
M_rp = +4.86e+000
M_uw = +2.40e+001
M_uq = -2.00e+000
N_dv = +1.93e+000
N_dr = -4.88e+000 
Nvav = -3.18e+000
Nrar = -9.40e+001
N_wp = -1.93e+000
N_pq = -4.86e+000
N_uv = -2.40e+001
N_ur = -2.00e+000
Y_uudr = +9.64e+000
Z_uude = -9.64e+000
M_uude = -6.15e+000
N_uudr = -6.15e+000
X_T = +9.25e+000
K_T = -5.43e-001


def AUV(states, inputs):
    
    u, v, w = states[0], states[1], states[2]
    p, q, r = states[3], states[4], states[5]
   # x, y, z = states[6], states[7], states[8]
    phi, theta, psi = states[9], states[10], states[11]
        
    dele_ac, delr_ac = inputs[0], inputs[1]    
        
    Cd = 0.193*(math.fabs(u))**(-0.14)
    Xuau=-0.5*rho*Af*Cd*1.5  
    
    M = np.array([[m-X_du,    0.0,         0.0,         0.0,      m*zG,       -m*yG],
        [0.0,        m-Y_dv,      0.0,       -m*zG,     0.0,       m*xG-Y_dr],
        [0.0,          0.0,       m-Z_dw,     m*yG,  -m*xG-Z_dq,     0.0],
        [0.0,      -m*zG,        m*yG,    Ix-K_dp,    0.0,           0.0],
        [m*zG,       0.0,    -m*xG-M_dw,     0.0,     Iy-M_dq,       0.0],
        [-m*yG,  m*xG-N_dv,     0.0,         0.0,       0.0,        Iz-N_dr]])
    
    F1 = (-(W-B)*math.sin( theta) + Xuau*math.fabs( u)* u + 
         (X_wq-m)* w* q + (X_qq+m*xG)* q**2 + 
         (X_vr+m)* v* r + (X_rr+m*xG)* r**2 
         - m*yG* p* q - m*zG* p* r + X_T) 
      
    F2 = ((W-B)*math.cos( theta)*math.sin( phi) + 
          Yvav*math.fabs( v)* v +Yrar*math.fabs( r)* r + 
          m*yG* r**2 + (Y_ur-m)*u*r+(Y_wp+m)*w*p+(Y_pq-m*xG)*p*q +
          Y_uv* u* v + m*yG* p**2 - m*zG* q* r + Y_uudr* u**2*delr_ac)
            
    F3 = ((W-B)*math.cos( theta)*math.cos( phi) + 
          Zwaw*math.fabs( w)* w + Zqaq*math.fabs( q)* q + 
          (Z_uq+m)* u* q + (Z_vp-m)* v* p + 
          (Z_rp-m*xG)* r* p + Z_uw* u* w + 
          m*zG*( q**2+ p**2) - m*yG* r* q
          + Z_uude*u**2*dele_ac)
        
    F4 = ((yG*W-yB*B)*math.cos( theta)*math.cos( phi) - 
          (zG*W-zB*B)*math.cos( theta)*math.sin( phi) +
          Kpap*math.fabs( p)* p - (Iz-Iy)* q* r + 
          m*yG*( u* q- v* p) - 
          m*zG*( w* p- u* r)+K_T)
      
    F5 = (-(zG*W-zB*B)*math.sin( theta) - 
          (xG*W-xB*B)*math.cos( theta)*math.cos( phi) +
          Mwaw*math.fabs( w)* w + Mqaq*math.fabs( q)* q +
          (M_uq-m*xG)* u* q + (M_vp+m*xG)* v* p +
          (M_rp-(Ix-Iz))* r* p + m*zG*( v* r- q* w) + 
          M_uw* u* w + M_uude*u**2*dele_ac)
     
    F6 = ((xG*W-xB*B)*math.cos(theta)*math.sin(phi) + 
          (yG*W-yB*B)*math.sin(theta) + Nvav*math.fabs(v)* v +
          Nrar*math.fabs(r)* r + (N_ur-m*xG)* u* r + 
          (N_wp+m*xG)* w* p + (N_pq-(Iy-Ix))* p* q - 
          m*yG*(v*r- w*q) + N_uv* u* v + 
          N_uudr*u**2*delr_ac)
        
    F = np.array([F1,F2,F3,F4,F5,F6]).T
        
    M_inv = np.linalg.inv(M)
    S1 = np.matmul(M_inv, F)
        
    x_dot = ((math.cos( psi)*math.cos( theta))* u + 
             (-math.sin( psi)*math.cos( phi) + 
              math.cos( psi)*math.sin( theta)*math.sin( phi))* v + 
             (math.sin( psi)*math.sin( phi) + 
              math.cos( psi)*math.cos( phi)*math.sin( theta))* w)
                                      
    y_dot = ((math.sin( psi)*math.cos( theta))* u + 
              (math.cos( psi)*math.cos( phi) + 
               math.sin( phi)*math.sin( theta)*math.sin( psi))* v +
              (-math.cos( psi)*math.sin( phi) + 
               math.sin( theta)*math.sin( psi)*math.cos( phi))* w)
     
    z_dot = ((-math.sin( theta))* u + 
              (math.cos( theta)*math.sin( phi))* v + 
              (math.cos( theta)*math.cos( phi))* w)
    
    phi_dot = ( p + math.sin( phi)*math.tan( theta)* q +
               math.cos( phi)*math.tan( theta)* r)
    
    theta_dot = math.cos( phi)* q - math.sin( phi)* r
        
    psi_dot = ((math.sin( phi)/math.cos( theta))* q +
               (math.cos( phi)/math.cos( theta))* r)
        
    S2 = np.array([x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]).T
    
    X_dot = np.concatenate((S1, S2), axis=None)
    
    return X_dot