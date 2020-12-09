"""
(6DOF) Six Degrees Of Freedom simulation of an AUV (Autonomous Underwater Vehicle)
with rate feedback PID controllers for pitch and yaw   
Code by: Pouria Sarhadi 
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import AUV_model
except:
    raise

dt = 0.01 #solver and sampling time [s]
Tf = 80.0 #simulation time [s]

class Guidance():
    """
    AUV Guidance (simple setpoint generator)
    """
    
    def Longitudinal(time):
        z_com = 0.0
        theta_c=0.0
        
        if t >= 1.0:
            theta_c = np.deg2rad(3.0)
        
        if t>=11.0:
            theta_c = np.deg2rad(-3.0)
            
        if t>=23.0:
            theta_c = np.deg2rad(5.0)
            
        if t>=41.0:
            theta_c = np.deg2rad(-5.0)
            
        if t>=51.0:
            theta_c = np.deg2rad(2.0)
            
        if t>=62.0:
            theta_c = np.deg2rad(-2.0)
            
            
        return z_com,theta_c

    def Lateral(time):
        psi_c=0.0
        
        if t >= 30.0:
            psi_c = np.deg2rad(5.0)
        
        
        return psi_c
    
class Control():
    """
    AUV Control
    """
    
    def Longitudinal(theta_com,z,theta,q,u_i_theta1):
        Kp = 3.0
        Ki = 1.0
        Kq = 2.0
                
        e_theta = theta_com-theta
        u_i_theta = dt*e_theta + u_i_theta1
        del_e = -(Kp*e_theta + Ki*u_i_theta - Kq*q) 
        
        return del_e, u_i_theta

    def Lateral(psi_com,psi,r,u_i_psi1):
        Kp = 4.0
        Ki = 0.1
        Kr = 3.0
                
        e_psi = psi_com-psi
        u_i_psi = dt*e_psi + u_i_psi1
        del_r = -(Kp*e_psi + Ki*u_i_psi - Kr*r) 
        
        #del_r = 0.0
        return del_r, u_i_psi
    
def Runge_Kutta(dx, state, inputs, dt):
    f1 = dx(state, inputs)
    x1 = state + (dt/2.0)*f1

    f2 = dx(x1, inputs)
    x2 = state + (dt/2.0)*f2

    f3 = dx(x2, inputs)
    x3 = state + dt*f3

    f4 = dx(x3, inputs)
    return state + (dt/6.0)*(f1 + 2*f2 + 2*f3 + f4)

def main():
    print("Start" + __file__)
    
    global t 
    # Set Initial parameters
    t = 0.0
    u_i_theta1 = 0.0
    u_i_psi1 = 0.0
    time = [0.0]
    delta_e_ac = [0.0]
    delta_r_ac = [0.0]
    AUX= [0.0]
 
    X0 =[1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,50.0,0.0,0.0,0.0]
    X = X0
    
    u, v, w = [X0[0]], [X0[1]], [X0[2]]
    p, q, r = [X0[3]], [X0[4]], [X0[5]]
    x, y, z = [X0[6]], [X0[7]], [X0[8]]
    phi, theta, psi = [X0[9]], [X0[10]], [X0[11]]
    
    theta_com, psi_com = [0.0], [0.0]
    
    # Simulation start   
    while Tf >= t:
                
        t += dt  
        
        # Guidance
        z_com,theta_c = Guidance.Longitudinal(t)
        psi_c = Guidance.Lateral(t)
        
        #Control
        del_e, u_i_theta1 = Control.Longitudinal(theta_c,X[8],X[10],X[4],u_i_theta1)
        del_r, u_i_psi1 = Control.Lateral(psi_c,X[11],X[5],u_i_psi1)
        inputs = [del_e, del_r]
        
        #Integration            
        X = Runge_Kutta(AUV_model.AUV,X0,inputs,dt)
        X0 = X
        
        #Save variables
        time.append(t)
        u.append(X[0]), v.append(X[1]), w.append(X[2])
        p.append(X[3]), q.append(X[4]), r.append(X[5])
        x.append(X[6]), y.append(X[7]), z.append(X[8])
        phi.append(X[9]), theta.append(X[10]), psi.append(X[11])
        
        theta_com.append(theta_c), psi_com.append(psi_c)
        delta_e_ac.append(del_e),  delta_r_ac.append(del_r)
        AUX.append(np.rad2deg(u_i_theta1))
            
    # Plot signals    
    plt.close("all")
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, np.rad2deg(theta_com),':r', label=r'$\theta_{com}$', linewidth=3)
    plt.plot(time, np.rad2deg(theta), label=r'$\theta$', linewidth=3)
    plt.ylabel(r'$\theta (deg)$', fontsize=16)
    plt.legend(loc='best')
   
    
    plt.subplot(212)
    plt.plot(time, np.rad2deg(delta_e_ac), linewidth=3)
    plt.xlabel('time (sec)', fontsize=16)
    plt.ylabel(r'$\delta_e (deg)$', fontsize=16)
    plt.show()
    
    
    plt.figure(2)
    plt.subplot(211)
    plt.plot(time, np.rad2deg(psi_com),':r', label=r'$\psi_{com}$', linewidth=3)
    plt.plot(time, np.rad2deg(psi), label=r'$\psi$', linewidth=3)
    plt.ylabel(r'$\psi (deg)$', fontsize=16)
    plt.legend(loc='best')
    
    plt.subplot(212)
    plt.plot(time, np.rad2deg(delta_r_ac), linewidth=3)
    plt.xlabel('time (sec)', fontsize=16)
    plt.ylabel(r'$\delta_r (deg)$', fontsize=16)
    plt.show()


if __name__ == '__main__':
    main()
    
   