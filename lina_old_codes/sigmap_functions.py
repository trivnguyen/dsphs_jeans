import numpy as np

def lnlike(projected_vel,vel_err,theta):
    vbar,sigmap = theta
    nstars = len(projected_vel)
    first_term = np.sum(np.log(vel_err**2+sigmap**2))
    second_term = np.sum((projected_vel-vbar)**2/(vel_err**2+sigmap**2))
    
    return -1/2*(first_term+second_term+nstars*np.log(2*np.pi))

def Fisher(projected_vel,vel_err,vbar,sigmap):
    denom_base = (vel_err**2+sigmap**2)
    
    Ainv_00 = np.sum(-1/denom_base)
    Ainv_01 = np.sum((-2*(projected_vel-vbar)*sigmap)/denom_base**2)
    
    numerator = (sigmap**4-vel_err**4+(projected_vel-vbar)**2*(vel_err**2-3*sigmap**2))
    Ainv_11 = np.sum(numerator/denom_base**3)
    
    return [[-Ainv_00,-Ainv_01],[-Ainv_01,-Ainv_11]]

def A(projected_vel,vel_err,vbar,sigmap):
    return np.linalg.inv(Fisher(projected_vel,vel_err,vbar,sigmap))

def one_sig(projected_vel,vel_err,vbar,sigmap):
    return np.sqrt(A(projected_vel,vel_err,vbar,sigmap)[1,1])