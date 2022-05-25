import numpy as np

# General cosmological constants for Milky Way
G = 43007.1 # Gravitational constant in units [(km/s)**2*(kpc/(1e10*M_s))]
H0 = 0.1 # Hubble in units of [(km/s)/(kpc/h)]
h = 0.7 # Dimensionless Hubble parameter
r0 = 8*h # Position of sun in [kpc/h]
r_vir = 213.5*h # r200 sfor MW in [kpc/h]

# Parameters for MW NFW profile
rho_c = 3*H0**2/(8*np.pi*G) # Critical density with units from above
delta_c = 200. # Virial overdensity

# Parameters for MW Einasto profile
# Use conversion [1M_s/pc^3] = 37.96[GeV/cm**3], 
# i.e. [1e10*M_s/(kpc**3/h**2)] = 379.6*h**2[GeV/cm**3]
rho_s = 0.4/(379.6*h**2) # Einasto_MW local density at r0 in [1e10M_s/(kpc**3/h**2)]