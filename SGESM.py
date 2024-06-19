import torch
import numpy as np


class SGESM1d:
    """1D Stochastic Generalized Ekman-Stokes model."""
    
    def __init__(self, param):
        
        # Data types
        self.farr_kwargs = {'dtype': torch.float64, 'device': param['device']}
        self.carr_kwargs = {'dtype': torch.complex128, 'device': param['device']}
        
        # Input parameters
        for key in param:
            val = param[key]
            if (type(val) == float):
                val = torch.tensor(val, **self.farr_kwargs)
            if (type(val) == complex):
                val = torch.tensor(val, **self.carr_kwargs)
            setattr(self, key, val) 
        
        # Derived parameters
        self.cdrrho = self.Cd * self.rhoat / self.rhooc
        self.sqrt_dt = torch.sqrt(self.dt)

        # Grid and derivative
        self.set_grid() 
        
        # Wind
        self.ua = self.ua_mean.clone() # (m/s)
        if self.rand_wind:
            self.ua_drift_coef = torch.exp(-self.dt/self.Ta)
            self.ua_diffu_coef = torch.sqrt(1 - torch.exp(-2*self.dt/self.Ta)) * self.ua_std

        # Wave
        self.us = self.us0 * torch.exp(self.z/self.hs + 1j*self.theta_mean)
        if self.rand_wave:
            self.us = self.us * torch.exp(1j*self.theta_std * torch.randn(self.ne,1,1,**self.farr_kwargs))
        
        # Init. model
        self.ue = torch.zeros(self.ne,self.nz,1, **self.carr_kwargs) # Ekman velocity (m/s)
        self.set_model() 
        

    def set_grid(self):
        """Chebyshev nodes, differentiation matrix and quadrature coefficients."""
        # Chebyshev-Lobatto points
        theta = torch.linspace(0, torch.pi, self.nz, **self.farr_kwargs).reshape(-1,1)
        self.z = torch.cos(theta)
        # Diff. matrix
        c = torch.ones_like(self.z)
        c[[0,-1]] *= 2.
        c *= (-1)**torch.arange(self.nz, **self.farr_kwargs).reshape(-1,1)
        Z = self.z.tile(1,self.nz)
        dZ = Z - Z.T
        self.D = c @ (1./c).T / (dZ + torch.eye(self.nz, **self.farr_kwargs))
        self.D -= torch.diag(self.D.sum(dim=-1)) 
        # Quadrature coef. (weight for integration) 
        self.W = torch.zeros_like(self.z)
        v = torch.ones((self.nz-2,1), **self.farr_kwargs)
        n = self.nz - 1
        if n%2 == 0:
            self.W[0] = 1/(n*n-1)
            self.W[-1] = 1/(n*n-1)
            for k in range(1,n//2):
                v = v - 2*torch.cos(2*k*theta[1:-1])/(4*k*k-1)
            v = v - torch.cos(n*theta[1:-1])/(n*n-1)
        else:
            self.W[0] = 1/(n*n)
            self.W[-1] = 1/(n*n)
            for k in range(1,(n-1)//2+1):
                v = v - 2*torch.cos(2*k*theta[1:-1])/(4*k*k-1)
        self.W[1:-1] = 2*v/n
        # Change coord.
        self.z = self.H/2 * (self.z - 1) # from [1,-1] to [0,-H]
        self.D *= 2/self.H
        self.W /= 2/self.H
        # Update shape and type
        self.z.unsqueeze_(0) # for ensemble operations
        self.D = (self.D.unsqueeze(0)).type(torch.complex128)
        self.W = (self.W.T.unsqueeze(0)).type(torch.complex128)


    def set_model(self):
        """Define model parameters."""
        # K-profile param. (KPP) for eddy viscosity
        u = self.ua - self.ue[:,0:1] if self.tauw_diff else self.ua.clone()
        ustar = torch.sqrt(self.cdrrho) * abs(u) # friction velocity (m/s)
        h = self.c2 * ustar / self.f # boundary layer depth (m)
        zn = -self.z / h # normalized height
        mask = torch.where(abs(self.z) <= h, 1., 0.)
        self.a = (zn*(1-zn)**2 + torch.heaviside(self.z0-zn, zn[-1])*(zn-self.z0)**2/(2*self.z0))*mask \
               * self.c1*ustar*h + self.kappa # KPP visc (m^2/s)
        # Noise correlations in random model
        if self.rand_model:
            self.sigma = torch.sqrt(2*self.a) # vertical component (m/s^(1/2))
            self.ifSigma = 2j*self.f*self.hs*self.us*torch.nan_to_num(1./self.sigma,posinf=0.,neginf=0.) # horizontal
        # Wind and wave stresses for surface boundary
        self.tauw = self.cdrrho * abs(u) * u # wind stress (m^2/s^2)
        self.tauw.squeeze_(1) if len(self.tauw.shape)>1 else None
        self.taus = self.a[:,0,:] * self.us[:,0,:] / self.hs # wave stress (m^2/s^2)
        # Linear operator for unsteady model
        I = torch.eye(self.nz, **self.farr_kwargs)
        aD = self.a * self.D
        self.L = I - self.dt*(self.D @ aD - (self.r + 1j*self.f)*I)
        self.L[:,0,:] = aD[:,0,:]  # surface boundary
        self.L[:,-1,:] = aD[:,-1,:] # bottom boundary


    def steady_solver(self):
        """Solve the steady Ekman model."""
        # LHS linear operator
        aD = self.a * self.D
        L = self.D @ aD - (self.r + 1j*self.f)*torch.eye(self.nz, **self.farr_kwargs)
        L[:,0,:] = aD[:,0,:]  # for surface BC
        L[:,-1,:] = aD[:,-1,:] # for bottom BC
        # RHS forces
        rhs = 1j*self.f*self.us
        if self.full_model:
            rhs -= self.D @ aD @ self.us
        rhs[:,0,:] = self.tauw - self.taus if self.wave_sbc else self.tauw # surface boundary
        rhs[:,-1,:] = torch.zeros_like(self.taus) # bottom boundary
        # Solve linear system and diagnostics
        self.ue = torch.linalg.solve(L, rhs)


    def unsteady_solver(self):
        """Solve the unsteady Ekman model."""
        # Ornstein-Uhlenbeck process for wind fluctuations
        if self.rand_wind:
            rn = ( torch.randn(self.ne,**self.farr_kwargs) \
               +1j*torch.randn(self.ne,**self.farr_kwargs) ).unsqueeze(-1).unsqueeze(-1)
            self.ua = self.ua_mean + self.ua_drift_coef*(self.ua - self.ua_mean) + self.ua_diffu_coef*rn 
            self.set_model()
        
        # RHS forces
        I = torch.eye(self.nz, **self.farr_kwargs)
        A = self.D @ (self.a*self.D) - 1j*self.f*I if self.full_model else -1j*self.f*I 
        rhs = self.ue + A @ self.us * self.dt 
        if self.rand_model:
            u = self.ue + self.us if self.full_model else self.ue.clone()
            rhs -= ((self.sigma*self.D)@u + self.ifSigma) * \
                    self.sqrt_dt*torch.randn(self.ue.shape, **self.farr_kwargs)
        rhs[:,0,:] = self.tauw - self.taus if self.wave_sbc else self.tauw # surface boundary
        rhs[:,-1,:] = torch.zeros_like(self.taus) # bottom boundary
        
        # Solve linear system
        self.ue = torch.linalg.solve(self.L, rhs)


    def diag_model(self):
        """Diagnostics of Ekman model."""
        ids = (abs(self.ue.squeeze()) <= 1e-4).int().argmax(dim=-1)
        self.he = abs(self.z.squeeze()[ids]) # Ekman layer depth (m)
        self.ue_int = (self.W @ self.ue).squeeze() # Ekman transport (m^2/s)
        self.degw = torch.angle(self.tauw).squeeze() * 180/torch.pi # wind stress angle (deg)
        self.degs = torch.angle(self.taus).squeeze() * 180/torch.pi # surface wave stress angle (deg)
        self.dege = torch.angle(self.ue_int) * 180/torch.pi # Ekman transport angle (deg)

