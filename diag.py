import os
import sys
import glob
import torch
import numpy as np
from SGESM import SGESM1d
from netCDF4 import Dataset
from scipy import stats 

def init_netcdf(nc):
    # Create variables
    nc.createVariable('t','f8',('t',))
    nc.createVariable('z','f8',('z',))
    nc.createVariable('u_mean','f8',('t','z',))
    nc.createVariable('u_std','f8',('t','z',))
    nc.createVariable('u_skew','f8',('t','z',))
    nc.createVariable('u_kurt','f8',('t','z',))
    nc.createVariable('v_mean','f8',('t','z',))
    nc.createVariable('v_std','f8',('t','z',))
    nc.createVariable('v_skew','f8',('t','z',))
    nc.createVariable('v_kurt','f8',('t','z',))
    nc.createVariable('uv_cov','f8',('t','z',))
    nc.createVariable('mked','f8',('t','z',))
    nc.createVariable('eked','f8',('t','z',))
    nc.createVariable('mke','f8',('t',))
    nc.createVariable('eke','f8',('t',))
    nc.createVariable('us_mean','f8',('z',))
    nc.createVariable('us_std','f8',('z',))
    nc.createVariable('vs_mean','f8',('z',))
    nc.createVariable('vs_std','f8',('z',))
    nc.createVariable('uvs_cov','f8',('z',))
    nc.createVariable('ua_mean','f8',('t',))
    nc.createVariable('ua_std','f8',('t',))
    nc.createVariable('va_mean','f8',('t',))
    nc.createVariable('va_std','f8',('t',))
    nc.createVariable('uva_cov','f8',('t',))
    nc.createVariable('Tx_mean','f8',('t',))
    nc.createVariable('Tx_std','f8',('t',))
    nc.createVariable('Tx_skew','f8',('t',))
    nc.createVariable('Tx_kurt','f8',('t',))
    nc.createVariable('Ty_mean','f8',('t',))
    nc.createVariable('Ty_std','f8',('t',))
    nc.createVariable('Ty_skew','f8',('t',))
    nc.createVariable('Ty_kurt','f8',('t',))
    nc.createVariable('Txy_cov','f8',('t',))
    nc.createVariable('twx_mean','f8',('t',))
    nc.createVariable('twx_std','f8',('t',))
    nc.createVariable('twx_skew','f8',('t',))
    nc.createVariable('twx_kurt','f8',('t',))
    nc.createVariable('twy_mean','f8',('t',))
    nc.createVariable('twy_std','f8',('t',))
    nc.createVariable('twy_skew','f8',('t',))
    nc.createVariable('twy_kurt','f8',('t',))
    nc.createVariable('twxy_cov','f8',('t',))
    nc.createVariable('tsx_mean','f8',('t',))
    nc.createVariable('tsx_std','f8',('t',))
    nc.createVariable('tsx_skew','f8',('t',))
    nc.createVariable('tsx_kurt','f8',('t',))
    nc.createVariable('tsy_mean','f8',('t',))
    nc.createVariable('tsy_std','f8',('t',))
    nc.createVariable('tsy_skew','f8',('t',))
    nc.createVariable('tsy_kurt','f8',('t',))
    nc.createVariable('tsxy_cov','f8',('t',))
    nc.createVariable('h_mean','f8',('t',))
    nc.createVariable('h_std','f8',('t',))
    nc.createVariable('argT_mean','f8',('t',))
    nc.createVariable('argT_std','f8',('t',))
    nc.createVariable('argU_mean','f8',('t',))
    nc.createVariable('argU_std','f8',('t',))
    nc.createVariable('argw_mean','f8',('t',))
    nc.createVariable('argw_std','f8',('t',))
    nc.createVariable('args_mean','f8',('t',))
    nc.createVariable('args_std','f8',('t',))
    nc.createVariable('absT_mean','f8',('t',))
    nc.createVariable('absT_std','f8',('t',))
    nc.createVariable('absU_mean','f8',('t',))
    nc.createVariable('absU_std','f8',('t',))
    nc.createVariable('a_mean','f8',('t','z',))
    nc.createVariable('a_std','f8',('t','z',))
    nc.createVariable('a_skew','f8',('t','z',))
    nc.createVariable('a_kurt','f8',('t','z',))
    nc.createVariable('ua_end','f8',('ens',))
    nc.createVariable('va_end','f8',('ens',))
    nc.createVariable('us0_end','f8',('ens',))
    nc.createVariable('vs0_end','f8',('ens',))
    nc.createVariable('u0_end','f8',('ens',))
    nc.createVariable('v0_end','f8',('ens',))
    nc.createVariable('Tx_end','f8',('ens',))
    nc.createVariable('Ty_end','f8',('ens',))
    nc.createVariable('twx_end','f8',('ens',))
    nc.createVariable('twy_end','f8',('ens',))
    nc.createVariable('tsx_end','f8',('ens',))
    nc.createVariable('tsy_end','f8',('ens',))
    # Add attributes
    nc.variables['t'].long_name = 'Time axis'
    nc.variables['t'].units = 'days'
    nc.variables['z'].long_name = 'Vertical axis'
    nc.variables['z'].units = 'm'
    nc.variables['u_mean'].long_name = 'Mean of zonal Ekman velocity'
    nc.variables['u_mean'].units = 'm/s'
    nc.variables['u_std'].long_name = 'Standard deviation of zonal Ekman velocity'
    nc.variables['u_std'].units = 'm/s'
    nc.variables['u_skew'].long_name = 'Skewness of zonal Ekman velocity'
    nc.variables['u_kurt'].long_name = 'Kurtosis of zonal Ekman velocity'
    nc.variables['v_mean'].long_name = 'Mean of meridional Ekman velocity'
    nc.variables['v_mean'].units = 'm/s'
    nc.variables['v_std'].long_name = 'Standard deviation of meridional Ekman velocity'
    nc.variables['v_std'].units = 'm/s'
    nc.variables['v_skew'].long_name = 'Skewness of meridional Ekman velocity'
    nc.variables['v_kurt'].long_name = 'Kurtosis of meridional Ekman velocity'
    nc.variables['uv_cov'].long_name = 'Covariance of Ekman velocity'
    nc.variables['uv_cov'].units = 'm^2/s^2'
    nc.variables['mked'].long_name = 'Mean kinetic energy density'
    nc.variables['mked'].units = 'm^2/s^2'
    nc.variables['eked'].long_name = 'Eddy kinetic energy density'
    nc.variables['eked'].units = 'm^2/s^2'
    nc.variables['mke'].long_name = 'Mean kinetic energy (global)'
    nc.variables['mke'].units = 'J/m^2'
    nc.variables['eke'].long_name = 'Eddy kinetic energy (global)'
    nc.variables['eke'].units = 'J/m^2'
    nc.variables['us_mean'].long_name = 'Mean of zonal Stokes drift'
    nc.variables['us_mean'].units = 'm/s'
    nc.variables['us_std'].long_name = 'Standard deviation of zonal Stokes drift'
    nc.variables['us_std'].units = 'm/s'
    nc.variables['vs_mean'].long_name = 'Mean of meridional Stokes drift'
    nc.variables['vs_mean'].units = 'm/s'
    nc.variables['vs_std'].long_name = 'Standard deviation of meridional Stokes drift'
    nc.variables['vs_std'].units = 'm/s'
    nc.variables['uvs_cov'].long_name = 'Covariance of Stokes drift'
    nc.variables['uvs_cov'].units = 'm^2/s^2'
    nc.variables['ua_mean'].long_name = 'Mean of zonal wind'
    nc.variables['ua_mean'].units = 'm/s'
    nc.variables['ua_std'].long_name = 'Standard deviation of zonal wind'
    nc.variables['ua_std'].units = 'm/s'
    nc.variables['va_mean'].long_name = 'Mean of meridional wind'
    nc.variables['va_mean'].units = 'm/s'
    nc.variables['va_std'].long_name = 'Standard deviation of meridional wind'
    nc.variables['va_std'].units = 'm/s'
    nc.variables['uva_cov'].long_name = 'Covariance of wind'
    nc.variables['uva_cov'].units = 'm^2/s^2'
    nc.variables['Tx_mean'].long_name = 'Mean of zonal Ekman transport'
    nc.variables['Tx_mean'].units = 'm^2/s'
    nc.variables['Tx_std'].long_name = 'Standard deviation of zonal Ekman transport'
    nc.variables['Tx_std'].units = 'm^2/s'
    nc.variables['Tx_skew'].long_name = 'Skewness of zonal Ekman transport'
    nc.variables['Tx_kurt'].long_name = 'Kurtosis of zonal Ekman transport'
    nc.variables['Ty_mean'].long_name = 'Mean of meridional Ekman transport'
    nc.variables['Ty_mean'].units = 'm^2/s'
    nc.variables['Ty_std'].long_name = 'Standard deviation of meridional Ekman transport'
    nc.variables['Ty_std'].units = 'm^2/s'
    nc.variables['Ty_skew'].long_name = 'Skewness of meridional Ekman transport'
    nc.variables['Ty_kurt'].long_name = 'Kurtosis of meridional Ekman transport'
    nc.variables['Txy_cov'].long_name = 'Covariance of Ekman transport'
    nc.variables['Txy_cov'].units = 'm^4/s^2'
    nc.variables['twx_mean'].long_name = 'Mean of zonal wind stress'
    nc.variables['twx_mean'].units = 'Pa'
    nc.variables['twx_std'].long_name = 'Standard deviation of zonal wind stress'
    nc.variables['twx_std'].units = 'Pa'
    nc.variables['twx_skew'].long_name = 'Skewness of zonal wind stress'
    nc.variables['twx_kurt'].long_name = 'Kurtosis of zonal wind stress'
    nc.variables['twy_mean'].long_name = 'Mean of meridional wind stress'
    nc.variables['twy_mean'].units = 'Pa'
    nc.variables['twy_std'].long_name = 'Standard deviation of meridional wind stress'
    nc.variables['twy_std'].units = 'Pa'
    nc.variables['twy_skew'].long_name = 'Skewness of meridional wind stress'
    nc.variables['twy_kurt'].long_name = 'Kurtosis of meridional wind stress'
    nc.variables['twxy_cov'].long_name = 'Covariance of wind stress'
    nc.variables['twxy_cov'].units = 'Pa^2'
    nc.variables['tsx_mean'].long_name = 'Mean of zonal wave stress'
    nc.variables['tsx_mean'].units = 'Pa'
    nc.variables['tsx_std'].long_name = 'Standard deviation of zonal wave stress'
    nc.variables['tsx_std'].units = 'Pa'
    nc.variables['tsx_skew'].long_name = 'Skewness of zonal wave stress'
    nc.variables['tsx_kurt'].long_name = 'Kurtosis of zonal wave stress'
    nc.variables['tsy_mean'].long_name = 'Mean of meridional wave stress'
    nc.variables['tsy_mean'].units = 'Pa'
    nc.variables['tsy_std'].long_name = 'Standard deviation of meridional wave stress'
    nc.variables['tsy_std'].units = 'Pa'
    nc.variables['tsy_skew'].long_name = 'Skewness of meridional wave stress'
    nc.variables['tsy_kurt'].long_name = 'Kurtosis of meridional wave stress'
    nc.variables['tsxy_cov'].long_name = 'Covariance of wave stress'
    nc.variables['tsxy_cov'].units = 'Pa^2'
    nc.variables['h_mean'].long_name = 'Mean of Ekman layer depth'
    nc.variables['h_mean'].units = 'm'
    nc.variables['h_std'].long_name = 'Standard deviation of Ekman layer depth'
    nc.variables['h_std'].units = 'm'
    nc.variables['argT_mean'].long_name = 'Mean of Ekman transport angle (from east)'
    nc.variables['argT_mean'].units = 'rad'
    nc.variables['argT_std'].long_name = 'Standard deviation of Ekman transport angle (from east)'
    nc.variables['argT_std'].units = 'rad'
    nc.variables['argU_mean'].long_name = 'Mean of surface current angle (from east)'
    nc.variables['argU_mean'].units = 'rad'
    nc.variables['argU_std'].long_name = 'Standard deviation of surface current angle (from east)'
    nc.variables['argU_std'].units = 'rad'
    nc.variables['argw_mean'].long_name = 'Mean of wind stress angle (from east)'
    nc.variables['argw_mean'].units = 'rad'
    nc.variables['argw_std'].long_name = 'Standard deviation of wind stress angle (from east)'
    nc.variables['argw_std'].units = 'rad'
    nc.variables['args_mean'].long_name = 'Mean of surface wave stress angle (from east)'
    nc.variables['args_mean'].units = 'rad'
    nc.variables['args_std'].long_name = 'Standard deviation of surface wave stress angle (from east)'
    nc.variables['args_std'].units = 'rad'
    nc.variables['absT_mean'].long_name = 'Mean of Ekman transport magnitude'
    nc.variables['absT_mean'].units = 'm^2/s'
    nc.variables['absT_std'].long_name = 'Standard deviation of Ekman transport magnitude'
    nc.variables['absT_std'].units = 'm^2/s'
    nc.variables['absU_mean'].long_name = 'Mean of surface current speed'
    nc.variables['absU_mean'].units = 'm/s'
    nc.variables['absU_std'].long_name = 'Standard deviation of surface current speed'
    nc.variables['absU_std'].units = 'm/s'
    nc.variables['a_mean'].long_name = 'Mean of eddy viscosity'
    nc.variables['a_mean'].units = 'm^2/s'
    nc.variables['a_std'].long_name = 'Standard deviation of eddy viscosity'
    nc.variables['a_std'].units = 'm^2/s'
    nc.variables['a_skew'].long_name = 'Skewness of eddy viscosity'
    nc.variables['a_kurt'].long_name = 'Kurtosis of eddy viscosity'
    nc.variables['ua_end'].long_name = 'Zonal wind at the end'
    nc.variables['ua_end'].units = 'm/s'
    nc.variables['va_end'].long_name = 'Meridional wind at the end'
    nc.variables['va_end'].units = 'm/s'
    nc.variables['us0_end'].long_name = 'Zonal surface Stokes drift at the end'
    nc.variables['us0_end'].units = 'm/s'
    nc.variables['vs0_end'].long_name = 'Meridional surface Stokes drift at the end'
    nc.variables['vs0_end'].units = 'm/s'
    nc.variables['u0_end'].long_name = 'Zonal surface Ekman velocity at the end'
    nc.variables['u0_end'].units = 'm/s'
    nc.variables['v0_end'].long_name = 'Meridional surface Ekman velocity at the end'
    nc.variables['v0_end'].units = 'm/s'
    nc.variables['twx_end'].long_name = 'Zonal wind stress at the end'
    nc.variables['twx_end'].units = 'Pa'
    nc.variables['twy_end'].long_name = 'Meridional wind stress at the end'
    nc.variables['twy_end'].units = 'Pa'
    nc.variables['tsx_end'].long_name = 'Zonal wave stress at the end'
    nc.variables['tsx_end'].units = 'Pa'
    nc.variables['tsy_end'].long_name = 'Meridional wave stress at the end'
    nc.variables['tsy_end'].units = 'Pa'
    nc.variables['Tx_end'].long_name = 'Zonal Ekman transport at the end'
    nc.variables['Tx_end'].units = 'm^2/s'
    nc.variables['Ty_end'].long_name = 'Meridional Ekman transport at the end'
    nc.variables['Ty_end'].units = 'm^2/s'

def complex_stats(f):
    """Compute statistics of complex tensors."""
    kwargs_sta = {'dim':0, 'keepdim':True}
    kwargs_nan = {'posinf':0, 'neginf':0}
    fmean = f.real.mean(**kwargs_sta) + 1j*f.imag.mean(**kwargs_sta)
    fstd = f.real.std(**kwargs_sta) + 1j*f.imag.std(**kwargs_sta)
    fcov = torch.sum((f.real - fmean.real)*(f.imag - fmean.imag), **kwargs_sta) / (f.shape[0]-1)
    fm3 = ((f.real - fmean.real)**3).mean(**kwargs_sta) + 1j*((f.imag - fmean.imag)**3).mean(**kwargs_sta)
    fm4 = ((f.real - fmean.real)**4).mean(**kwargs_sta) + 1j*((f.imag - fmean.imag)**4).mean(**kwargs_sta)
    fskew = torch.nan_to_num(fm3.real/(fstd.real**3),posinf=0,neginf=0) + 1j*torch.nan_to_num(fm3.imag/(fstd.imag**3),posinf=0,neginf=0)
    fkurt = torch.nan_to_num(fm4.real/(fstd.real**4), **kwargs_nan) + 1j*torch.nan_to_num(fm4.imag/(fstd.imag**4), **kwargs_nan)
    return fmean.squeeze(), fstd.squeeze(), fcov.squeeze(), fskew.squeeze(), fkurt.squeeze()


####################################################################################


# Set param.
dirm = '/srv/storage/ithaca@storage2.rennes.grid5000.fr/lli/ekman/rand_wind_wave'
dirs = os.listdir(dirm)

for s in dirs:
    datdir = os.path.join(dirm, s) 
    print(f'Read data from {datdir}')

    # Read param.
    param = torch.load(os.path.join(datdir,'param.pth'))
    param['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    nt = len(glob.glob(os.path.join(datdir,'*.npz')))

    # Create model
    spde = SGESM1d(param)
    dtype = spde.carr_kwargs['dtype']
    device = spde.carr_kwargs['device']
    rhoW = spde.rhooc * spde.W.real.squeeze()

    # Init. output
    file1 = os.path.join(datdir,'diag.nc')
    if os.path.exists(file1):
        os.remove(file1)
    nc = Dataset(file1,'w',format='NETCDF4')
    nc.createDimension('t',nt)
    nc.createDimension('z',spde.nz)
    nc.createDimension('ens',spde.ne)
    init_netcdf(nc)
    nc.variables['z'][:] = spde.z.squeeze().cpu().numpy()

    # Time iterations
    for n in range(nt):
        
        # Read input data
        print(f'file: t_{n}.npz')
        data = np.load(os.path.join(datdir,f't_{n}.npz'))
        nc.variables['t'][n] = float(data['t'])
        spde.ue = torch.from_numpy(data['ue']).type(dtype).to(device)
        if (n==0):
            spde.ua = spde.ua_mean.tile(spde.ne,1,1)
            # Stats of Stokes
            spde.us = torch.from_numpy(data['us']).type(dtype).to(device)
            us_mean, us_std, us_cov = complex_stats(spde.us)[:3]
            nc.variables['us_mean'][:] = us_mean.real.cpu().numpy() 
            nc.variables['vs_mean'][:] = us_mean.imag.cpu().numpy() 
            nc.variables['us_std'][:] = us_std.real.cpu().numpy() 
            nc.variables['vs_std'][:] = us_std.imag.cpu().numpy()  
            nc.variables['uvs_cov'][:] = us_cov.cpu().numpy()
        else:
            spde.ua = torch.from_numpy(data['ua']).type(dtype).to(device)
        data.close()
        
        # Stats of wind
        ua_mean, ua_std, ua_cov = complex_stats(spde.ua)[:3]
        nc.variables['ua_mean'][n] = ua_mean.real.cpu().numpy() 
        nc.variables['va_mean'][n] = ua_mean.imag.cpu().numpy() 
        nc.variables['ua_std'][n] = ua_std.real.cpu().numpy() 
        nc.variables['va_std'][n] = ua_std.imag.cpu().numpy()  
        nc.variables['uva_cov'][n] = ua_cov.cpu().numpy()

        # Set model
        spde.set_model()
        
        # Stats of wind stress
        tw_mean, tw_std, tw_cov, tw_skew, tw_kurt = complex_stats(spde.rhooc*spde.tauw)
        nc.variables['twx_mean'][n] = tw_mean.real.cpu().numpy() 
        nc.variables['twy_mean'][n] = tw_mean.imag.cpu().numpy() 
        nc.variables['twx_std'][n] = tw_std.real.cpu().numpy() 
        nc.variables['twy_std'][n] = tw_std.imag.cpu().numpy() 
        nc.variables['twxy_cov'][n] = tw_cov.cpu().numpy() 
        nc.variables['twx_skew'][n] = tw_skew.real.cpu().numpy() 
        nc.variables['twy_skew'][n] = tw_skew.imag.cpu().numpy() 
        nc.variables['twx_kurt'][n] = tw_kurt.real.cpu().numpy() - 3 
        nc.variables['twy_kurt'][n] = tw_kurt.imag.cpu().numpy() - 3  
        tauw = (spde.rhooc*spde.tauw.squeeze()).cpu().numpy()
        theta = np.angle(tauw)
        nc.variables['argw_mean'][n] = stats.circmean(theta)
        nc.variables['argw_std'][n] = stats.circstd(theta)
        
        # Stats of wave stress
        ts_mean, ts_std, ts_cov, ts_skew, ts_kurt = complex_stats(spde.rhooc*spde.taus)
        nc.variables['tsx_mean'][n] = ts_mean.real.cpu().numpy() 
        nc.variables['tsy_mean'][n] = ts_mean.imag.cpu().numpy() 
        nc.variables['tsx_std'][n] = ts_std.real.cpu().numpy() 
        nc.variables['tsy_std'][n] = ts_std.imag.cpu().numpy() 
        nc.variables['tsxy_cov'][n] = ts_cov.cpu().numpy() 
        nc.variables['tsx_skew'][n] = ts_skew.real.cpu().numpy() 
        nc.variables['tsy_skew'][n] = ts_skew.imag.cpu().numpy() 
        nc.variables['tsx_kurt'][n] = ts_kurt.real.cpu().numpy() - 3 
        nc.variables['tsy_kurt'][n] = ts_kurt.imag.cpu().numpy() - 3 
        taus = (spde.rhooc*spde.taus.squeeze()).cpu().numpy()
        theta = np.angle(taus)
        nc.variables['args_mean'][n] = stats.circmean(theta)
        nc.variables['args_std'][n] = stats.circstd(theta)

        # Stats of Ekman depth
        ids = (abs(spde.ue.squeeze()) <= 1e-4).int().argmax(dim=-1)
        h = abs(spde.z.squeeze()[ids])
        nc.variables['h_mean'][n] = h.mean().cpu().numpy() 
        nc.variables['h_std'][n] = h.std().cpu().numpy() 
        mask = torch.where(abs(spde.z.squeeze()) <= h.mean(), 1., 0.).cpu().numpy()

        # Stats of Ekman transport
        T = spde.W @ spde.ue
        T_mean, T_std, T_cov, T_skew, T_kurt = complex_stats(T)
        nc.variables['Tx_mean'][n] = T_mean.real.cpu().numpy() 
        nc.variables['Ty_mean'][n] = T_mean.imag.cpu().numpy() 
        nc.variables['Tx_std'][n] = T_std.real.cpu().numpy() 
        nc.variables['Ty_std'][n] = T_std.imag.cpu().numpy() 
        nc.variables['Txy_cov'][n] = T_cov.cpu().numpy() 
        nc.variables['Tx_skew'][n] = T_skew.real.cpu().numpy() 
        nc.variables['Ty_skew'][n] = T_skew.imag.cpu().numpy() 
        nc.variables['Tx_kurt'][n] = T_kurt.real.cpu().numpy() - 3 
        nc.variables['Ty_kurt'][n] = T_kurt.imag.cpu().numpy() - 3  
        T = T.squeeze().cpu().numpy()
        theta = np.angle(T)
        nc.variables['argT_mean'][n] = stats.circmean(theta)
        nc.variables['argT_std'][n] = stats.circstd(theta)
        nc.variables['absT_mean'][n] = abs(T).mean()
        nc.variables['absT_std'][n] = abs(T).std()

        # Stats of eddy viscosity
        a_mean, a_std, _, a_skew, a_kurt = complex_stats(spde.a+1j*torch.zeros_like(spde.a))   
        nc.variables['a_mean'][n] = a_mean.real.cpu().numpy()
        nc.variables['a_std'][n] = a_std.real.cpu().numpy() 
        nc.variables['a_skew'][n] = mask * a_skew.real.cpu().numpy() 
        nc.variables['a_kurt'][n] = mask * a_kurt.real.cpu().numpy() - 3 
        
        # Stats of Ekman current
        u_mean, u_std, u_cov, u_skew, u_kurt = complex_stats(spde.ue)
        nc.variables['u_mean'][n] = u_mean.real.cpu().numpy() 
        nc.variables['v_mean'][n] = u_mean.imag.cpu().numpy() 
        nc.variables['u_std'][n] = u_std.real.cpu().numpy() 
        nc.variables['v_std'][n] = u_std.imag.cpu().numpy() 
        nc.variables['uv_cov'][n] = u_cov.cpu().numpy() 
        nc.variables['u_skew'][n] = mask * u_skew.real.cpu().numpy() 
        nc.variables['v_skew'][n] = mask * u_skew.imag.cpu().numpy() 
        nc.variables['u_kurt'][n] = mask * u_kurt.real.cpu().numpy() - 3 
        nc.variables['v_kurt'][n] = mask * u_kurt.imag.cpu().numpy() - 3 
        u0 = spde.ue[:,0].squeeze().cpu().numpy()
        theta = np.angle(u0)
        nc.variables['argU_mean'][n] = stats.circmean(theta)
        nc.variables['argU_std'][n] = stats.circstd(theta)
        nc.variables['absU_mean'][n] = abs(u0).mean()
        nc.variables['absU_std'][n] = abs(u0).std() 

        # Kinetic energy
        mke, eke = abs(u_mean)**2/2, abs(u_std)**2/2
        nc.variables['mked'][n] = mke.cpu().numpy() 
        nc.variables['mke'][n] = (rhoW @ mke).cpu().numpy() 
        nc.variables['eked'][n] = eke.cpu().numpy() 
        nc.variables['eke'][n] = (rhoW @ eke).cpu().numpy() 

    # Ensemble values at the end
    nc.variables['Tx_end'][:], nc.variables['Ty_end'][:] = T.real, T.imag
    nc.variables['u0_end'][:], nc.variables['v0_end'][:] = u0.real, u0.imag
    nc.variables['twx_end'][:], nc.variables['twy_end'][:] = tauw.real, tauw.imag
    nc.variables['tsx_end'][:], nc.variables['tsy_end'][:] = taus.real, taus.imag
    ua = spde.ua.squeeze().cpu().numpy()
    nc.variables['ua_end'][:], nc.variables['va_end'][:] = ua.real, ua.imag
    us = spde.us[:,0].squeeze().cpu().numpy()
    nc.variables['us0_end'][:], nc.variables['vs0_end'][:] = us.real, us.imag

    # Close output
    nc.close()
    print(f'NetCDF file saved in {datdir} \n')
