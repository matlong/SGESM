import torch
import numpy as np
from SGESM import SGESM1d

torch.backends.cudnn.deterministic = True

param = {
        'ne': 1000, # number of ensemble samples
        'nz': 512, # number of vertical levels (T-grid)    
        'H': 256., # finite depth (m) 
        'dt': 1800., # time step (s)
        'f': 0.73e-4, # Coriolis parameter (s^-1)
        'rhooc': 1.0e3, # water density (kg/m^3)
        'rhoat': 1., # air density (kg/m^3)
        'Cd': 1.3e-3, # air-sea drag coef.
        'c1': 0.4, # von Karman constant
        'c2': 0.7, # constant to determine spde.n layer thickness
        'z0': 0.05, # constant to regularize log singularity
        'rand_wind': True, # option to include random wind fluctuations
        'ua_mean': 5.+0.j, # mean wind speed (m/s)
        'ua_std': 5., # std of wind speed (m/s)
        'Ta': 86400., # memory time for wind fluc. (s)
        'rand_wave': True, # option to include random wave angles
        'us0': 0.068, # Stokes amplititude (m/s)
        'hs': 4.775, # Stokes thickness (m)
        'theta_mean': 0*np.pi/180, # mean of Stokes angular direction
        'theta_std': 5*np.pi/180, # std of Stokes angular direction
        'wave_sbc': False, # include wave stress to surface BC
        'r': 0., # damping rate due to radiation of internal waves (s^-1)
        'kappa': 1.0e-4, # background uniform viscosity (m^2/s) 
        'tauw_diff': False, # option to use relative wind stress
        'rand_model': True, # option to use stochastic dynamical model
        'full_model': True, # option to include vertical mixing of waves 
        'device': 'cuda', #if torch.cuda.is_available() else 'cpu',
}

spde = SGESM1d(param)

# Control param.
t = 0.
dt = param['dt']
n_steps = int(30*24*3600/dt) + 1
freq_checknan = int(2*3600/dt)
freq_log = int(2*3600/dt)
freq_plot = 0*int(2*3600/dt)
freq_save = int(2*3600/dt) 
n_steps_save = 0
dirm = '/srv/storage/ithaca@storage2.rennes.grid5000.fr/lli/ekman/tauw_diff'

# Init. logout
if freq_log > 0:
    rho = param['rhooc']
    spde.diag_model()
    log_str = '*********************************************\n' \
              '  Stochastic Generalized Ekman-Stokes Model  \n' \
              '*********************************************\n\n'
    log_str += 'Input parmeters\n' \
               '---------------\n' \
               f'{param}\n\n' \
               'Output log\n' \
               '----------\n' \
               'Time, Surface current amp. (mean,std), Ekman transport amp. (mean,std), '\
               'Ekman layer depth (mean,std), Ekman transport angle (mean,std)'
    print(log_str)

# Init. outputs
if freq_save > 0:
    import os
    mode = 'rand_wind_wave_mode' if spde.rand_model else 'rand_wind_wave'
    if spde.full_model:
        mode = f'{mode}_full'
    ua, us, hs, ts = param['ua_std'], param['us0'], param['hs'], \
            round(np.rad2deg(param['theta_mean']))
    outdir = os.path.join(dirm, mode, f'ua{ua:.1f}_us{us:.3f}_hs{hs:.3f}_ts{ts}')
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(outdir, f't_{n_steps_save}.npz')
    np.savez(filename, t=t/(24*3600), ua=spde.ua.cpu().numpy(), \
             ue=spde.ue.cpu().numpy(), us=spde.us.cpu().numpy())
    n_steps_save += 1

# Init. figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
    plt.ion()
    z = spde.z.squeeze().cpu().numpy()
    fig, ax = plt.subplots(constrained_layout=True,figsize=(4,6))
    ax.set(xlim=(-0.1,0.1), xlabel='(m/s)', ylabel='$z$ (m)', \
           title=f'Ekman current after {int(t//86400):03d} days {int(t%86400//3600):02d} hours')
    ax.grid()
    plt.pause(0.1)

# Time-steppings    
for n in range(1, n_steps+1):

    spde.unsteady_solver()
    t += dt

    if (n % freq_checknan == 0) and (torch.isnan(spde.ue.abs()).any()): 
        raise ValueError(f'Stopping, NAN number at iteration {n}.')
    
    if freq_plot > 0 and n % freq_plot == 0:
        
        u = spde.ue.real.squeeze().cpu().numpy()
        v = spde.ue.imag.squeeze().cpu().numpy()
        um, us = np.mean(u,axis=0), np.std(u,axis=0) 
        vm, vs = np.mean(v,axis=0), np.std(v,axis=0) 
        
        ax.clear()
        ax.plot(um[::-1], z[::-1], label='$u$')
        ax.fill_betweenx(z[::-1], (um-1.96*us)[::-1], (um+1.96*us)[::-1], alpha=0.15)
        ax.plot(vm[::-1], z[::-1], label='$v$')
        ax.fill_betweenx(z[::-1], (vm-1.96*vs)[::-1], (vm+1.96*vs)[::-1], alpha=0.15)
        
        ax.set(xlim=(-0.1,0.1), xlabel='(m/s)', ylabel='$z$ (m)', \
               title=f'Ekman current after {int(t//86400):03d} days {int(t%86400//3600):02d} hours')
        ax.legend(loc='lower right')
        ax.grid()
        plt.yticks(rotation=90)
        plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        spde.diag_model()
        he = spde.he.cpu().numpy()
        de = spde.dege.cpu().numpy()
        ui = abs(spde.ue_int).cpu().numpy()
        u0 = abs(spde.ue[:,0]).squeeze().cpu().numpy()
        log_str = f't={int(t//86400):03d}d{int(t%86400//3600):02d}h, ' \
                  f'u0=({np.mean(u0):.3f},{np.std(u0):.3f})m/s, ' \
                  f'ui=({np.mean(ui):.3f},{np.std(ui):.3f})m^2/s, ' \
                  f'he=({np.mean(he):.2f},{np.std(he):.2f})m, ' \
                  f'de=({int(np.mean(de))},{int(np.std(de))})deg'
        """
        if spde.rand_wind:
            dw = spde.degw.cpu().numpy()
            tw = rho * abs(spde.tauw).squeeze().cpu().numpy()
            log_str += f', dw=({int(np.mean(dw))},{int(np.std(dw))}) deg, ' \
                    f'tw=({np.mean(tw):.3f},{np.std(tw):.3f}) Pa'
        """
        print(log_str)
        
    if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
        filename = os.path.join(outdir, f't_{n_steps_save}.npz')
        if spde.rand_wind:
            np.savez(filename, t=t/(24*3600), ua=spde.ua.cpu().numpy(), ue=spde.ue.cpu().numpy())
        else:
            np.savez(filename, t=t/(24*3600), ue=spde.ue.cpu().numpy())
        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'Data saved to {filename}')

if freq_log > 0:
    log_str = '\n*********************************************\n' \
                '                      END                    \n' \
                '*********************************************'
    print(log_str)
