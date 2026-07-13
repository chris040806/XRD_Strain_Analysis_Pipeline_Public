import numpy as np
import tifffile as tiff
from nexusformat.nexus import *
import os
import six
#from nexpy.gui.pyqt import QtCore
from scipy import optimize
from numpy.linalg import inv, norm
from ImageD11 import labelimage,cImageD11,unitcell
import pyFAI
import nxrefine_lab
import h5py
import matplotlib.pyplot as plt
import itertools

def LabCreate(file = None, calib_path = None, imsize = 3450, imstart = 1, imend = 400, det_bg = 10,
             frame_time = 60, orientation_matrix = np.eye(3), chi = -90, omega = 0,
             phi_start = 0, phi_end = 20, phi_step = 0.05, two_theta = 0, sample_label =None,
             sample_name = None, sample_temperature = 300, unit_cell_group = 'tetragonal',
             lattice_centring = 'I', unitcell_a = 1.0, unitcell_b = 1.0, unitcell_c = 1.0,
             unitcell_alpha = 90, unitcell_beta = 90, unitcell_gamma = 90, calibrant = 'Si powder',
             calibration_date = None, datafile_path = None, data_base_name = None, chunk_size = 10, 
             null_image = False, imstartdiff = 0, gapstart = [0], gapend = [0]):
    if file is None:
        print('filename error')
    NXfile = NXroot()
    NXfile['/entry'] = NXentry()
    NXfile.save(file + '.nxs')
    
    NXfilecreate=nxrefine_lab.NXCreate(NXfile,calib_path = calib_path, imsize = imsize, 
                                       imstart = imstart, imend = imend, det_bg = det_bg,
                                       frame_time = frame_time, orientation_matrix = orientation_matrix, 
                                       chi = chi, omega = omega, phi_start = phi_start, phi_end = phi_end, 
                                       phi_step = phi_step, two_theta = two_theta, sample_label =sample_label,
                                       sample_name = sample_name, sample_temperature = sample_temperature, 
                                       unit_cell_group = unit_cell_group, lattice_centring = lattice_centring, 
                                       unitcell_a = unitcell_a, unitcell_b = unitcell_b, unitcell_c = unitcell_c,
                                       unitcell_alpha = unitcell_alpha, unitcell_beta = unitcell_beta, 
                                       unitcell_gamma = unitcell_gamma, calibrant = calibrant, 
                                       calibration_date = calibration_date, datafile_path = datafile_path, 
                                       data_base_name = data_base_name, chunk_size = chunk_size, 
                                       null_image = null_image, imstartdiff = imstartdiff, 
                                       gapstart = gapstart, gapend = gapend)
    NXfilecreate.write_instrument()
    NXfilecreate.write_sample()
    #NXfilecreate.write_data_prepare()
    NXfilecreate.write_data()


def LabReduce(file = None, maxcount = True, find = True, refine = True, lattice = True, 
              imstart = 1, imend = 400, threshold = None, null_image = False, imstartdiff = 0,
              peaklist = [], xwidth = 25, ywidth = 25, zwidth = 5, peaklist_filename="peaklist.txt"):
    if file is None:
        print('filename error')
        
    NXfile = nxload(file+'.nxs','rw')
    NXfilereduce = nxrefine_lab.NXReduce(NXfile, maxcount = maxcount, find = find, refine = refine, lattice = lattice, 
                                         imstart = imstart, imend = imend, threshold = threshold, 
                                         null_image = null_image, imstartdiff = imstartdiff)
    NXfilereduce.nxmax()
    peaks = NXfilereduce.nxfind()
    
    NXfile = nxload(file+'.nxs','rw')
    filepeaks = NXfile.entry.peaks
    indices = np.argsort(filepeaks.polar_angle.nxdata)
    satpeaks = []
    bdpeaks = []
    peakchoice = []
    peaks2file = []

    for i in range(len(filepeaks.x)):
        print('#%s\t\tpeak#%s' %(i,indices[i]))
        print('x,y,z_frame = %.1f\t\t%.1f\t\t%.1f' %(filepeaks.x[indices[i]],filepeaks.y[indices[i]],filepeaks.z_frame[indices[i]]))
        print('polar,azimuthal = %.2f\t\t%.2f' %(filepeaks.polar_angle[indices[i]],filepeaks.azimuthal_angle[indices[i]]))    
        peakx = np.int(filepeaks.x[indices[i]])
        peaky = np.int(filepeaks.y[indices[i]])
        peakz = np.int(filepeaks.z[indices[i]])
        imsize = NXfile.entry.data.shape[2]
        #if (peakx-imsize/2)**2 + (peaky-imsize/2)**2 < (imsize/2-11)**2 and 10<=peakz<NXfile.entry.data.shape[0]-10:
        if (peakx-imsize/2)**2 + (peaky-imsize/2)**2 < (imsize/2-11)**2 and 10<=peakz<imend-imstart+imstartdiff+1-10:
            maxinbox = np.max(NXfile.entry.data.intensity[peakz-10:peakz+10,peaky-10:peaky+10,peakx-10:peakx+10].nxdata)
            print('intensity = %d' %maxinbox)
            peaks2file.append([peakz, peaky, peakx, maxinbox])
            
            if np.any(NXfile.entry.data.intensity[peakz-10:peakz+10,peaky-10:peaky+10,peakx-10:peakx+10].nxdata>65530):
                print('saturated')
                satpeaks.append(indices[i])
        else:
            print('touch the boundary')
            bdpeaks.append(indices[i])
        if np.any(peaklist):
            idz = np.where(np.logical_and(peaklist[:,0] >= peakz+imstart-imstartdiff-zwidth, peaklist[:,0] <= peakz+imstart-imstartdiff+zwidth ))[0]
            if idz.size:
                idy = np.where(np.logical_and(peaklist[idz,1] >= peaky-ywidth, peaklist[idz,1] <= peaky+ywidth ))[0]
                if idy.size:
                    idx = np.where(np.logical_and(peaklist[idz[idy],2] >= peakx-xwidth, peaklist[idz[idy],2] <= peakx+xwidth ))[0]
                    if idx.size:
                        print('in the peak list')
                        peakchoice.append(indices[i])
        print('\n')
    print('saturated peaks:\t',np.sort(satpeaks).tolist())
    print('boundary peaks:\t', np.sort(bdpeaks).tolist())
    with open(peaklist_filename, "w") as out:
        peaks2file = np.vstack(peaks2file)
        print(peaks2file)
        assert peaks2file.shape[1] == 4
        print(f"saving to {peaklist_filename}")
        np.savetxt(out, peaks2file, delimiter=',')
        
    if np.any(peaklist):
        print('peaks not in list:\t', [i for i in range(len(filepeaks.x)) if i not in peakchoice])
    return peaks


def LabReduce_peakselect(file = None, imstart = 1, imend = 400, threshold = None, 
                         null_image = False, imstartdiff = 0, peaks = [], bgpeaks = []):
    if file is None:
        print('filename error') 

    NXfile = nxload(file+'.nxs','rw')
    NXfilereduce = nxrefine_lab.NXReduce(NXfile, imstart = imstart, imend = imend, threshold = threshold, 
                                         null_image = null_image, imstartdiff = imstartdiff)  
    NXfilereduce.write_postpeaks(peaks,bgpeaks)   

    NXfile = nxload(file+'.nxs','rw')
    filepeaks = NXfile.entry.postpeaks
    indices = np.argsort(filepeaks.polar_angle.nxdata)
    oldindices = np.where(np.in1d(NXfile.entry.peaks.z, NXfile.entry.postpeaks.z))[0]   #both peaks.z and postpeaks.z are sorted
    for i in range(len(filepeaks.x)):
        print('#%s\t\tpeak#%s\t\told peak#%s' %(i,indices[i],oldindices[indices[i]]))
        print('x,y,z_frame = %.1f\t\t%.1f\t\t%.1f' %(filepeaks.x[indices[i]],filepeaks.y[indices[i]],filepeaks.z_frame[indices[i]]))
        print('polar,azimuthal = %.2f\t\t%.2f' %(filepeaks.polar_angle[indices[i]],filepeaks.azimuthal_angle[indices[i]]))
        peakx = np.int(filepeaks.x[indices[i]])
        peaky = np.int(filepeaks.y[indices[i]])
        peakz = np.int(filepeaks.z[indices[i]])
        maxinbox = np.max(NXfile.entry.data.intensity[peakz-10:peakz+10,peaky-10:peaky+10,peakx-10:peakx+10].nxdata)
        print('intensity = %d' %maxinbox)
        print('\n')


def LabReduce_peaklist(file = None, imstart = 1, imend = 400, threshold = None, null_image = False, imstartdiff = 0, peaklist = []):
    if file is None:
        print('filename error') 
    
    NXfile = nxload(file+'.nxs','rw')
    NXfilereduce = nxrefine_lab.NXReduce(NXfile, imstart = imstart, imend = imend, threshold = threshold, 
                                         null_image = null_image, imstartdiff = imstartdiff)  
    NXfilereduce.write_peaklist(peaklist)

    NXfile = nxload(file+'.nxs','rw')
    filepeaks = NXfile.entry.postpeaks
    indices = np.argsort(filepeaks.polar_angle.nxdata)
    for i in range(len(filepeaks.x)):
        print('#%s\t\tpeak#%s' %(i,indices[i]))
        print('x,y,z_frame = %.1f\t\t%.1f\t\t%.1f' %(filepeaks.x[indices[i]],filepeaks.y[indices[i]],filepeaks.z_frame[indices[i]]))
        print('polar,azimuthal = %.2f\t\t%.2f' %(filepeaks.polar_angle[indices[i]],filepeaks.azimuthal_angle[indices[i]]))
        print('intensity = %d' %filepeaks.intensity[indices[i]])
        print('\n')


def LabReduce_check(file = None):
    if file is None:
        print('filename error')

    NXfile = nxload(file+'.nxs','rw')
    angle = NXfile.entry.radial_maxsum.polar_angle.nxdata
    intensity = NXfile.entry.radial_maxsum.radial_maxsum.nxdata
    powder = np.zeros((len(intensity),2))
    powder[:,0] = angle
    powder[:,1] = intensity
    label = NXfile.entry.sample.name.nxdata + '_' +NXfile.entry.sample.label.nxdata + '_' +str(NXfile.entry.sample.temperature.nxdata) + 'K'
    np.savetxt(label + 'powder.txt', powder)
    polarpeaks = np.sort(NXfile.entry.postpeaks.polar_angle.nxdata)
    writepolar = np.c_[polarpeaks,np.zeros((len(polarpeaks),7))]
    np.savetxt(label + '.pkslst', writepolar, delimiter = ', ', fmt = ['%.4f', '%1d', '%1d', '%1d', '%1d', '%1d', '%1d', '%1d'])


def LabReduce_unitcell(file = None, unitcell_a = 4.0, unitcell_b = 4.0, unitcell_c = 4.0,
                       unitcell_alpha = 90.0, unitcell_beta = 9.0, unitcell_gamma = 90.0):
    if file is None:
        print('filename error')

    NXfile = nxload(file+'.nxs','rw')
    with NXfile.entry.nxfile:
        if 'unitcell_a' in NXfile.entry['sample']:
            del NXfile.entry['sample/unitcell_a']
        NXfile.entry['sample/unitcell_a'] = unitcell_a
        if 'unitcell_b' in NXfile.entry['sample']:
            del NXfile.entry['sample/unitcell_b']
        NXfile.entry['sample/unitcell_b'] = unitcell_b
        if 'unitcell_c' in NXfile.entry['sample']:
            del NXfile.entry['sample/unitcell_c']
        NXfile.entry['sample/unitcell_c'] = unitcell_c
        if 'unitcell_alpha' in NXfile.entry['sample']:
            del NXfile.entry['sample/unitcell_alpha']
        NXfile.entry['sample/unitcell_alpha'] = unitcell_alpha
        if 'unitcell_beta' in NXfile.entry['sample']:
            del NXfile.entry['sample/unitcell_beta']
        NXfile.entry['sample/unitcell_beta'] = unitcell_beta
        if 'unitcell_gamma' in NXfile.entry['sample']:
            del NXfile.entry['sample/unitcell_gamma']
        NXfile.entry['sample/unitcell_gamma'] = unitcell_gamma


def LabRefine_prepare(file = None, hklselect = True, peak_tolerance = 5.0, ring_tolerance = 1.0, hkl_tolerance = 0.05, mode = 1):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    NXfilerefine = nxrefine_lab.NXRefine(NXfile, peak_tolerance = peak_tolerance, ring_tolerance = ring_tolerance, hkl_tolerance = hkl_tolerance)
    try:
        NXfilerefine.generate_grains_lab(hklselect = hklselect, mode = mode)   ###verbose = 1 to print the calculation process
        return NXfilerefine
    except:
        print('generate_grains_lab error')
        return NXfilerefine


def LabRefine_check(refinevars = None, hkl_tolerance = 0.1, write_peaklist = False):
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars
    graintot = len(filerefine.grains_lab)
    indices = np.searchsorted(np.sort(filerefine.entry.postpeaks.polar_angle.nxdata),filerefine.entry.postpeaks.polar_angle.nxdata)
    if not write_peaklist:
        oldindices = np.where(np.in1d(filerefine.entry.peaks.z, filerefine.entry.postpeaks.z))[0]
    for igrain in range(graintot):
        print('grain#%s' %igrain)
        print(filerefine.grains_lab[igrain])
        filerefine.Umat = filerefine.grains_lab[igrain].Umat
        h,k,l = filerefine.get_hkls()
        peaktot = len(h)
        if any(np.isnan(h)) or any(np.isnan(k)) or any(np.isnan(l)):
            print('nan')
            continue
        for ipeak in range(peaktot):
            hdiff = abs(h[ipeak]-round(h[ipeak]))
            kdiff = abs(k[ipeak]-round(k[ipeak]))
            ldiff = abs(l[ipeak]-round(l[ipeak]))
            if hdiff > hkl_tolerance or kdiff > hkl_tolerance or ldiff > hkl_tolerance:
                if not write_peaklist:
                    print('peak#%s\t\tpolar#%s\t\told peak#%s' %(ipeak,indices[ipeak],oldindices[ipeak]))
                else:
                    print('peak#%s\t\tpolar#%s' %(ipeak,indices[ipeak]))
                print('h,k,l = %s\t%s\t%s' % (h[ipeak],k[ipeak],l[ipeak]))
        print('\n')

def LabRefine_grain(refinevars = None, grain = 0):
    if refinevars is None:
        print('refinevars error')    
    filerefine = refinevars   
    filerefine.Umat = filerefine.grains_lab[grain].Umat
    filerefine.write_parameters()
    with filerefine.entry.nxfile:
        if 'h' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/h']
        if 'k' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/k']
        if 'l' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/l']
        h,k,l = filerefine.get_hkls()
        filerefine.entry['postpeaks/h'] = NXfield(h)
        filerefine.entry['postpeaks/k'] = NXfield(k)
        filerefine.entry['postpeaks/l'] = NXfield(l)


def LabRefine_init(refinevars = None):
    if refinevars is None:
        print('refinevars error')    
    filerefine = refinevars
    return filerefine.a,filerefine.b,filerefine.c,filerefine.alpha,filerefine.beta,filerefine.gamma,filerefine.omega,filerefine.chi,filerefine.Umat


def LabRefine(file = None, refinevars = None, grain = 0, maxcount = True, 
              find = True, refine = True, lattice = True, hkl_tolerance = 0.05, posthkl_tolerance = 0.05, write_peaklist = False):
    if refinevars is None:
        print('refinevars error')    
    filerefine = refinevars   
    filerefine.Umat = filerefine.grains_lab[grain].Umat
    filerefine.write_parameters()
    
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    NXfilereduce = nxrefine_lab.NXReduce(NXfile, maxcount = maxcount, find = find, refine = refine, lattice = lattice)
    NXfilereduce.nxrefine(posthkl_tolerance = posthkl_tolerance)
    
    NXfile = nxload(file+'.nxs','rw')
    NXfilerefine = nxrefine_lab.NXRefine(NXfile)
    h,k,l = NXfilerefine.get_hkls()
    peaktot = len(h)
    indices = np.searchsorted(np.sort(NXfilerefine.entry.postpeaks.polar_angle.nxdata),NXfilerefine.entry.postpeaks.polar_angle.nxdata)
    if not write_peaklist:
        oldindices = np.where(np.in1d(NXfilerefine.entry.peaks.z, NXfilerefine.entry.postpeaks.z))[0]
    if any(np.isnan(h)) or any(np.isnan(k)) or any(np.isnan(l)):
        print('nan')
    for ipeak in range(peaktot):
        hdiff = abs(h[ipeak]-round(h[ipeak]))
        kdiff = abs(k[ipeak]-round(k[ipeak]))
        ldiff = abs(l[ipeak]-round(l[ipeak]))
        if hdiff > hkl_tolerance or kdiff > hkl_tolerance or ldiff > hkl_tolerance:
            if not write_peaklist:
                print('peak#%s\t\tpolar#%s\t\told peak#%s' %(ipeak,indices[ipeak],oldindices[ipeak]))
            else:
                print('peak#%s\t\tpolar#%s' %(ipeak,indices[ipeak]))
            print('h,k,l = %s\t%s\t%s' % (h[ipeak],k[ipeak],l[ipeak]))

    with NXfilerefine.entry.nxfile:
        if 'h' in NXfilerefine.entry['postpeaks']:
            del NXfilerefine.entry['postpeaks/h']
        if 'k' in NXfilerefine.entry['postpeaks']:
            del NXfilerefine.entry['postpeaks/k']
        if 'l' in NXfilerefine.entry['postpeaks']:
            del NXfilerefine.entry['postpeaks/l']
        NXfilerefine.entry['postpeaks/h'] = NXfield(h)
        NXfilerefine.entry['postpeaks/k'] = NXfield(k)
        NXfilerefine.entry['postpeaks/l'] = NXfield(l)


def LabRefine_restore(refinevars = None,a_init=4,b_init=4,c_init=4,alpha_init=90,beta_init=90,gamma_init=90,
                      omega_init=0,chi_init=-90,Umat_init=np.eye(3)):
    if refinevars is None:
        print('refinevars error')    
    filerefine = refinevars
    filerefine.a = a_init
    filerefine.b = b_init
    filerefine.c = c_init
    filerefine.alpha = alpha_init
    filerefine.beta = beta_init
    filerefine.gamma = gamma_init
    filerefine.omega = omega_init
    filerefine.chi = chi_init
    filerefine.Umat =  Umat_init
    with filerefine.entry.nxfile:
        filerefine.write_parameters()


def LabRefine_post(file = None, hklselect = True, peak_tolerance = 5.0, hkl_tolerance = 0.05, write_peaklist = False):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    NXfilerefine = nxrefine_lab.NXRefine(NXfile, peak_tolerance = peak_tolerance, hkl_tolerance = hkl_tolerance)
    try:
        NXfilerefine.generate_grains_lab(hklselect = hklselect)
        for grain in NXfilerefine.grains_lab:
            NXfilerefine.Umat = grain.Umat = np.matrix(NXfile.entry.instrument.detector.orientation_matrix.nxdata)
        LabRefine_check(refinevars = NXfilerefine, hkl_tolerance = hkl_tolerance, write_peaklist = write_peaklist)
        with NXfilerefine.entry.nxfile:
            if 'h' in NXfilerefine.entry['postpeaks']:
                del NXfilerefine.entry['postpeaks/h']
            if 'k' in NXfilerefine.entry['postpeaks']:
                del NXfilerefine.entry['postpeaks/k']
            if 'l' in NXfilerefine.entry['postpeaks']:
                del NXfilerefine.entry['postpeaks/l']
            h,k,l = NXfilerefine.get_hkls()
            NXfilerefine.entry['postpeaks/h'] = NXfield(h)
            NXfilerefine.entry['postpeaks/k'] = NXfield(k)
            NXfilerefine.entry['postpeaks/l'] = NXfield(l)    
        return NXfilerefine
    except:
        print('generate_grains_lab error')
        return NXfilerefine


def LabRefine_transform_prepare(file = None, itn = 1000, chunkx = 1, chunky = 1, chunkz = 1):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    filerefine = nxrefine_lab.NXRefine(NXfile)

    astar = 2*np.pi*(filerefine.Bmat[0,0]**2 + filerefine.Bmat[0,1]**2 + filerefine.Bmat[0,2]**2)**0.5
    bstar = 2*np.pi*(filerefine.Bmat[1,0]**2 + filerefine.Bmat[1,1]**2 + filerefine.Bmat[1,2]**2)**0.5
    cstar = 2*np.pi*(filerefine.Bmat[2,0]**2 + filerefine.Bmat[2,1]**2 + filerefine.Bmat[2,2]**2)**0.5
    print('astar,bstar,cstar (A-1) = %.4f\t%.4f\t%.4f' %(astar,bstar,cstar))

    diffh = np.zeros(itn)
    diffk = np.zeros(itn)
    diffl = np.zeros(itn)
    if chunkx == 1 and chunky == 1 and chunkz == 1:
        for i in range(itn):
            idx = np.random.randint(0,filerefine.shape[0])
            idy = np.random.randint(0,filerefine.shape[1])
            idz = np.random.randint(0,filerefine.entry.data.last)
            h0,k0,l0 = filerefine.get_hkl(idx,idy,idz)
            ranidx = np.random.randint(0,3)
            ranarr = [[1,0,0],[0,1,0],[0,0,1]]
            h1,k1,l1 = filerefine.get_hkl(idx+ranarr[ranidx][0],idy+ranarr[ranidx][1],idz+ranarr[ranidx][2])
            diffh[i] = h1-h0
            diffk[i] = k1-k0
            diffl[i] = l1-l0
    else:
        for i in range(itn):
            idx = np.random.randint(0,filerefine.shape[0])
            idy = np.random.randint(0,filerefine.shape[1])
            idz = np.random.randint(0,filerefine.entry.data.last)
            hbox = np.zeros((chunkx,chunky,chunkz))
            kbox = np.zeros((chunkx,chunky,chunkz))
            lbox = np.zeros((chunkx,chunky,chunkz))
            for ix in range(chunkx):
                for iy in range(chunky):
                    for iz in range(chunkz):
                        h,k,l = filerefine.get_hkl(idx+ix,idy+iy,idz+iz)
                        hbox[ix,iy,iz] = h
                        kbox[ix,iy,iz] = k
                        lbox[ix,iy,iz] = l
            diffh[i] = np.max(hbox)-np.min(hbox)
            diffk[i] = np.max(kbox)-np.min(kbox)
            diffl[i] = np.max(lbox)-np.min(lbox)
            #print('%.3f\t%.3f\t%.3f'%(diffh[i],diffk[i],diffl[i]))

    #fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(20,10))
    fig, axs = plt.subplots(1, 3,sharey=True, figsize=(20,10))

    axs[0].hist(diffh,bins = int(itn/10))
    axs[0].set_xlabel(r'$\Delta$'+'H(r.l.u.)',fontsize=20)
    axs[0].tick_params(labelsize=20) 
    axs[1].hist(diffk,bins = int(itn/10))
    axs[1].set_xlabel(r'$\Delta$'+'K(r.l.u.)',fontsize=20)
    axs[1].tick_params(labelsize=20) 
    axs[2].hist(diffl,bins = int(itn/10))
    axs[2].set_xlabel(r'$\Delta$'+'L(r.l.u.)',fontsize=20)
    axs[2].tick_params(labelsize=20) 

    filerefine.make_mask(chunkx,chunky,chunkz)


def LabRefine_transform_writemask(file = None, chunkx = 1, chunky = 1, chunkz = 1):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    filerefine = nxrefine_lab.NXRefine(NXfile)
    filerefine.make_mask(chunkx,chunky,chunkz)


def LabRefine_transform_local(file = None, xstart = 0, xend = 1, ystart = 0, yend = 1, zstart = 0, zend = 1):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    filerefine = nxrefine_lab.NXRefine(NXfile)
    data_t,_h,_k,_l = filerefine.transform_localdata(xstart,xend,ystart,yend,zstart,zend)
    return data_t,_h,_k,_l


def LabRefine_combine_transform(file = None, data_t = None, _h = None, _k = None,  _l = None, deltah = 0.01, deltak = 0.01, deltal = 0.01):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    filerefine = nxrefine_lab.NXRefine(NXfile)
    transform_data,Hrange,Krange,Lrange = filerefine.combine_transform(data_t,_h,_k,_l,deltah,deltak,deltal)
    maxindex = np.where(transform_data == np.max(transform_data))
    print('peak index:\tH = %f,\tK = %f,\tL = %f'%(Hrange[maxindex[0][0]],Krange[maxindex[1][0]],Lrange[maxindex[2][0]]))
    return transform_data,Hrange,Krange,Lrange

def HKmap(Hrange,Krange,Lrange,Lval,transform_data):
    fig,ax=plt.subplots()
    im = ax.pcolormesh(Krange, Hrange, np.log10(transform_data[:,:,np.searchsorted(Lrange, Lval, side="left")]))
    ax.set_xlabel('K(r.l.u.)',fontsize = 20)
    ax.set_ylabel('H(r.l.u.)',fontsize = 20)
    ax.tick_params(labelsize=15) 
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=15)

def KLmap(Hrange,Krange,Lrange,Hval,transform_data):
    fig,ax=plt.subplots()
    im = ax.pcolormesh(Lrange, Krange, np.log10(transform_data[np.searchsorted(Hrange, Hval, side="left"),:,:]))
    ax.set_xlabel('L(r.l.u.)',fontsize = 20)
    ax.set_ylabel('K(r.l.u.)',fontsize = 20)
    ax.tick_params(labelsize=15) 
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=15)

def HLmap(Hrange,Krange,Lrange,Kval,transform_data):
    fig,ax=plt.subplots()
    im = ax.pcolormesh(Lrange, Hrange, np.log10(transform_data[:,np.searchsorted(Krange, Kval, side="left"),:]))
    ax.set_xlabel('L(r.l.u.)',fontsize = 20)
    ax.set_ylabel('H(r.l.u.)',fontsize = 20)
    ax.tick_params(labelsize=15) 
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=15)

def Hscan(hmin,hmax,kstep,lstep,Hrange,transform_data):
    kstep = int(kstep)
    lstep = int(lstep)
    h0 = np.searchsorted(Hrange, hmin, side="left")
    h1 = np.searchsorted(Hrange, hmax, side="left")
    maxindex = np.where(transform_data == np.max(transform_data))
    scan = np.sum(transform_data[h0:h1,maxindex[1][0]-kstep:maxindex[1][0]+kstep,maxindex[2][0]-lstep:maxindex[2][0]+lstep],axis = (1,2))
    fig,ax=plt.subplots()
    ax.plot(Hrange[h0:h1], scan,'o:')
    ax.set_xlabel('H(r.l.u.)',fontsize = 20)
    ax.set_ylabel('Intensity(a.u.)',fontsize = 20)
    ax.tick_params(labelsize=15)

def Kscan(kmin,kmax,hstep,lstep,Krange,transform_data):
    hstep = int(hstep)
    lstep = int(lstep)
    k0 = np.searchsorted(Krange, kmin, side="left")
    k1 = np.searchsorted(Krange, kmax, side="left")
    maxindex = np.where(transform_data == np.max(transform_data))
    scan = np.sum(transform_data[maxindex[0][0]-hstep:maxindex[0][0]+hstep,k0:k1,maxindex[2][0]-lstep:maxindex[2][0]+lstep],axis = (0,2))
    fig,ax=plt.subplots()
    ax.plot(Krange[k0:k1], scan,'o:')
    ax.set_xlabel('K(r.l.u.)',fontsize = 20)
    ax.set_ylabel('Intensity(a.u.)',fontsize = 20)
    ax.tick_params(labelsize=15)

def Lscan(lmin,lmax,hstep,kstep,Lrange,transform_data):
    hstep = int(hstep)
    kstep = int(kstep)
    l0 = np.searchsorted(Lrange, lmin, side="left")
    l1 = np.searchsorted(Lrange, lmax, side="left")
    maxindex = np.where(transform_data == np.max(transform_data))
    scan = np.sum(transform_data[maxindex[0][0]-hstep:maxindex[0][0]+hstep,maxindex[1][0]-kstep:maxindex[1][0]+kstep,l0:l1],axis = (0,1))
    fig,ax=plt.subplots()
    ax.plot(Lrange[l0:l1], scan,'o:')
    ax.set_xlabel('L(r.l.u.)',fontsize = 20)
    ax.set_ylabel('Intensity(a.u.)',fontsize = 20)
    ax.tick_params(labelsize=15)


def LabRefine_compress(file = None, calib_path = None, imsize = 3450, imstart = 1, imend = 400, det_bg = 10,
                        sample_name = None, sample_temperature = 300, datafile_path = None, data_base_name = None, 
                        chunk_size = 10, null_image = False, imstartdiff = 0, gapstart = [0], gapend = [0]):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    NXfilecreate=nxrefine_lab.NXCreate(NXfile, calib_path = calib_path, imsize = imsize, imstart = imstart, 
                                       imend = imend, det_bg = det_bg, sample_name = sample_name, 
                                       sample_temperature = sample_temperature, datafile_path = datafile_path, 
                                       data_base_name = data_base_name, chunk_size = chunk_size, 
                                       null_image = null_image, imstartdiff = imstartdiff, 
                                       gapstart = gapstart, gapend = gapend)
    NXfilecreate.write_data(compress = True, backup = True)
    os.remove(NXfilecreate.sample_name + '_' + str(NXfilecreate.sample_temperature) + 'K.hdf5')
    os.rename(NXfilecreate.sample_name + '_' + str(NXfilecreate.sample_temperature) + 'K_backup.hdf5',NXfilecreate.sample_name + '_' + str(NXfilecreate.sample_temperature) + 'K.hdf5')
    chunk_size = NXfile.entry.data.intensity.chunks[0]
    last = NXfile.entry.data.last
    print('dataidzfinal = %s' %(int(np.ceil(last/chunk_size))-1))


def LabRefine_uncompress(file = None):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')

    f = h5py.File(NXfile.entry.sample.name.nxdata + '_' + str(NXfile.entry.sample.temperature.nxdata) + 'K_backup.hdf5')
    _nxentry = f.create_group('entry')
    _nxdata = _nxentry.create_dataset('intensity', NXfile.entry.data.intensity.shape, dtype='uint16', chunks = NXfile.entry.data.intensity.chunks, compression = None, shuffle = False)
    imagetot = NXfile.entry.data.intensity.shape[0]
    chunk_size = NXfile.entry.data.intensity.chunks[0]
    for i in range(0,imagetot,chunk_size):
        _nxdata[i:i+chunk_size,:,:] = NXfile.entry.data.intensity[i:i+chunk_size,:,:]
    f.close()

    os.remove(NXfile.entry.sample.name.nxdata + '_' + str(NXfile.entry.sample.temperature.nxdata) + 'K.hdf5')
    os.rename(NXfile.entry.sample.name.nxdata + '_' + str(NXfile.entry.sample.temperature.nxdata) + 'K_backup.hdf5',NXfile.entry.sample.name.nxdata + '_' + str(NXfile.entry.sample.temperature.nxdata) + 'K.hdf5')


def LabRefine_transform(file = None, deltah = 0.01, deltak = 0.01, deltal = 0.01):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    
    def get_closest(array, values):
        #make sure array is a numpy array
        #array = np.array(array)

        # get insert positions
        idxs = np.searchsorted(array, values, side="left")

        # find indexes where previous index is closer
        prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
        idxs[prev_idx_is_less] -= 1

        #return array[idxs]
        return idxs

    hlist = NXfile.entry.transform.h.nxdata
    klist = NXfile.entry.transform.k.nxdata
    llist = NXfile.entry.transform.l.nxdata
    Hrange = np.linspace(np.min(hlist),np.max(hlist),int((np.max(hlist)-np.min(hlist))/deltah+1))
    Krange = np.linspace(np.min(klist),np.max(klist),int((np.max(klist)-np.min(klist))/deltak+1))
    Lrange = np.linspace(np.min(llist),np.max(llist),int((np.max(llist)-np.min(llist))/deltal+1))
    axis0 = NXfield(Hrange,name = 'H',units = 'r.l.u.')
    axis1 = NXfield(Krange,name = 'K',units = 'r.l.u.')
    axis2 = NXfield(Lrange,name = 'L',units = 'r.l.u.')

    idhkl = np.zeros((len(hlist),3))
    idhkl[:,0] = get_closest(Hrange,hlist)
    idhkl[:,1] = get_closest(Krange,klist)
    idhkl[:,2] = get_closest(Lrange,llist)

    u,indices = np.unique(idhkl, return_inverse = True, axis = 0)
    u = u.astype(int)

    del hlist,klist,llist,idhkl
    countsum = np.bincount(indices,NXfile.entry.transform.counts.nxdata)
    weightsum = np.bincount(indices,NXfile.entry.transform.weights.nxdata)

    # 16GB RAM could allocate up to (2500,2500,2500) array with dtype=np.float32
    # poor performance/bug for fancy indexing in large array in hdf5 is the reason for temparr
    with NXfile.entry.nxfile:
        if 'transform_combine' in NXfile.entry:
            del NXfile.entry['transform_combine']   
        transform_data = NXfield(shape = (len(axis0),len(axis1),len(axis2)),name = 'intensity',dtype = np.float32)
        NXfile.entry.transform_combine = NXdata(transform_data,(axis0,axis1,axis2))
        
        temparr = np.zeros((len(axis0),len(axis1),len(axis2)),dtype = np.float32)
        temparr[u[:,0],u[:,1],u[:,2]] = (countsum/weightsum)
        NXfile.entry.transform_combine.intensity[:,:,:]=temparr
        del temparr
        
        transform_weights = NXfield(shape = (len(axis0),len(axis1),len(axis2)),name = 'weights',dtype = np.uint32)
        NXfile.entry.transform_combine.weights = transform_weights 
        
        temparr = np.zeros((len(axis0),len(axis1),len(axis2)),dtype = np.uint32)
        temparr[u[:,0],u[:,1],u[:,2]] = weightsum
        NXfile.entry.transform_combine.weights[:,:,:]=temparr
        del temparr


def LabRefine_getxyz(file = None, h = 1, k = 1, l = 1):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    filerefine = nxrefine_lab.NXRefine(NXfile)
    return filerefine.get_xyz(h,k,l)


def LabRefine_gethkl(file = None, x = 1, y = 1, z_frame = 1):
    if file is None:
        print('filename error')
    NXfile = nxload(file+'.nxs','rw')
    first_im_num = NXfile.entry.data.frame_number.nxdata[0]
    filerefine = nxrefine_lab.NXRefine(NXfile)
    h0,k0,l0 = filerefine.get_hkl(x,y,z_frame-first_im_num)
    polar,azimuthal = filerefine.calculate_angles(np.array([x]), np.array([y]))
    print('peak index:\tH = %f,\tK = %f,\tL = %f'%(h0,k0,l0))
    print('polar = %f,\tazimuthal = %f'%(polar,azimuthal))


def UBpeaks(refinevars = None, i = 0, j = 0, ring_tolerance = 1):
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars    
    #print('xp')
    #print(*filerefine.xp, sep = '\n')
    #print('\nyp')
    #print(*filerefine.yp, sep = '\n')
    #print('\nzp')
    #print(*filerefine.zp, sep = '\n')

    i = int(i)
    j = int(j)
    for iring in range(-int(ring_tolerance),int(ring_tolerance)+1):
        print('\npossible hkls of peak i_ring = %s' %iring)
        print(*filerefine.unitcell.ringhkls[filerefine.unitcell.ringds[filerefine.rp[i]+iring]], sep = '\n')
    for jring in range(-int(ring_tolerance),int(ring_tolerance)+1):
        print('\npossible hkls of peak j_ring = %s' %jring)
        print(*filerefine.unitcell.ringhkls[filerefine.unitcell.ringds[filerefine.rp[j]+jring]], sep = '\n')


def UBorientcheck(refinevars = None, i = 0, j = 0, i_ring = 0, j_ring = 0):
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars 
    g1 = np.array(filerefine.Gvec(filerefine.xp[i], filerefine.yp[i], filerefine.zp[i]).T)[0]
    g2 = np.array(filerefine.Gvec(filerefine.xp[j], filerefine.yp[j], filerefine.zp[j]).T)[0]
    filerefine.unitcell.orient(filerefine.rp[i]+i_ring, g1, filerefine.rp[j]+j_ring, g2, verbose=1)


###This function is out-of-date
def UBonscore(refinevars = None, i = 0, j = 0, ring_tolerance = 1.0, hkl_tolerance = 0.1):   
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars  
    filerefine.assign_rings()
    i = int(i)
    j = int(j)
    
    filerefine.ring_tolerance = int(ring_tolerance)
    filerefine.orient_lab(i,j)
    h,k,l = filerefine.get_hkls()
    peaktot = len(h)
    if any(np.isnan(h)) or any(np.isnan(k)) or any(np.isnan(l)):
        print('nan')
    for ipeak in range(peaktot):
        hdiff = abs(h[ipeak]-round(h[ipeak]))
        kdiff = abs(k[ipeak]-round(k[ipeak]))
        ldiff = abs(l[ipeak]-round(l[ipeak]))
        if hdiff > hkl_tolerance or kdiff > hkl_tolerance or ldiff > hkl_tolerance:
            print('peak number = %s' %ipeak)
            print('h,k,l = %s\t%s\t%s' % (h[ipeak],k[ipeak],l[ipeak]))
    filerefine.write_parameters()
    with filerefine.entry.nxfile:
        if 'h' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/h']
        if 'k' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/k']
        if 'l' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/l']
        h,k,l = filerefine.get_hkls()
        filerefine.entry['postpeaks/h'] = NXfield(h)
        filerefine.entry['postpeaks/k'] = NXfield(k)
        filerefine.entry['postpeaks/l'] = NXfield(l)

def UBmanual_prepare_loop(peaks, refinevars = None, hkl_tolerance = 0.1,write_peaklist = False):
    if refinevars is None:
        print('refinevars error')

    for peak1,peak2 in list(itertools.combinations(peaks,2)):
        print('(i,j) = (%d,%d)'%(peak1,peak2))
        Ulist = UBmanual_prepare(refinevars = refinevars, i = peak1, j = peak2, 
                                 hkl_tolerance = hkl_tolerance, write_peaklist = write_peaklist)

def UBmanual_prepare(refinevars = None, i = 0, j = 0, hkl_tolerance = 0.1, write_peaklist = False):
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars  
    filerefine.assign_rings()
    i = int(i)
    j = int(j)
    
    Ulist = filerefine.orient_manual(i,j,filerefine.rp[i],filerefine.rp[j])
    Utot = len(Ulist)
    indices = np.searchsorted(np.sort(filerefine.entry.postpeaks.polar_angle.nxdata),filerefine.entry.postpeaks.polar_angle.nxdata)
    if not write_peaklist:
        oldindices = np.where(np.in1d(filerefine.entry.peaks.z, filerefine.entry.postpeaks.z))[0]
    offlimitlist = []
    for iU in range(Utot):
        print('\norientation matrix#%s' %iU)
        filerefine.Umat = Ulist[iU]
        h,k,l = filerefine.get_hkls()
        peaktot = len(h)
        offlimit = 0
        if any(np.isnan(h)) or any(np.isnan(k)) or any(np.isnan(l)):
            print('nan')
            continue
        for ipeak in range(peaktot):
            hdiff = abs(h[ipeak]-round(h[ipeak]))
            kdiff = abs(k[ipeak]-round(k[ipeak]))
            ldiff = abs(l[ipeak]-round(l[ipeak]))
            if hdiff > hkl_tolerance or kdiff > hkl_tolerance or ldiff > hkl_tolerance:
                if not write_peaklist:
                    print('peak#%s\t\tpolar#%s\t\told peak#%s' %(ipeak,indices[ipeak],oldindices[ipeak]))
                else:
                    print('peak#%s\t\tpolar#%s' %(ipeak,indices[ipeak]))
                print('h,k,l = %s\t%s\t%s' % (h[ipeak],k[ipeak],l[ipeak]))
                offlimit += 1
        print('offlimit=%s'%offlimit)
        offlimitlist.append(offlimit)
        print('\n')    
    if np.any(offlimitlist):
        print('minimum offlimit = %s'%np.min(offlimitlist))
    return Ulist


def UBmanual_ring_tolerance_prepare_loop(peaks, refinevars = None, ring_tolerance = 1.0, hkl_tolerance = 0.1, write_peaklist = False):
    if refinevars is None:
        print('refinevars error')
    for peak1,peak2 in list(itertools.combinations(peaks,2)):
        print('(i,j) = (%d,%d)'%(peak1,peak2))
        Ulist = UBmanual_ring_tolerance_prepare(refinevars = refinevars, 
                                                i = peak1, j = peak2, ring_tolerance = ring_tolerance, 
                                                hkl_tolerance = hkl_tolerance, write_peaklist = write_peaklist)    

def UBmanual_ring_tolerance_prepare(refinevars = None, i = 0, j = 0, ring_tolerance = 1.0, hkl_tolerance = 0.1, write_peaklist = False):
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars  
    filerefine.assign_rings()
    i = int(i)
    j = int(j)
    
    indices = np.searchsorted(np.sort(filerefine.entry.postpeaks.polar_angle.nxdata),filerefine.entry.postpeaks.polar_angle.nxdata)
    if not write_peaklist:
        oldindices = np.where(np.in1d(filerefine.entry.peaks.z, filerefine.entry.postpeaks.z))[0]

    filerefine.ring_tolerance = int(ring_tolerance)
    UBlist = filerefine.get_UBmat_lab(i,j,filerefine.ring_tolerance)
    Ulist = []
    for ilist in range(len(UBlist)):
        filerefine.Umat = UBlist[ilist] * filerefine.Bimat
        Ulist.append(filerefine.Umat)
    Utot = len(Ulist)
    offlimitlist = []
    for iU in range(Utot):
        print('\norientation matrix#%s' %iU)
        filerefine.Umat = Ulist[iU]
        h,k,l = filerefine.get_hkls()
        peaktot = len(h)
        offlimit = 0
        if any(np.isnan(h)) or any(np.isnan(k)) or any(np.isnan(l)):
            print('nan')
            print('\n')
            continue
        for ipeak in range(peaktot):
            hdiff = abs(h[ipeak]-round(h[ipeak]))
            kdiff = abs(k[ipeak]-round(k[ipeak]))
            ldiff = abs(l[ipeak]-round(l[ipeak]))
            if hdiff > hkl_tolerance or kdiff > hkl_tolerance or ldiff > hkl_tolerance:
                if not write_peaklist:
                    print('peak#%s\t\tpolar#%s\t\told peak#%s' %(ipeak,indices[ipeak],oldindices[ipeak]))
                else:
                    print('peak#%s\t\tpolar#%s' %(ipeak,indices[ipeak]))
                print('h,k,l = %s\t%s\t%s' % (h[ipeak],k[ipeak],l[ipeak]))
                offlimit += 1
        print('offlimit=%s'%offlimit)
        offlimitlist.append(offlimit)
        print('\n')
    if np.any(offlimitlist):
        print('minimum offlimit = %s'%np.min(offlimitlist))
    return Ulist


def UBmanual(refinevars = None, Ulist = None, iU = 0):
    if refinevars is None:
        print('refinevars error')
    filerefine = refinevars 
    if Ulist is None:
        print('Ulist error')

    filerefine.Umat = Ulist[iU]
    filerefine.write_parameters()
    with filerefine.entry.nxfile:
        if 'h' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/h']
        if 'k' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/k']
        if 'l' in filerefine.entry['postpeaks']:
            del filerefine.entry['postpeaks/l']
        h,k,l = filerefine.get_hkls()
        filerefine.entry['postpeaks/h'] = NXfield(h)
        filerefine.entry['postpeaks/k'] = NXfield(k)
        filerefine.entry['postpeaks/l'] = NXfield(l)

def UBcopy(parent_file = None, file = None):
    if parent_file is None:
        print('parent_filename error')
    if file is None:
        print('filename error')
    NXparent_file = nxload(parent_file +'.nxs','rw')
    NXfile = nxload(file+'.nxs','rw')
    NXfilerefine = nxrefine_lab.NXRefine(NXfile)
    NXfilerefine.Umat = np.matrix(NXparent_file.entry.instrument.detector.orientation_matrix.nxdata)
    NXfilerefine.write_parameters()
