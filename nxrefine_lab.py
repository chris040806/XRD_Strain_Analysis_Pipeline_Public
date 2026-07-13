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
import h5py
import matplotlib.pyplot as plt
import itertools


class NXCreate(object):

    def __init__(self, node=None, calib_path=None, imsize = 3450, imstart = 1, imend = 400, det_bg = 10,
                 frame_time = 60, orientation_matrix = np.eye(3), chi = -90, omega = 0,
                 phi_start = 0, phi_end = 20, phi_step = 0.05, two_theta = 0, sample_label = None,
                 sample_name = None, sample_temperature = 300, unit_cell_group = 'tetragonal',
                 lattice_centring = 'I', unitcell_a = 1.0, unitcell_b = 1.0, unitcell_c = 1.0,
                 unitcell_alpha = 90, unitcell_beta = 90, unitcell_gamma = 90, calibrant = 'Si powder',
                 calibration_date = None, datafile_path = None, data_base_name = None, chunk_size = 10, 
                 null_image = False, imstartdiff = 0, gapstart = [0], gapend = [0]):
        
        if node is not None:
            self.entry = node.entry
        else:
            print('NXCreate node error')
            
        if calib_path is not None:
            self.calib = pyFAI.load(calib_path + '.poni')
            self.calibimg = tiff.imread(calib_path + '.tif')
        else:
            print('NXCreate calib_path error')
            
        if datafile_path is not None:
            self.file_path = datafile_path
        else:
            print('NXCreate datafile_path error')

        if data_base_name is not None:
            self.base_name = data_base_name
        else:
            print('NXCreate data_base_name error')
            
        self.imsize = imsize
        self.imstartdiff = imstartdiff
        self.first_im_num = imstart-imstartdiff
        self.last_im_num = imend
        self.imagetot = self.last_im_num - self.first_im_num + 1
        if null_image:
            if 150 < self.imagetot < 400:
                self.imagetot = 400
        self.det_bg = det_bg
        self.axis0 = NXfield(np.arange(0,self.imagetot),name = 'frame_number') ###convert to the real image number afterwards
        self.axis0_frame = NXfield(np.arange(np.int(imstart-imstartdiff),np.int(self.imagetot+imstart-imstartdiff)),name = 'frame_number')
        self.axis1 = NXfield(np.arange(0,self.imsize),name = 'Albulay')
        self.axis2 = NXfield(np.arange(0,self.imsize),name = 'Albulax')
        self.frame_time = frame_time
        self.orientation_matrix = orientation_matrix
        self.goniometer_chi = chi
        self.goniometer_omega = omega
        self.goniometer_phi_start = phi_start  #defined as th in fourc
        self.goniometer_phi_end = phi_end
        self.goniometer_phi_step = phi_step
        self.goniometer_two_theta = two_theta
        self.sample_label = sample_label
        self.sample_name = sample_name
        self.sample_temperature = sample_temperature
        self.unit_cell_group = unit_cell_group
        self.lattice_centring = lattice_centring
        self.unitcell_a = unitcell_a
        self.unitcell_b = unitcell_b
        self.unitcell_c = unitcell_c
        self.unitcell_alpha = unitcell_alpha
        self.unitcell_beta = unitcell_beta
        self.unitcell_gamma = unitcell_gamma
        self.calibrant = calibrant
        self.calibration_date = calibration_date
        self.chunk_size = chunk_size
        self.null_image = null_image
        self.gapstart = gapstart
        self.gapend = gapend
    
    def write_instrument(self):
        with self.entry.nxfile:
            Detconfig = self.calib.get_config()
            DetFit2D = self.calib.getFit2D()
            Calibimg = NXfield(self.calibimg, name = 'intensity')
            self.entry.instrument = NXinstrument()
            self.entry.instrument.calibration = NXdata(Calibimg,(self.axis1, self.axis2))
            self.entry.instrument.calibration.calibrant = NXfield(self.calibrant)
            self.entry.instrument.calibration.refinement = NXentry()
            self.entry.instrument.calibration.refinement.date = NXfield(self.calibration_date)
            self.entry.instrument.calibration.refinement.parameters = NXcollection()
            self.entry.instrument.calibration.refinement.parameters.Detector = Detconfig['detector']
            self.entry.instrument.calibration.refinement.parameters.Distance = Detconfig['dist']
            self.entry.instrument.calibration.refinement.parameters.PixelSize1 = Detconfig['detector_config']['pixel1']
            self.entry.instrument.calibration.refinement.parameters.PixelSize2 = Detconfig['detector_config']['pixel2']
            self.entry.instrument.calibration.refinement.parameters.Poni1 = Detconfig['poni1']
            self.entry.instrument.calibration.refinement.parameters.Poni2 = Detconfig['poni2']
            self.entry.instrument.calibration.refinement.parameters.Rot1 = Detconfig['rot1']
            self.entry.instrument.calibration.refinement.parameters.Rot2 = Detconfig['rot2']
            self.entry.instrument.calibration.refinement.parameters.Rot3 = Detconfig['rot3']
            self.entry.instrument.calibration.refinement.parameters.Wavelength = Detconfig['wavelength']
            self.entry.instrument.calibration.refinement.program = NXfield('pyFAI')
            
            self.entry.instrument.detector = NXdetector()
            self.entry.instrument.detector.beam_center_x = DetFit2D['centerX']
            self.entry.instrument.detector.beam_center_y = DetFit2D['centerY']
            self.entry.instrument.detector.description = Detconfig['detector']
            self.entry.instrument.detector.distance = NXfield(DetFit2D['directDist'],units = 'mm')
            self.entry.instrument.detector.frame_time = NXfield(self.frame_time,units = 'seconds')
            self.entry.instrument.detector.orientation_matrix = self.orientation_matrix
            #self.entry.instrument.detector.pitch = Detconfig['rot2']
            #self.entry.instrument.detector.roll = Detconfig['rot3']
            #self.entry.instrument.detector.yaw = -Detconfig['rot1']
            self.entry.instrument.detector.pitch = Detconfig['rot2']* 180.0 / np.pi
            self.entry.instrument.detector.roll = Detconfig['rot3']* 180.0 / np.pi
            self.entry.instrument.detector.yaw = -Detconfig['rot1']* 180.0 / np.pi
            self.entry.instrument.detector.pixel_size = NXfield(DetFit2D['pixelX']/1000,units = 'mm')
            self.entry.instrument.detector.shape = NXfield([self.imsize,self.imsize])
            
            self.entry.instrument.goniometer = NXgoniometer()
            self.entry.instrument.goniometer.chi = self.goniometer_chi
            self.entry.instrument.goniometer.goniometer_pitch = 0
            self.entry.instrument.goniometer.omega = self.goniometer_omega
            self.entry.instrument.goniometer.phi = NXfield(self.goniometer_phi_start, start = self.goniometer_phi_start, 
                                                           end = self.goniometer_phi_end, step = self.goniometer_phi_step)
            self.entry.instrument.goniometer.two_theta = self.goniometer_two_theta
            
    def write_sample(self):
        with self.entry.nxfile:
            self.entry.sample = NXsample()
            self.entry.sample.label = self.sample_label
            self.entry.sample.name = self.sample_name
            self.entry.sample.temperature = NXfield(self.sample_temperature,units = 'K')
            self.entry.sample.unit_cell_group = self.unit_cell_group
            self.entry.sample.lattice_centring = self.lattice_centring
            self.entry.sample.unitcell_a = self.unitcell_a
            self.entry.sample.unitcell_b = self.unitcell_b
            self.entry.sample.unitcell_c = self.unitcell_c
            self.entry.sample.unitcell_alpha = self.unitcell_alpha
            self.entry.sample.unitcell_beta = self.unitcell_beta
            self.entry.sample.unitcell_gamma = self.unitcell_gamma
    
    def write_data_prepare(self):
        with self.entry.nxfile:
            #tiffdata = NXfield(shape = (self.imagetot,self.imsize,self.imsize), name = 'intensity', dtype = np.float64)
            tiffdata = NXfield(shape = (self.imagetot,self.imsize,self.imsize), name = 'intensity', dtype = np.uint16, chunks = (self.chunk_size,self.imsize,self.imsize), compression = None, shuffle = False)
            #tiffdata = NXfield(shape = (self.imagetot,self.imsize,self.imsize), name = 'intensity', dtype = np.uint16, chunks = (self.chunk_size,self.imsize,self.imsize), compression = 'gzip', shuffle = True)
            #tiffdata = NXfield(shape = (self.imagetot,self.imsize,self.imsize), name = 'intensity', dtype = np.uint16, chunks = True, compression = None, shuffle = False)
            #self.entry.data = NXdata(tiffdata,(self.axis0,self.axis1,self.axis2))
            self.entry.data = NXdata(tiffdata,(self.axis0_frame,self.axis1,self.axis2))
    
    def write_data(self,compress = False, backup = False):
        if self.gapstart[0]>0:
            gaparr = np.arange(int(self.gapstart[0]),int(self.gapend[0])+1)
            for i in range(1,len(self.gapstart)):
                gaparr = np.concatenate((gaparr,np.arange(int(self.gapstart[i]),int(self.gapend[i])+1)))
        else:
            gaparr = np.array([0])          
        if backup:
            f = h5py.File(self.sample_name + '_' + str(self.sample_temperature) + 'K_backup.hdf5')
        else:
            f = h5py.File(self.sample_name + '_' + str(self.sample_temperature) + 'K.hdf5')
        _nxentry = f.create_group('entry')
        if compress:
            _nxdata = _nxentry.create_dataset('intensity', (self.imagetot,self.imsize,self.imsize), dtype='uint16', chunks = (self.chunk_size,self.imsize,self.imsize), compression = 'gzip', shuffle = True)
        else:
            _nxdata = _nxentry.create_dataset('intensity', (self.imagetot,self.imsize,self.imsize), dtype='uint16', chunks = (self.chunk_size,self.imsize,self.imsize), compression = None, shuffle = False)
        with self.entry.nxfile:
            i = int(self.imstartdiff)
            immax = np.zeros((self.imsize,self.imsize),dtype = np.uint16)
            imnum = np.zeros((self.imsize,self.imsize),dtype = np.uint16)
            im0 = np.zeros((self.imsize,self.imsize),dtype = np.uint16)
            imtemp = np.zeros((self.chunk_size,self.imsize,self.imsize),dtype = np.uint16)
            
            _x = np.arange(0,self.imsize)
            _y = np.arange(0,self.imsize)
            _x0 = self.imsize/2
            _y0 = self.imsize/2
            _r = 80
            _mask = (_x[np.newaxis,:]-_x0)**2 + (_y[:,np.newaxis]-_y0)**2 < _r**2
            if i > 0:
                for _it in range(0,i,self.chunk_size):
                    _nxdata[_it:_it+self.chunk_size,:,:] = np.zeros((self.chunk_size,self.imsize,self.imsize),dtype = np.uint16)
            for it in range(self.first_im_num+self.imstartdiff, self.last_im_num+1):
                if i == int(self.imagetot/4):
                    print('finish 1/4')
                elif i == int(self.imagetot/2):
                    print('finish 1/2')
                elif i == int(self.imagetot*3/4):
                    print('finish 3/4')
                
                if it < 10:
                    file_name = self.base_name + '_00' + str(it) + '.tif'
                elif it < 100:
                    file_name = self.base_name + '_0' + str(it) + '.tif'
                else:
                    file_name = self.base_name + '_' + str(it) + '.tif'

                if it in gaparr:
                    im0 = np.zeros((self.imsize,self.imsize),dtype = np.uint16)
                else:
                    im0 = tiff.imread(self.file_path + file_name)
                #im0 = im0.T    #image from 0~3450-1, match convention in ALBULA
                #im0 = im0.astype(np.float64)
                im0[im0 < self.det_bg] = 0
                #mask detector center (3450,3450) format
                #im0[1725-80:1725+80,1725-80:1725+80] = 0
                im0[_mask] = 0

                #self.entry.data.intensity[i,:,:] = im0
                _i = i%self.chunk_size
                if _i == 0:
                    imtemp = np.zeros((self.chunk_size,self.imsize,self.imsize),dtype = np.uint16)
                imtemp[_i,:,:] = im0
                if _i == self.chunk_size-1:
                    #self.entry.data.intensity[i-(self.chunk_size-1):i+1,:,:] = imtemp
                    _nxdata[i-(self.chunk_size-1):i+1,:,:] = imtemp
                #if i == self.imagetot -1:
                if i == np.int(self.last_im_num - self.first_im_num):
                    #self.entry.data.intensity[i-_i:,:,:] = imtemp[:_i+1,:,:]
                    #_nxdata[i-_i:,:,:] = imtemp[:_i+1,:,:]
                    _nxdata[i-_i:np.int(self.last_im_num - self.first_im_num + 1),:,:] = imtemp[:_i+1,:,:]

                imnum[im0>immax] = it
                immax = np.maximum(immax,im0)
                i += 1
            if self.null_image:
                for i in range(np.int(self.last_im_num - self.first_im_num + 1),self.imagetot,self.chunk_size):
                    if self.imagetot - i < self.chunk_size:
                        _nxdata[i:,:,:] = np.zeros((int(self.imagetot - i),self.imsize,self.imsize),dtype = np.uint16)
                    else:
                        _nxdata[i:i+self.chunk_size,:,:] = np.zeros((self.chunk_size,self.imsize,self.imsize),dtype = np.uint16)
            if backup:
                f.close()
            else:
                self.immax = NXfield(immax, name = 'max_intensity')
                self.imnum = NXfield(imnum, name = 'max_zframe')
                self.entry['max_data'] = NXdata(self.immax,(self.axis1,self.axis2))
                self.entry['max_data/max_zframe'] = self.imnum 
                f.close()
                tiffdata = NXlink('/entry/intensity',file = self.sample_name + '_' + str(self.sample_temperature) + 'K.hdf5')
                self.entry.data = NXdata(tiffdata,(self.axis0_frame,self.axis1,self.axis2))




### threshold = maximum/10
### pixel tolerance & frame tolerance
#class NXReduce(QtCore.QObject):
class NXReduce(object):
    
    def __init__(self, node=None, entry='entry', path='/entry/data/intensity', threshold=None, 
                 first=None, imstart = 1, last=None, imend = 400, null_image = False, imstartdiff = 0,
                 background = None, chunk_size = 1, pixel_tolerance = 50, frame_tolerance = 1,
                 Qh=None, Qk=None, Ql=None, maxcount=False, find = False, refine=False, lattice=False
                 ):
        
        #super(NXReduce, self).__init__()
        
        if node is not None:
            self.entry_name = node.entry.nxname
            self.wrapper_file = node.entry.nxfilename
            self._root = node.entry.nxroot
        else:
            print('Error loading file')
        self.node = node
        self.path = path
        self._data = node[path]
        if null_image:
            self._shape = (int(imend-(imstart-imstartdiff)+1),node[path].shape[1],node[path].shape[2])
        else:
            self._shape = node[path].shape
        self._threshold = threshold
        self._maximum = None
        self.summed_data = None
        self._first = first
        self.imstartdiff = imstartdiff 
        self.first_im_num = imstart-imstartdiff
        self.last_im_num = imend
        self._last = last
        self.background = background
        if self._data.chunks:
            self.chunk_size = self._data.chunks[0]
        else:
            self.chunk_size = chunk_size
        self.pixel_tolerance = pixel_tolerance
        self.frame_tolerance = frame_tolerance
        self.Qh = Qh
        self.Qk = Qk
        self.Ql = Ql
        self.maxcount = maxcount
        self.find = find
        self.refine = refine
        self.lattice = lattice
        self.null_image = null_image
    def __repr__(self):
        return "NXReduce('" + self.wrapper_file + "')" 

    @property
    def data(self):
        return self.node[self.path]

    @property
    def shape(self):
        return self._shape
    
    @property
    def first(self):
        _first = self._first
        if _first is None:
            if 'peaks' in self.node.entry and 'first' in self.node.entry['peaks'].attrs:
                _first = np.int32(self.node.entry['peaks'].attrs['first'])
            elif 'first' in self.node.entry['data'].attrs:
                _first = np.int32(self.node.entry['data'].attrs['first'])
        try:
            self._first = np.int(_first)
        except Exception as error:
            self._first = None
            #print('select start frame for data analysis')
        return self._first

    @first.setter
    def first(self, value):
        try:
            self._first = np.int(value)
        except ValueError:
            pass
        
    @property
    def last(self):
        _last = self._last
        if _last is None:
            if 'peaks' in self.node.entry and 'last' in self.node.entry['peaks'].attrs:
                _last = np.int32(self.node.entry['peaks'].attrs['last'])
            elif 'last' in self.node.entry['data'].attrs:
                _last = np.int32(self.node.entry['data'].attrs['last'])
        try:
            self._last = np.int(_last)
        except Exception as error:
            self._last = None
            #print('select end frame for data analysis')
        return self._last

    @last.setter
    def last(self, value):
        try:
            self._last = np.int(value)
        except ValueError:
            pass
        
    @property
    def threshold(self):
        _threshold = self._threshold
        if _threshold is None:
            if 'peaks' in self.node.entry and 'threshold' in self.node.entry['peaks'].attrs:
                _threshold = np.int32(self.node.entry['peaks'].attrs['threshold'])
        if _threshold is None:
            if self.maximum is not None:
                _threshold = self.maximum / 10
        try:
            self._threshold = np.float(_threshold)
            if self._threshold <= 0.0:
                self._threshold = None
        except:
            self._threshold = None
        #if self._threshold is None:
            #print('select intensity minimum threshold')
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def maximum(self):
        if self._maximum is None:
            if 'maximum' in self.node.entry['data'].attrs:
                self._maximum = self.node.entry['data'].attrs['maximum']
        return self._maximum

    def nxmax(self):
        if self.maxcount:
            maximum = self.find_maximum()
            self.write_maximum(maximum)
    
    def find_maximum(self):
        with self.node.nxfile:
            maximum = 0.0
            chunk_size = self.chunk_size
            nframes = self.shape[0]
            if self.first == None:
                self.first = 0
            if self.last == None:
                self.last = nframes
            data = self.data.nxdata
            fsum = np.zeros(nframes, dtype=np.float64)
            
            for i in range(self.first, self.last, chunk_size):
                try:
                    if i+chunk_size > nframes and self.null_image:
                        v = data[i:nframes,:,:]
                    else:        
                        v = data[i:i+chunk_size,:,:]
                except IndexError as error:
                    pass
                if i == self.first:
                    vsum = v.sum(0)
                else:
                    vsum += v.sum(0)
                fsum[i:i+chunk_size] = v.sum((1,2))
                if maximum < v.max():
                    maximum = v.max()
                del v

        self.summed_data = NXfield(vsum, name='summed_data')
        self.summed_frames = NXfield(fsum, name='summed_frames')

        del data
        return maximum

    def write_maximum(self, maximum):
        with self.node.nxfile:
            self.node.entry['data'].attrs['maximum'] = maximum
            self.node.entry['data'].attrs['first'] = self.first
            self.node.entry['data'].attrs['last'] = self.last
            if 'summed_data' in self.node.entry:
                del self.node.entry['summed_data']
            self.node.entry['summed_data'] = NXdata(self.summed_data,
                                               self.node.entry['data'].nxaxes[-2:])
            if 'summed_frames' in self.node.entry:
                del self.node.entry['summed_frames']
            self.node.entry['summed_frames'] = NXdata(self.summed_frames,
                                                 self.node.entry['data'].nxaxes[0])
            try:
                from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
                parameters = self.node.entry['instrument/calibration/refinement/parameters']
                cake = AzimuthalIntegrator(dist=parameters['Distance'].nxvalue,
                                           poni1=parameters['Poni1'].nxvalue,
                                           poni2=parameters['Poni2'].nxvalue,
                                           rot1=parameters['Rot1'].nxvalue,
                                           rot2=parameters['Rot2'].nxvalue,
                                           rot3=parameters['Rot3'].nxvalue,
                                           pixel1=parameters['PixelSize1'].nxvalue,
                                           pixel2=parameters['PixelSize2'].nxvalue,
                                           wavelength = parameters['Wavelength'].nxvalue)
                counts = self.node.entry['summed_data/summed_data'].nxvalue
                polar_angle, intensity = cake.integrate1d(counts, 2048,
                                                          unit='2th_deg',
                                                          correctSolidAngle=True)
                if 'radial_sum' in self.node.entry:
                    del self.node.entry['radial_sum']
                self.node.entry['radial_sum'] = NXdata(NXfield(intensity, name='radial_sum'),
                                                  NXfield(polar_angle, name='polar_angle'))
                
                maxcounts = self.node.entry['max_data/max_intensity'].nxvalue
                polar_angle, intensity = cake.integrate1d(maxcounts, 2048,
                                                          unit='2th_deg',
                                                          correctSolidAngle=True)
                if 'radial_maxsum' in self.node.entry:
                    del self.node.entry['radial_maxsum']
                self.node.entry['radial_maxsum'] = NXdata(NXfield(intensity, name='radial_maxsum'),
                                                  NXfield(polar_angle, name='polar_angle'))

                
            except Exception as error:
                print('Unable to create radial sum')

    def nxfind(self):
        if self.find:
            peaks = self.find_peaks()
            self.write_peaks(peaks)
            return peaks
                
    def find_peaks(self):
        with self.node.nxfile:
            self._threshold, self._maximum = self.threshold, self.maximum

        if self.threshold is None:
            if self.maximum is None:
                self.maxcount = True
                self.nxmax()
            self.threshold = self.maximum / 10

        with self.node.nxfile:
            if self.first == None:
                self.first = 0
            if self.last == None:
                self.last = self.shape[0]
            z_min, z_max = self.first, self.last

            lio = labelimage.labelimage(self.shape[-2:], flipper = labelimage.flip1)
            allpeaks = []
            if len(self.shape) == 2:
                res = None
            else:
                chunk_size = self.chunk_size
                pixel_tolerance = self.pixel_tolerance
                frame_tolerance = self.frame_tolerance
                nframes = z_max-z_min
                data = self.data.nxdata
                for i in range(z_min, z_max, chunk_size):
                    try:
                        if i + chunk_size > z_min and i < z_max:
                            v = data[i:i+chunk_size,:,:]
                            for j in range(chunk_size):
                                if i+j >= z_min and i+j <= z_max:
                                    omega = np.float32(i+j)
                                    lio.peaksearch(v[j], self.threshold, omega)
                                    if lio.res is not None:
                                        cImageD11.blob_moments(lio.res)
                                        for k in range(lio.res.shape[0]):
                                            res = lio.res[k]
                                            peak = NXBlob(res[0], res[22],
                                                res[23], res[24], omega,
                                                res[27], res[26], res[29],
                                                self.threshold,
                                                pixel_tolerance,
                                                frame_tolerance)
                                            allpeaks.append(peak)
                    except IndexError as error:
                        pass
                del data

        if not allpeaks:
            print('No peaks found')
            return None

        allpeaks = sorted(allpeaks)

        merged_peaks = []
        for z in range(z_min, z_max+1):
            frame = [peak for peak in allpeaks if peak.z == z]
            if not merged_peaks:
                merged_peaks.extend(frame)
            else:
                for peak1 in frame:
                    combined = False
                    for peak2 in last_frame:
                        if peak1 == peak2:
                            for idx in range(len(merged_peaks)):
                                if peak1 == merged_peaks[idx]:
                                    break
                            peak1.combine(merged_peaks[idx])
                            merged_peaks[idx] = peak1
                            combined = True
                            break
                    if not combined:
                        reversed_peaks = [p for p in reversed(merged_peaks)
                                          if p.z >= peak1.z - frame_tolerance]
                        for peak2 in reversed_peaks:
                            if peak1 == peak2:
                                for idx in range(len(merged_peaks)):
                                    if peak1 == merged_peaks[idx]:
                                        break
                                peak1.combine(merged_peaks[idx])
                                merged_peaks[idx] = peak1
                                combined = True
                                break
                        if not combined:
                            merged_peaks.append(peak1)

            if frame:
                last_frame = frame

        merged_peaks = sorted(merged_peaks)
        for peak in merged_peaks:
            peak.merge()

        merged_peaks = sorted(merged_peaks)
        peaks = merged_peaks
        return peaks
    
    def write_peaks(self, peaks):
        group = NXreflections()
        shape = (len(peaks),)
        group['npixels'] = NXfield([peak.np for peak in peaks], dtype=np.float32)
        group['intensity'] = NXfield([peak.intensity for peak in peaks],
                                        dtype=np.float32)
        group['x'] = NXfield([peak.x for peak in peaks], dtype=np.float32)
        group['y'] = NXfield([peak.y for peak in peaks], dtype=np.float32)
        group['z'] = NXfield([peak.z for peak in peaks], dtype=np.float32)
        group['z_frame'] = NXfield([peak.z+self.first_im_num for peak in peaks], dtype=np.float32)
        group['sigx'] = NXfield([peak.sigx for peak in peaks], dtype=np.float32)
        group['sigy'] = NXfield([peak.sigy for peak in peaks], dtype=np.float32)
        group['covxy'] = NXfield([peak.covxy for peak in peaks], dtype=np.float32)
        group.attrs['first'] = self.first
        group.attrs['last'] = self.last
        group.attrs['threshold'] = self.threshold
        with self.node.nxfile:
            if 'peaks' in self.node.entry:
                del self.node.entry['peaks']
            self.node.entry['peaks'] = group
            refine = NXRefine(self.node)
            polar_angles, azimuthal_angles = refine.calculate_angles(refine._xp,
                                                                     refine._yp)
            refine.write_angles(polar_angles, azimuthal_angles)


    def write_postpeaks(self, peaks, bgpeaks = []):
        outlist = list(range(len(peaks)))
        for ele in sorted(bgpeaks,reverse = True):
            del outlist[ele]
        postpeaks = list(peaks[i] for i in outlist)
        group = NXreflections()
        shape = (len(postpeaks),)
        group['npixels'] = NXfield([peak.np for peak in postpeaks], dtype=np.float32)
        group['intensity'] = NXfield([peak.intensity for peak in postpeaks],
                                        dtype=np.float32)
        group['x'] = NXfield([peak.x for peak in postpeaks], dtype=np.float32)
        group['y'] = NXfield([peak.y for peak in postpeaks], dtype=np.float32)
        group['z'] = NXfield([peak.z for peak in postpeaks], dtype=np.float32)
        group['z_frame'] = NXfield([peak.z+self.first_im_num for peak in postpeaks], dtype=np.float32)
        group['sigx'] = NXfield([peak.sigx for peak in postpeaks], dtype=np.float32)
        group['sigy'] = NXfield([peak.sigy for peak in postpeaks], dtype=np.float32)
        group['covxy'] = NXfield([peak.covxy for peak in postpeaks], dtype=np.float32)
        group.attrs['first'] = self.first
        group.attrs['last'] = self.last
        group.attrs['threshold'] = self.threshold
        with self.node.nxfile:
            if 'postpeaks' in self.node.entry:
                del self.node.entry['postpeaks']
            self.node.entry['postpeaks'] = group
            refine = NXRefine(self.node)
            polar_angles, azimuthal_angles = refine.calculate_angles(refine.xp,
                                                                     refine.yp)
            refine.write_postangles(polar_angles, azimuthal_angles)


    def write_peaklist(self,peaklist = []):
        if not np.any(peaklist):
            print('empty list')
        peaklist = np.unique(peaklist,axis = 0)
        idz = np.where(np.logical_and(peaklist[:,0] >= self.first_im_num + self.imstartdiff, peaklist[:,0] <= self.last_im_num ))[0]
        group = NXreflections() 
        group['intensity'] = NXfield(peaklist[idz,3],dtype = np.float32)
        group['x'] = NXfield(peaklist[idz,2],dtype = np.float32)
        group['y'] = NXfield(peaklist[idz,1],dtype = np.float32)
        group['z'] = NXfield(peaklist[idz,0]-self.first_im_num,dtype = np.float32)
        group['z_frame'] = NXfield(peaklist[idz,0],dtype = np.float32)
        group.attrs['first'] = self.first
        group.attrs['last'] = self.last
        group.attrs['threshold'] = np.min(peaklist[idz,3])
        with self.node.nxfile:
            if 'postpeaks' in self.node.entry:
                del self.node.entry['postpeaks']
            self.node.entry['postpeaks'] = group
            refine = NXRefine(self.node)
            polar_angles, azimuthal_angles = refine.calculate_angles(refine.xp,
                                                                     refine.yp)
            refine.write_postangles(polar_angles, azimuthal_angles)


    def nxrefine(self, posthkl_tolerance = 0.05):
        if self.refine:
            if self.lattice:
                lattice = True
            else:
                lattice = False
            result = self.refine_parameters(lattice=lattice, posthkl_tolerance = posthkl_tolerance)
            if result:
                self.write_refinement(result)
            else:
                print('nxrefine_fail')

    def refine_parameters(self, lattice=False,posthkl_tolerance = 0.05):
        with self.node.nxfile:
            refine = NXRefine(self.node)
            refine.posthkl_tolerance = posthkl_tolerance
            refine.refine_hkls(lattice=lattice, chi=True, omega=True)
            fit_report=refine.fit_report
            refine.refine_hkls(chi=True, omega=True)
            fit_report = fit_report + '\n' + refine.fit_report
            refine.refine_orientation_matrix()
            fit_report = fit_report + '\n' + refine.fit_report
            if refine.result.success:
                refine.fit_report = fit_report
                return refine
            else:
                print('HKL refinement not successful')
                return None

    def write_refinement(self, refine):
        with self.node.nxfile:
            refine.write_parameters()

    
class NXBlob(object):

    def __init__(self, np, average, x, y, z, sigx, sigy, covxy, threshold,
                 pixel_tolerance, frame_tolerance):
        self.np = np
        self.average = average
        self.intensity = np * average
        self.x = x
        self.y = y
        self.z = z
        self.sigx = sigx
        self.sigy = sigy
        self.covxy = covxy
        self.threshold = threshold
        self.peaks = [self]
        self.pixel_tolerance = pixel_tolerance**2
        self.frame_tolerance = frame_tolerance
        self.combined = False

    def __str__(self):
        return "NXBlob x=%f y=%f z=%f np=%i avg=%f" % (self.x, self.y, self.z, self.np, self.average)

    def __repr__(self):
        return "NXBlob x=%f y=%f z=%f np=%i avg=%f" % (self.x, self.y, self.z, self.np, self.average)

    def __lt__(self, other):
        return self.z < other.z

    def __eq__(self, other):
        if abs(self.z - other.z) <= self.frame_tolerance:
            if (self.x - other.x)**2 + (self.y - other.y)**2 <= self.pixel_tolerance:
                return True
            else:
                return False
        else:
            return False

    def __ne__(self, other):
        if abs(self.z - other.z) > self.frame_tolerance:
            if (self.x - other.x)**2 + (self.y - other.y)**2 > self.pixel_tolerance:
                return True
            else:
                return False
        else:
            return False

    def combine(self, other):
        self.peaks.extend(other.peaks)
        self.combined = True
        other.combined = False

    def merge(self):
        np = sum([p.np for p in self.peaks])
        intensity = sum([p.intensity for p in self.peaks])
        self.x = sum([p.x * p.intensity for p in self.peaks]) / intensity
        self.y = sum([p.y * p.intensity for p in self.peaks]) /intensity
        self.z = sum([p.z * p.intensity for p in self.peaks]) / intensity
        self.sigx = sum([p.sigx * p.intensity for p in self.peaks]) / intensity
        self.sigy = sum([p.sigy * p.intensity for p in self.peaks]) / intensity
        self.covxy = sum([p.covxy * p.intensity for p in self.peaks]) / intensity
        self.np = np
        self.intensity = intensity
        self.average = self.intensity / self.np







#polar_max
#initialize_grid_step
#peak_tolerance   (angle_tolerance between peaks)
#ring_tolerance   (integer tolerance of sets of rings)
#Uiterate  #(not used)
#hkl_tolerance/posthkl_tolerance
#angle_threshold #(not used) (angles between two peaks for orientation)

degrees = 180.0 / np.pi
radians = np.pi / 180.0

def find_nearest(array, value):
    """Return array value closest to the requested value."""
    idx = (np.abs(array-value)).argmin()
    return array[idx]
 

def rotmat(axis, angle):
    """Return a rotation matrix for rotation about the specified axis."""
    mat = np.eye(3) 
    if angle is None or np.isclose(angle, 0.0):
        return mat
    cang = np.cos(angle*radians)
    sang = np.sin(angle*radians)
    if axis == 1:
        mat = np.array(((1,0,0), (0,cang,-sang), (0, sang, cang)))
    elif axis == 2:
        mat = np.array(((cang,0,sang), (0,1,0), (-sang,0,cang)))
    else:
        mat = np.array(((cang,-sang,0), (sang,cang,0), (0,0,1)))
    return np.matrix(mat)


def vec(x, y=0.0, z=0.0):
    return np.matrix((x, y, z)).T


def norm_vec(vec):
    return vec / norm(vec)


class NXRefine(object):

    symmetries = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 
                  'monoclinic', 'triclinic']
    centrings = ['P', 'A', 'B', 'C', 'I', 'F', 'R']

    def __init__(self, node=None, peak_tolerance = 5.0, ring_tolerance = 1, hkl_tolerance = 0.05, posthkl_tolerance = 0.05):
        if node is not None:
            self.entry = node.entry
            if 'data' in self.entry:
                 self.data = self.entry['data/intensity']
        else:
            print('NXRefine node error')

        self.a = 4.0
        self.b = 4.0
        self.c = 4.0
        self.alpha = 90.0
        self.beta = 90.0
        self.gamma = 90.0
        self.wavelength = 7.092998626781036e-11 * 1e10
        self.distance = 100.0
        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self.twotheta = 0.0
        self._gonpitch = 0.0
        self._omega = 0.0
        self._chi = 0.0
        self.phi = 0.0
        self.phi_step = 0.05
        self.xc = 256.0
        self.yc = 256.0
        self.frame_time = 0.05
        self.symmetry = 'cubic'
        self.centring = 'P'
        self.peak = None
        self._xp = None
        self._yp = None
        self._zp = None
        self.xp = None
        self.yp = None
        self.zp = None
        self.x = None
        self.y = None
        #self.z = None
        self.polar_angle = None
        self.azimuthal_angle = None
        #self.rotation_angle = None
        self.intensity = None
        self.pixel_size = 0.1
        self.shape = [3450, 3450]
        self.polar_max = None
        self.Umat = None
        self.primary = None
        self.secondary = None
        self._unitcell = None
        #self.polar_tolerance = 0.1
        #self.peak_tolerance = 5.0
        self.peak_tolerance = peak_tolerance
        #self.ring_tolerance = int(1)
        self.ring_tolerance = int(ring_tolerance)
        self.Uiterate = int(10.0)
        #self.hkl_tolerance = 0.05
        self.hkl_tolerance = hkl_tolerance
        #self.posthkl_tolerance = 0.02
        self.posthkl_tolerance = posthkl_tolerance
        self.angle_threshold = 20.0
        self.grid_origin = None
        self.grid_basis = None
        self.grid_shape = None
        self.grid_step = None
        self.standard = True
        self._name = ""
        self._idx = None
        self._Dmat_cache = inv(rotmat(1, self.roll) * rotmat(2, self.pitch) *
                               rotmat(3, self.yaw))
        self._Gmat_cache = (rotmat(2,self.gonpitch) * rotmat(3, self.omega) * 
                            rotmat(1, self.chi))
        self.parameters = None
        self.grains = None
        self.grains_lab = None
        if self.entry is not None:
            self.read_parameters()
        if self.data.chunks:
            self.chunk_size = self.data.chunks[0]
        else:
            self.chunk_size = 1


    def __repr__(self):
        return "NXRefine('" + self._name + "')"
    
    def read_parameter(self, path, default=None, attr=None):
        try:
            if attr:
                return self.entry[path].attrs[attr]
            else:
                return self.entry[path].nxdata
        except NeXusError:
            return default

    def read_parameters(self, entry=None):
        if entry:
            self.entry = entry
        with self.entry.nxfile:
            self._name = self.entry.nxroot.nxname + "/" + self.entry.nxname
            self.a = self.read_parameter('sample/unitcell_a', self.a)
            self.b = self.read_parameter('sample/unitcell_b', self.b)
            self.c = self.read_parameter('sample/unitcell_c', self.c)
            self.alpha = self.read_parameter('sample/unitcell_alpha', self.alpha)
            self.beta = self.read_parameter('sample/unitcell_beta', self.beta)
            self.gamma = self.read_parameter('sample/unitcell_gamma', self.gamma)
            self.wavelength = self.read_parameter('instrument/calibration/refinement/parameters/Wavelength', 
                                                  self.wavelength) * 1e10
            self.distance = self.read_parameter('instrument/detector/distance', 
                                                self.distance)
            self.yaw = self.read_parameter('instrument/detector/yaw', self.yaw)
            self.pitch = self.read_parameter('instrument/detector/pitch', 
                                             self.pitch)
            self.roll = self.read_parameter('instrument/detector/roll', self.roll)
            self.xc = self.read_parameter('instrument/detector/beam_center_x', 
                                          self.xc)
            self.yc = self.read_parameter('instrument/detector/beam_center_y', 
                                          self.yc)
            self.frame_time = self.read_parameter('instrument/detector/frame_time', 
                                                  self.frame_time)
            self.shape = self.read_parameter('instrument/detector/shape', self.shape)
            self.phi = self.read_parameter('instrument/goniometer/phi', self.phi)
            try:
                self.phi_step = self.read_parameter('instrument/goniometer/phi', 
                                                    self.phi, attr='step')
            except Exception:
                pass
            self.chi = self.read_parameter('instrument/goniometer/chi', self.chi)
            self.omega = self.read_parameter('instrument/goniometer/omega', 
                                             self.omega)
            self.twotheta = self.read_parameter('instrument/goniometer/two_theta', 
                                                self.twotheta)
            self.gonpitch = self.read_parameter('instrument/goniometer/goniometer_pitch', 
                                                self.gonpitch)
            self.symmetry = self.read_parameter('sample/unit_cell_group', 
                                                self.symmetry)
            self.centring = self.read_parameter('sample/lattice_centring', 
                                                self.centring)
            self._xp = self.read_parameter('peaks/x')
            self._yp = self.read_parameter('peaks/y')
            self._zp = self.read_parameter('peaks/z')
            self.xp = self.read_parameter('postpeaks/x')
            self.yp = self.read_parameter('postpeaks/y')
            self.zp = self.read_parameter('postpeaks/z')           
            self.polar_angle = self.read_parameter('postpeaks/polar_angle')
            self.azimuthal_angle = self.read_parameter('postpeaks/azimuthal_angle')
            self.intensity = self.read_parameter('postpeaks/intensity')
            self.pixel_size = self.read_parameter('instrument/detector/pixel_size', 
                                                  self.pixel_size) 
            #self.rotation_angle = self.read_parameter('postpeaks/rotation_angle')
            self.primary = self.read_parameter('postpeaks/primary_reflection')
            self.secondary = self.read_parameter('postpeaks/secondary_reflection')
            self.Umat = self.read_parameter('instrument/detector/orientation_matrix')
            if isinstance(self.polar_angle, np.ndarray):
                try:
                    self.set_polar_max(np.sort(self.polar_angle)[200] + 0.1)
                except IndexError:
                    self.set_polar_max(self.polar_angle.max())
            else:
                self.set_polar_max(10.0)

    def write_parameter(self, path, value, attr=None):
        if path.startswith('sample'):
            entry = self.entry.nxroot['entry']
        else:
            entry = self.entry
        if value is not None:
            if attr and path in entry:
                entry[path].attrs[attr] = value
            elif path in entry:
                entry[path].replace(value)
            elif attr is None:
                entry[path] = value

    def write_parameters(self, entry=None):
        if entry:
            self.entry = entry
        with self.entry.nxfile:
            if 'sample' not in self.entry:
                self.entry['sample'] = NXsample()
            self.write_parameter('sample/unit_cell_group', self.symmetry)
            self.write_parameter('sample/lattice_centring', self.centring)
            self.write_parameter('sample/unitcell_a', self.a)
            self.write_parameter('sample/unitcell_b', self.b)
            self.write_parameter('sample/unitcell_c', self.c)
            self.write_parameter('sample/unitcell_alpha', self.alpha)
            self.write_parameter('sample/unitcell_beta', self.beta)
            self.write_parameter('sample/unitcell_gamma', self.gamma)
            if 'instrument' not in self.entry:
                self.entry['instrument'] = NXinstrument()
            if 'detector' not in self.entry['instrument']:
                self.entry['instrument/detector'] = NXdetector()
            if 'goniometer' not in self.entry['instrument']:
                self.entry['instrument/goniometer'] = NXgoniometer()
            self.write_parameter('instrument/detector/distance', self.distance) 
            self.write_parameter('instrument/detector/yaw', self.yaw)
            self.write_parameter('instrument/detector/pitch', self.pitch)
            self.write_parameter('instrument/detector/roll', self.roll)
            self.write_parameter('instrument/detector/beam_center_x', self.xc)
            self.write_parameter('instrument/detector/beam_center_y', self.yc)
            self.write_parameter('instrument/detector/pixel_size', self.pixel_size) 
            self.write_parameter('instrument/detector/frame_time', self.frame_time) 
            if self.Umat is not None:
                self.write_parameter('instrument/detector/orientation_matrix', 
                                     np.array(self.Umat))
            self.write_parameter('instrument/goniometer/phi', self.phi) 
            self.write_parameter('instrument/goniometer/phi', self.phi_step, 
                                 attr='step')
            self.write_parameter('instrument/goniometer/chi', self.chi)
            self.write_parameter('instrument/goniometer/omega', self.omega)
            self.write_parameter('instrument/goniometer/two_theta', self.twotheta)
            self.write_parameter('instrument/goniometer/goniometer_pitch', 
                                 self.gonpitch)
            self.write_parameter('postpeaks/primary_reflection', self.primary)
            self.write_parameter('postpeaks/secondary_reflection', self.secondary)        
#             if isinstance(self.z, np.ndarray):
#                 self.rotation_angle = self.phi + (self.phi_step * self.z)

    def write_angles(self, polar_angles, azimuthal_angles):
        with self.entry.nxfile:
            if 'sample' not in self.entry:
                self.entry['sample'] = NXsample()
            if 'peaks' not in self.entry:
                self.entry['peaks'] = NXdata()            
            else:
                if 'polar_angle' in self.entry['peaks']:
                    del self.entry['peaks/polar_angle']
                if 'azimuthal_angle' in self.entry['peaks']:
                    del self.entry['peaks/azimuthal_angle']
            self.write_parameter('peaks/polar_angle', polar_angles)
            self.write_parameter('peaks/azimuthal_angle', azimuthal_angles)
            
    def write_postangles(self, polar_angles, azimuthal_angles):
        with self.entry.nxfile:
            if 'sample' not in self.entry:
                self.entry['sample'] = NXsample()
            if 'postpeaks' not in self.entry:
                self.entry['postpeaks'] = NXdata()            
            else:
                if 'polar_angle' in self.entry['postpeaks']:
                    del self.entry['postpeaks/polar_angle']
                if 'azimuthal_angle' in self.entry['postpeaks']:
                    del self.entry['postpeaks/azimuthal_angle']
            self.write_parameter('postpeaks/polar_angle', polar_angles)
            self.write_parameter('postpeaks/azimuthal_angle', azimuthal_angles)

    def initialize_peaks(self):
        peaks=list(zip(self.xp,  self.yp, self.zp, self.intensity))
        self.peak = dict(zip(range(len(peaks)),[NXPeak(*args) for args in peaks]))

    def initialize_grid(self):
        polar_max = self.polar_max
        try:
            self.set_polar_max(self.polar_angle.max())
        except:
            pass
        self.h_stop = np.round(self.ds_max * self.a)
        h_range = np.round(2*self.h_stop)
        self.h_start = -self.h_stop
        self.h_step = np.round(h_range/1000, 2)
        self.k_stop = np.round(self.ds_max * self.b)
        k_range = np.round(2*self.k_stop)
        self.k_start = -self.k_stop
        self.k_step = np.round(k_range/1000, 2)
        self.l_stop = np.round(self.ds_max * self.c)
        l_range = np.round(2*self.l_stop)
        self.l_start = -self.l_stop
        self.l_step = np.round(l_range/1000, 2)
        self.polar_max = polar_max
        self.define_grid()
       
    def define_grid(self):
        self.h_shape = np.int32(np.round((self.h_stop - self.h_start) / 
                                          self.h_step, 2)) + 1
        self.k_shape = np.int32(np.round((self.k_stop - self.k_start) / 
                                          self.k_step, 2)) + 1
        self.l_shape = np.int32(np.round((self.l_stop - self.l_start) / 
                                          self.l_step, 2)) + 1
        self.grid_origin = [self.h_start, self.k_start, self.l_start]
        self.grid_step = [np.int32(np.rint(1.0/self.h_step)),    
                          np.int32(np.rint(1.0/self.k_step)),
                          np.int32(np.rint(1.0/self.l_step))]
        self.grid_shape = [self.h_shape, self.k_shape, self.l_shape]
        self.grid_basis = [[1,0,0],[0,1,0],[0,0,1]]

    def set_symmetry(self):
        if self.symmetry == 'cubic':
            self.c = self.b = self.a
            self.alpha = self.beta = self.gamma = 90.0 
        elif self.symmetry == 'tetragonal':
            self.b = self.a       
            self.alpha = self.beta = self.gamma = 90.0
        elif self.symmetry == 'orthorhombic':
            self.alpha = self.beta = self.gamma = 90.0 
        elif self.symmetry == 'hexagonal':
            self.b = self.a
            self.alpha = self.beta = 90.0 
            self.gamma = 120.0
        elif self.symmetry == 'monoclinic':
            self.alpha = self.gamma = 90.0

            
    @property
    def lattice_parameters(self):
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    @property
    def lattice_settings(self):
        return (self.a, self.b, self.c, 
                self.alpha*radians, self.beta*radians, self.gamma*radians)

    @property
    def tilts(self):
        return self.yaw, self.pitch, self.roll

    @property
    def centers(self):
        return self.xc, self.yc

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, value):
        self._roll = value
        try:
            self._Dmat_cache = inv(rotmat(1, self.roll) * rotmat(2, self.pitch) *
                                   rotmat(3, self.yaw))

        except:
            pass
        
    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value
        try:
            self._Dmat_cache = inv(rotmat(1, self.roll) * rotmat(2, self.pitch) *
                                   rotmat(3, self.yaw))

        except:
            pass
        
    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        try:
            self._Dmat_cache = inv(rotmat(1, self.roll) * rotmat(2, self.pitch) *
                                   rotmat(3, self.yaw))

        except:
            pass
        
    @property
    def chi(self):
        return self._chi

    @chi.setter
    def chi(self, value):
        self._chi = value
        try:
            self._Gmat_cache = (rotmat(2,self.gonpitch) * rotmat(3, self.omega) * 
                                rotmat(1, self.chi))
        except:
            pass
        
    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value
        try:
            self._Gmat_cache = (rotmat(2,self.gonpitch) * rotmat(3, self.omega) * 
                                rotmat(1, self.chi))
        except:
            pass

    @property
    def gonpitch(self):
        return self._gonpitch

    @gonpitch.setter
    def gonpitch(self, value):
        self._gonpitch = value
        try:
            self._Gmat_cache = (rotmat(2,self.gonpitch) * rotmat(3, self.omega) * 
                                rotmat(1, self.chi))
        except:
            pass

    @property
    def phi_start(self):
        return self.phi

    @property
    def ds_max(self):
        return 2 * np.sin(self.polar_max*radians/2) / self.wavelength
   
    @property
    def unitcell(self):
        if self._unitcell is None:
            self._unitcell = unitcell.unitcell(self.lattice_parameters, self.centring)
        self._unitcell.makerings(self.ds_max)
        return self._unitcell
    
    def absent(self, h, k, l):
        return unitcell.outif[self.centring](h, k, l)

    @property
    def npks(self):
        try:
            return self.xp.size
        except Exception:
            return 0

    @property
    def rings(self):
        return 2 * np.arcsin(np.array(self.unitcell.ringds) * 
                             self.wavelength/2) * degrees

    @property
    def UBmat(self):
        """Determine the U matrix using the defined UB matrix and B matrix
        calculated from the lattice parameters
        """
        if self.Umat is not None:
            return self.Umat * self.Bmat
        else:
            return np.matrix(np.eye(3))

    @property
    def Bimat(self):
        """Create a B matrix containing the column basis vectors of the direct 
        unit cell.
        """
        a, b, c, alpha, beta, gamma = self.lattice_parameters
        alpha = alpha * radians
        beta = beta * radians
        gamma = gamma * radians
        B23 = c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
        B33 = np.sqrt(c**2-(c*np.cos(beta))**2-B23**2)
        return np.matrix(((a, b*np.cos(gamma), c*np.cos(beta)),
                         (0, b*np.sin(gamma),  B23),
                         (0, 0, B33)))

    @property
    def Bmat(self):
        """Create a B matrix containing the column basis vectors of the direct 
        unit cell.
        """
        return inv(self.Bimat)

    @property
    def Omat(self):
        """Define the transform that rotates detector axes into lab axes.
        The inverse transforms detector coords into lab coords.
        When all angles are zero,
            +X(det) = -y(lab), +Y(det) = +z(lab), and +Z(det) = -x(lab)
        """
        if self.standard:
            #return np.matrix(((0,-1,0), (0,0,1), (-1,0,0)))
            return np.matrix(((0,-1,0), (0,0,-1), (1,0,0)))
        else:
            return np.matrix(((0,0,1), (0,1,0), (-1,0,0)))

    @property
    def Dmat(self):
        """Define the matrix, whose inverse physically orients the detector.
        It also transforms detector coords into lab coords.
        Operation order:    yaw -> pitch -> roll
        """
        return self._Dmat_cache

    def Gmat(self, phi):
        """Define the matrix that physically orients the goniometer head.
    
        It performs the inverse transform of lab coords into head coords.
        """
        return self._Gmat_cache * rotmat(3, phi)

    @property
    def Cvec(self):
        return vec(self.xc, self.yc)

    @property
    def Dvec(self):
        """Define the vector from the detector center to the sample position.
        
        Svec is the vector from the goniometer center to the sample position, 
        i.e., t_gs. From this is subtracted the vector from the goniometer 
        center to the detector center, i.e., t_gd
        """
        return vec(-self.distance)

    @property
    def Evec(self):
        return vec(1.0 / self.wavelength)

    def Gvec(self, x, y, z):
        phi = self.phi + self.phi_step * z
        v1 = vec(x, y)
        v2 = self.pixel_size * inv(self.Omat) * (v1 - self.Cvec)
        v3 = inv(self.Dmat) * v2 - self.Dvec
        return (inv(self.Gmat(phi)) * 
                ((norm_vec(v3) / self.wavelength) - self.Evec))

    def get_Gvecs(self, idx):
        self.Gvecs = [self.Gvec(x,y,z) for x,y,z 
                      in zip(self.xp[idx], self.yp[idx], self.zp[idx])]
        return self.Gvecs
    
    def set_polar_max(self, polar_max):
        try:
            if not isinstance(self.polar_angle, np.ndarray):
                self.polar_angle, self.azimuthal_angle = self.calculate_angles(self.xp, self.yp)
            self.x = []
            self.y = []
            for i in range(self.npks):
                if self.polar_angle[i] <= polar_max:
                    self.x.append(self.xp[i])
                    self.y.append(self.yp[i])
        except Exception:
            pass
        self.polar_max = polar_max
        self._idx = None

    def calculate_angles(self, x, y):
        """Calculate the polar and azimuthal angles of the specified pixels"""
        Oimat = inv(self.Omat)
        Mat = self.pixel_size * inv(self.Dmat) * Oimat
        polar_angles = []
        azimuthal_angles = []
        for i in range(len(x)):
            peak = Oimat * (vec(x[i], y[i]) - self.Cvec)
            v = norm(Mat * peak)
            polar_angle = np.arctan(v / self.distance)
            polar_angles.append(polar_angle)
            azimuthal_angles.append(np.arctan2(-peak[1,0], peak[2,0]))
        return (np.array(polar_angles) * degrees, 
                np.array(azimuthal_angles) * degrees)
    
    def calculate_rings(self, polar_max=None):
        """Calculate the polar angles of the Bragg peak rings"""
        if polar_max is None:
            polar_max = self.polar_max
        ds_max = 2 * np.sin(polar_max*radians/2) / self.wavelength
        dss = set(sorted([np.around(x[0],3) 
                          for x in self.unitcell.gethkls(ds_max)]))
        peaks = []
        for ds in dss:
            peaks.append(2*np.arcsin(self.wavelength*ds/2)*degrees)
        return sorted(peaks)

    def angle_peaks(self, i, j):
        """Calculate the angle (in degrees) between two peaks"""
        g1 = norm_vec(self.Gvec(self.xp[i], self.yp[i], self.zp[i]))
        g2 = norm_vec(self.Gvec(self.xp[j], self.yp[j], self.zp[j]))
        return np.around(np.arccos(float(g1.T*g2)) * degrees, 3)

    def angle_hkls(self, h1, h2):
        """Calculate the angle (in degrees) between two (hkl) tuples: 
        h1 = (H1,K1,L1), h2 = (H2,K2,L2)
        """
        return self.unitcell.anglehkls(h1, h2)[0]
    
    def angle_rings(self, ring1, ring2):
        """Calculate the set of angles allowed between peaks in two rings, input ring number"""
        return set(np.around(np.arccos(
                                 self.unitcell.getanglehkls(ring1, ring2)[1])
                             * degrees, 3))

    def assign_rings(self):
        """Assign all the peaks to rings (stored in 'rp')"""
        polar_max = self.polar_max
        self.set_polar_max(max(self.polar_angle))
        rings = self.rings
        self.rp = np.zeros((self.npks), np.int16)
        for i in range(self.npks):
            self.rp[i] = (np.abs(self.polar_angle[i] - rings)).argmin()
        self.set_polar_max(polar_max)
        
    def compatible(self, i, j):
        """Determine if the angle between two peaks is contained in the set of
        angles between their respective rings"""
        if i == j:
            return False
        angle = self.angle_peaks(i, j)
        angles = self.angle_rings(self.rp[i], self.rp[j])
        close = [a for a in angles if abs(a-angle) < self.peak_tolerance]
        if close:
            return True
        else:
            return False
        
    def generate_grains(self):
        self.assign_rings()
        grains = []
        peaks = [i for i in range(self.npks) 
                 if self.polar_angle[i] < self.polar_max]
        assigned = set()
        for (i, j) in [(i, j) for i in peaks for j in peaks if j > i]:
            if self.compatible(i,j):
                if i not in assigned and j not in assigned:
                    grains.append([i,j])
                    assigned.add(i)
                    assigned.add(j)
                else:
                    for grain in grains:
                        if i not in grain:
                            bad = [k for k in grain if not self.compatible(i,k)]
                            if not bad:
                                grain.append(i)
                                assigned.add(i)
                        if j not in grain:
                            bad = [k for k in grain if not self.compatible(j,k)]
                            if not bad:
                                grain.append(j)
                                assigned.add(j)
        self.grains = sorted([NXgrain(grain) 
                              for grain in grains if len(grain) > 2])
        for grain in self.grains:
            self.orient(grain)
            grain.peaks = [i for i in range(self.npks) 
                           if self.diff(i) < self.hkl_tolerance]
            grain.score = self.score(grain)
            
    def generate_grains_lab(self, hklselect = True, mode = 1):
        self.assign_rings()
        grains = []
        peaks = [i for i in range(self.npks) 
                 if self.polar_angle[i] < self.polar_max]
        assigned = set()
        for (i, j) in [(i, j) for i in peaks for j in peaks if j > i]:
            if self.compatible(i,j):
                if i not in assigned and j not in assigned:
                    grains.append([i,j])
                    assigned.add(i)
                    assigned.add(j)
                else:
                    for grain in grains:
                        if i not in grain:
                            bad = [k for k in grain if not self.compatible(i,k)]
                            if not bad:
                                grain.append(i)
                                assigned.add(i)
                        if j not in grain:
                            bad = [k for k in grain if not self.compatible(j,k)]
                            if not bad:
                                grain.append(j)
                                assigned.add(j)
        self.grains_lab = sorted([NXgrain(grain) 
                              for grain in grains if len(grain) > 2])
        if not np.any(self.grains_lab):
            self.grains_lab = [NXgrain(peaks)]
            print('no grain found')
        for grain in self.grains_lab:        
            Ulist = []
            Uscore = []
            Uij = []
            #its = np.min([len(grain.peaks)*self.Uiterate,len(grain.peaks)**2])
            #for it in range(its):
            #    i = np.random.randint(len(grain.peaks))
            #    j = np.random.randint(len(grain.peaks))
            #    while abs(i-j) < 1:
            #        j = np.random.randint(len(grain.peaks))
            #    score = self.orient_lab(grain.peaks[i],grain.peaks[j])
            #    if score > 0.0:
            #        Ulist.append(self.Umat)
            #        Uscore.append(score)
            #        Uij.append([grain.peaks[i],grain.peaks[j]])
#                 for (i, j) in [(i,j) for i in self.grains_lab.peaks 
#                        for j in self.grains_lab.peaks if j > i]:
#                     score = self.orient_lab(i,j)
#                     if score > 0.0:
#                         Ulist.append(self.Umat)
#                         Uscore.append(score)
            for (i, j) in [(i,j) for i in grain.peaks for j in grain.peaks if j > i]:
                if mode == 1:
                    weight = True
                    numthresh = False
                    score = self.orient_lab(i,j,weight = weight, numthresh = numthresh)
                if mode == 2:
                    weight = False
                    numthresh = False
                    score = self.orient_lab(i,j,weight = weight, numthresh = numthresh)
                if mode == 3:
                    weight = False
                    numthresh = True
                    score = self.orient_lab(i,j,weight = weight, numthresh = numthresh)
                if score > 0.0:
                    Ulist.append(self.Umat)
                    Uscore.append(score)
                    Uij.append([i,j])
            if Ulist:
                self.Umat = grain.Umat = Ulist[Uscore.index(min(Uscore))]
                grain.primary  = Uij[Uscore.index(min(Uscore))][0]
                grain.secondary = Uij[Uscore.index(min(Uscore))][1]
                if hklselect:
                    grain.peaks = [i for i in range(self.npks) 
                                if self.diff(i) < self.hkl_tolerance]
                grain.score = min(Uscore)
            else:
                self.Umat = grain.Umat = np.eye(3)
                grain.primary = 0
                grain.secondary = 0
                grain.score = 0.0



    def orient(self, grain=None):
        """Determine the UB matrix (optionally for the specified grain)"""
        if grain:
            for (i, j) in [(i,j) for i in grain.peaks 
                           for j in grain.peaks if j > i]:
                angle = self.angle_peaks(i, j)
                if abs(angle) > self.angle_threshold and abs(angle-180.0) > self.angle_threshold:
                    break
            grain.primary, grain.secondary = i, j
            self.Umat = grain.Umat = self.get_UBmat(i, j) * self.Bimat
        elif self.primary is not None and self.secondary is not None:
            self.Umat = (self.get_UBmat(self.primary, self.secondary) 
                         * self.Bimat)

    def orient_lab(self,i,j,weight = True, numthresh = False):
        """Determine the UB matrix (optionally for the specified grain)"""
        UBlist = self.get_UBmat_lab(i, j, self.ring_tolerance)
        score = np.zeros(len(UBlist))
        Ulist = []
        for ilist in range(len(UBlist)):
            self.Umat = UBlist[ilist] * self.Bimat
            Ulist.append(self.Umat)
            score[ilist] = self.score(weight = weight, numthresh = numthresh)
        if np.min(score) > 0.0:
            self.Umat = Ulist[np.argmin(score)]
            return np.min(score)
        else:
            return 0.0
            
    def orient_manual(self, i, j, ring1, ring2):
        """Determine the UB matrix (optionally for the specified grain)"""
        UBlist = self.get_UBmat_manual(i, j, ring1, ring2)
        Ulist = []
        for ilist in range(len(UBlist)):
            self.Umat = UBlist[ilist] * self.Bimat
            Ulist.append(self.Umat)
        return Ulist
            
    def unitarity(self):
        if self.Umat is not None:
            return np.matrix(self.Umat) * np.matrix(self.Umat.T)
        else:
            return None

    def get_UBmat(self, i, j):
        """Determine a UBmatrix using the specified peaks"""
        ring1 = np.abs(self.polar_angle[i] - self.rings).argmin()
        g1 = np.array(self.Gvec(self.xp[i], self.yp[i], self.zp[i]).T)[0]
        ring2 = np.abs(self.polar_angle[j] - self.rings).argmin()
        g2 = np.array(self.Gvec(self.xp[j], self.yp[j], self.zp[j]).T)[0]
        self.unitcell.orient(ring1, g1, ring2, g2, verbose=1)
        return np.matrix(self.unitcell.UB)
    
    def get_UBmat_lab(self, i, j, ring_tolerance):
        """Determine a UBmatrix using the specified peaks"""
        ring1 = np.abs(self.polar_angle[i] - self.rings).argmin()
        g1 = np.array(self.Gvec(self.xp[i], self.yp[i], self.zp[i]).T)[0]
        ring2 = np.abs(self.polar_angle[j] - self.rings).argmin()
        g2 = np.array(self.Gvec(self.xp[j], self.yp[j], self.zp[j]).T)[0]
        UBlist = []
        for iring1 in range(-ring_tolerance,ring_tolerance+1):
            for iring2 in range(-ring_tolerance,ring_tolerance+1):
                limit1 = (ring1+iring1) < 0 or (ring1+iring1)>=len(self.rings)
                limit2 = (ring2+iring2) < 0 or (ring2+iring2)>=len(self.rings)
                if limit1 or limit2:
                    continue
                self.unitcell.orient(ring1+iring1, g1, ring2+iring2, g2, verbose=0)
                UBIlist = self.unitcell.UBIlist
                for ilist in range(len(UBIlist)):
                    UBlist.append(inv(np.matrix(UBIlist[ilist])))
        return UBlist
    
    def get_UBmat_manual(self, i, j, ring1, ring2):
        """Determine a UBmatrix using the specified peaks"""
        g1 = np.array(self.Gvec(self.xp[i], self.yp[i], self.zp[i]).T)[0]
        g2 = np.array(self.Gvec(self.xp[j], self.yp[j], self.zp[j]).T)[0]
        self.unitcell.orient(ring1, g1, ring2, g2, verbose=0)
        UBlist = []
        UBIlist = self.unitcell.UBIlist
        for ilist in range(len(UBIlist)):
            UBlist.append(inv(np.matrix(UBIlist[ilist])))
        return UBlist

    def get_hkl(self, x, y, z):
        """Determine hkl for the specified pixel coordinates"""
        if self.Umat is not None:
            v5 = self.Gvec(x, y, z)
#            v6 = inv(self.Umat) * v5
#            v7 = inv(self.Bmat) * v6
            v7 = inv(self.UBmat) * v5
            return list(np.array(v7.T)[0])
        else:
            return [0.0, 0.0, 0.0]

    def get_hkls(self):
        """Determine the set of hkls for all the peaks as three columns: list() to read"""
        return zip(*[self.hkl(i) for i in range(self.npks)])
        
    @property
    def hkls(self):
        """Determine the set of hkls for all the peaks"""
        return [self.get_hkl(self.xp[i], self.yp[i], self.zp[i]) 
                for i in range(self.npks)]

    def hkl(self, i):
        """Return the calculated (hkl) for the specified peak"""
        return self.get_hkl(self.xp[i], self.yp[i], self.zp[i])

    def get_xyz(self, h, k, l):

        v7 = vec(h, k, l)
        v6 = self.Bmat * v7
        v5 = self.Umat * v6
        
        ewald_condition = lambda phi: (norm(self.Evec)**2 - norm(self.Gmat(phi)*v5 +
                                       self.Evec)**2)

        phis = []
        if h == 0 and k == 0 and l == 0:
            pass
        elif optimize.fsolve(ewald_condition, 45.0, full_output=1)[2] == 1:
            phis = list(np.unique(np.around([optimize.fsolve(ewald_condition, phi) % 360 
                                             for phi in np.arange(30, 390, 15)], 
                                            decimals=4)))

        def get_ij(phi):
            v4 = self.Gmat(phi) * v5
            p = norm_vec(v4 + self.Evec)
            v3 = -(self.Dvec[0,0] / p[0,0]) * p
            v2 = self.Dmat * (v3 + self.Dvec)
            v1 = (self.Omat * v2 / self.pixel_size) + self.Cvec
            return v1[0,0], v1[1,0]

        peaks = []
        for phi in phis:
            x, y = get_ij(phi)
            z = ((phi - self.phi_start) / self.phi_step) % (360/self.phi_step)
            if x > 0 and x < self.shape[1] and y > 0 and y < self.shape[0]:
                peaks.append(NXPeak(x, y, z, H=h, K=k, L=l))

        peaks = [peak for peak in peaks if peak.z > 0 and peak.z < (360/self.phi_step)]

        return peaks
    
    def polar(self, i):
        """Return the polar angle for the specified peak"""
        Oimat = inv(self.Omat)
        Mat = self.pixel_size * inv(self.Dmat) * Oimat
        peak = Oimat * (vec(self.xp[i], self.yp[i]) - self.Cvec)
        v = norm(Mat * peak)
        return np.arctan(v / self.distance)

    def score(self, grain=None, weight=True, numthresh = False):
        self.set_idx()
        if self.idx:
            h,k,l = self.get_hkls()
            if any(np.isnan(h)) or any(np.isnan(k)) or any(np.isnan(l)):
                return 0.0
            else:
                diffs = self.diffs()
                if weight:
                    weights = self.weights
                    return np.sum(weights * diffs) / np.sum(weights)
                else:
                    if not numthresh:
                        return np.sum(diffs)
                    else:
                        _num = 0.01
                        for i in self.idx:
                            if self.diff(i)>self.hkl_tolerance:
                                _num += 1
                        return _num
        else:
            return 0.0

    @property
    def idx(self):
        if self._idx is None:
            self._idx = list(np.where(self.polar_angle < self.polar_max)[0])
        return self._idx
        
    def set_idx(self, hkl_tolerance=None):
        if hkl_tolerance is None:
            hkl_tolerance = self.hkl_tolerance
        _idx = list(np.where(self.polar_angle < self.polar_max)[0])
        self._idx = [i for i in _idx if self.diff(i) < hkl_tolerance]

    @property
    def weights(self):
        return np.array(self.intensity[self.idx])

    def diffs(self):
        """Return the set of reciproal space differences for all the peaks"""
        return np.array([self.diff(i) for i in self.idx])

    def diff(self, i):
        """Determine the reciprocal space difference between the calculated 
        (hkl) and the closest integer (hkl) of the specified peak"""
        h, k, l = self.hkl(i)
        Q = np.matrix((h, k, l)).T
        Q0 = np.matrix((np.rint(h), np.rint(k), np.rint(l))).T
        return norm(self.Bmat * (Q - Q0))
    
    def angle_diffs(self):
        """Return the set of polar angle differences for all the peaks"""
        return np.array([self.angle_diff(i) for i in self.idx])

    def angle_diff(self, i):
        """Determine the polar angle difference between the calculated 
        (hkl) and the closest integer (hkl) of the specified peak"""
        h, k, l = self.hkl(i)
        (h0, k0, l0) = (np.rint(h), np.rint(k), np.rint(l))
        polar0 = 2 * np.arcsin(self.unitcell.ds((h0,k0,l0))*self.wavelength/2)
        return np.abs(self.polar(i) - polar0)  ###def polar

    def xyz(self, i):
        """Return the pixel coordinates of the specified peak"""
        return self.xp[i], self.yp[i], self.zp[i]
   
    def get_peaks(self, polar_max=None):
        """Return tuples containing the peaks and their respective parameters"""
        peaks = np.array([i for i in range(self.npks) 
                          if self.polar_angle[i] < self.polar_max])
        x, y, z = (np.rint(self.xp[peaks]).astype(np.int16), 
                   np.rint(self.yp[peaks]).astype(np.int16), 
                   np.rint(self.zp[peaks]).astype(np.int16))
        polar, azi = self.polar_angle[peaks], self.azimuthal_angle[peaks]
        intensity = self.intensity[peaks]
        if self.Umat is not None:
            h, k, l = self.get_hkls()
            h = np.array(h)[peaks]
            k = np.array(k)[peaks]
            l = np.array(l)[peaks]
            diffs = np.array([self.diff(i) for i in peaks])
        else:
            h = k = l = diffs = np.zeros(peaks.shape, dtype=np.float32)
        return list(zip(peaks, x, y, z, polar, azi, intensity, h, k, l, diffs))
    
    def get_ring_hkls(self):
        polar_max = self.polar_max
        self.set_polar_max(max(self.polar_angle))
        dss = sorted([ringds for ringds in self.unitcell.ringds])
        hkls=[self.unitcell.ringhkls[ds] for ds in dss]
        self.set_polar_max(polar_max)
        return hkls
    
    def define_parameters(self, **opts):
        from lmfit import Parameters
        self.parameters = Parameters()
        if 'lattice' in opts:
            self.define_lattice_parameters()
            del opts['lattice']
        for opt in opts:
            self.parameters.add(opt, getattr(self, opt), vary=opts[opt])
        return self.parameters
    
    def define_lattice_parameters(self, lattice=True):
        if self.symmetry == 'cubic':
            self.parameters.add('a', self.a, vary=lattice)
        elif self.symmetry == 'tetragonal' or self.symmetry == 'hexagonal':
            self.parameters.add('a', self.a, vary=lattice)
            self.parameters.add('c', self.c, vary=lattice)
        elif self.symmetry == 'orthorhombic':
            self.parameters.add('a', self.a, vary=lattice)
            self.parameters.add('b', self.b, vary=lattice)
            self.parameters.add('c', self.c, vary=lattice)
        elif self.symmetry == 'monoclinic':
            self.parameters.add('a', self.a, vary=lattice)
            self.parameters.add('b', self.b, vary=lattice)
            self.parameters.add('c', self.c, vary=lattice)
            self.parameters.add('beta', self.beta, vary=lattice)
        else:
            self.parameters.add('a', self.a, vary=lattice)
            self.parameters.add('b', self.b, vary=lattice)
            self.parameters.add('c', self.c, vary=lattice)
            self.parameters.add('alpha', self.alpha, vary=lattice)
            self.parameters.add('beta', self.beta, vary=lattice)
            self.parameters.add('gamma', self.gamma, vary=lattice)
    
    def get_parameters(self, parameters):
        for p in parameters:
            setattr(self, p, parameters[p].value)
        self.set_symmetry()
    
    def restore_parameters(self):
        for p in self.parameters:
            setattr(self, p, self.parameters[p].init_value)
        self.set_symmetry()

    def refine_hkls(self, method='leastsq', **opts):
        self.set_idx(self.posthkl_tolerance)
        from lmfit import minimize, fit_report
        if self.Umat is None:
            raise NeXusError('No orientation matrix defined')
        p0 = self.define_parameters(**opts)
        self.result = minimize(self.hkl_residuals, p0, method=method)
        self.fit_report = fit_report(self.result)
        if self.result.success:
            self.get_parameters(self.result.params)
    
    def hkl_residuals(self, parameters):
        self.get_parameters(parameters)
        return self.diffs()
    
    def refine_angles(self, method='nelder', **opts):
        self.set_idx()
        from lmfit import minimize, fit_report
        p0 = self.define_parameters(lattice=True, **opts)
        self.result = minimize(self.angle_residuals, p0, method=method)
        self.fit_report = fit_report(self.result)
        if self.result.success:
            self.get_parameters(self.result.params)

    def angle_residuals(self, parameters):
        self.get_parameters(parameters)
        return self.angle_diffs()
        
    def define_orientation_matrix(self):
        from lmfit import Parameters
        p = Parameters()
        for i in range(3):
            for j in range(3):
                p.add('U%d%d' % (i,j), self.Umat[i,j])
        self.init_p = self.Umat
        return p
    
    def get_orientation_matrix(self, p):
        for i in range(3):
            for j in range(3):
                self.Umat[i,j] = p['U%d%d' % (i,j)].value
                
    def refine_orientation_matrix(self, **opts):
        self.set_idx(self.posthkl_tolerance)
        from lmfit import minimize, fit_report
        p0 = self.define_orientation_matrix()
        self.result = minimize(self.orient_residuals, p0, **opts)
        self.fit_report = fit_report(self.result)
        if self.result.success:
            self.get_orientation_matrix(self.result.params)
        
    def restore_orientation_matrix(self):
        self.Umat = self.init_p

    def orient_residuals(self, p):
        self.get_orientation_matrix(p)
        return self.diffs()

    def make_mask(self,chunkx,chunky,chunkz):
        x = np.arange(0, self.shape[0])
        y = np.arange(0, self.shape[1])
        cx = self.shape[0]/2
        cy = self.shape[1]/2
        r = self.shape[0]/2
        mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2

        #self.chunkx = chunkx
        #self.chunky = chunky
        #self.chunkz = chunkz
        chunkxind = int(self.shape[0]/chunkx)
        chunkyind = int(self.shape[1]/chunky)
        chunkmask = np.full((chunkxind,chunkyind),False)
        chunkcount = np.zeros((chunkxind,chunkyind),dtype=np.uint64)
        for idx in range(chunkxind):
            for idy in range(chunkyind):
                if np.any(mask[idx*chunkx:(idx+1)*chunkx,idy*chunky:(idy+1)*chunky]):
                    chunkmask[idx,idy] = True
                    chunkcount[idx,idy] = np.sum(mask[idx*chunkx:(idx+1)*chunkx,idy*chunky:(idy+1)*chunky])
        row = np.where(chunkmask)[0]
        col = np.where(chunkmask)[1]
        u,indices = np.unique(row,return_index = True)

        idxarr = np.arange(row[0],row[-1]+1)
        idystart = np.zeros(len(idxarr),dtype=np.int64)
        idyend = np.zeros(len(idxarr),dtype=np.int64)
        for i in range(len(idxarr)):
            if i < len(idxarr)-1:
                idystart[i] = col[indices[i]]
                idyend[i] = col[indices[i+1]-1]
            else:
                idystart[i] = col[indices[i]]
                idyend[i] = col[-1]
                
        #self.idxarr = idxarr
        #self.idystart = idystart
        #self.idyend = idyend
        #self.chunkcount = chunkcount
        #self.subdatalen = len(row)*int(self.chunk_size/chunkz)
        subdatalen = len(row)*int(self.chunk_size/chunkz)
        last = self.entry.data.last
        #datalen = subdatalen*int(self.data.shape[0]/self.chunk_size)
        #datalen = subdatalen*int(last/self.chunk_size)
        #datalen = np.int(subdatalen*(last//self.chunk_size) + len(row)*(last%self.chunk_size))
        datalen = np.int(subdatalen*np.ceil(last/self.chunk_size))
        group = NXgroup()
        group['chunkx'] = NXfield(chunkx,dtype = np.int64)
        group['chunky'] = NXfield(chunky,dtype = np.int64)
        group['chunkz'] = NXfield(chunkz,dtype = np.int64)
        group['idxarr'] = NXfield(idxarr,dtype = np.int64)
        group['idystart'] = NXfield(idystart,dtype = np.int64)
        group['idyend'] = NXfield(idyend,dtype = np.int64)
        group['chunkcount'] = NXfield(chunkcount,dtype = np.int64)
        group['subdatalen'] = NXfield(subdatalen,dtype = np.int64)
        group['datalen'] = NXfield(datalen,dtype = np.int64)

        with self.entry.nxfile:
            if 'transform_prepare' in self.entry:
                del self.entry['transform_prepare']
            self.entry['transform_prepare'] = group
    
    def transform_localdata(self,xstart,xend,ystart,yend,zstart,zend):
        z0 = self.entry.data.frame_number[0].nxdata
        xstart = int(xstart)
        xend = int(xend)
        ystart = int(ystart)
        yend = int(yend)
        zstart = int(zstart)
        zend = int(zend)
        z0 = int(z0)
        data_t = self.data[(zstart-z0):(zend-z0),ystart:yend,xstart:xend].nxdata #each pixel has same weight, then ignore it
        indices = np.unravel_index(np.arange(len(data_t.flatten())),data_t.shape)
        _x = indices[2] + xstart
        _y = indices[1] + ystart
        _z = indices[0] + zstart-z0
        _h,_k,_l = zip(*[self.get_hkl(_x[i],_y[i],_z[i]) for i in range(len(_x))])
        _h = np.array(_h).astype(np.float32)
        _k = np.array(_k).astype(np.float32)
        _l = np.array(_l).astype(np.float32)
        return data_t.flatten().astype(np.uint16),_h,_k,_l
    
    def combine_transform(self,data_t,_h,_k,_l,deltah,deltak,deltal):
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

        Hrange = np.linspace(np.min(_h),np.max(_h),int((np.max(_h)-np.min(_h))/deltah+1))
        Krange = np.linspace(np.min(_k),np.max(_k),int((np.max(_k)-np.min(_k))/deltak+1))
        Lrange = np.linspace(np.min(_l),np.max(_l),int((np.max(_l)-np.min(_l))/deltal+1))

        idhkl = np.zeros((len(_h),3))
        idhkl[:,0] = get_closest(Hrange,_h)
        idhkl[:,1] = get_closest(Krange,_k)
        idhkl[:,2] = get_closest(Lrange,_l)

        u,indices,weights = np.unique(idhkl, return_inverse = True, return_counts = True, axis = 0)
        u = u.astype(int)
        countsum = np.bincount(indices,data_t)
        transform_data = np.zeros((len(Hrange),len(Krange),len(Lrange)),dtype = np.float32)
        transform_data[u[:,0],u[:,1],u[:,2]] = (countsum/weights)
        return transform_data,Hrange,Krange,Lrange
        
        
class NXPeak(object):

    def __init__(self, x, y, z, intensity=None, pixel_count=None, 
                 H=None, K=None, L=None,  
                 polar_angle=None, azimuthal_angle=None, rotation_angle=None):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
        self.pixel_count = pixel_count
        self.H = H
        self.K = K
        self.L = L
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.rotation_angle = rotation_angle
        self.ring = None
        self.Umat = None

    def __repr__(self):
        return "NXPeak(x=%.2f, y=%.2f, z=%.2f)" % (self.x, self.y, self.z)
    
class NXgrain(object):

    def __init__(self, peaks, Umat=None, primary=None, secondary=None):
        self.peaks = sorted(list(set(sorted(peaks))))
        self.primary = primary
        self.secondary = secondary
        self.Umat = Umat
        self.score = 0

    def __repr__(self):
        return "NXgrain(%s)" % self.peaks

    def __lt__(self, other):
        return len(self.peaks) < len(other.peaks)

    def __contains__(self, peak):
        return peak in self.peaks

    def __len__(self):
        return len(self.peaks)
