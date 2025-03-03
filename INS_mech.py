#!/usr/bin/env python
"""
Property of: NavINST Laboratory
Author: Hany Ragab

Description:
    Validation file of the mechanization class on Kingston 2008 dataset.
    The validation will be run at:
        Frequency : 10 Hz
        Start time : 600 seconds
    Biases :
        fx = -70.88e-4; fy = 40.80e-4; fz = 61.00e-4
        wx = -40.84e-007; wy = 12.3e-007; wz = 2.06e-006

    Note: fy and wy from Novatel should be negative, hence:
        wy = - wy, and fy = -fy
      should always be done before applying them to the mechanization.
"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Initialization Parameters Classes
# -----------------------------------------------------------------------------
class InitINS:
    def __init__(self, latitude, longitude, altitude, roll, pitch, azimuth):
        """
        Input : The initial 3D position components (Latitude, Longitude, Altitude)
                and 3D Attitude Angles (Roll, Pitch, Yaw).
        Output: Initializes local variables for the mechanization.
        """
        # Position components
        self.Init_Lat  = np.deg2rad(latitude)
        self.Init_Long = np.deg2rad(longitude)
        self.Init_Alt  = altitude
        # Attitude angles
        self.Init_Roll    = np.deg2rad(roll)
        self.Init_Pitch   = np.deg2rad(pitch)
        self.Init_Azimuth = np.deg2rad(azimuth)

        # Earth / gravity constants
        self.a  = 6378137.0  # Earth semi-major axis
        self.fa = 1 / 298.257223563
        self.b  = self.a * (1 - self.fa)
        self.e2 = 1 - (self.b ** 2) / (self.a ** 2)
        self.We = 7.292115e-5  # Earth rotation rate [rad/s]

        # Local gravity computation constants
        self.a1, self.a2, self.a3 = 9.7803267714, 0.0052790414, 0.0000232718
        self.a4, self.a5, self.a6 = -0.000003087691089, 0.000000004397731, 0.000000000000721

        # Initialization of previous delta velocity
        self._Delta_Vl_previous = np.zeros(3)

    def Init_Rbl(self):
        """
        Computes the initial rotation matrix (Rb_l) from Body frame to Local-Level frame.
        """
        # For convenience in referencing
        roll    = self.Init_Roll
        pitch   = self.Init_Pitch
        azimuth = self.Init_Azimuth

        # First row
        self.rbl11 = np.cos(roll) * np.cos(azimuth) + np.sin(roll) * np.sin(pitch) * np.sin(azimuth)
        self.rbl12 = np.sin(azimuth) * np.cos(pitch)
        self.rbl13 = np.cos(azimuth)*np.sin(roll) - np.sin(azimuth)*np.sin(pitch)*np.cos(roll)
        # Second row
        self.rbl21 = - np.sin(azimuth)*np.cos(roll) + np.cos(azimuth)*np.sin(pitch)*np.sin(roll)
        self.rbl22 = np.cos(azimuth)*np.cos(pitch)
        self.rbl23 = - np.sin(azimuth)*np.sin(roll) - np.cos(azimuth)*np.sin(pitch)*np.cos(roll)
        # Third row
        self.rbl31 = - np.cos(pitch)*np.sin(roll)
        self.rbl32 = np.sin(pitch)
        self.rbl33 = np.cos(roll) * np.cos(pitch)

        self._Rb_l = np.array([
            [self.rbl11, self.rbl12, self.rbl13],
            [self.rbl21, self.rbl22, self.rbl23],
            [self.rbl31, self.rbl32, self.rbl33]
        ], dtype=float)

        print("Initial Rb_l matrix:\n", self._Rb_l)

    def Init_Quaternion(self):
        """
        Computes the initial quaternion vector from the initial Rb_l matrix.
        """
        # Diagonal elements
        tr = 1 + self.rbl11 + self.rbl22 + self.rbl33
        Fourth_Quaternion = 0.5 * np.sqrt(tr)
        First_Quaternion  = 0.25 * (self.rbl32 - self.rbl23) / Fourth_Quaternion
        Second_Quaternion = 0.25 * (self.rbl13 - self.rbl31) / Fourth_Quaternion
        Third_Quaternion  = 0.25 * (self.rbl21 - self.rbl12) / Fourth_Quaternion

        self._Quaternion = np.array([
            First_Quaternion,
            Second_Quaternion,
            Third_Quaternion,
            Fourth_Quaternion
        ], dtype=float)

        print("Initial Quaternion (raw):", self._Quaternion)
        self._Quaternion /= np.linalg.norm(self._Quaternion)  # Normalize
        print("Initial Quaternion (normalized):", self._Quaternion)

    def Init_Localg(self):
        """
        Calculates local gravity based on initial latitude & altitude.
        """
        Proj_Lat = np.sin(self.Init_Lat)
        self._Local_g = ( self.a1 * (1 + (self.a2*(Proj_Lat**2)) + self.a3*(Proj_Lat**4)) 
                        + ( (self.a4 + self.a5*(Proj_Lat**2)) ) * self.Init_Alt
                        + self.a6 * (self.Init_Alt**2) )
        self._gVector = np.array([0, 0, -self._Local_g])

    def Init_Velocity(self, Initial_ve=0, Initial_vn=0, Initial_vu=0):
        """
        Initializes the velocity vector in local-level frame.
        """
        self._vl = np.array([Initial_ve, Initial_vn, Initial_vu])
        print("Initial velocity vector:", self._vl)

# -----------------------------------------------------------------------------
# 2. Mechanization Class
# -----------------------------------------------------------------------------
class Mechanization(InitINS):
    def __init__(self, latitude, longitude, altitude, roll, pitch, azimuth, dt):
        super().__init__(latitude, longitude, altitude, roll, pitch, azimuth)
        # Initialize rotation matrix, quaternion, local gravity
        self.Init_Rbl()
        self.Init_Quaternion()
        self.Init_Localg()

        # Store states
        self._latitude   = latitude
        self._longitude  = longitude
        self._altitude   = altitude
        self._roll       = roll
        self._pitch      = pitch
        self._azimuth    = azimuth
        self._delta_time = dt

    # --- Earth Radii ---
    def RadiiM(self):
        lat_rad = np.deg2rad(self._latitude)
        self._Rm = ( self.a*(1 - self.e2) ) / ( (1 - self.e2*(np.sin(lat_rad)**2))**1.5 )

    def RadiiN(self):
        lat_rad = np.deg2rad(self._latitude)
        self._Rn = self.a / np.sqrt(1 - self.e2*(np.sin(lat_rad)**2))

    # --- ECEF to Local-Level transformation ---
    def R_EL(self):
        """
        Updates the rotation matrix from ECEF to local-level frame, based on lat & lon.
        """
        lat_rad  = np.deg2rad(self._latitude)
        lon_rad  = np.deg2rad(self._longitude)

        # Hardcoded rotation for lat/lon -> local-level
        self.rel11 = -(np.sin(0)*np.sin(lat_rad)*np.cos(lon_rad)) - (np.cos(0)*np.sin(lon_rad))
        self.rel12 = -(np.sin(0)*np.sin(lat_rad)*np.sin(lon_rad)) + (np.cos(0)*np.cos(lon_rad))
        self.rel13 = np.sin(0)*np.cos(lat_rad)

        self.rel21 = -(np.cos(0)*np.sin(lat_rad)*np.cos(lon_rad)) + (np.sin(0)*np.sin(lon_rad))
        self.rel22 = -(np.cos(0)*np.sin(lat_rad)*np.sin(lon_rad)) - (np.sin(0)*np.cos(lon_rad))
        self.rel23 = np.cos(0)*np.cos(lat_rad)

        self.rel31 = np.cos(lat_rad)*np.cos(lon_rad)
        self.rel32 = np.cos(lat_rad)*np.sin(lon_rad)
        self.rel33 = np.sin(lat_rad)

        self._Re_l = np.array([
            [self.rel11, self.rel12, self.rel13],
            [self.rel21, self.rel22, self.rel23],
            [self.rel31, self.rel32, self.rel33]
        ])

    def WIE_L(self):
        """
        Transforms Earth rotation rate from ECEF frame to local-level frame.
        """
        we_transform = np.array([0, 0, self.We])
        self._wie_l  = self._Re_l.dot(we_transform)

    def WEL_L(self):
        """
        Computes angular velocity due to movement over Earth (transport rate).
        """
        lat_rad = np.deg2rad(self._latitude)
        # - (vn)/Rm+alt , (ve)/Rn+alt, (ve tanLat)/Rn+alt
        wel_l1 = - self._vl[1] / (self._Rm + self._altitude)
        wel_l2 =   self._vl[0] / (self._Rn + self._altitude)
        wel_l3 =   self._vl[0]*np.tan(lat_rad) / (self._Rn + self._altitude)
        self._wel_l = np.array([wel_l1, wel_l2, wel_l3])

    def WLB_B(self, wx, wy, wz):
        """
        Computes the body-rate vector minus Earth/transport rates in the local-level frame.
        """
        wib_b = np.array([wx, wy, wz])
        # wlb_b = wib_b - Rb_l^T (wel_l + wie_l)
        self._wlb_b = wib_b - self._Rb_l.T.dot(self._wel_l + self._wie_l)

    def SkewMatrix_WLB_B(self):
        """
        Builds the skew-symmetric matrix of _wlb_b for quaternion derivative.
        """
        p, q, r = self._wlb_b
        # 4x4 skew matrix for quaternion updates
        self._skewmatrix_wlb_b = np.array([
            [   0,    r,  -q,   p],
            [  -r,    0,   p,   q],
            [   q,   -p,   0,   r],
            [  -p,   -q,  -r,   0]
        ], dtype=float)

    def UpdateQuaternion(self):
        """
        Integrates the quaternion using the skew-symmetric matrix of body rates.
        """
        Q_dot = 0.5 * (self._skewmatrix_wlb_b @ self._Quaternion)
        self._Quaternion += self._delta_time * Q_dot
        self._Quaternion /= np.linalg.norm(self._Quaternion)

    def UpdateRBL_Q(self):
        """
        Rebuilds the Rb_l matrix from the updated quaternion.
        """
        q0, q1, q2, q3 = self._Quaternion
        self.rbl11 = q0**2 - q1**2 - q2**2 + q3**2
        self.rbl12 = 2*((q0*q1) - (q2*q3))
        self.rbl13 = 2*((q0*q2) + (q1*q3))

        self.rbl21 = 2*((q0*q1) + (q2*q3))
        self.rbl22 = - (q0**2) + (q1**2) - (q2**2) + (q3**2)
        self.rbl23 = 2*((q1*q2) - (q0*q3))

        self.rbl31 = 2*((q0*q2) - (q1*q3))
        self.rbl32 = 2*((q1*q2) + (q0*q3))
        self.rbl33 = - (q0**2) - (q1**2) + (q2**2) + (q3**2)

        self._Rb_l = np.array([
            [self.rbl11, self.rbl12, self.rbl13],
            [self.rbl21, self.rbl22, self.rbl23],
            [self.rbl31, self.rbl32, self.rbl33]
        ], dtype=float)

    def UpdateAttitude(self):
        """
        Extracts roll, pitch, and azimuth from the updated Rb_l.
        """
        self._pitch = np.rad2deg(
            np.arctan2(self.rbl32, np.sqrt(self.rbl12**2 + self.rbl22**2))
        )
        self._roll = - np.rad2deg( np.arctan2(self.rbl31, self.rbl33) )
        self._azimuth = np.rad2deg( np.arctan2(self.rbl12, self.rbl22) )

    def CorrectAzimuth(self):
        """
        Ensures azimuth remains within [0, 360).
        """
        if self._azimuth >= 360.0:
            self._azimuth -= 360.0
        elif self._azimuth < 0.0:
            self._azimuth += 360.0

    def OMEGA_IE_L(self):
        """
        Builds the skew matrix of Earth rotation in local level frame.
        """
        wx, wy, wz = self._wie_l
        self._omega_ie_l = np.array([
            [0,   -wz,  wy],
            [wz,   0,  -wx],
            [-wy, wx,   0]
        ], dtype=float)

    def OMEGA_EL_L(self):
        """
        Builds the skew matrix of transport rate in local level frame.
        """
        wx, wy, wz = self._wel_l
        self._omega_el_l = np.array([
            [0,   -wz,  wy],
            [wz,   0,  -wx],
            [-wy, wx,   0]
        ], dtype=float)

    def UpdateAccelerometers(self, fx, fy, fz):
        """
        Saves current accelerometer reading in local scope.
        """
        self._fb = np.array([fx, fy, fz], dtype=float)

    def UpdateG(self):
        """
        Recomputes local gravity if needed (currently not called in the loop).
        """
        lat_rad = np.deg2rad(self._latitude)
        Proj_Lat = np.sin(lat_rad)
        self._Local_g = ( self.a1*(1 + self.a2*(Proj_Lat**2) + self.a3*(Proj_Lat**4))
                        + (self.a4 + self.a5*(Proj_Lat**2))*self._altitude
                        + self.a6*(self._altitude**2) )
        self._gVector = np.array([0, 0, - self._Local_g], dtype=float)

    def UpdateDeltaVelocity(self):
        """
        Updates change in velocity in local-level frame, accounting for Earth rotation and gravity.
        """
        component_a = self._Rb_l @ self._fb
        component_b = (2*self._omega_ie_l + self._omega_el_l) @ self._vl
        delta_v_t   = component_a - component_b + self._gVector
        self._Delta_Vl_current = delta_v_t * self._delta_time

    def UpdateVelocity(self):
        """
        Integrates velocity using trapezoidal rule.
        """
        self._prev_vl = self._vl.copy()
        self._vl += 0.5 * (self._Delta_Vl_current + self._Delta_Vl_previous)
        self._Delta_Vl_previous = self._Delta_Vl_current

    def UpdatePosition(self):
        """
        Integrates position (lat, lon, alt) using velocity.
        """
        ve_prev, vn_prev, vu_prev = self._prev_vl
        ve, vn, vu = self._vl
        half_dt = 0.5*self._delta_time

        # dLon
        self._longitude += np.rad2deg(
            half_dt*(ve + ve_prev) / ((self._Rn + self._altitude)*np.cos(np.deg2rad(self._latitude)))
        )
        # dLat
        self._latitude += np.rad2deg(
            half_dt*(vn + vn_prev) / (self._Rm + self._altitude)
        )
        # dAlt
        self._altitude += half_dt*(vu + vu_prev)

    def compile_standalone(self, wx, wy, wz, fx, fy, fz, odometer_speed=None):
        """
        Performs one epoch of open-loop INS mechanization.
        Optionally, if an odometer speed is available, the horizontal (East, North)
        velocity is recomputed using the INS-derived azimuth such that its magnitude
        exactly matches the odometer speed.
        """
        # 1. Compute Earth radii
        self.RadiiM()
        self.RadiiN()
        
        # 2. Compute Earth rates and related matrices
        self.R_EL()
        self.WIE_L()
        self.WEL_L()
        self.WLB_B(wx, wy, wz)
        self.SkewMatrix_WLB_B()
        
        # 3. Update quaternion
        self.UpdateQuaternion()
        
        # 4. Update attitude based on updated quaternion
        self.UpdateRBL_Q()
        self.UpdateAttitude()
        self.CorrectAzimuth()
        
        # 5. Update velocity using accelerometers and delta velocity
        self.OMEGA_IE_L()
        self.OMEGA_EL_L()
        self.UpdateAccelerometers(fx, fy, fz)
        self.UpdateDeltaVelocity()
        self.UpdateVelocity()
        
        # -----------------------------
        # Odometer integration logic:
        # If an odometer speed is provided, assume it is the forward speed in the
        # body frame. Form the body-frame velocity vector [odo_speed, 0, 0],
        # transform it using the current rotation matrix, and update the INS velocity.
        if odometer_speed is not None:
            odo_body = np.array([odometer_speed, 0, 0])
            # Transform to local-level frame using the body-to-local rotation matrix.
            odo_local = self._Rb_l @ odo_body
            self._vl[0] = odo_local[0]
            self._vl[1] = odo_local[1]
            self._vl[2] = odo_local[2]
        
        # 7. Update position using the (possibly odometer-corrected) velocity
        self.UpdatePosition()


    # -----------------------------------------------------------
    # Properties for direct access to states
    # -----------------------------------------------------------
    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def altitude(self):
        return self._altitude

    @property
    def velocity_vector(self):
        return self._vl

    @property
    def roll(self):
        return self._roll

    @property
    def pitch(self):
        return self._pitch

    @property
    def azimuth(self):
        return self._azimuth

    @property
    def delta_time(self):
        return self._delta_time

    # Setters
    @latitude.setter
    def latitude(self, val):
        self._latitude = val

    @longitude.setter
    def longitude(self, val):
        self._longitude = val

    @altitude.setter
    def altitude(self, val):
        self._altitude = val

    @velocity_vector.setter
    def velocity_vector(self, val):
        self._vl = val

    @roll.setter
    def roll(self, val):
        self._roll = val

    @pitch.setter
    def pitch(self, val):
        self._pitch = val

    @azimuth.setter
    def azimuth(self, val):
        self._azimuth = val

    @delta_time.setter
    def delta_time(self, val):
        self._delta_time = val

# -----------------------------------------------------------------------------
# 3. Helper Functions (downsample, sync, etc.)
# -----------------------------------------------------------------------------
def down_sample(signal, down_sample_bin):
    L = len(signal)
    L_new = int(np.floor(L/down_sample_bin))
    signal_new = np.zeros(L_new)
    for i in range(L_new):
        frm = i * down_sample_bin
        to  = (i + 1) * down_sample_bin
        signal_new[i] = np.mean(signal[frm:to])
    return signal_new

def down_sample_time(signal, down_sample_bin):
    L = len(signal)
    L_new = int(np.floor(L/down_sample_bin))
    signal_new = np.zeros(L_new)
    for i in range(L_new):
        pick_time = i * down_sample_bin
        signal_new[i] = signal[pick_time]
    return signal_new

def upsample_signal(signal, up_sample_factor):
    d = 1.0 / up_sample_factor
    return np.interp(
        np.arange(0, len(signal), d),
        np.arange(0, len(signal)),
        signal
    )

def shift_calc(time_signal, t0):
    time_min_diff = np.min(np.abs(time_signal - t0))
    idx = np.where(np.abs(time_signal - t0) == time_min_diff)[0][0]
    return idx

# -----------------------------------------------------------------------------
# 4. Data Loading / Synchronization Routines
# -----------------------------------------------------------------------------
def load_IMU_dataset(imu_file_path, IMU_CHOICE, Freq_INS):
    """
    Loads and downsamples the IMU data.
    """
    data = sio.loadmat(imu_file_path)
    time_imu = data['IMU_second'][:, 0]
    Freq_imu = np.round(1 / np.mean(time_imu[1:] - time_imu[:-1]))
    print('IMU frequency =', Freq_imu, 'Hz')

    down_sampleBin_imu = int(np.round(Freq_imu / Freq_INS))
    print('Downsampling IMU by:', down_sampleBin_imu)

    time_imu = down_sample_time(time_imu, down_sampleBin_imu)

    # Acc
    fx_imu = data['f']['x'][0, 0][:, 0]
    fy_imu = data['f']['y'][0, 0][:, 0]
    fz_imu = data['f']['z'][0, 0][:, 0]
    if IMU_CHOICE == 0:  # KVH
        fy_imu *= -1
    fx_imu = down_sample(fx_imu, down_sampleBin_imu)
    fy_imu = down_sample(fy_imu, down_sampleBin_imu)
    fz_imu = down_sample(fz_imu, down_sampleBin_imu)

    # Gyros
    wx_imu = data['w']['x'][0, 0][:, 0]
    wy_imu = data['w']['y'][0, 0][:, 0]
    wz_imu = data['w']['z'][0, 0][:, 0]
    if IMU_CHOICE == 0:  # KVH
        wy_imu *= -1

    wx_imu = down_sample(wx_imu, down_sampleBin_imu)
    wy_imu = down_sample(wy_imu, down_sampleBin_imu)
    wz_imu = down_sample(wz_imu, down_sampleBin_imu)

    imu_data = {
        'time': time_imu,
        'fx':   fx_imu,
        'fy':   fy_imu,
        'fz':   fz_imu,
        'wx':   wx_imu,
        'wy':   wy_imu,
        'wz':   wz_imu
    }
    return imu_data

def load_ODO_dataset(odo_file_path):
    data = sio.loadmat(odo_file_path)
    odo_time  = data['CarChip_second_1HZ'].squeeze()
    odo_speed = data['CarChip_Speed_1HZ'].squeeze()
    return {
        'time':  odo_time,
        'speed': odo_speed
    }

def load_REF_dataset(ref_file_path, Freq_INS):
    """
    Loads and (possibly) downsamples the reference (INS) data.
    """
    ref_data = sio.loadmat(ref_file_path)

    time_ref = ref_data['INS_second'][:,0]
    lat_ref  = ref_data['INS_Lat'][:,0]
    lon_ref  = ref_data['INS_Long'][:,0]
    alt_ref  = ref_data['INS_Alt'][:,0]

    roll_ref    = ref_data['INS_Roll'][:,0]
    pitch_ref   = ref_data['INS_Pitch'][:,0]
    azimuth_ref = ref_data['INS_Azi'][:,0]

    ve_ref = ref_data['INS_ve'][:,0]
    vn_ref = ref_data['INS_vn'][:,0]
    vu_ref = ref_data['INS_vu'][:,0]

    Freq_ref = np.round(1 / np.mean(time_ref[1:] - time_ref[:-1]))
    print('Ref frequency =', Freq_ref, 'Hz')

    if Freq_ref > Freq_INS:
        sampling_factor = int(np.round(Freq_ref / Freq_INS))
        sample_fn = down_sample
    elif Freq_ref < Freq_INS:
        sampling_factor = int(Freq_INS / Freq_ref)
        sample_fn = upsample_signal
    else:
        sampling_factor = 1

    if sampling_factor != 1:
        time_ref    = sample_fn(time_ref, sampling_factor)
        lat_ref     = sample_fn(lat_ref, sampling_factor)
        lon_ref     = sample_fn(lon_ref, sampling_factor)
        alt_ref     = sample_fn(alt_ref, sampling_factor)
        roll_ref    = sample_fn(roll_ref, sampling_factor)
        pitch_ref   = sample_fn(pitch_ref, sampling_factor)
        azimuth_ref = sample_fn(azimuth_ref, sampling_factor)
        ve_ref      = sample_fn(ve_ref, sampling_factor)
        vn_ref      = sample_fn(vn_ref, sampling_factor)
        vu_ref      = sample_fn(vu_ref, sampling_factor)

    ref_dict = {
        'time':    time_ref,
        'lat':     lat_ref,
        'lon':     lon_ref,
        'alt':     alt_ref,
        'roll':    roll_ref,
        'pitch':   pitch_ref,
        'azimuth': azimuth_ref,
        'Ve':      ve_ref,
        'Vn':      vn_ref,
        'Vu':      vu_ref
    }
    return ref_dict

# def synchronize_general(data, time_from, time_to):
#     """
#     Crops data dict so that data['time'] is within [time_from, time_to].
#     """
#     dt = data['time'][1] - data['time'][0]

#     # Crop start
#     if (time_from - data['time'][0]) > dt:
#         start_idx = shift_calc(data['time'], time_from)
#         for k in data.keys():
#             data[k] = data[k][start_idx:]

#     # Crop end
#     if (data['time'][-1] - time_to) > dt:
#         end_idx = shift_calc(data['time'], time_to) + 1
#         for k in data.keys():
#             data[k] = data[k][:end_idx]

def synchronize_general(data, time_from, time_to):
    dt = data['time'][1] - data['time'][0]
    # Use >= instead of > to force cropping when difference equals dt
    if (time_from - data['time'][0]) >= dt:
        start_idx = shift_calc(data['time'], time_from)
        for k in data:
            data[k] = data[k][start_idx:]
    if (data['time'][-1] - time_to) >= dt:
        end_idx = shift_calc(data['time'], time_to) + 1
        for k in data:
            data[k] = data[k][:end_idx]

    # Crop end
    if (data['time'][-1] - time_to) > dt:
        end_idx = shift_calc(data['time'], time_to) + 1
        for k in data.keys():
            data[k] = data[k][:end_idx]


def sync_INS(ref_data, imu_data):
    """
    Synchronizes reference and IMU data to the overlapping time region.
    """
    time_from = max(ref_data['time'][0], imu_data['time'][0])
    time_to   = min(ref_data['time'][-1], imu_data['time'][-1])

    synchronize_general(ref_data, time_from, time_to)
    print('* Ref ->', len(ref_data['time']), ref_data['time'][0], ref_data['time'][-1])

    synchronize_general(imu_data, time_from, time_to)
    print('* IMU ->', len(imu_data['time']), imu_data['time'][0], imu_data['time'][-1])

    return ref_data, imu_data

def syncronize_INS_ODO(ref_data, imu_data, odo_data):
    """
    Synchronizes reference, IMU, and ODO data all to the overlapping time region.
    """
    # 1. Find the common overlapping time interval among ref, IMU, and ODO
    overall_start = max(ref_data['time'][0], imu_data['time'][0], odo_data['time'][0])
    overall_end   = min(ref_data['time'][-1], imu_data['time'][-1], odo_data['time'][-1])

    # 2. Crop each dataset to this interval
    synchronize_general(ref_data, overall_start, overall_end)
    synchronize_general(imu_data, overall_start, overall_end)
    synchronize_general(odo_data, overall_start, overall_end)

    print("After ODO sync:")
    print("Ref =>", len(ref_data['time']), ref_data['time'][0], ref_data['time'][-1])
    print("IMU =>", len(imu_data['time']), imu_data['time'][0], imu_data['time'][-1])
    print("ODO =>", len(odo_data['time']), odo_data['time'][0], odo_data['time'][-1])

    return ref_data, imu_data, odo_data
# -----------------------------------------------------------------------------
# 5. Plotting Utilities
# -----------------------------------------------------------------------------
def plot_to_compare(sig_ref, time_ref, sig_imu, time_imu, title_text, ylabel, solution_label, gps_outage_info=None):
    fig = plt.figure(dpi=100, figsize=(7,4))
    time_ref_plot = (time_ref - time_ref.min())/60.0
    time_imu_plot = (time_imu - time_imu.min())/60.0

    plt.plot(time_imu_plot, sig_imu, 'b', label=solution_label)
    plt.plot(time_ref_plot, sig_ref, 'r--', label='Reference')
    plt.title(title_text)
    plt.xlabel('Time [mins]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    if gps_outage_info is not None:
        for s, e in zip(gps_outage_info['outage_stime'], gps_outage_info['outage_etime']):
            plt.axvspan(s, e, color='yellow', alpha=0.8)
    plt.tight_layout()
    plt.show()

def plot_error(sig_ref, time_ref, sig_imu, time_imu, title_text, ylabel, gps_outage_info=None):
    fig = plt.figure(dpi=100, figsize=(7,4))
    time_imu_plot = (time_imu - time_imu.min())/60.0
    abs_err = np.abs(sig_ref - sig_imu)
    plt.plot(time_imu_plot, abs_err, 'k--')
    plt.title(title_text)
    plt.xlabel('Time [mins]')
    plt.ylabel(ylabel)
    plt.grid()
    if gps_outage_info is not None:
        for s, e in zip(gps_outage_info['outage_stime'], gps_outage_info['outage_etime']):
            plt.axvspan(s, e, color='yellow', alpha=0.8)
    plt.tight_layout()
    plt.show()

def plot_bi_trajectory(sig_ref, sig_imu, title_text, solution_label):
    fig = plt.figure(dpi=100, figsize=(7,4))
    plt.plot(sig_imu['lon'], sig_imu['lat'], 'b', label=solution_label)
    plt.plot(sig_ref['lon'], sig_ref['lat'], 'r--', label='Reference')
    plt.title(title_text)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def full_diagnosis(ref_data, sig_data, label):
    """
    Generates multiple comparative plots for diagnosing the INS performance.
    """
    solution_label = label
    # 1) Trajectory
    plot_bi_trajectory(ref_data, sig_data, 'Trajectory', solution_label)
    # 2) Position Errors
    plot_error(ref_data['lat'], ref_data['time'], sig_data['lat'], sig_data['time'], 'Latitude Errors', 'Error [deg]')
    plot_error(ref_data['lon'], ref_data['time'], sig_data['lon'], sig_data['time'], 'Longitude Errors', 'Error [deg]')
    plot_error(ref_data['alt'], ref_data['time'], sig_data['alt'], sig_data['time'], 'Altitude Errors', 'Error [m]')

    # 3) Attitude Errors
    plot_error(ref_data['roll'],    ref_data['time'], sig_data['roll'],    sig_data['time'], 'Roll Errors',    'Error [deg]')
    plot_error(ref_data['pitch'],   ref_data['time'], sig_data['pitch'],   sig_data['time'], 'Pitch Errors',   'Error [deg]')
    plot_error(ref_data['azimuth'], ref_data['time'], sig_data['azimuth'], sig_data['time'], 'Azimuth Errors', 'Error [deg]')

    # 4) Velocity Errors
    plot_error(ref_data['Ve'], ref_data['time'], sig_data['ve'], sig_data['time'], 'East Velocity Errors',  'Error [m/s]')
    plot_error(ref_data['Vn'], ref_data['time'], sig_data['vn'], sig_data['time'], 'North Velocity Errors', 'Error [m/s]')
    plot_error(ref_data['Vu'], ref_data['time'], sig_data['vu'], sig_data['time'], 'Up Velocity Errors',    'Error [m/s]')

    # 5) Direct Comparisons
    plot_to_compare(ref_data['lat'], ref_data['time'], sig_data['lat'], sig_data['time'],     'Latitude',  'Degrees', solution_label)
    plot_to_compare(ref_data['lon'], ref_data['time'], sig_data['lon'], sig_data['time'],     'Longitude', 'Degrees', solution_label)
    plot_to_compare(ref_data['alt'], ref_data['time'], sig_data['alt'], sig_data['time'],     'Altitude',  'Meters',  solution_label)

    plot_to_compare(ref_data['roll'],    ref_data['time'], sig_data['roll'],    sig_data['time'], 'Roll',    'Degrees', solution_label)
    plot_to_compare(ref_data['pitch'],   ref_data['time'], sig_data['pitch'],   sig_data['time'], 'Pitch',   'Degrees', solution_label)
    plot_to_compare(ref_data['azimuth'], ref_data['time'], sig_data['azimuth'], sig_data['time'], 'Azimuth', 'Degrees', solution_label)

    plot_to_compare(ref_data['Ve'], ref_data['time'], sig_data['ve'], sig_data['time'], 'East Velocity',  'm/s', solution_label)
    plot_to_compare(ref_data['Vn'], ref_data['time'], sig_data['vn'], sig_data['time'], 'North Velocity', 'm/s', solution_label)
    plot_to_compare(ref_data['Vu'], ref_data['time'], sig_data['vu'], sig_data['time'], 'Up Velocity',    'm/s', solution_label)


# -----------------------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 16})

    # File paths: adjust to your actual directories
    ref_file_path     = r'.\2008-06-23_Downtown_Kingston\rover\Mat\INSPVA'
    imu_or_file_path  = r'.\2008-06-23_Downtown_Kingston\rover\Mat\RAWIMU_denoised_LOD3_interpolated2'
    imu_file_path     = r'.\2008-06-23_Downtown_Kingston\f_w_IMSseconds_dt.mat'
    sample_file_path  = r'.\2008-06-23_Downtown_Kingston\sample_time.mat'
    odo_file_path     = r'C:\Users\hanym\OneDrive\Documents\INSfullMech-main\INSfullMech-main\2008-06-23_Downtown_Kingston\Sensors\CarChip_Speed_interpolated.mat'

    # Frequency and data loading
    Freq_INS = 10
    KVH      = 0  # IMU type
    ref_data = load_REF_dataset(ref_file_path, Freq_INS)
    IMU_data = load_IMU_dataset(imu_or_file_path, KVH, Freq_INS)
    ODO_data = load_ODO_dataset(odo_file_path)
    ODO_data_time = ODO_data['time']
    ODO_data_speed = ODO_data['speed']
    ref_data_speed = np.sqrt(ref_data['Ve']**2 + ref_data['Vn']**2 + ref_data['Vu']**2)
    ref_data_time = ref_data['time']
    plt.figure()
    plt.plot(ref_data_time, ref_data_speed, color = 'red')
    plt.plot(ODO_data_time, ODO_data_speed, 'go', markersize=2)
    plt.show()
    import pdb; pdb.set_trace()
    # Synchronize reference & IMU data
    # ref_data, IMU_data = sync_INS(ref_data, IMU_data)
    ref_data, IMU_data, ODO_data = syncronize_INS_ODO(ref_data, IMU_data, ODO_data)
    pdb.set_trace()
    # Display some info
    print("IMU first timestamp:", IMU_data['time'][0])
    print("Ref first timestamp:", ref_data['time'][0])

    # Biases (Disable if using Dr. Malek's IMU data with no needed biases)
    fx_bias = -70.88e-4
    fy_bias =  40.80e-4
    fz_bias =  61.00e-4
    wx_bias = -40.84e-007
    wy_bias =  12.3e-007
    wz_bias =  2.06e-006

    # Remove biases from IMU signals
    IMU_data['fx'] -= fx_bias
    IMU_data['fy'] -= fy_bias
    IMU_data['fz'] -= fz_bias
    IMU_data['wx'] -= wx_bias
    IMU_data['wy'] -= wy_bias
    IMU_data['wz'] -= wz_bias

    # Sample time array
    sample_data    = sio.loadmat(sample_file_path)
    sample_time_new = sample_data['sample_time'][0][:]
    
    # Mechanization initialization parameters
    start_time = 600  # in seconds
    start      = start_time * Freq_INS
    Initial    = start - 1

    Init_lat      = ref_data['lat'][start]
    Init_lon      = ref_data['lon'][start]
    Init_alt      = ref_data['alt'][start]
    Init_roll     = ref_data['roll'][start]
    Init_pitch    = ref_data['pitch'][start]
    Init_azimuth  = ref_data['azimuth'][start]
    Init_ve       = ref_data['Ve'][start]
    Init_vn       = ref_data['Vn'][start]
    Init_vu       = ref_data['Vu'][start]
    dt            = sample_time_new[start]  # Alternatively: IMU_data['time'][1] - IMU_data['time'][0]

    # Create Mechanization instance
    INS_Mechanize = Mechanization(
        Init_lat, Init_lon, Init_alt,
        Init_roll, Init_pitch, Init_azimuth,
        dt
    )
    INS_Mechanize.Init_Velocity(Init_ve, Init_vn, Init_vu)

    duration = len(IMU_data['fx'])
    Lat, Lon, Alt = [Init_lat], [Init_lon], [Init_alt]
    Roll, Pitch, Azimuth = [Init_roll], [Init_pitch], [Init_azimuth]
    ve, vn, vu = [Init_ve], [Init_vn], [Init_vu]

    # Initialize pointer for odometer data (ODO is 1Hz)
    j_odo = 0
    n_odo = len(ODO_data['time'])

    # Main Mechanization Loop
    for i in range(start, duration):
        current_imu_time = IMU_data['time'][i]
        print('IMU Time:', current_imu_time)
        wx = IMU_data['wx'][i]
        wy = IMU_data['wy'][i]
        wz = IMU_data['wz'][i]
        fx = IMU_data['fx'][i]
        fy = IMU_data['fy'][i]
        fz = IMU_data['fz'][i]
        INS_Mechanize.delta_time = IMU_data['time'][i] - IMU_data['time'][i - 1]
        # INS_Mechanize.delta_time = sample_time_new[i]  # If you prefer this approach

        # Check if odometer data is available at this IMU time:
        odo_value = None
        # While the next odometer sample is less than or equal to current IMU time, update odo_value.
        while j_odo < n_odo and ODO_data['time'][j_odo] <= current_imu_time:
            odo_value = ODO_data['speed'][j_odo]
            j_odo += 1

        if odo_value is not None:
            print("ODO Speed available at IMU time:", odo_value)
            # You can pass odo_value to your mechanization if you want to incorporate it.
            # For example:
            INS_Mechanize.compile_standalone(wx, wy, wz, fx, fy, fz, odometer_speed = odo_value)
            # Otherwise, if not used, simply print or process as needed.
        else:
            # No new ODO sample at this epoch
            # Mechanize
            INS_Mechanize.compile_standalone(wx, wy, wz, fx, fy, fz, odometer_speed = odo_value)
        # Update delta_time from consecutive IMU stamps or sample_time array
        

        

        # Store results
        Lat.append(INS_Mechanize.latitude)
        Lon.append(INS_Mechanize.longitude)
        Alt.append(INS_Mechanize.altitude)
        Roll.append(INS_Mechanize.roll)
        Pitch.append(INS_Mechanize.pitch)
        Azimuth.append(INS_Mechanize.azimuth)
        ve.append(INS_Mechanize.velocity_vector[0])
        vn.append(INS_Mechanize.velocity_vector[1])
        vu.append(INS_Mechanize.velocity_vector[2])

    # Prepare results for plotting
    imu_data_mechanized = {}
    ref_data_trimmed    = {}

    imu_data_mechanized['time']  = IMU_data['time'][Initial:duration]
    ref_data_trimmed['time']     = ref_data['time'][Initial:duration]

    imu_data_mechanized['lat']   = Lat
    ref_data_trimmed['lat']      = ref_data['lat'][Initial:duration]

    imu_data_mechanized['lon']   = Lon
    ref_data_trimmed['lon']      = ref_data['lon'][Initial:duration]

    imu_data_mechanized['alt']   = Alt
    ref_data_trimmed['alt']      = ref_data['alt'][Initial:duration]

    imu_data_mechanized['roll']  = Roll
    ref_data_trimmed['roll']     = ref_data['roll'][Initial:duration]

    imu_data_mechanized['pitch'] = Pitch
    ref_data_trimmed['pitch']    = ref_data['pitch'][Initial:duration]

    imu_data_mechanized['azimuth'] = Azimuth
    ref_data_trimmed['azimuth']    = ref_data['azimuth'][Initial:duration]

    imu_data_mechanized['ve']     = ve
    ref_data_trimmed['ve']        = ref_data['Ve'][Initial:duration]

    imu_data_mechanized['vn']     = vn
    ref_data_trimmed['vn']        = ref_data['Vn'][Initial:duration]

    imu_data_mechanized['vu']     = vu
    ref_data_trimmed['vu']        = ref_data['Vu'][Initial:duration]

    pdb.set_trace()
    ve_list = imu_data_mechanized['ve']  # Python list
    vn_list = imu_data_mechanized['vn']
    vu_list = imu_data_mechanized['vu']

    # Convert to NumPy arrays
    ve = np.array(ve_list, dtype=float)
    vn = np.array(vn_list, dtype=float)
    vu = np.array(vu_list, dtype=float)

    imu_data_mechanized_speed = np.sqrt(ve**2 + vn**2 + vu**2)
    imu_data_mechanized_time = imu_data_mechanized['time']

    plt.figure()
    plt.plot(ref_data_time, ref_data_speed, color = 'red')
    plt.plot(imu_data_mechanized_time, imu_data_mechanized_speed, 'go', markersize=2)
    plt.show()
    # Print some checks
    print("IMU first timestamp (mechanized):", imu_data_mechanized['time'][0])
    print("REF first timestamp (trimmed):", ref_data_trimmed['time'][0])

    # Final diagnosis plots
    full_diagnosis(ref_data_trimmed, imu_data_mechanized, 'INS standalone')
