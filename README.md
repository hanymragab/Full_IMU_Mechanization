# Full INS Mechanization with Odometer/Speed Aiding

This repository contains a validation script for an Inertial Navigation System (INS) mechanization algorithm using the Kingston 2008 dataset. The algorithm integrates inertial measurements and—optionally—fuses low-rate odometer speed measurements to update the INS velocity. **Note:** The odometer speed is acquired using the **OBD-II Vehicle Diagnostic protocol**, which is not as accurate as CAN-derived data or dedicated Distance Measuring Instruments (DMIs). See [arXiv](https://arxiv.org/abs/2501.00242) for a comprehensive overview of automotive speed sensing technologies.

## Overview

The main Python file, `INS_mech.py`, performs the following tasks:

- **Data Loading:**  
  Loads MATLAB data files for:
  - **Reference Data** (INSPVA):  
    - `INS_second` – Time stamps  
    - `INS_Lat`, `INS_Long`, `INS_Alt` – Position data  
    - `INS_Roll`, `INS_Pitch`, `INS_Azi` – Attitude data  
    - `INS_ve`, `INS_vn`, `INS_vu` – Velocity components  
  - **IMU Data:**  
    - `IMU_second` – Time stamps  
    - `f.x`, `f.y`, `f.z` – Accelerometer measurements  
    - `w.x`, `w.y`, `w.z` – Gyroscope measurements  
    *(For KVH IMU, the y-axis signals are inverted.)*
  - **Sample Time Data:**  
    - `sample_time` – Array of sample intervals  
  - **Odometer Data:**  
    - `CarChip_second_1HZ` – Time stamps (at 1 Hz)
    - `CarChip_second_10HZ` – Time stamps (upsampled at 10 Hz)  
    - `CarChip_Speed_1HZ` – Odometer speed measurements (via OBD-II)
    - `CarChip_Speed_10HZ` – Upsampled odometer speed measurements

- **Data Synchronization:**  
  The script synchronizes the reference, IMU, and odometer datasets to a common overlapping time interval.

- **INS Mechanization:**  
  The INS mechanization algorithm computes the navigation state (attitude, velocity, and position) at a 10 Hz update rate. Processing starts at 600 seconds into the dataset.

- **Odometer Aiding:**  
  If an odometer speed measurement is available, it is assumed to represent the forward (body-frame) speed. The speed is transformed into the local-level frame using the current body-to-local rotation matrix and is fused into the INS velocity update.

- **Plotting:**  
  The script produces:
  - A full diagnostic set of plots comparing INS outputs with the reference.
  - A time-series plot that overlays the INS-derived speed (continuous line) with the odometer speed (displayed as dots due to its lower sampling rate).

## Data Requirements

The code expects the following MATLAB files with these variable names:

- **Reference Data (`INSPVA`):**
  - `INS_second` – Time stamps (typically GPS time)
  - `INS_Lat`, `INS_Long`, `INS_Alt` – Position (latitude, longitude, altitude)
  - `INS_Roll`, `INS_Pitch`, `INS_Azi` – Attitude (roll, pitch, azimuth)
  - `INS_ve`, `INS_vn`, `INS_vu` – Velocity components (east, north, up)

- **IMU Data (`RAWIMU_denoised_LOD3_interpolated2`):**
  - `IMU_second` – Time stamps
  - `f.x`, `f.y`, `f.z` – Accelerometer measurements
  - `w.x`, `w.y`, `w.z` – Gyroscope measurements  
    *(For KVH IMU, the y-axis signals are inverted.)*

- **Sample Time Data (`sample_time.mat`):**
  - `sample_time` – Array of sample intervals

- **Odometer Data (`CarChip_Speed_interpolated.mat`):**
  - `CarChip_second_1HZ` – Time stamps (approx. 1 Hz)
  - `CarChip_Speed_1HZ` – Odometer speed measurements  
    **Note:** These speeds are obtained via the **OBD-II Vehicle Diagnostic protocol** and are not as accurate as CAN-derived data or DMIs.

## Dependencies

This project requires:
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)

Install them via pip:

```bash
pip install numpy scipy matplotlib
```

## Installation

Clone the repository:

```bash
git clone https://github.com/hanymragab/Full_INS_Mechanization.git
cd Full_INS_Mechanization
```

Place your MATLAB data files in the appropriate directories as referenced in the code (or update the file paths accordingly).

## Usage

To run the script, execute:

```bash
python INS_mech.py
```

### What the Script Does

1. **Data Loading:**  
   Loads reference, IMU, sample time, and odometer data from MATLAB files.

2. **Data Synchronization:**  
   Crops and synchronizes the datasets to the overlapping time interval.

3. **INS Mechanization:**  
   Processes IMU data at 10 Hz starting at 600 seconds into the dataset to compute navigation states.

4. **Odometer Aiding:**  
   If an odometer speed is available, it is transformed from the body frame into the local-level frame using the current rotation matrix and fused into the INS velocity update.

5. **Plotting:**  
   The script produces:
   - A full diagnosis comparing INS outputs to reference data.
   - A time-series plot comparing the INS-derived speed (continuous line) with the odometer speed (dots).

## Algorithm Details

- **INS Mechanization:**  
  Integrates gyroscope and accelerometer data to compute the navigation state. The algorithm:
  - Computes Earth radii based on the current latitude.
  - Transforms Earth rotation rates into the local-level frame.
  - Updates the INS quaternion and extracts attitude (roll, pitch, azimuth).
  - Integrates velocity (with gravitational compensation).
  - Updates position based on the integrated velocity.

- **Odometer Aiding:**  
  The odometer speed (assumed to be the forward body-frame speed) is transformed into the local-level frame using the current body-to-local rotation matrix. The INS velocity vector is then replaced with this transformed vector before updating position.

## Customization

- **Start Time:**  
  The mechanization starts at 600 seconds. Adjust the `start_time` variable if needed.

- **Bias Correction:**  
  The IMU biases are hard-coded; update these if your sensor calibration differs.

- **Odometer Fusion:**  
  The odometer aiding logic is contained in the `compile_standalone` method of the `Mechanization` class. Modify this method if you need a different fusion approach.

## Contact

For any questions or further assistance, please contact me at [hany.ragab@queensu.ca].
