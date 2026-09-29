# MOTION DX
> **"Objective movement measurement using computer vision."**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Intel RealSense](https://img.shields.io/badge/Intel%20RealSense-D405%20RGB--D-critical.svg)](https://www.intelrealsense.com/depth-camera-d405/)
[![MediaPipe Tasks API](https://img.shields.io/badge/MediaPipe-Tasks%20API%20v1.0-brightgreen.svg)](https://ai.google.dev/edge/mediapipe/solutions/guide)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Clinical-red.svg)](https://streamlit.io/)
[![Safety](https://img.shields.io/badge/Clinical%20Safety-Non--Diagnostic-orange.svg)](#clinical-safety-statement)

---

## 1. Project Overview & Clinical Context

### The Real-World Problem
Physical therapists, orthopedic surgeons, sports biomechanists, and dentists/orthodontists frequently evaluate joint Range of Motion (ROM), jaw excursions, and gait mechanics using manual tools (plastic goniometers, calipers, visual inspection). These manual methods suffer from:
- **Inter-rater variability:** Subjective parallax and landmark palpation differences between examiners.
- **Static snapshots:** Inability to continuously quantify full dynamic excursion curves across continuous motion.
- **Manual transcription burden:** High paperwork overhead without automated digital export.

### The Solution: MOTION DX
**MOTION DX** is a computer-vision clinical assistance prototype engineered to turn visible human movement into **objective, repeatable measurements** without contact.

Utilizing the **Intel RealSense D405** optical-axis RGB-D sensor coupled with the **Google MediaPipe Tasks API**, MOTION DX samples high-frequency 2D landmarks and aligns them with spatial depth fields using neighborhood median filtering. It computes anatomical joint angles, active excursion limits, and bilateral symmetries in real time.

> **CRITICAL CLINICAL NOTICE:**  
> **MOTION DX does NOT diagnose medical conditions or diseases.**  
> It provides verified, objective kinematic measurements and observations to support a licensed clinician's evaluation.

---

## 2. Core Modules & Clinical Capabilities

### Module 1: Limb Range of Motion (ROM)
- **Selectable Joints:**
  - **Knee:** Hip $\rightarrow$ Knee (vertex) $\rightarrow$ Ankle sagittal angle.
  - **Elbow:** Shoulder $\rightarrow$ Elbow (vertex) $\rightarrow$ Wrist angle.
  - **Shoulder:** Hip $\rightarrow$ Shoulder (vertex) $\rightarrow$ Elbow elevation/abduction.
- **Dynamic Telemetry:** Current instantaneous angle, peak flexion (minimum), peak extension (maximum), and observed active ROM ($Max - Min$).
- **Visual Feedback:** Vibrant clinical green skeleton with double-ring joint vertex nodes and real-time angle callouts.
- **Bilateral Comparison:** Concurrent tracking of the contralateral limb when both limbs are within the camera frustum.

### Module 2: Temporomandibular Joint (TMD) / Jaw Kinematics
- **Guided 5-Step Clinical Protocol:**
  1. *Step 1 — Neutral Baseline:* Calibrates resting oral aperture and facial midline.
  2. *Step 2 — Active Opening:* Records peak maximum incisal opening aperture.
  3. *Step 3 — Closing Return:* Evaluates return to resting dental contact.
  4. *Step 4 — Left Lateral Excursion:* Quantifies leftward mandibular displacement from midline.
  5. *Step 5 — Right Lateral Excursion:* Quantifies rightward mandibular displacement from midline.
- **Physical Millimeter Quantification:** Leverages Intel RealSense D405 close-range sub-millimeter depth deprojection. (Gracefully falls back to projected pixels on 2D webcams, explicitly labeled).
- **Non-Diagnostic Reporting:** Employs strictly descriptive terminology such as *"Observed lateral asymmetry"* rather than diagnostic labels.

### Module 3: Basic Gait Observation
- **Lower-Limb Kinematic Tracking:** Bilateral Hip, Knee, and Ankle kinetic chains.
- **Continuous Kinematic Cycles:** Detects flexion/extension oscillations, step cycle count estimates, and session duration.
- **Strictly Descriptive:** Formatted as basic gait observation for clinician review; does not attempt disease classification (e.g. Parkinson's or antalgic gait).

---

## 3. Hardware Architecture & Depth Pipeline

### Primary Hardware: Intel RealSense™ D405
The Intel RealSense D405 is an optical-depth sub-millimeter camera designed for high-resolution close-range inspection (ideal operating range: 7 cm to 50 cm, extensible to 1.5 m).

```
Intel RealSense D405
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
Color Stream (RGB)             Depth Stream (Z16)
(640x480 @ 30fps)              (640x480 @ 30fps)
   │                               │
   └───────────────┬───────────────┘
                   ▼
       Hardware Stream Alignment
          rs.align(color)
                   │
                   ▼
      MediaPipe Tasks Vision
        Landmark Extraction
       (u, v) Pixel Coordinates
                   │
                   ▼
    Neighborhood Median Depth Filter
      (5x5 / 7x7 Patch Sampling)
                   │
                   ▼
      Pinhole Back-Deprojection
       (X, Y, Z) 3D Metric Point
                   │
                   ▼
      3D Euclidean Vector Kinematics
```

### Depth Robustness & Fault Tolerance
1. **Neighborhood Median Filtering:** Rather than trusting a single depth pixel (susceptible to infrared specular reflection, edge dropouts, or sensor noise), MOTION DX takes a $5 \times 5$ or $7 \times 7$ window around each landmark and computes the median of valid depths ($0.05\text{ m} < Z < 5.0\text{ m}$).
2. **Missing Depth Fallback:** If depth is obstructed or invalid, the measurement engine automatically falls back to 2D projection vectors without dropping frames or crashing.
3. **Camera Disconnection Immunity:** State machines gracefully recover from USB transient drops and display an intuitive repositioning prompt.

---

## 4. Software Architecture

```
motion_dx/
├── app.py                     # Streamlit application entrypoint & routing
├── camera/                    # Hardware sensor drivers
│   ├── base.py                # Abstract BaseCamera and CameraFrame models
│   ├── realsense.py           # Intel RealSense D405 driver with stream alignment
│   ├── webcam.py              # Fallback OpenCV UVC webcam driver
│   └── demo_camera.py         # Synthetic kinematic generator for zero-hardware demo
├── vision/                    # Computer vision & landmark detection
│   ├── pose.py                # MediaPipe Pose Landmarker (Tasks API)
│   ├── face.py                # MediaPipe Face Landmarker (Tasks API)
│   └── renderer.py            # High-visibility clinical green skeleton renderer
├── measurements/              # Kinematics mathematics (separated from UI)
│   ├── angles.py              # 2D & 3D vector angle algorithms
│   ├── distance.py            # 2D/3D Euclidean distance & perpendicular line deviation
│   ├── rom.py                 # Dynamic Range of Motion tracking engine
│   └── symmetry.py            # Bilateral symmetry index calculations
├── assessments/               # Assessment session controllers
│   ├── base.py                # Session data models & lifecycle state machines
│   ├── limb_rom.py            # Knee, Elbow, and Shoulder ROM assessment
│   ├── tmd.py                 # 5-step guided TMD/Jaw assessment protocol
│   └── gait.py                # Basic lower-limb gait observation
├── reports/                   # Clinical documentation exporters
│   ├── csv_report.py          # Timestamped time-series CSV export
│   └── pdf_report.py          # Publication-quality ReportLab clinical PDF report
├── validation/                # Empirical calibration & validation mode
│   └── validation.py          # System vs. Reference MAE calculation engine
├── storage/                   # 100% Local data persistence
│   └── session_store.py       # JSON session archive manager (no cloud)
├── ui/                        # Clinical user interface components
│   ├── styles.py              # Medical CSS theme & typography tokens
│   ├── home.py                # Clinical dashboard
│   ├── camera_check.py        # Hardware diagnostics & positioning guide
│   ├── assessment_view.py     # Live assessment split-screen UI
│   ├── results_view.py        # Post-assessment review & notes
│   └── validation_view.py     # Validation benchmarking screen
├── models/                    # MediaPipe .task bundle assets
│   ├── download_models.py     # Automated model downloader
│   ├── pose_landmarker.task   # MediaPipe Pose Task Model
│   └── face_landmarker.task   # MediaPipe Face Task Model
├── config/
│   └── references.json        # Configurable reference ranges (strictly annotated)
├── tests/                     # Automated unit test suite
│   ├── test_angles.py
│   ├── test_distance.py
│   ├── test_rom.py
│   ├── test_symmetry.py
│   └── test_safety.py
├── requirements.txt
└── README.md
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 (64-bit)
- Windows 10/11, macOS, or Linux
- USB 3.0 / 3.1 port (required for Intel RealSense D405 high-bandwidth depth streaming)

### Step 1: Clone or Navigate to Project
```bash
cd motion_dx
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify or Download Model Assets
```bash
python models/download_models.py
```
*(The models `pose_landmarker.task` and `face_landmarker.task` are downloaded directly into the `models/` directory).*

### Step 4: Run Automated Verification Tests
```bash
python -m unittest discover tests -v
```
All 22 unit tests will execute and confirm angle math, median depth filtering, and safety assertions.

---

## 6. Running the Application

Launch the Streamlit clinical interface:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Operating Modes:
1. **Intel RealSense D405 (Hardware Connected):**
   Select `📷 Intel RealSense D405 (RGB-D)` from the sidebar to activate the sub-millimeter depth pipeline.
2. **Standard Webcam (Fallback):**
   If no D405 is plugged in, select `💻 Fallback Webcam (2D)` to run the full pipeline using 2D projective geometry.
3. **Demo Mode (Zero-Hardware Synthetic):**
   Select `🎮 Demo Mode (Synthetic Feed)` to evaluate or present the complete kinematic pipeline, charts, and report generation without requiring any physical camera. Watermarked as **DEMO MODE — SYNTHETIC DATA**.

---

## 7. Clinical Measurement Methodology

### Angle Calculations
For a joint vertex $B$ with proximal landmark $A$ and distal landmark $C$:
$$\mathbf{v}_1 = A - B, \quad \mathbf{v}_2 = C - B$$
$$\theta = \arccos\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}\right) \times \frac{180^\circ}{\pi}$$

When aligned depth is available, $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^3$ are calculated using metric 3D coordinates.

### Bilateral Symmetry Index (SI)
$$\text{Difference} = |L - R|$$
$$\text{Symmetry Index (\%)} = \max\left(0, 100 - \frac{|L - R|}{\frac{L + R}{2}} \times 100\right)$$

### Midline Lateral Excursion (TMD)
The facial midline is modeled as the directional vector $\mathbf{m}$ passing from the Glabella to the Subnasale. The lateral excursion of the Menton (chin point $P$) is calculated as the signed perpendicular distance:
$$d_{\perp} = \frac{\mathbf{m}_x(P_y - m_{1y}) - \mathbf{m}_y(P_x - m_{1x})}{\|\mathbf{m}\|}$$

---

## 8. Validation Methodology & Benchmarking

MOTION DX includes an **Empirical Validation Mode** allowing clinical researchers to enter paired measurements against gold-standard tools (e.g., manual universal goniometers, optical mo-cap).

The system calculates:
$$\text{Mean Absolute Error (MAE)} = \frac{1}{N}\sum_{i=1}^N |y_{\text{system}, i} - y_{\text{reference}, i}|$$
$$\text{Root Mean Square Error (RMSE)} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_{\text{system}, i} - y_{\text{reference}, i})^2}$$

> **Important:** The system displays accuracy metrics **strictly computed from entered empirical trials**, avoiding fabricated accuracy claims.

---

## 9. Privacy & Local Storage

- **100% Local Device Processing:** Camera frames and depth arrays are processed strictly in RAM on the local workstation.
- **No Cloud Transmission:** No images, video recordings, or patient identifiers are ever sent to external cloud APIs, Firebase, or remote servers.
- **Local Persistence:** Session summaries and time-series are stored locally in `data/sessions/*.json`.

---

## 10. Clinical Safety Statement

> ### ⚠️ MANDATORY CLINICAL NOTICE
> 1. **Non-Diagnostic:** MOTION DX is a clinical-assistance and research quantification prototype. It does **NOT** provide a medical diagnosis, identify specific pathologies, or formulate medical prescriptions.
> 2. **Clinician Interpretation Required:** All outputs are designated as **OBSERVED MEASUREMENTS** and are intended to assist a qualified, licensed clinician who exercises independent clinical judgment.
> 3. **Non-Standardized Reference Values:** Reference ranges displayed in configuration files are derived from general literature and must not be used as clinical diagnostic cutoffs.
