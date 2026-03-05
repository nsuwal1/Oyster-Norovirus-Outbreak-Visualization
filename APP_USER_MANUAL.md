# Oyster Linked Norovirus Outbreak Prediction Dashboard
## User Manual and Supplementary Material

---

## Table of Contents
1. [Introduction](#introduction)
2. [Overview](#overview)
3. [System Requirements](#system-requirements)
4. [Installation & Setup](#installation--setup)
5. [Application Features](#application-features)
6. [About Data Tab](#about-data-tab)
7. [Run Model Tab](#run-model-tab)
8. [Results Tab](#results-tab)
9. [Model Details](#model-details)
10. [Data Format Requirements](#data-format-requirements)
11. [Interpretation Guide](#interpretation-guide)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)

---

## Introduction

The **Oyster Linked Norovirus Outbreak Prediction Dashboard** is an interactive web-based tool designed to predict the probability of norovirus outbreaks in oyster harvesting areas using hydro-meteorological and spatial indicators. This application integrates machine learning (LightGBM) with particle swarm optimization (PSO) to provide accurate, real-time outbreak risk predictions.

### Purpose
- Predict norovirus outbreak probability (0-1 scale)
- Support decision-making for oyster harvesting management
- Provide visual analysis of temporal and spatial risk patterns
- Enable threshold-based outbreak classification

### Intended Users
- Shellfish safety managers
- Public health officials
- Epidemiological researchers
- Oyster harvesting area managers

---

## Overview

### Key Features

1. **Interactive Dashboard** - User-friendly web interface built with Streamlit
2. **Real-time Predictions** - Generate outbreak probability scores for new data
3. **Temporal Visualization** - Animated time-series of outbreak risk
4. **Spatial Visualization** - Interactive maps showing regional risk assessment
5. **Multi-Region Support** - Predictions for BC, WA, FR, and LA regions
6. **Customizable Threshold** - Adjust outbreak classification threshold (0-1)

### Application Architecture

- **Backend**: Python 3.8+, LightGBM, scikit-learn
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Geopandas

---

## System Requirements

### Hardware
- **Processor**: 2+ GHz dual-core processor
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: 500 MB free space
- **Display**: 1366 x 768 minimum resolution

### Software
- Python 3.8 or higher
- pip (Python package manager)
- Web browser (Chrome, Firefox, Safari, Edge - latest versions)

### Python Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
joblib>=1.3.0
lightgbm>=4.0.0
plotly>=5.0.0
scikit-learn>=1.0.0
geopandas>=0.12.0
shapely>=2.0.0
pillow>=9.0.0
```

---

## Installation & Setup

### Step 1: Install Python
Download and install Python 3.8+ from https://www.python.org/

### Step 2: Clone or Download Application Files
Ensure you have the following files in your working directory:
- `app3.py` or `Visualization_10.py` (main application file)
- `models/lightgbm_model_20260302_150003.pkl` (trained model)
- `models/lightgbm_model_20260302_150003_params.json` (model parameters)
- `models/threshold.json` (default outbreak threshold)
- `Testing.csv` (sample data for predictions)
- `lightgbm_PSO.txt` (model documentation)
- Shapefile data (BC.tif, WA.tif, FR.tif, LA.tif)

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy joblib lightgbm plotly scikit-learn geopandas shapely pillow
```

### Step 4: Run the Application
```bash
streamlit run app3.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## Application Features

### Main Navigation
The application consists of three main sections, accessible via the sidebar:

1. **About Data** - Documentation and data information
2. **Run Model** - Prediction interface
3. **Results** - Analysis and visualization of predictions

---

## About Data Tab

### Overview Sub-tab

Provides general information about the application including:

- **Objective**: Predict norovirus outbreak probability using hydro-meteorological indicators
- **Data Sources**: References to USGS, NOAA, NASA POWER, and other agencies
- **Target Variable**: Binary indicator (1 = outbreak, 0 = no outbreak)
- **Model Type**: LightGBM classifier with PSO-optimized hyperparameters
- **Decision Rule**: Outbreak confirmed if Probability ≥ Threshold

**Key Data Sources:**
| Data Type | Organization | Source |
|-----------|--------------|--------|
| Gage Height | USGS, Canadian Water Office | https://apps.usgs.gov/nwismapper/ |
| Precipitation | NOAA, LSU AgCenter | https://weather.lsuagcenter.com/ |
| Temperature | NASA POWER | https://power.larc.nasa.gov/ |
| Salinity | CDMO | https://cdmo.baruch.sc.edu/dges/ |
| Location | Manual Entry | Latitude/Longitude coordinates |

### Input Variables Sub-tab

Lists all model input features organized by category:

#### Main Variables (Raw Features)
- Solar Radiation (W/m²)
- Sea Surface Temperature (°C)
- Gage Height (m)
- Precipitation (mm)
- Salinity (ppt)
- Latitude (decimal degrees)
- Longitude (decimal degrees)

#### Derived Features (Time-Aggregated)

**Solar Radiation Features:**
- SR1: Mean solar radiation from 4-29 days prior
- SR2: Mean solar radiation from 14-30 days prior

**Temperature Features:**
- T1: Mean maximum temperature from 14-30 days prior
- T2: Mean average temperature from 14-21 days prior
- T3: Temperature fluctuation 2 days prior
- T4: Average water temperature 30 days prior

**Gage Height Features:**
- GH1: Mean gage height from 4-30 days prior
- GH2: Average gage height 2 days prior
- GH3: Gage height variation 17 days prior
- GH4: Minimum gage height difference (days 11-12 prior)

**Rainfall Features:**
- R1: Total rainfall from 4-9 days prior
- R2: Cumulative rainfall in 10 days prior

**Salinity Features:**
- S1: Total daily average salinity from 4-29 days prior
- S2: Daily average salinity 30 days prior

**Location Features:**
- Latitude
- Longitude
- Region (BC, WA, FR, LA)

### Data Preprocessing Sub-tab

Details the normalization process applied to all continuous features:

#### Min-Max Normalization (0-1 Scaling)

**Formula:**
$$X_{\text{normalized}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

**Parameters:**
- X = original feature value
- X_min = minimum value observed in training data
- X_max = maximum value observed in training data
- X_normalized = scaled value in range [0, 1]

**Benefits:**
1. Ensures uniform feature scaling
2. Prevents large-magnitude features from dominating the model
3. Improves model convergence and prediction accuracy
4. Preserves the shape of the original distribution
5. Enables fair comparison between features

**Example:**
If Solar Radiation ranges from 0 to 300 W/m² in the training data:
- A value of 150 W/m² becomes: (150 - 0) / (300 - 0) = 0.5
- A value of 75 W/m² becomes: (75 - 0) / (300 - 0) = 0.25
- A value of 225 W/m² becomes: (225 - 0) / (300 - 0) = 0.75

### Global Oyster Harvesting Areas Sub-tab

Links to regional oyster harvesting area maps and boundaries:

- **Washington**: https://fortress.wa.gov/doh/oswpviewer/index.html
- **Louisiana**: https://ladhh.maps.arcgis.com/apps/webappviewer/
- **British Columbia**: https://egisp.dfo-mpo.gc.ca/vertigisstudio/web/
- **Global Oyster Atlas**: https://symbio6.nl/en/apps/oyster-map.html

---

## Run Model Tab

### Data Upload Interface

1. **Threshold Slider** (0.0 - 1.0)
   - Default: Model-optimized threshold
   - Adjust to change outbreak classification probability
   - Lower threshold → More sensitive (higher recall)
   - Higher threshold → More specific (higher precision)

2. **Testing Data Preview**
   - Displays first 5 rows of input data
   - Shows all columns including Date, ID, Region, and features

3. **Run Prediction Button**
   - Click to generate predictions
   - Processing time: Typically 1-5 seconds
   - Results saved to `norovirus_predictions.csv`

### Input Data Requirements

**Required Columns:**
- Date (YYYY-MM-DD format)
- ID (unique event identifier)
- Region (BC, WA, FR, or LA)
- All 16 feature columns: SR1, SR2, T1, T2, T3, T4, GH1, GH2, GH3, GH4, R1, R2, S1, S2, Lat, Long

**Data Format:**
- CSV file with headers
- Numeric values (decimals acceptable)
- Missing values: Use blank cells or NaN
- Record one observation per row

**Example CSV Structure:**
```
Date,ID,Region,SR1,SR2,T1,T2,T3,T4,GH1,GH2,GH3,GH4,R1,R2,S1,S2,Lat,Long
2024-01-15,OY001,BC,180.5,175.2,12.3,11.8,2.1,10.5,1.2,1.15,0.05,0.08,15.3,45.2,28.5,29.1,49.25,-123.45
2024-01-20,OY002,WA,195.3,188.7,14.1,13.2,1.8,12.3,1.5,1.42,0.08,0.10,12.5,38.7,30.2,31.5,47.25,-124.15
```

### Output Files

**Prediction CSV**: `norovirus_predictions.csv`

Contains original input data plus:
- `Outbreak_Probability`: Probability score (0-1)
- `Outbreak_Flag`: Binary classification (0 = No, 1 = Yes)

---

## Results Tab

### Result Display Options

#### Filters

1. **Filter by Region**
   - Select specific region: BC, WA, FR, LA
   - Default: "All" (shows all regions)
   - Dynamically updates other filters

2. **Filter by Event (ID)**
   - Select specific event ID
   - Default: "All"
   - Shows all events within selected region

3. **Filter by Date**
   - Date picker showing date range
   - Only available if Date column present
   - Defaults to earliest date in dataset

### Data Preview

- **Prediction Output Table** - Shows first 5 rows of filtered results
- **Columns displayed**: All input features plus Outbreak_Probability and Outbreak_Flag

### Temporal Analysis

**Time Series Plot** - Available when Date column present

Shows:
- **X-axis**: Date (chronological)
- **Y-axis**: Outbreak Probability (0-1)
- **Blue line**: Predicted probability trend
- **Red marker**: Current date selection
- **Black dashed line**: Outbreak threshold

**Interactive Features:**
- Hover to see exact probability and status
- Click legend items to toggle traces
- Zoom in/out by dragging and scrolling
- Animation controls:
  - ▶ Start Animation: Auto-play through dates
  - ⏸ Pause Animation: Freeze at current date
- Manual date selection via date picker

**Animation Details:**
- Frame duration: 700ms per date
- Auto-loops after reaching end
- Synchronized with spatial map

### Spatial Analysis

**Interactive Map** - Available when Lat/Long columns present

Shows:
- **Map style**: OpenStreetMap
- **Color coding**: Rectangle areas showing outbreak probability
  - Green: Low probability (0-0.25)
  - Yellow: Medium probability (0.25-0.75)
  - Red: High probability (0.75-1.0)
- **Text labels**: Exact probability values at location centers

**Interactive Features:**
- **Hover**: Display location details and probability
- **Zoom**: Mouse wheel or pinch to zoom
- **Pan**: Click and drag to move map
- **Frame selector**: Slider to change time frame
- **Animation controls**:
  - ▶ Play Spatial Animation: Visualize risk evolution
  - ⏸ Pause Spatial Animation: Freeze current view
- **Risk indicator**: Shows average risk for current time frame

### Color Gradient Guide

| Color | Probability | Risk Level |
|-------|-------------|-----------|
| 🟢 Green | 0.0 - 0.25 | Low Risk |
| 🟡 Yellow | 0.25 - 0.75 | Medium Risk |
| 🔴 Red | 0.75 - 1.0 | High Risk |

---

## Model Details

### LightGBM Classifier

**Model Type:** Gradient Boosting Decision Tree (GBDT)

**Key Attributes:**
- Framework: LightGBM (Light Gradient Boosting Machine)
- Task: Binary Classification
- Loss Function: Binary Cross-entropy
- Optimization: Early stopping with validation set

**Advantages:**
- Fast training and prediction
- Handles mixed feature types
- Built-in feature importance
- Memory efficient
- Interpretable feature relationships

### Hyperparameter Optimization

**Method:** Particle Swarm Optimization (PSO)

**Optimized Parameters:**
- num_leaves: Number of leaves in decision trees
- learning_rate: Shrinkage parameter (eta)
- n_estimators: Number of boosting rounds
- subsample: Row sampling for stochastic boosting
- colsample_bytree: Feature sampling per tree
- reg_alpha: L1 regularization
- reg_lambda: L2 regularization

**Optimization Process:**
1. PSO explores hyperparameter space with multiple particles
2. Each particle evaluates candidate parameters on validation set
3. Particles update positions based on best performance
4. Process iterates for specified number of generations
5. Best-performing hyperparameters selected for final model

### Model Performance Metrics

**Standard Metrics:**
- AUC-ROC: Area Under Receiver Operating Characteristic Curve
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: Harmonic mean of precision and recall
- Accuracy: (TP + TN) / Total

**Optimal Threshold:**
- Determined using Matthews Correlation Coefficient (MCC)
- Balances true positives and true negatives
- Default: Model-optimized value (adjustable in app)

### Model Calibration

- **Probability calibration**: Ensures predicted probabilities match empirical frequencies
- **Output range**: Calibrated to [0, 1] scale
- **Interpretability**: Probability directly reflects outbreak likelihood

---

## Data Format Requirements

### Input Data Specifications

#### Column Headers (Required)
```
Date, ID, Region, SR1, SR2, T1, T2, T3, T4, GH1, GH2, GH3, GH4, R1, R2, S1, S2, Lat, Long
```

#### Data Types

| Column | Type | Range | Unit | Format |
|--------|------|-------|------|--------|
| Date | String | Any | - | YYYY-MM-DD |
| ID | String | Any | - | Alphanumeric |
| Region | String | BC, WA, FR, LA | - | 2 characters |
| SR1, SR2 | Float | 0-450 | W/m² | Decimal |
| T1, T2, T3, T4 | Float | -30 to 40 | °C | Decimal |
| GH1, GH2, GH3, GH4 | Float | -5 to 10 | m | Decimal |
| R1, R2 | Float | 0-500 | mm | Decimal |
| S1, S2 | Float | 0-40 | ppt | Decimal |
| Lat | Float | 30-55 | Degrees | Decimal (e.g., 49.25) |
| Long | Float | -130 to -60 | Degrees | Decimal (e.g., -123.45) |

#### Data Quality Standards

- **Missing values**: Should be rare (<5% per column)
- **Text encoding**: UTF-8
- **File size**: Recommended <10 MB for optimal performance
- **Records**: Up to 10,000 rows per prediction run recommended

### File Format

**Accepted format**: CSV (Comma-separated values)

**Example file structure:**
```csv
Date,ID,Region,SR1,SR2,T1,T2,T3,T4,GH1,GH2,GH3,GH4,R1,R2,S1,S2,Lat,Long
2024-01-15,EVENT_001,BC,181.2,175.8,12.5,11.9,2.0,10.6,1.18,1.16,0.05,0.08,14.8,44.5,28.3,29.0,49.25,-123.45
2024-01-16,EVENT_001,BC,185.3,180.2,13.1,12.4,1.9,11.2,1.22,1.19,0.06,0.09,13.2,42.1,28.5,29.2,49.25,-123.45
2024-01-20,EVENT_002,WA,192.5,187.3,14.3,13.5,1.8,12.5,1.52,1.45,0.08,0.11,11.6,37.8,30.4,31.6,47.25,-124.15
```

---

## Interpretation Guide

### Understanding Outbreak Probability

**Probability Scale:**

| Probability | Interpretation | Recommended Action |
|-------------|-----------------|-------------------|
| 0.0 - 0.20 | Very Low Risk | Continue normal monitoring |
| 0.20 - 0.40 | Low Risk | Maintain standard precautions |
| 0.40 - 0.60 | Moderate Risk | Increase surveillance frequency |
| 0.60 - 0.80 | High Risk | Implement enhanced monitoring |
| 0.80 - 1.00 | Very High Risk | Consider harvesting restrictions |

### Threshold Decision-Making

**Lower Threshold (e.g., 0.30):**
- Pros: Catches more outbreaks (higher sensitivity)
- Cons: More false alarms (lower specificity)
- Use when: Preventing outbreaks is priority

**Higher Threshold (e.g., 0.70):**
- Pros: Fewer false alarms (higher specificity)
- Cons: May miss some outbreaks (lower sensitivity)
- Use when: Avoiding operational disruptions is priority

**Balanced Threshold (Default):**
- Maximizes overall accuracy
- Best for general-purpose monitoring
- Recommended for most users

### Temporal Patterns

When viewing the time series animation:

1. **Upward trends** = Increasing outbreak risk
   - Monitor more frequently
   - Prepare mitigation plans

2. **Downward trends** = Decreasing outbreak risk
   - Risk subsiding
   - Resume normal operations gradually

3. **Sudden spikes** = Abrupt risk increase
   - Often corresponds to environmental changes
   - Investigate contributing factors

4. **Seasonal patterns** = Recurring cycles
   - Common in oyster-related diseases
   - Plan interventions accordingly

### Spatial Patterns

When viewing the interactive map:

1. **Red concentrated areas** = Geographic hotspots
   - High-risk zones requiring close monitoring
   - May need targeted interventions

2. **Green widespread areas** = Low-risk zones
   - Safe for normal harvesting
   - Standard monitoring sufficient

3. **Mixed colors** = Heterogeneous risk
   - Different areas have different risks
   - Zone-specific management recommended

4. **Map evolution** = Risk migration
   - Watch how risk moves spatially over time
   - Important for predicting spread patterns

### Feature Importance

Model importance ranking (in descending order):
- Environmental factors (temperature, precipitation) - High impact
- Hydrological factors (gage height, salinity) - High impact
- Solar radiation - Moderate impact
- Location (latitude/longitude) - Context-dependent impact
- Region - Context-dependent impact

---

## Troubleshooting

### Common Issues & Solutions

#### Issue 1: Application Won't Start
**Symptom:** Error when running `streamlit run app3.py`

**Solutions:**
1. Verify Python installation: `python --version`
2. Reinstall dependencies: `pip install -r requirements.txt --upgrade`
3. Check for port conflicts: Default port 8501
4. Try alternate port: `streamlit run app3.py --server.port 8502`

#### Issue 2: Model File Not Found
**Symptom:** "Model file not found: models/lightgbm_model_20260302_150003.pkl"

**Solutions:**
1. Verify model file exists in `models/` directory
2. Check file name matches exactly (case-sensitive)
3. Ensure path is correct relative to app location
4. Verify file is not corrupted: Check file size (should be >1 MB)

#### Issue 3: Predictions Fail
**Symptom:** "Model prediction failed" error message

**Solutions:**
1. **Check input data**:
   - All required columns present
   - No empty rows
   - Numeric values in numeric columns
   
2. **Verify feature consistency**:
   - Column names match exactly
   - Data types correct (numbers, not text)
   
3. **Check data ranges**:
   - Values within expected ranges
   - No extreme outliers causing issues

4. **Validate CSV format**:
   - Opened and re-saved in Excel/Pandas
   - Removed any special characters
   - Used UTF-8 encoding

#### Issue 4: Memory Error
**Symptom:** "Memory error" or application crashes with large datasets

**Solutions:**
1. Reduce dataset size (process in batches)
2. Close other applications to free RAM
3. Upgrade to machine with more RAM
4. Process data in smaller chunks (<5000 rows)

#### Issue 5: Visualizations Not Displaying
**Symptom:** Maps or charts appear blank or don't load

**Solutions:**
1. Check internet connection (needed for map tiles)
2. Clear browser cache: Ctrl+Shift+Delete (Chrome)
3. Try different browser
4. Verify Date/Lat/Long columns are present
5. Ensure Lat/Long values are valid (decimal degrees)

#### Issue 6: Slow Performance
**Symptom:** App is laggy or animations stutter

**Solutions:**
1. Close unnecessary browser tabs
2. Reduce animation frame duration (modify code if needed)
3. Use smaller dataset for testing
4. Upgrade browser to latest version
5. Reduce map zoom level

### Error Messages Reference

| Error | Cause | Solution |
|-------|-------|----------|
| ModuleNotFoundError | Missing Python package | Run: `pip install [package]` |
| FileNotFoundError | Missing data/model file | Check file path and name |
| ValueError: Shape mismatch | Wrong number of features | Verify all 16+ features are present |
| TypeError: Cannot convert | Data type error | Ensure numeric columns are numbers |
| MemoryError | Insufficient RAM | Reduce dataset size |
| Connection timeout | Internet/map server issue | Check connection, try later |

---

## FAQ

### General Questions

**Q1: What is the purpose of this dashboard?**
A: To predict norovirus outbreak probability in oyster harvesting areas and support management decisions about harvesting, testing, and closure decisions.

**Q2: How accurate are the predictions?**
A: The model achieves AUC-ROC of [insert value] on validation data. Accuracy depends on input data quality and feature values representative of conditions where outbreaks occurred.

**Q3: Can I use this for other diseases?**
A: This model is specific to norovirus in oysters. Different diseases/organisms would require retraining with appropriate data.

**Q4: How often should I run predictions?**
A: Frequency depends on your management needs. Recommendations: Daily or weekly during high-risk seasons; Monthly during low-risk periods.

**Q5: Is historical data included?**
A: Testing.csv contains sample historical data. Users can replace with their own recent data for predictions.

---

### Data Questions

**Q6: What if my data is missing some features?**
A: The model requires all 16 features. Missing values should be imputed using:
- Mean/median of available values
- Forward-fill from time series
- Domain expert estimates

**Q7: Can I use different date formats?**
A: The app expects YYYY-MM-DD format. Reformat before uploading if needed using Excel or Python.

**Q8: What coordinate system should I use?**
A: Decimal degrees (latitude/longitude). Example: 49.25°N, -123.45°W

**Q9: How precise do coordinates need to be?**
A: At least 2 decimal places (±1.1 km precision) recommended. More precision (4+ decimals) improves spatial analysis.

**Q10: Can I update the model with new data?**
A: Yes, the lightgbm_PSO.txt file contains the training code. Rerun with new data to retrain and regenerate the .pkl model file.

---

### Technical Questions

**Q11: What Python version do I need?**
A: Python 3.8 or higher. Version 3.10+ recommended for best performance.

**Q12: Can I deploy this on a server?**
A: Yes. Use Streamlit Cloud (free tier available) or deploy on your own Linux/Windows server.

**Q13: How do I modify the threshold?**
A: Use the slider in "Run Model" tab to adjust threshold (0.0-1.0) before running predictions.

**Q14: Can I batch process multiple datasets?**
A: Currently, the app processes one CSV at a time. For batch processing, modify app code to loop through multiple files.

**Q15: How do I export results?**
A: Results are automatically saved to `norovirus_predictions.csv` after each prediction run.

---

### Results & Interpretation

**Q16: What does the red/yellow/green color mean on the map?**
A: Green = low probability (safe), Yellow = medium probability (caution), Red = high probability (high risk).

**Q17: Can I zoom into specific regions on the map?**
A: Yes, use mouse wheel (scroll up to zoom in, down to zoom out) or pinch gesture on touch devices.

**Q18: What does the animation show?**
A: The animation displays how outbreak probability changes over time, showing temporal evolution of outbreak risk.

**Q19: Why does my prediction differ from last week's?**
A: Changes in environmental conditions (temperature, precipitation, gage height) directly affect outbreak probability estimates.

**Q20: Should I always follow the outbreak flag?**
A: The outbreak flag is a recommendation tool. Combine with expert judgment, local knowledge, and monitoring data for final decisions.

---

### Troubleshooting Help

**Q21: The app won't open. What's wrong?**
A: See Troubleshooting section above. Most common: Port 8501 already in use or Python environment issues.

**Q22: How do I update to a new version?**
A: Download latest app files and replace old ones. Keep your data and model files separate from app code.

**Q23: Can I run this offline?**
A: Yes, except the map requires internet for map tiles. All predictions and analysis work offline.

**Q24: Where can I get sample data?**
A: The Testing.csv file in the workspace contains sample data with proper formatting and feature values.

**Q25: Who should I contact for issues?**
A: Document the error message and steps to reproduce, contact the development team or refer to the documentation.

---

## References

### Key Literature
1. [Insert relevant norovirus outbreak literature]
2. [Insert machine learning model development papers]
3. [Insert oyster-disease surveillance references]

### Data Sources
- USGS Water Data: https://apps.usgs.gov/nwismapper/
- NASA POWER: https://power.larc.nasa.gov/data-access-viewer/
- NOAA Tides & Currents: https://tidesandcurrents.noaa.gov/
- CDMO Salinity Data: https://cdmo.baruch.sc.edu/dges/

### Tools & Libraries
- Streamlit: https://streamlit.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Plotly: https://plotly.com/
- Geopandas: https://geopandas.org/

---

## Appendices

### Appendix A: Quick Reference Card

**Starting the app:**
```bash
streamlit run app3.py
```

**Default address:** http://localhost:8501

**Main tabs:** About Data | Run Model | Results

**Required features:** Date, ID, Region, SR1, SR2, T1, T2, T3, T4, GH1, GH2, GH3, GH4, R1, R2, S1, S2, Lat, Long

**Output file:** norovirus_predictions.csv

---

### Appendix B: Data Preparation Template

```python
import pandas as pd
from datetime import datetime

# Load your raw data
df = pd.read_csv('your_data.csv')

# Ensure required columns
required_cols = ['Date', 'ID', 'Region', 'SR1', 'SR2', 'T1', 'T2', 'T3', 'T4', 
                 'GH1', 'GH2', 'GH3', 'GH4', 'R1', 'R2', 'S1', 'S2', 'Lat', 'Long']

# Format date to YYYY-MM-DD
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# Remove rows with missing features (keep ID, Region)
df = df.dropna(subset=['SR1', 'SR2', 'T1', 'T2', 'T3', 'T4', 'GH1', 'GH2', 'GH3', 'GH4', 'R1', 'R2', 'S1', 'S2', 'Lat', 'Long'])

# Save as CSV
df.to_csv('prepared_data.csv', index=False)
```

---

### Appendix C: File Structure

```
project_folder/
├── app3.py                              # Main application file
├── Visualization_10.py                  # Alternative app file
├── requirements.txt                     # Python dependencies
├── Testing.csv                          # Sample input data
├── norovirus_predictions.csv           # Output (generated after prediction)
├── lightgbm_PSO.txt                    # Model training code
├── APP_USER_MANUAL.md                  # This file
├── models/
│   ├── lightgbm_model_20260302_150003.pkl          # Trained model
│   ├── lightgbm_model_20260302_150003_params.json  # Model parameters
│   └── threshold.json                   # Default threshold value
├── BC.tif                               # British Columbia study area
├── WA.tif                               # Washington study area
├── FR.tif                               # France study area
├── LA.tif                               # Louisiana study area
└── Shapefiles/
    ├── British Columbia.shp/.shx/.dbf
    ├── WA.shp/.shx/.dbf
    └── ...
```

---

## Version Information

**Application Version:** 2.0  
**Build Date:** March 5, 2026  
**Model Training Date:** March 2, 2026  
**Last Updated:** March 5, 2026

---

## License & Citation

[Insert your license information here]

**Suggested Citation:**
```
[Your Name(s)] (2026). Oyster Linked Norovirus Outbreak Prediction Dashboard v2.0. 
[Institution]. [URL if applicable]
```

---

## Contact & Support

**For technical support, feature requests, or bug reports:**
- Email: [your.email@institution.edu]
- Documentation: APP_USER_MANUAL.md
- Code Repository: [GitHub/repository link if applicable]

---

**End of User Manual**

---
