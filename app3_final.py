import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components
from pathlib import Path

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Oyster Linked Norovirus Outbreak Prediction Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_PATH = Path("models/lightgbm_model_20260302_150003.pkl")
PARAMS_PATH = Path("models/lightgbm_model_20260302_150003_params.json")
THRESHOLD_PATH = Path("models/threshold.json")
TEST_DATA_PATH = Path("Testing.csv")
OUTPUT_PATH = Path("norovirus_predictions.csv")

# --------------------------------------------------
# Load model & threshold
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found: models/lightgbm_model_20260302_150003.pkl")
        st.stop()
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict):
        # Handle common training artifacts where the model is stored in a dict
        for key in ("model", "estimator", "clf", "classifier", "lightgbm_model"):
            if key in loaded:
                return loaded[key]
        st.error(
            "Model file contains a dict but no known model key. "
            f"Available keys: {list(loaded.keys())}"
        )
        st.stop()
    return loaded

@st.cache_resource
def load_params():
    if not PARAMS_PATH.exists():
        st.error("Params file not found: models/lightgbm_model_20260302_150003_params.json")
        st.stop()
    with open(PARAMS_PATH, "r") as f:
        return json.load(f)

@st.cache_resource
def load_threshold():
    if not THRESHOLD_PATH.exists():
        st.error("Threshold file not found: models/threshold.json")
        st.stop()
    with open(THRESHOLD_PATH, "r") as f:
        return json.load(f)["threshold"]

model = load_model()
params = load_params()
default_threshold = load_threshold()

# --------------------------------------------------
# Prediction helper
# --------------------------------------------------
def predict_positive_proba(model_obj, features):
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(features)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.asarray(proba).ravel()
    if hasattr(model_obj, "predict"):
        preds = model_obj.predict(features)
        return np.asarray(preds).ravel()
    raise AttributeError("Model object has no predict or predict_proba method")

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["About Data", "Run Model", "Results"]
)

# --------------------------------------------------
# Global session storage
# --------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

# ==================================================
# PAGE 1: ABOUT DATA
# ==================================================
if page == "About Data":

    st.title("Oyster Linked Norovirus Outbreak Modeling Framework")

    about_tabs = st.tabs(["Overview", "Input Variables", "Data Preprocessing", "Global Oyster Harvesting Areas"])

    with about_tabs[0]:
        st.markdown("""
        ## Objective
        Predict the **probability of norovirus outbreaks (0–1)** using
        hydro-meteorological and location indicators.

        ## Data Sources
        - **Hydrology**: [USGS](https://apps.usgs.gov/nwismapper/) (gage height, Salinity)
        - **Meteorology**: [NOAA](https://tidesandcurrents.noaa.gov/), [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) (TMAX, TMIN, precipitation, solar radiation)
        - **Spatial**: Latitude & Longitude

        ### Additional Data Sources
        - **Salinity**: [CDMO](https://cdmo.baruch.sc.edu/dges/), [Canada Open Data](https://open.canada.ca/data/en/dataset/719955f2-bf8e-44f7-bc26-6bd623e82884), [Marine In Situ](https://marineinsitu.eu/dashboard/)
        - **Precipitation**: [LSU AgCenter](https://weather.lsuagcenter.com/), [Pacific Climate Data Portal](https://services.pacificclimate.org/met-data-portal-pcds/app/#close)
        - **Gage Height**: [Canadian Water Office](https://wateroffice.ec.gc.ca/), [SHOM Data](https://data.shom.fr/)
        - **Sea Surface Temperature**: [Climate Weather Canada](https://climate.weather.gc.ca/), [Geographic.org](https://geographic.org/)

        ## Target Variable
        - Binary outbreak indicator  
        - `1` = confirmed outbreak  
        - `0` = no outbreak  

        ## Model
        - LightGBM classifier
        - Hyperparameters optimized using particle swarm optimization (PSO)
        - Outputs calibrated probability ∈ [0, 1]

        ## Decision Rule
        - Outbreak confirmed if  
          **Probability ≥ Threshold**
        """)

        # Study Area Shapefiles
        st.subheader("🗺️ Study Area Maps")
        st.markdown("Click on each region to view and download the study area map:")
        
        study_areas = {
            "BC (British Columbia)": Path("BC.tif"),
            "LA (Gulf of Mexico)": Path("LA.tif"),
            "FR (Southeast France Coast)": Path("FR.tif"),
            "WA (Washington Coast)": Path("WA.tif")
        }
        
        col1, col2 = st.columns(2)
        
        for idx, (name, tif_path) in enumerate(study_areas.items()):
            col = col1 if idx % 2 == 0 else col2
            with col:
                with st.expander(f"📍 {name}"):
                    if tif_path.exists():
                        try:
                            from PIL import Image
                            img = Image.open(tif_path)
                            st.image(img, caption=name, use_container_width=True)
                            
                            with open(tif_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {tif_path.name}",
                                    data=f.read(),
                                    file_name=tif_path.name,
                                    mime="image/tiff",
                                    key=f"download_{tif_path.stem}"
                                )
                        except ImportError:
                            st.error("PIL/Pillow library required. Install with: pip install Pillow")
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                            with open(tif_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {tif_path.name}",
                                    data=f.read(),
                                    file_name=tif_path.name,
                                    mime="image/tiff",
                                    key=f"download_{tif_path.stem}"
                                )
                    else:
                        st.warning(f"{name} not found")

        # LightGBM PSO Code
        st.subheader("💻 Model Code")
        code_path = Path("lightgbm_PSO.txt")
        if code_path.exists():
            with open(code_path, "r", encoding="utf-8") as f:
                code_content = f.read()
            with st.expander("Click to view LightGBM PSO Code"):
                st.code(code_content, language="python")
        else:
            st.warning(f"Code file not found: {code_path}")

    with about_tabs[1]:
        st.subheader("Input Features")
        
        st.markdown("""
        ## Main Variables
        - Solar Radiation
        - Sea Surface Temperature
        - Gage Height
        - Precipitation
        - Salinity
        - Latitude
        - Longitude
        
        ---
        
        ## Derived Input Features
        
        ### Solar Radiation
        - **SR1**: Total mean solar radiation from 4–29 days prior to outbreak
        - **SR2**: Mean solar radiation from 14–30 days prior to outbreak
        
        ### Temperature
        - **T1**: Mean maximum temperature from 14–30 days prior to outbreak
        - **T2**: Mean average temperature from 14–21 days prior to outbreak
        - **T3**: Temperature fluctuation 2 days prior to outbreak
        - **T4**: Average water temperature 30 days prior to outbreak
        
        ### Gage Height
        - **GH1**: Total mean gage height from 4–30 days prior to outbreak
        - **GH2**: Average gage height 2 days prior to outbreak
        - **GH3**: Gage height variation 17 days prior to outbreak
        - **GH4**: Minimum gage height difference between 11 and 12 days prior to outbreak
        
        ### Rainfall
        - **R1**: Total rainfall from 4–9 days prior to outbreak
        - **R2**: Cumulative rainfall in 10 days prior to outbreak
        
        ### Salinity
        - **S1**: Total daily average salinity from 4–29 days prior to outbreak
        - **S2**: Daily average salinity 30 days prior to outbreak
        
        ### Location
        - **Latitude**
        - **Longitude**
        """)

    with about_tabs[2]:
        st.subheader("Data Preprocessing & Normalization")
        
        st.markdown("""
        ## Min-Max Normalization (Standardization)
        All continuous features are normalized using the **Min-Max scaling** method to ensure uniform feature importance during model training.
        
        **Formula:**
        
        $$X_{\\text{normalized}} = \\frac{X - X_{\\text{min}}}{X_{\\text{max}} - X_{\\text{min}}}$$
        
        **Where:**
        - $X$ = original feature value
        - $X_{\\text{min}}$ = minimum value of the feature
        - $X_{\\text{max}}$ = maximum value of the feature
        - $X_{\\text{normalized}}$ = normalized value in range [0, 1]
        
        **Benefits:**
        - Scales all features to a range of [0, 1]
        - Preserves the shape of the original distribution
        - Prevents features with larger scales from dominating the model
        - Improves model convergence and prediction accuracy
        
        **Example:**
        - If Solar Radiation ranges from 0 to 300 W/m²
        - A value of 150 W/m² becomes: $(150 - 0) / (300 - 0) = 0.5$
        - A value of 75 W/m² becomes: $(75 - 0) / (300 - 0) = 0.25$
        """)

    with about_tabs[3]:
        st.subheader("Global oyster harvesting areas")
        st.markdown(
            "Washington areas: https://fortress.wa.gov/doh/oswpviewer/index.html"
        )
        st.markdown(
            "Louisiana areas: https://ladhh.maps.arcgis.com/apps/webappviewer/index.html?id=a5577e62649d4242aeca3c2083280e5a"
        )
        st.markdown(
            "British Columbia areas: https://egisp.dfo-mpo.gc.ca/vertigisstudio/web/?app=84572c5703a24d0cbe062d2d7ba126d7&locale=en"
        )
        st.markdown(
            "Texas areas: https://tpwd.maps.arcgis.com/apps/webappviewer/index.html?id=d5e366e86a894a6cbe06ad70e9befd95"
        )
        st.markdown(
            "Florida areas: https://experience.arcgis.com/experience/00ad5ac144a9437ea94a6a809bbe9ef7/"
        )
        st.markdown(
            "Global Oyster Atlas: https://symbio6.nl/en/apps/oyster-map.html"
        )
        st.markdown(
            "Maryland areas: https://symbio6.nl/en/apps/oyster-map.html"
        )

# ==================================================
# PAGE 2: RUN MODEL
# ==================================================
elif page == "Run Model":

    st.title("Run Norovirus Outbreak Prediction")

    # Initialize session state for workflow
    if "imported_data" not in st.session_state:
        st.session_state["imported_data"] = None
    if "data_viewed" not in st.session_state:
        st.session_state["data_viewed"] = False

    # Threshold slider (always available)
    threshold = st.slider(
        "Outbreak Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01
    )

    st.markdown("---")

    # STEP 1: IMPORT DATA
    st.subheader("📥 Step 1: Import Data")
    col1, col2 = st.columns(2)

    with col1:
        import_option = st.radio("Choose data source:", ["Use default (Testing.csv)", "Upload CSV file"])

    with col2:
        if import_option == "Upload CSV file":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_upload")
            if uploaded_file is not None:
                try:
                    st.session_state["imported_data"] = pd.read_csv(uploaded_file)
                    st.success(f"✓ Data imported: {uploaded_file.name} ({len(st.session_state['imported_data'])} rows)")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            if not TEST_DATA_PATH.exists():
                st.error(f"Default test file not found: {TEST_DATA_PATH}")
            else:
                try:
                    st.session_state["imported_data"] = pd.read_csv(TEST_DATA_PATH)
                    st.success(f"✓ Default data loaded: {TEST_DATA_PATH} ({len(st.session_state['imported_data'])} rows)")
                except Exception as e:
                    st.error(f"Error loading default file: {e}")

    if st.button("✓ Import Data Confirmed", key="import_confirm"):
        if st.session_state["imported_data"] is not None:
            st.session_state["data_viewed"] = False
            st.success("Data imported successfully!")
        else:
            st.error("Please select data before confirming.")

    st.markdown("---")

    # STEP 2: VIEW DATA
    if st.session_state["imported_data"] is not None:
        st.subheader("👁️ Step 2: View Data")

        if st.button("📊 View Data Preview", key="view_data_btn"):
            st.session_state["data_viewed"] = True

        if st.session_state["data_viewed"]:
            st.write(f"**Total Rows:** {len(st.session_state['imported_data'])}")
            st.write(f"**Total Columns:** {len(st.session_state['imported_data'].columns)}")
            st.dataframe(st.session_state["imported_data"].head(10), use_container_width=True)

            # Show column info
            with st.expander("📋 Column Information"):
                st.write(st.session_state["imported_data"].dtypes)
                st.write(f"**Missing Values:**\n{st.session_state['imported_data'].isnull().sum()}")

        st.markdown("---")

        # STEP 3: RUN PREDICTION
        st.subheader("🚀 Step 3: Run Prediction")

        if st.button("▶️ Run Prediction", key="run_prediction"):
            if st.session_state["imported_data"] is None:
                st.error("Please import data first.")
            else:
                with st.spinner("Running prediction..."):
                    try:
                        df = st.session_state["imported_data"].copy()
                        features_df = df.copy()
                   
                        drop_cols = []
                        if "Date" in features_df.columns:
                            # LightGBM expects numeric features; drop Date for prediction.
                            features_df["Date"] = pd.to_datetime(features_df["Date"], errors="coerce")
                            drop_cols.append("Date")
                        if "ID" in features_df.columns:
                            drop_cols.append("ID")
                        if drop_cols:
                            features_df = features_df.drop(columns=drop_cols)

                        # Select only the features used in training
                        feature_names = params.get("feature_names", [])
                        if feature_names:
                            features_df = features_df[feature_names]

                        # Convert categorical features to category dtype
                        categorical_features = params.get("categorical_features", [])
                        for col in categorical_features:
                            if col in features_df.columns:
                                features_df[col] = features_df[col].astype('category')

                        probs = predict_positive_proba(model, features_df)

                        df["Outbreak_Probability"] = probs
                        df["Outbreak_Flag"] = (probs >= threshold).astype(int)

                        df.to_csv(OUTPUT_PATH, index=False)
                        st.session_state["results_df"] = df

                        st.success(f"✓ Prediction completed! Output saved to {OUTPUT_PATH}")
                        
                        # Show results summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", len(df))
                        with col2:
                            outbreak_count = (df["Outbreak_Flag"] == 1).sum()
                            st.metric("Outbreaks Detected", outbreak_count)
                        with col3:
                            avg_prob = df["Outbreak_Probability"].mean()
                            st.metric("Avg. Probability", f"{avg_prob:.3f}")
                        
                        st.subheader("📈 Prediction Results Preview")
                        st.dataframe(df[["ID", "Outbreak_Probability", "Outbreak_Flag"]].head(10) if "ID" in df.columns 
                                    else df[["Outbreak_Probability", "Outbreak_Flag"]].head(10), 
                                    use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name=OUTPUT_PATH.name,
                            mime="text/csv",
                            key="download_predictions"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
    else:
        st.info("👇 Please import data in Step 1 to proceed.")

# ==================================================
# PAGE 3: RESULTS & VISUALIZATION
# ==================================================
elif page == "Results":

    st.title("Prediction Results & Visualization")

    if st.session_state["results_df"] is None:
        st.warning("No results available. Run the model first.")
        st.stop()

    df = st.session_state["results_df"]

    # Filters
    if "Region" in df.columns:
        st.subheader("Filter by Region")
        region_options = sorted(df["Region"].dropna().unique().tolist())
        selected_region = st.selectbox("Select Region", ["All"] + region_options)
        if selected_region != "All":
            df_filtered = df[df["Region"] == selected_region].copy()
        else:
            df_filtered = df.copy()
    else:
        selected_region = "All"  # Default to "All" if Region column doesn't exist
        df_filtered = df.copy()

    if "ID" in df_filtered.columns:
        st.subheader("Filter by Event (ID)")
        id_options = sorted(df_filtered["ID"].dropna().unique().tolist())
        selected_id = st.selectbox("Event ID", ["All"] + id_options)
        if selected_id != "All":
            df_view = df_filtered[df_filtered["ID"] == selected_id].copy()
        else:
            df_view = df_filtered.copy()
    else:
        df_view = df_filtered.copy()

    st.subheader("Prediction Output (Preview)")
    st.dataframe(df_view.head())

    manual_date = None
    min_date = None
    max_date = None
    if "Date" in df_view.columns:
        date_series = pd.to_datetime(df_view["Date"], errors="coerce").dt.date
        valid_dates = date_series.dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            manual_date = st.date_input(
                "Select Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="manual_date"
            )

    if "sim_playing" not in st.session_state:
        st.session_state["sim_playing"] = False

    # -------------------------------
    # Time series (if Date exists)
    # -------------------------------
    if "Date" in df_view.columns:
        st.subheader("Temporal Outbreak Risk")

        temporal_df = df_view.copy()
        temporal_df["Date"] = pd.to_datetime(temporal_df["Date"])
        temporal_df = temporal_df.sort_values("Date")
        temporal_df["Date_Frame"] = temporal_df["Date"].dt.strftime("%m/%d/%Y")
        temporal_df["Date_Only"] = temporal_df["Date"].dt.date
        temporal_df["Outbreak_Status_Label"] = np.where(
            temporal_df["Outbreak_Flag"] == 1,
            "Outbreak",
            "No Outbreak"
        )
        temporal_df["Outbreak_Probability_Label"] = temporal_df["Outbreak_Probability"].map(
            lambda v: f"{v:.2f}"
        )
        temporal_df["Temporal_Label"] = (
            temporal_df["Outbreak_Probability_Label"]
            + " | "
            + temporal_df["Outbreak_Status_Label"]
        )

        line_trace = go.Scatter(
            x=temporal_df["Date"],
            y=temporal_df["Outbreak_Probability"],
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            marker=dict(symbol="star", size=8, color="#1f77b4"),
            name="Predicted Outbreak Risk"
        )

        if manual_date and manual_date in temporal_df["Date_Only"].values:
            first_row = temporal_df[temporal_df["Date_Only"] == manual_date].iloc[0]
        else:
            first_row = temporal_df.iloc[0]
        marker_trace = go.Scatter(
            x=[first_row["Date"]],
            y=[first_row["Outbreak_Probability"]],
            mode="markers+text",
            text=[first_row["Temporal_Label"]],
            textposition="top center",
            textfont=dict(color="black", size=12, family="Arial Black"),
            marker=dict(size=12, color="red"),
            hovertemplate="Probability: %{y:.2f}<br>Status: %{text}<extra></extra>",
            showlegend=False
        )

        threshold_trace = go.Scatter(
            x=[temporal_df["Date"].min(), temporal_df["Date"].max()],
            y=[default_threshold, default_threshold],
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name=f"Threshold ({default_threshold:.2f})",
            showlegend=True
        )

        fig2 = go.Figure(data=[line_trace, marker_trace, threshold_trace])

        frames = []
        for _, row in temporal_df.iterrows():
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=[row["Date"]],
                            y=[row["Outbreak_Probability"]],
                            mode="markers+text",
                            text=[row["Temporal_Label"]],
                            textposition="top center",
                            textfont=dict(color="black", size=12, family="Arial Black"),
                            marker=dict(size=12, color="red")
                        )
                    ],
                    name=row["Date_Frame"],
                    traces=[1]
                )
            )

        fig2.frames = frames

        fig2.update_layout(
            title=dict(text="Temporal Norovirus Outbreak Probability", font=dict(color="black", size=18, family="Arial Black")),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", font=dict(color="black", size=14, family="Arial")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                title=dict(text="Date", font=dict(size=16, family="Arial Black", color="black")),
                tickfont=dict(size=14, family="Arial", color="black"),
                tickformat="%m/%d/%Y",
                gridcolor="lightgray",
                showline=True,
                linecolor="black"
            ),
            yaxis=dict(
                title=dict(text="Outbreak Probability", font=dict(size=16, family="Arial Black", color="black")),
                tickfont=dict(size=14, family="Arial", color="black"),
                gridcolor="lightgray",
                showline=True,
                linecolor="black"
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.02,
                    "y": 1.05,
                    "xanchor": "left",
                    "yanchor": "top",
                    "direction": "left",
                    "font": {"color": "black", "size": 12},
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                        }
                    ],
                }
            ],
            sliders=[
                {
                    "x": 0.1,
                    "y": -0.15,
                    "len": 0.8,
                    "pad": {"t": 30, "b": 10},
                    "currentvalue": {
                        "prefix": "Date: ",
                        "font": {"size": 14, "family": "Arial", "color": "black"}
                    },
                    "font": {"color": "black"},
                    "steps": [
                        {
                            "label": frame.name,
                            "method": "animate",
                            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        }
                        for frame in frames
                    ],
                }
            ]
        )

        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "png", "width": 1200, "height": 600}})
        
        # Download buttons for Temporal chart
        if KALEIDO_AVAILABLE:
            col_png, col_tiff = st.columns(2)
            with col_png:
                try:
                    png_image = pio.to_image(fig2, format="png", width=1200, height=600)
                    st.download_button(
                        label="📥 Download Temporal Chart (PNG)",
                        data=png_image,
                        file_name="temporal_outbreak_risk.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"PNG download error: {e}")
            with col_tiff:
                try:
                    tiff_image = pio.to_image(fig2, format="tiff", width=1200, height=600, scale=5)
                    st.download_button(
                        label="📥 Download Temporal Chart (TIFF 400 DPI)",
                        data=tiff_image,
                        file_name="temporal_outbreak_risk_400dpi.tiff",
                        mime="image/tiff"
                    )
                except Exception as e:
                    st.warning(f"TIFF download error: {e}")

    # -------------------------------
    # Spatial visualization (if lat/lon exists)
    # -------------------------------
    if {"Lat", "Long"}.issubset(df_view.columns):
        st.subheader("Spatial Outbreak Risk")

        df_view = df_view.copy()
        df_view["Outbreak_Status_Label"] = np.where(
            df_view["Outbreak_Flag"] == 1,
            "Outbreak",
            "No Outbreak"
        )
        df_view["Outbreak_Probability_Label"] = df_view["Outbreak_Probability"].map(
            lambda v: f"{v:.2f}"
        )
        df_view["Outbreak_Map_Label"] = (
            df_view["Outbreak_Probability_Label"]
            + "\n"
            + df_view["Outbreak_Status_Label"]
        )

        # Prepare data for animation if Date exists
        if "Date" in df_view.columns:
            df_view["Date"] = pd.to_datetime(df_view["Date"])
            df_view = df_view.sort_values("Date")  # Sort by date chronologically
            df_view["Date_Frame"] = df_view["Date"].dt.strftime("%m/%d/%Y")
            frame_values = df_view["Date_Frame"].dropna().unique().tolist()  # Already sorted
        else:
            frame_values = []

        center_lat = float(df_view["Lat"].mean()) if not df_view["Lat"].isna().all() else 0.0
        center_lon = float(df_view["Long"].mean()) if not df_view["Long"].isna().all() else 0.0



        def create_rectangle_coords(lat, lon, delta=0.3):
            """Create rectangle coordinates for ±delta degrees around a point."""
            # Create rectangle: bottom-left -> bottom-right -> top-right -> top-left -> close
            lats = [lat - delta, lat - delta, lat + delta, lat + delta, lat - delta]
            lons = [lon - delta, lon + delta, lon + delta, lon - delta, lon - delta]
            return lats, lons

        def get_color_from_probability(prob):
            """Map probability to color gradient (0=green, 1=dark red)."""
            # Green (0) -> Yellow (0.5) -> Red (1.0)
            if prob < 0.5:
                # Interpolate green to yellow
                r = int(255 * (prob * 2))
                g = 255
                b = 0
            else:
                # Interpolate yellow to dark red
                r = 255
                g = int(255 * (2 - 2 * prob))
                b = 0
            return f"rgb({r},{g},{b})"



        def build_traces(frame_df):
            traces = []
            
            # Handle all regions uniformly with rectangles
            for idx, row in frame_df.iterrows():
                lat, lon = row["Lat"], row["Long"]
                prob = row["Outbreak_Probability"]
                
                # Create rectangle coordinates
                rect_lats, rect_lons = create_rectangle_coords(lat, lon, delta=0.3)
                
                # Get color based on probability
                color = get_color_from_probability(prob)
                
                # Create rectangle trace
                rect_trace = go.Scattermapbox(
                    lat=rect_lats,
                    lon=rect_lons,
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    opacity=0.6,
                    line=dict(color=color, width=2),
                    customdata=[[
                        row["Outbreak_Probability_Label"],
                        row["Outbreak_Status_Label"],
                        row["Date_Frame"],
                        row["Outbreak_Map_Label"]
                    ]] * len(rect_lats),
                    hovertemplate=(
                        "<b>Probability: %{customdata[0]}</b><br>"
                        "Status: %{customdata[1]}<br>"
                        "Date: %{customdata[2]}<br>"
                        "Area: %{customdata[3]}<extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor="white", font=dict(size=14, family="Arial", color="black")),
                    showlegend=False
                )
                traces.append(rect_trace)
                
                # Add text label at the center of the rectangle showing the probability
                text_trace = go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode="text",
                    text=[f"{prob:.2f}"],
                    textposition="middle center",
                    textfont=dict(
                        size=14,
                        color="black",
                        family="Arial Black"
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
                traces.append(text_trace)
            
            return traces

        if frame_values:
            if manual_date:
                manual_frame = manual_date.strftime("%Y-%m-%d")
            else:
                manual_frame = None
            if manual_frame in frame_values:
                initial_df = df_view[df_view["Date_Frame"] == manual_frame]
            else:
                initial_df = df_view[df_view["Date_Frame"] == frame_values[0]]
        else:
            initial_df = df_view

        initial_traces = build_traces(initial_df)
        fig4 = go.Figure(data=initial_traces)

        if frame_values:
            frames = []
            # Calculate average risk for each frame for display
            frame_risk_map = {}
            for frame_value in frame_values:
                frame_df = df_view[df_view["Date_Frame"] == frame_value]
                avg_risk = frame_df["Outbreak_Probability"].mean()
                frame_risk_map[frame_value] = avg_risk
                frame_traces = build_traces(frame_df)
                frames.append(go.Frame(data=frame_traces, name=frame_value))
            fig4.frames = frames

            fig4.update_layout(
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "x": 0.02,
                        "y": 0.02,
                        "xanchor": "left",
                        "yanchor": "bottom",
                        "direction": "left",
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]
                            }
                        ],
                    }
                ],
                sliders=[
                    {
                        "x": 0.1,
                        "y": 0.02,
                        "len": 0.8,
                        "pad": {"t": 30, "b": 10},
                        "currentvalue": {"prefix": "Date: | Risk: "},
                        "steps": [
                            {
                                "label": f"{frame_value} | Risk: {frame_risk_map[frame_value]:.2f}",
                                "method": "animate",
                                "args": [[frame_value], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]
                            }
                            for frame_value in frame_values
                        ],
                    }
                ]
            )

        # Configure mapbox layout with satellite imagery
        mapbox_dict = dict(
            style="open-street-map",  # More colorful, realistic map style
            zoom=3,
            center={"lat": center_lat, "lon": center_lon}
        )
        
        # Add invisible scatter trace for colorbar legend
        legend_trace = go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode="markers",
            marker=dict(
                size=1,
                color=[0, 0.5, 1],  # Dummy values for colorscale
                colorscale=[[0, "rgb(0,255,0)"], [0.5, "rgb(255,255,0)"], [1, "rgb(255,0,0)"]],
                showscale=True,
                colorbar=dict(
                    title="Outbreak<br>Probability",
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    xanchor="left",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%", "25%", "50%", "75%", "100%"]
                )
            ),
            showlegend=False
        )
        fig4.add_trace(legend_trace)
        
        fig4.update_layout(
            title="Spatial Outbreak Risk Simulation",
            showlegend=False,
            mapbox=mapbox_dict,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # Adjust figure height based on full-screen mode
        fig4.update_layout(height=700)
        st.plotly_chart(
            fig4,
            use_container_width=True,
            config={"displaylogo": False, "displayModeBar": True, "responsive": True}
        )

    # -------------------------------
    # Download results
    # -------------------------------
    st.subheader("Download Results")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction Results (CSV)",
        data=csv,
        file_name="norovirus_predictions.csv",
        mime="text/csv"
    )
