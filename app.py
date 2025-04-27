import streamlit as st
import pandas as pd
import hashlib
import os
import io
from datetime import datetime
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
import random
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta
import sqlite3
import requests
import shutil
import numpy as np

# --- File Paths ---
USERS_FILE = "users.csv"
CRIMES_FILE = "Crime_Data_from_2020_to_Present.csv"
REPORT_FILE = "reports.csv"

@st.cache_data
def load_crime_data(max_rows=200000):
    endpoint = "https://data.lacity.org/resource/2nrs-mtv8.json"
    limit = 50000
    dfs = []

    # --- NEW: Set cutoff date dynamically ---
    cutoff_reference_date = datetime.today() - relativedelta(months=3)
    cutoff_date = (cutoff_reference_date - relativedelta(months=15)).strftime("%Y-%m-%dT00:00:00.000")

    # --- NEW: Hardcoded cutoff dates because data is only up to Feb 2025 ---
    start_date = "2024-01-01T00:00:00.000"  # 12 months ago
    end_date = "2025-01-31T23:59:59.999"     # End of Jan 2025

    where_clause = f"DATE_OCC >= '{start_date}' AND DATE_OCC <= '{end_date}'"

    for offset in range(0, max_rows, limit):
        url = (
            f"{endpoint}?$limit={limit}&$offset={offset}"
            f"&$where={where_clause}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            break  # Exit on error or no more data
        data = response.json()
        if not data:
            break
        df = pd.DataFrame(data)
        dfs.append(df)

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.columns = full_df.columns.str.strip().str.upper()
        return full_df
    else:
        return pd.DataFrame()

# Utility: create tables if not exist
def init_db():
    # Check if database files already exist
    if not os.path.exists("users.db"):
        shutil.copyfile("starter_db/users.db", "users.db")
    if not os.path.exists("reports.db"):
        shutil.copyfile("starter_db/reports.db", "reports.db")
    if not os.path.exists("feedback.db"):
        shutil.copyfile("starter_db/feedback.db", "feedback.db")
        
    with sqlite3.connect("users.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )""")

    with sqlite3.connect("reports.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS reports (
            date_occ TEXT,
            crime_type TEXT,
            location TEXT,
            mocodes TEXT,
            description TEXT,
            lat REAL,
            lon REAL
        )""")

    with sqlite3.connect("feedback.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            user TEXT,
            timestamp TEXT,
            message TEXT
        )""")

    # --- Ensure admin user exists ---
    with sqlite3.connect("users.db") as conn:
        cursor = conn.execute("SELECT * FROM users WHERE username = 'admin'")
        if cursor.fetchone() is None:
            admin_password = hashlib.sha256("admin".encode()).hexdigest()
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", admin_password))


# --- Authentication Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_login(username, password):
    hashed = hash_password(password)
    with sqlite3.connect("users.db") as conn:
        cursor = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
        return cursor.fetchone() is not None

def register_user(username, password):
    hashed = hash_password(password)
    try:
        with sqlite3.connect("users.db") as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

# --- Login Page ---
def login_page():
    st.title("Community Crime Watch Portal")
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if validate_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with register_tab:
        with st.form("Register Form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.form_submit_button("Create Account"):
                if register_user(new_user, new_pass):
                    st.success("Account created! Please login")
                else:
                    st.error("Username already exists")

# --- Crime Analysis Functions ---
def categorize_crime(description):
    violent_keywords = ['ASSAULT', 'HOMICIDE', 'RAPE', 'ROBBERY']
    property_keywords = ['THEFT', 'BURGLARY', 'VEHICLE', 'SHOPLIFTING']
    desc = str(description).upper()
    
    if any(kw in desc for kw in violent_keywords):
        return "Violent Crimes"
    elif any(kw in desc for kw in property_keywords):
        return "Property Crimes"
    return "Other"

# --- Enhanced Home Page ---
def homepage():
    st.title(f"Crime Dashboard - Welcome {st.session_state.user}")

    crimes = load_crime_data()
    crimes["LAT"] = pd.to_numeric(crimes["LAT"], errors="coerce")
    crimes["LON"] = pd.to_numeric(crimes["LON"], errors="coerce")
    crimes['DATE_OCC'] = pd.to_datetime(crimes['DATE_OCC'], errors='coerce')
    crimes['Category'] = crimes['CRM_CD_DESC'].apply(categorize_crime)

    min_date = crimes['DATE_OCC'].min().date() if not crimes.empty else datetime.today().date()
    max_date = crimes['DATE_OCC'].max().date() if not crimes.empty else datetime.today().date()
    col_filter1, col_filter2 = st.columns([2, 3])
    with col_filter1:
        date_range = st.date_input("Select Date Range", [min_date, max_date])
    with col_filter2:
        selected_types = st.multiselect("Filter by Crime Type", crimes['CRM_CD_DESC'].unique())

    filtered_crimes = crimes[
        (crimes['DATE_OCC'].dt.date >= date_range[0]) & 
        (crimes['DATE_OCC'].dt.date <= date_range[1]) &
        (crimes['CRM_CD_DESC'].isin(selected_types) if selected_types else True)
    ] if len(date_range) == 2 else crimes

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crimes", len(filtered_crimes))
    with col2:
        st.metric("Violent Crimes", len(filtered_crimes[filtered_crimes['Category'] == "Violent Crimes"]))
    with col3:
        st.metric("Property Crimes", len(filtered_crimes[filtered_crimes['Category'] == "Property Crimes"]))
    with col4:
        clearance_rate = filtered_crimes['STATUS_DESC'].str.contains('Arrest', na=False).mean() * 100
        st.metric("Arrest Rate", f"{clearance_rate:.1f}%")

    st.markdown("### Top 5 Reported Crime Types")
    top_crimes = filtered_crimes['CRM_CD_DESC'].value_counts().head(5)
    st.table(top_crimes.reset_index().rename(columns={'index': 'Crime Type', 'CRM_CD_DESC': 'Reports'}))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Crimes by Day of Week")
        day_counts = filtered_crimes['DATE_OCC'].dt.day_name().value_counts().reindex([
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        st.bar_chart(day_counts)

    with col2:
        st.markdown("### Crime Activity by Hour of Day")
        filtered_crimes['Hour'] = pd.to_numeric(
    filtered_crimes['TIME_OCC'].fillna(0).astype(int).astype(str).str.zfill(4).str[:2],
    errors='coerce'
).fillna(0).astype(int)
        hour_counts = filtered_crimes['Hour'].value_counts().sort_index()
        st.line_chart(hour_counts)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Crime Type Distribution")
        crime_types = filtered_crimes['CRM_CD_DESC'].value_counts()
        st.bar_chart(crime_types)

    with col2:
        st.markdown("### Monthly Crime Trend")
        time_data = filtered_crimes.set_index('DATE_OCC').resample('M').size().rename("count")
        st.area_chart(time_data)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Violent Crimes Trend")
        time_data_violent = filtered_crimes[filtered_crimes['Category'] == "Violent Crimes"].set_index('DATE_OCC').resample('ME').size().rename("count")
        st.area_chart(time_data_violent)

    with col4:
        st.markdown("### Property Crimes Trend")
        time_data_property = filtered_crimes[filtered_crimes['Category'] == "Property Crimes"].set_index('DATE_OCC').resample('ME').size().rename("count")
        st.area_chart(time_data_property)

    with st.expander("📢 Community Safety Tips"):
        st.markdown("""
        - Always report suspicious activity promptly.
        - Stay alert in poorly lit areas, especially at night.
        - Join or follow updates from your local neighborhood watch.
        - Lock your vehicles and keep valuables out of sight.
        - Know the most common crimes in your neighborhood.
        """)

# --- Crime Reporting Page ---
def report_page():
    st.header("File New Crime Report")
    
    with st.form("Crime Report Form"):
        # UCR Code Selection
        crime_type = st.selectbox("Crime Type", options=[
            "Homicide", "Rape", "Robbery", "Aggravated Assault",
            "Burglary", "Motor Vehicle Theft", "Theft"
        ])
        
        # Location Details
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location Address")
        with col2:
            date_occ = st.date_input("Date Occurred", datetime.today())
        
        # MO Code Selection
        mo_codes = st.multiselect("Modus Operandi (MO Codes)", options=[
            "0100: Suspect Impersonate", "0200: Disguise Used",
            "0300: Forced Entry", "0400: Weapon Displayed"
        ])
        
        description = st.text_area("Incident Details")
        
        if st.form_submit_button("Submit Report"):
            try:
                with sqlite3.connect("reports.db") as conn:
                    conn.execute("""
                        INSERT INTO reports (date_occ, crime_type, location, mocodes, description, lat, lon)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        date_occ.strftime('%m/%d/%Y'),
                        crime_type,
                        location,
                        ', '.join(mo_codes),
                        description,
                        34.0522,
                        -118.2437
                    ))
                st.success("Report submitted successfully!")
            except Exception as e:
                st.error(f"Error saving report: {str(e)}")

    # --- Feedback Section ---
    st.markdown("### 🗣️ Resident Feedback")
    with st.form("feedback_form"):
        feedback = st.text_area("Have suggestions, concerns, or feedback about safety in your area?")
        if st.form_submit_button("Submit"):
            try:
                with sqlite3.connect("feedback.db") as conn:
                    conn.execute("""
                        INSERT INTO feedback (user, timestamp, message)
                        VALUES (?, ?, ?)
                    """, (
                        st.session_state.user,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        feedback
                    ))
                st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"Error saving feedback: {str(e)}")

# --- Interactive Map View Page ---
def map_view_page():
    st.header("Interactive Crime Map")
    crimes = load_crime_data().dropna(subset=['LAT', 'LON'])
    crimes["LAT"] = pd.to_numeric(crimes["LAT"], errors="coerce")
    crimes["LON"] = pd.to_numeric(crimes["LON"], errors="coerce")
    crimes['DATE_OCC'] = pd.to_datetime(crimes['DATE_OCC'], errors='coerce')
    crimes = crimes.dropna(subset=['DATE_OCC'])

    col1, col2 = st.columns(2)
    with col1:
        crime_filter = st.multiselect("Filter Crime Types", options=crimes['CRM_CD_DESC'].unique(), default="VEHICLE - STOLEN")
    with col2:
        min_date = crimes['DATE_OCC'].min().date()
        max_date = crimes['DATE_OCC'].max().date()
        date_filter = st.date_input("Filter by Date", [min_date, max_date])

    if crime_filter:
        crimes = crimes[crimes['CRM_CD_DESC'].isin(crime_filter)]
    if len(date_filter) == 2:
        crimes = crimes[(crimes['DATE_OCC'].dt.date >= date_filter[0]) & (crimes['DATE_OCC'].dt.date <= date_filter[1])]

    df = crimes[['CRM_CD', 'CRM_CD_DESC', 'LAT', 'LON', 'DATE_OCC', 'LOCATION', 'PREMIS_DESC']].head(500)
    if df.empty:
        st.info("No incidents found matching filters")
        return

    view_option = st.radio("Map View Type", ["Marker Clusters", "Heatmap"], horizontal=True)

    map_center = [df["LAT"].mean(), df["LON"].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

    if view_option == "Marker Clusters":
        crime_codes = df["CRM_CD"].unique()
        color_map = {code: f"#{random.randint(0, 0xFFFFFF):06x}" for code in crime_codes}
        marker_cluster = MarkerCluster().add_to(crime_map)

        for _, row in df.iterrows():
            crime_code = row["CRM_CD"]
            crime_desc = row["CRM_CD_DESC"]
            color = color_map[crime_code]
            popup_text = f"""
            <div style='font-size: 14px;'>
                <strong>Crime:</strong> {crime_desc}<br>
                <strong>Date:</strong> {row['DATE_OCC'].strftime('%Y-%m-%d')}<br>
                <strong>Location:</strong> {row['LOCATION']}<br>
                <strong>Description:</strong> {row['PREMIS_DESC']}
            </div>
            """
            folium.CircleMarker(
                location=[row["LAT"], row["LON"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=350),
            ).add_to(marker_cluster)
    else:
        heat_data = df[['LAT', 'LON']].values.tolist()
        HeatMap(heat_data).add_to(crime_map)

    folium_static(crime_map, width=1800, height=700)

    st.subheader("Filtered Crime Incidents")
    st.dataframe(df.rename(columns={
        'DATE_OCC': 'Date',
        'CRM_CD_DESC': 'Crime Type',
        'LOCATION': 'Location',
        'PREMIS_DESC': 'Description'
    }), use_container_width=True)

    # --- Download Option ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_crimes.csv',
        mime='text/csv'
    )

# Forecasting Crime 
def forecast_page():
    st.title("📈 Crime Forecasting")

    try:
        # --- 1. Load and Prepare Crime Data ---
        crimes = load_crime_data()
        crimes["LAT"] = pd.to_numeric(crimes["LAT"], errors="coerce")
        crimes["LON"] = pd.to_numeric(crimes["LON"], errors="coerce")
        crimes['DATE_OCC'] = pd.to_datetime(crimes['DATE_OCC'], errors='coerce')
        crimes = crimes.dropna(subset=["LAT", "LON"])

        # --- 2. Create Zones (rounded 0.1 degrees) ---
        crimes["LAT_ZONE"] = crimes["LAT"].apply(lambda x: round(x, 1))
        crimes["LON_ZONE"] = crimes["LON"].apply(lambda x: round(x, 1))
        crimes["ZONE_ID"] = crimes["LAT_ZONE"].astype(str) + "_" + crimes["LON_ZONE"].astype(str)

        # --- 3. Create Month Field ---
        crimes["Month"] = crimes["DATE_OCC"].dt.to_period("M")

        # --- 4. Set Forecast Target Month ---
        #max_date = crimes['DATE_OCC'].max()
        #forecast_month = max_date.replace(day=1) + relativedelta(months=1)
        forecast_month = datetime.today() + relativedelta(months=1)
        forecast_label = forecast_month.strftime("%B %Y")

        st.markdown(f"#### Forecasting for {forecast_label}")
        # ========== ZONE FORECASTING (only for Map internally) ==========

        zone_month_crimes = crimes.groupby(["ZONE_ID", "Month"]).size().reset_index(name="Crime_Count")

        forecast_zone_counts = {}

        for zone in zone_month_crimes["ZONE_ID"].unique():
            zone_data = zone_month_crimes[zone_month_crimes["ZONE_ID"] == zone]
            zone_data = zone_data.sort_values("Month")
            zone_data["Ordinal"] = zone_data["Month"].apply(lambda x: x.start_time.toordinal())

            X = zone_data[["Ordinal"]]
            y = zone_data["Crime_Count"]

            if len(X) >= 2:
                model = LinearRegression().fit(X, y)
                next_month = (zone_data["Month"].max().start_time + relativedelta(months=1)).toordinal()
                predicted_count = model.predict(np.array([[next_month]]))[0]
                forecast_zone_counts[zone] = max(int(predicted_count), 0)

        # Build map if any zones predicted
        if forecast_zone_counts:
            forecast_zone_df = pd.DataFrame(
                list(forecast_zone_counts.items()),
                columns=["Zone_ID", "Predicted_Crimes"]
            )
            forecast_zone_df[["Latitude", "Longitude"]] = forecast_zone_df["Zone_ID"].str.split("_", expand=True)
            forecast_zone_df["Latitude"] = forecast_zone_df["Latitude"].astype(float)
            forecast_zone_df["Longitude"] = forecast_zone_df["Longitude"].astype(float)

            # --- Map ---
            st.subheader("🗺️ Forecasted Crime Risk Map")

            forecast_map = folium.Map(location=[34.05, -118.25], zoom_start=11, tiles="OpenStreetMap")
            max_crimes = forecast_zone_df["Predicted_Crimes"].max()

            for _, row in forecast_zone_df.iterrows():
                lat = row["Latitude"]
                lon = row["Longitude"]
                predicted_crimes = row["Predicted_Crimes"]

                if predicted_crimes > 0:
                    square = [
                        [lat, lon],
                        [lat + 0.1, lon],
                        [lat + 0.1, lon + 0.1],
                        [lat, lon + 0.1],
                        [lat, lon]
                    ]

                    normalized_value = predicted_crimes / max_crimes
                    red = int(255 * normalized_value)
                    color = f"#{red:02x}0000"

                    folium.Polygon(
                        locations=square,
                        color=None,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5,
                        popup=folium.Popup(f"Predicted Crimes: {predicted_crimes}", max_width=250)
                    ).add_to(forecast_map)

            folium_static(forecast_map, width=1800, height=800)

        else:
            st.info("Not enough data to generate forecasts by zone.")

        # ========== CRIME TYPE FORECASTING (for Table only) ==========

        st.subheader("📋 Predicted Crimes by Type (City-wide)")

        crime_type_month = crimes.groupby(["Month", "CRM_CD_DESC"]).size().reset_index(name="Crime_Count")

        crime_type_forecasts = {}

        for crime_type in crime_type_month["CRM_CD_DESC"].unique():
            crime_data = crime_type_month[crime_type_month["CRM_CD_DESC"] == crime_type]
            crime_data = crime_data.sort_values("Month")
            crime_data["Ordinal"] = crime_data["Month"].apply(lambda x: x.start_time.toordinal())

            X = crime_data[["Ordinal"]]
            y = crime_data["Crime_Count"]

            if len(X) >= 2:
                model = LinearRegression().fit(X, y)
                next_month = (crime_data["Month"].max().start_time + relativedelta(months=1)).toordinal()
                predicted_count = model.predict(np.array([[next_month]]))[0]
                crime_type_forecasts[crime_type] = max(int(predicted_count), 0)

        crime_type_forecast_df = pd.DataFrame(
            list(crime_type_forecasts.items()),
            columns=["Crime Type", f"Predicted ({forecast_label})"]
        )

        crime_type_forecast_df = crime_type_forecast_df[
            crime_type_forecast_df[f"Predicted ({forecast_label})"] > 0
        ].sort_values(f"Predicted ({forecast_label})", ascending=False).reset_index(drop=True)

        st.dataframe(crime_type_forecast_df, use_container_width=True)

    except Exception as e:
        st.warning("Forecast unavailable: " + str(e))

def admin_page():
    st.header("🛡️ Admin Panel")

    st.subheader("Filed Crime Reports")
    try:
        with sqlite3.connect("reports.db") as conn:
            reports = pd.read_sql_query("SELECT * FROM reports", conn)
        st.dataframe(reports, use_container_width=True)

        csv_reports = reports.to_csv(index=False).encode('utf-8')
        st.download_button("Download Reports as CSV", data=csv_reports, file_name="reports.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error loading reports: {str(e)}")

    st.subheader("Resident Feedback")
    try:
        with sqlite3.connect("feedback.db") as conn:
            feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
        st.dataframe(feedback, use_container_width=True)

        csv_feedback = feedback.to_csv(index=False).encode('utf-8')
        st.download_button("Download Feedback as CSV", data=csv_feedback, file_name="feedback.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error loading feedback: {str(e)}")


# --- Main Application ---
def main():
    st.set_page_config(page_title="CrimeWatch", layout="wide")

    st.markdown("""
        <style>
        /* Background image with overlay */
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQfZ_L0V6oYyvKM9zVGf9UxhFrOFtq6_FuhmYPKzQHuo9fX-butRj9k6c&s");
            background-size: cover;
            background-attachment: fixed;
        }

        /* General layout spacing */
        .block-container {
            padding: 2rem 2.5rem 2rem 2.5rem;
        }

        /* Main content container */
        section.main > div {
            backdrop-filter: blur(10px);
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.2);
            padding: 1.25rem;
            overflow: visible;
            margin-top: 2rem;
        }

        /* Inputs and selectors */
        .stTextInput, .stDateInput, .stSelectbox, .stMultiSelect, .stNumberInput,
        .stRadio, .stForm, .stExpander {
            background-color: rgba(255, 255, 255, 0.07) !important;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white !important;
            border-radius: 8px;
            padding: 0.6rem;
            overflow: visible;
        }

        /* Metric containers */
        .stMetric {
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 1rem;
            color: white !important;
            overflow: visible;
        }

        /* Buttons */
        .stButton > button {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            overflow: visible;
            transition: 0.3s ease;
        }
        .stButton > button:hover {
            background-color: rgba(255, 255, 255, 0.35);
            transform: scale(1.02);
            cursor: pointer;
        }

        /* Text and headers */
        h1, h2, h3, h4, p, label, span, .stMarkdown {
            color: #ffffff !important;
        }

        /* Header fix */
        header {
            background: transparent;
            position: relative;
            z-index: 999;
            height: auto;
            min-height: 60px;
            padding-top: 0.5rem;
        }

        /* Chart/table containers */
        div[data-testid="stHorizontalBlock"] > div,
        div[data-testid="stVerticalBlock"] > div {
            background-color: transparent !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # --- CSS (keep the fixed version from before) ---
    st.markdown("""<style> ...your CSS here... </style>""", unsafe_allow_html=True)

    # --- Safe padding at top of page to prevent header clipping ---
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"

    if not st.session_state.logged_in:
        login_page()
    else:
        # --- Header Navigation ---
        if st.session_state.user == "admin":
            cols = st.columns([1.2, 1, 1, 1, 1, 1, 1])
        else:
            cols = st.columns([1.5, 1, 1, 1, 1, 1])
            
        with cols[0]:
            st.markdown(f"### 👮 CrimeWatch")
        with cols[1]:
            if st.button("Dashboard"):
                st.session_state.page = "Dashboard"
        with cols[2]:
            if st.button("Interactive Map"):
                st.session_state.page = "Interactive Map"
        with cols[3]:
            if st.button("Forecasting"):
                st.session_state.page = "Forecasting"
        with cols[4]:
            if st.button("File Report"):
                st.session_state.page = "File Report"
        with cols[5]:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.page = "Dashboard"
                st.rerun()
        # NEW Admin Button:
        if st.session_state.user == "admin":
            with cols[6]:
                if st.button("Admin Panel"):
                    st.session_state.page = "Admin Panel"

        st.markdown("---")  # Divider below the navbar

        # --- Page Routing Based on Selected Tab ---
        if st.session_state.page == "Dashboard":
            homepage()
        elif st.session_state.page == "Interactive Map":
            map_view_page()
        elif st.session_state.page == "Forecasting":
            forecast_page()
        elif st.session_state.page == "File Report":
            report_page()
        elif st.session_state.page == "Admin Panel":
            admin_page()

if __name__ == "__main__":
    main()
