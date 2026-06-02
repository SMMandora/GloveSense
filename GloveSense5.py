import os
import base64
import datetime
import time
import serial
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import zscore
from dash import Dash, html, dcc, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import threading

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server
app.title = "Glove Sense Dash"

# Gesture Mapping
gesture_map = {
    0: "RS", 1: "A", 2: "B", 3: "C", 4: "D", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "L", 10: "N", 11: "W",
    12: "Y", 13: "TR", 14: "TIM", 15: "TMR"
}

gesture_options = [{"label": f"{i} - {name}", "value": i} for i, name in gesture_map.items() if i != 0]

current_live_image_id = {"value": 0}
data_collection_control = {"pause": False, "abort": False, "data_collection_status" : "Collecting Data....."}
TRAIN_ROOT = "../GloveSenseDash/"
# Utility: load image from disk as base64
def get_image_base64(img_path):
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        return f"data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# Default session state
initial_session = {
    'T': "", 'cookies': 0, 'SelectedOption': 1, 'SelectedOptionRepeat': 1,
    'a1': [], 'a2': [], 'a3': [], 'a4': [], 'a5': [], 'a6': [], 'a7': [], 'a8': [],
    'Energies': {}, 'Gst': [], 'tFeature': [], 'tFeature_Df': None, 'tTarget': [],
    'tFeature1': None, 'tTarget1': [], 'll': 0, 'isFirstClick': True,
    'Trial': [1], 'model_scores': {}, 'pause': False, 'abort': False,
    'data_collection_complete': False
}

# Serial initialization helper
def init_serial():
    try:
        return serial.Serial(port='COM3', baudrate=115200, timeout=1)
    except Exception as e:
        print(f"[ERROR] Serial connection failed: {e}")
        return None

# Layout
app.layout = dbc.Container([
    dcc.Store(id="session-store", data=initial_session),
    dcc.Store(id="current-letter", data=0),
    dcc.Store(id="prev-next-clicks", data=0),
    dcc.Store(id="prev-repeat-clicks", data=0),
    dcc.Store(id="live-image", data=0),
    dcc.Interval(id="interval-refresh", interval=100, n_intervals=0),

    html.H2("🧤 Glove Sense Dashboard", style={"color": "#FF5733", "fontFamily": "Comic Sans MS"}),
    html.Div(id="mascot-message", children="👋 Ready to play with gestures?",
             style={"fontSize": "20px", "color": "#007BFF"}),

    dbc.Row([
        # Create Folder Section
        dbc.Col([
            html.Label("Enter your name:", className="fw-bold"),
            dcc.Input(id="user-name", type="text", placeholder="Your name...", className="mb-2 w-100"),
            dbc.Button("🎨 Create Folder", id="create-folder-btn", color="success", className="mb-2 w-100"),
            html.Div(id="folder-status", className="text-success mt-1")
        ], width=3),

        # Train/Test Section
        dbc.Col([
            html.Div(id="gesture-display", style={"textAlign": "center", "paddingTop": "10px"}),
            html.Br(),
            dcc.Tabs(id="tabs", value='train', children=[
                dcc.Tab(label="🎯 Train", value='train'),
                dcc.Tab(label="🧪 Test", value='test'),
            ]),
            html.Div(id="tab-content")
        ], width=9)
    ])
], fluid=True, style={"backgroundColor": "#E8F6F3"})

#  Display static as well as live image
@app.callback(
    Output("gesture-display", "children"),
    [Input("live-image", "data"), Input("current-letter", "data")]
)
def display_combined_image(live_img_id, static_img_id):
    # Determine which callback triggered this one
    triggered = ctx.triggered_id

    # If triggered by live-image interval, show the live one
    if triggered == "live-image":
        gesture_id = int(live_img_id)
    else:
        gesture_id = int(static_img_id)

    gesture_name = gesture_map.get(gesture_id, "Unknown")
    image_path = f"gestures/{gesture_id}.jpg"
    return html.Div([
        html.Img(src=get_image_base64(image_path),
                 style={"width": "300px", "height": "400px", "borderRadius": "10px",
                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"}),
        html.H5(f"Letter: {gesture_name}", className="text-center mt-2 text-primary")
    ])

# # Show image and letter based on current gesture
# @app.callback(
#     Output("gesture-display", "children"),
#     Input("current-letter", "data"),
#     Input("live-image", "data")
# )
# def display_image(current_letter, live_image):
#     gesture_name = gesture_map[current_letter]
#     image_path = f"gestures/{current_letter}.jpg"
#     return html.Div([
#         html.Img(src=get_image_base64(image_path),
#                  style={"width": "300px", "height": "400px", "borderRadius": "10px",
#                         "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"}),
#         html.H5(f"Letter: {gesture_name}", className="text-center mt-2 text-primary")
#     ])

# Tab switching content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def update_tab_content(active_tab):
    if active_tab == 'train':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button("➡️ Next Gesture", id="next-btn", color="primary", className="w-100 mb-4"),
                    dbc.Button("📡 Start Data Collection", id="collect-data-btn", color="info", className="w-100 mb-4"),
                    dbc.Button("💪 Train Gesture", id="train-btn", color="success", className="w-100")
                ], width=3),
                dbc.Col([
                    dcc.Dropdown(id="repeat-gesture-dropdown", options=gesture_options,
                                 placeholder="🔁 Choose gesture to repeat...", className="mb-2"),
                    dbc.Button("🔁 Repeat Gesture", id="repeat-btn", color="warning", className="w-100")
                ], width=3),
                dbc.Col([
                    dbc.Button("⏸️ Pause Data Collection", id="toggle-pause-btn", n_clicks=0, color="danger", className="w-100 mb-2"),
                    dbc.Button("❌ Abort Data Collection", id="abort-btn", color="secondary", className="w-100")
                ], width=3)
            ]),
            html.Div(id="train-status", className="text-success mt-2")
        ])
    elif active_tab == 'test':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id="test-dropdown", options=gesture_options,
                                 placeholder="🎯 Select Gesture to Test", className="mb-2"),
                    dbc.Button("📥 Get Test Data", id="get-test-data", color="secondary", className="w-100 mb-2")
                ], width=3),
                dbc.Col(dbc.Button("📊 Test Accuracy", id="test-btn", color="info", className="w-100"), width=3),
                dbc.Col(html.Div(id="test-status", className="text-info mt-2"), width=3)
            ])
        ])

# Gesture change via dropdown
@app.callback(
    Output("current-letter", "data", allow_duplicate=True),
    Input("repeat-gesture-dropdown", "value"),
    prevent_initial_call=True
)
def change_image_train_dropdown(train_value):
    if train_value is not None:
        print(f"[DEBUG] Train dropdown selected gesture: {train_value}")
        return train_value
    raise PreventUpdate

@app.callback(
    Output("current-letter", "data", allow_duplicate=True),
    Input("test-dropdown", "value"),
    prevent_initial_call=True
)
def change_image_test_dropdown(test_value):
    if test_value is not None:
        print(f"[DEBUG] Test dropdown selected gesture: {test_value}")
        return test_value
    raise PreventUpdate

# Handle gesture changes ONLY by true button clicks
@app.callback(
    Output("train-status", "children"),
    Output("current-letter", "data"),
    Output("prev-next-clicks", "data"),
    Output("prev-repeat-clicks", "data"),
    Input("next-btn", "n_clicks"),
    Input("repeat-btn", "n_clicks"),
    State("repeat-gesture-dropdown", "value"),
    State("current-letter", "data"),
    State("prev-next-clicks", "data"),
    State("prev-repeat-clicks", "data"),
    prevent_initial_call=True
)
def handle_gesture_buttons(next_click, repeat_click, repeat_value, current_index, prev_next, prev_repeat):
    updated_next = prev_next
    updated_repeat = prev_repeat
    if next_click and next_click > prev_next:
        print("[DEBUG] Next Gesture button clicked")
        status = f"Next Gesture Button Clicked. Current Gesture is: {current_index}"
        updated_next = next_click
        return status, (current_index + 1) % len(gesture_map), updated_next, updated_repeat
    elif repeat_click and repeat_click > prev_repeat and repeat_value is not None:
        print(f"[DEBUG] Repeat Gesture selected: {repeat_value}")
        updated_repeat = repeat_click
        return repeat_value, updated_next, updated_repeat
    raise PreventUpdate

# Folder creation
@app.callback(
    Output("folder-status", "children"),
    Output("session-store", "data"),
    Input("create-folder-btn", "n_clicks"),
    State("user-name", "value"),
    State("session-store", "data")
)
def create_folder(n, username, session_data):
    if not n or not username:
        raise PreventUpdate
    folder = f"../GloveSenseDash/G_Alph_{username}"
    os.makedirs(folder, exist_ok=True)
    session_data['T'] = username
    print(f"[DEBUG] Folder created for user: {username}")
    return f"✅ Folder ready: {folder}", session_data

def create_path_folder(letter_id, Test, session_data):
    path_folder = f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}{Test}"
    try:
        os.mkdir(path_folder)
        return f"✅ Folder ready: {path_folder}", session_data
    except OSError:
        return f"The directory {path_folder} already exists."

def process_signal(signal, VR, VM):
    signal = signal.copy()
    window_size = 5
    num_windows = len(signal) // window_size
    print("nnum_windows", num_windows)

    # Loop over windows
    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        local_std = np.std(signal[start_idx:end_idx])
        threshold = 1 * local_std
        window_mean = np.mean(signal[start_idx:end_idx])
        print("window_mean", window_mean)
        print("local_std", local_std)
        print("threshold", threshold)

        # Detect spikes
        for i in range(start_idx + 1, end_idx):
            if abs(signal[i] - signal[i - 1]) > threshold:
                signal[i] = window_mean
                print("signal", signal)

    # Compute average amplitude per window
    avg_window_amplitude = np.zeros(num_windows)
    print("avg_window_amplitude", avg_window_amplitude)
    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        avg_window_amplitude[w] = np.mean(signal[start_idx:end_idx])
        print("avg_window_amplitude", avg_window_amplitude)

    # Replace each element in the window with the average amplitude
    print("num_windows", num_windows)
    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        signal[start_idx:end_idx] = avg_window_amplitude[w]

    # Compute standard deviations and means over VR and VM ranges
    stR = np.std(signal[VR] - np.mean(signal[VR]))
    stM = np.std(signal[VM] - np.mean(signal[VM]))
    MR = np.mean(signal[VR])
    MM = np.mean(signal[VM])
    print("stR", stR)
    print("stM", stM)
    print("MR", MR)
    print("MM", MM)

    # Adjust signal based on deviations
    num_sections = len(VR) // 5
    print("num_sections", num_sections)
    for i in range(num_sections):
        idx_vr = VR[i * 5:(i + 1) * 5]
        idx_vm = VM[i * 5:(i + 1) * 5]
        if abs(signal[idx_vr[0]] - MR) > 1.5 * stR:
            signal[idx_vr] = MR
        if abs(signal[idx_vm[0]] - MM) > 1.5 * stM:
            signal[idx_vm] = MM
    print("signal", signal)
    return signal

def fill_outliers(segment):
    z_scores = np.abs(zscore(segment))
    threshold = 3  # Adjust threshold as needed
    outlier_indices = np.where(z_scores > threshold)[0]
    if len(outlier_indices) > 0:
        segment_no_outliers = segment.copy()
        segment_no_outliers[outlier_indices] = np.nan
        nans = np.isnan(segment_no_outliers)
        not_nans = ~nans
        if np.sum(not_nans) >= 2:
            segment_no_outliers[nans] = np.interp(nans.nonzero()[0], not_nans.nonzero()[0], segment_no_outliers[not_nans])
        else:
            # Not enough points to interpolate, use mean
            segment_no_outliers[nans] = np.nanmean(segment_no_outliers)
        return segment_no_outliers
    else:
        return segment

def plot_multiple_csv(files, letter_id, session_data, Test=""):
    plt.figure(figsize=(12, 6))

    labels = ['Thumb', 'index', 'middle', 'ring', 'pinky', 'Palm', 'Wrist', 'Ulnar']
    # Define a list of markers
    markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D', 'h', '>', '<']

    # Loop through each file path and its corresponding marker
    for file_path, marker, label in zip(files, markers, labels):
        # Load data from the CSV file
        data = pd.read_csv(file_path, header=None)

        # Plotting the data with specified marker
        plt.plot(data[0], label=f'{label}', marker=marker, linestyle='-', markersize=5)

    # Customizing the plot
    plt.title('Plot of Multiple CSV Data')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}{Test}", "sensor_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def collect_data(session_data, letter_id, Test=""):
    if session_data['T'] == "":
        return "Enter UserName/Folder Name to Collect Data"

    create_path_folder(letter_id, Test, session_data)
    ser = init_serial()
    if not ser:
        return "❌ Serial connection failed."

    print("[SYNC] Collecting data synchronously for gesture:", gesture_map[letter_id])

    # Clear previous data
    for i in range(1, 9):
        session_data[f"a{i}"] = []

    gesture = gesture_map[letter_id]
    max_samples = 55 if Test == '' else 15

    while True:

        if data_collection_control["abort"]:
            print("[STATUS] Data collection aborted.")
            # Clear all Sensor arrays and close serial connection
            for i in range(1, 9):
                session_data[f"a{i}"] = []
            ser.close()
            break

        if data_collection_control["pause"]:
            ser.close()
            print("[STATUS] Pause detected. Cleaning up last incomplete group.")

            # Remove incomplete samples
            total_samples = len(session_data['a1'])
            if total_samples > 5:
                group_offset = (total_samples - 5) % 10
                if group_offset:
                    print(f"[STATUS] Discarding {group_offset} samples from current group.")
                    for i in range(1, 9):
                        session_data[f"a{i}"] = session_data[f"a{i}"][:-(group_offset)]

            # While paused, keep waiting
            while data_collection_control["pause"] and not data_collection_control["abort"]:
                time.sleep(0.1)

            # Once resume is clicked (pause becomes False), send 'R'
            if not data_collection_control["pause"] and not data_collection_control["abort"]:
                ser.open()
                print("[STATUS] Resumed data collection. Sent 'R' to Arduino.")

            continue

        if all(len(session_data[f"a{i}"]) >= max_samples for i in range(1, 9)):
            break

        try:
            ser.write(b'g')
            data = ser.readline().strip()  # Read the line of text from serial
            df = data.decode('utf-8', errors='ignore')
            try:
                sensor_number = int(df)
            except ValueError:
                continue  # Skip if cannot parse integer
            else:
                df = [sensor_number]

            # Read sensor data based on sensor number
            ser.write(b'g')
            data = ser.readline().strip()
            df_value = data.decode('utf-8', errors='ignore')
            try:
                sensor_value = int(df_value)
            except ValueError:
                continue

            # Append to appropriate sensor data array
            if df[0] == 1:
                session_data['a1'].append(sensor_value)
            elif df[0] == 2:
                session_data['a2'].append(sensor_value)
            elif df[0] == 3:
                session_data['a3'].append(sensor_value)
            elif df[0] == 4:
                session_data['a4'].append(sensor_value)
            elif df[0] == 5:
                session_data['a5'].append(sensor_value)
            elif df[0] == 6:
                session_data['a6'].append(sensor_value)
            elif df[0] == 7:
                session_data['a7'].append(sensor_value)
            elif df[0] == 8:
                session_data['a8'].append(sensor_value)
            else:
                # st.write(f"Unknown sensor number: {df[0]}")
                continue

            # if 1 <= sensor_number <= 8:
            #     session_data[f"a{sensor_number}"].append(sensor_value)

            if (len(session_data['a1']) % 5 == 0 and len(session_data['a2']) % 5 == 0 and
                len(session_data['a3']) % 5 == 0 and len(session_data['a4']) % 5 == 0 and
                len(session_data['a5']) % 5 == 0 and len(session_data['a6']) % 5 == 0 and
                len(session_data['a7']) % 5 == 0 and len(session_data['a8']) % 5 == 0):

                division = len(session_data['a1']) / 5
                if division in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
                    current_live_image_id["value"] = 0  # Display the rest image
                elif division in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                    current_live_image_id["value"] = letter_id # Display the gesture image

                a_values = [session_data[f"a{i}"][5:] for i in range(1, 9)]
                print("A1 to A8 (after 5 samples):", *a_values)
                log_filename = f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}{Test}/sensor_values.log"
                with open(log_filename, "a") as log_file:
                    # Convert the arrays to strings for logging from session_data dictionary
                    a1_str = str(session_data["a1"][5:])
                    a2_str = str(session_data["a2"][5:])
                    a3_str = str(session_data["a3"][5:])
                    a4_str = str(session_data["a4"][5:])
                    a5_str = str(session_data["a5"][5:])
                    a6_str = str(session_data["a6"][5:])
                    a7_str = str(session_data["a7"][5:])
                    a8_str = str(session_data["a8"][5:])

                    # Optionally write to log
                    log_file.write(f"A1: {a1_str}\nA2: {a2_str}\nA3: {a3_str}\nA4: {a4_str}\n"
                                   f"A5: {a5_str}\nA6: {a6_str}\nA7: {a7_str}\nA8: {a8_str}\n\n")

                A1 = np.array(session_data["a1"][5:], dtype=float)
                A2 = np.array(session_data["a2"][5:], dtype=float)
                A3 = np.array(session_data["a3"][5:], dtype=float)
                A4 = np.array(session_data["a4"][5:], dtype=float)
                A5 = np.array(session_data["a5"][5:], dtype=float)
                A6 = np.array(session_data["a6"][5:], dtype=float)
                A7 = np.array(session_data["a7"][5:], dtype=float)
                A8 = np.array(session_data["a8"][5:], dtype=float)

                A1 -= np.mean(A1[:5])
                A2 -= np.mean(A2[:5])
                A3 -= np.mean(A3[:5])
                A4 -= np.mean(A4[:5])
                A5 -= np.mean(A5[:5])
                A6 -= np.mean(A6[:5])
                A7 -= np.mean(A7[:5])
                A8 -= np.mean(A8[:5])

                VR = []
                VM = []
                data_stream = 50 if Test == "" else 10
                for i in range(0, data_stream, 10):
                    VR.extend(range(i, i + 5))
                    VM.extend(range(i + 5, i + 10))
                VR = np.array(VR)
                VM = np.array(VM)

                # Process each signal
                A1 = process_signal(A1, VR, VM)
                A2 = process_signal(A2, VR, VM)
                A3 = process_signal(A3, VR, VM)
                A4 = process_signal(A4, VR, VM)
                A5 = process_signal(A5, VR, VM)
                A6 = process_signal(A6, VR, VM)
                A7 = process_signal(A7, VR, VM)
                A8 = process_signal(A8, VR, VM)

                # Set VR indices to zero
                A1[VR] = 0
                A2[VR] = 0
                A3[VR] = 0
                A4[VR] = 0
                A5[VR] = 0
                A6[VR] = 0
                A7[VR] = 0
                A8[VR] = 0

                # Normalize VM indices
                A = np.vstack([A1, A2, A3, A4, A5, A6, A7, A8])
                for i in VM:
                    max_val = np.max(np.abs(A[:, i]))
                    if max_val != 0:
                        A[:, i] = A[:, i] / max_val

                log_filename_processing = f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}{Test}/sensor_values_after_processing.log"
                with open(log_filename_processing, "a") as log_file:
                    # Convert the arrays to strings for logging
                    a1_str = str(A1)
                    a2_str = str(A2)
                    a3_str = str(A3)
                    a4_str = str(A4)
                    a5_str = str(A5)
                    a6_str = str(A6)
                    a7_str = str(A7)
                    a8_str = str(A8)

                    # Create a single line of log output
                    log_line = (
                        "A1, A2, A3, A4, A5, A6, A7, A8: "
                        f"{a1_str} {a2_str} {a3_str} {a4_str} {a5_str} {a6_str} {a7_str} {a8_str}\n"
                    )

                    # Write the log line to the file
                    log_file.write(log_line)

                # Assuming image_number and Test are already defined variables
                user_folder = f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}{Test}"
                os.makedirs(user_folder, exist_ok=True)  # Ensure folder exists

                # Save each channel to CSV
                np.savetxt(f"{user_folder}/A1.csv", A1, delimiter=',')
                np.savetxt(f"{user_folder}/A2.csv", A2, delimiter=',')
                np.savetxt(f"{user_folder}/A3.csv", A3, delimiter=',')
                np.savetxt(f"{user_folder}/A4.csv", A4, delimiter=',')
                np.savetxt(f"{user_folder}/A5.csv", A5, delimiter=',')
                np.savetxt(f"{user_folder}/A6.csv", A6, delimiter=',')
                np.savetxt(f"{user_folder}/A7.csv", A7, delimiter=',')
                np.savetxt(f"{user_folder}/A8.csv", A8, delimiter=',')

                # Construct file paths
                file_paths = [f"{user_folder}/A{i}.csv" for i in range(1, 9)]

                # Plot the data (assumes function exists and works the same)
                plot_multiple_csv(file_paths, letter_id, session_data, Test)

                E1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                E8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                j_files = 5 if Test=='' else 1

                for sa in range(1, j_files + 1):
                    E1[sa - 1] = np.sum(A1[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E2[sa - 1] = np.sum(A2[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E3[sa - 1] = np.sum(A3[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E4[sa - 1] = np.sum(A4[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E5[sa - 1] = np.sum(A5[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E6[sa - 1] = np.sum(A6[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E7[sa - 1] = np.sum(A7[VM[(sa - 1) * 5: sa * 5]] ** 2)
                    E8[sa - 1] = np.sum(A8[VM[(sa - 1) * 5: sa * 5]] ** 2)

                Q = np.vstack([E1, E2, E3, E4, E5, E6, E7, E8])
                J0 = Q[:, 0] / np.max(Q[:, 0])
                J1 = Q[:, 1] / np.max(Q[:, 1])
                J2 = Q[:, 2] / np.max(Q[:, 2])
                J3 = Q[:, 3] / np.max(Q[:, 3])
                J4 = Q[:, 4] / np.max(Q[:, 4])
                print("Q,J0,J1,J2,J3,J4,J5,J6,J7,J8,J9", Q, J0, J1, J2, J3, J4)

                # Save the processed energy features into CSV files
                for k in range(j_files):
                    ii = str(letter_id)
                    jj = str(k + 1)
                    Filename = f'J{ii}.{jj}.csv'
                    my_list = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
                    energy_values = locals()[f'J{k}']

                    # Prepare data to save
                    # Convert energy values to a list of lists to match CSV writing format
                    data_to_save = [energy_values.tolist()]

                    save_path = f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}{Test}/{Filename}"
                    with open(save_path, 'w', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow(my_list)
                        wr.writerows(data_to_save)

        except Exception as e:
            print(f"[SYNC ERROR] {e}")
            continue
    current_live_image_id["value"] = 0
    data_collection_control["data_collection_status"] = "✅ Data collection completed."
    ser.close()
    print("[SYNC] Completed data collection for:", gesture)

    # Save raw data as CSV
    folder = f"../GloveSenseDash/G_Alph_{session_data['T']}/Gesture {letter_id}"
    os.makedirs(folder, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"{gesture}_{ts}.csv")

    df = pd.DataFrame({f"a{i}": session_data[f"a{i}"] for i in range(1, 9)})
    df.to_csv(file_path, index=False)

    return f"✅ Synchronous data collected for gesture {gesture} and saved to {file_path}"

@app.callback(
    Output("get-test-data", "disabled"),
    Input("test-dropdown", "value")
)
def toggle_test_button_state(gesture_selected):
    return gesture_selected is None

# @app.callback(
#     Output("gesture-display", "children", allow_duplicate=True),
#     Input("live-image", "data"),
#     prevent_initial_call=True
# )
# def update_image(img_id):
#     name = gesture_map.get(img_id, "Unknown")
#     path = f"gestures/{img_id}.jpg"
#     return html.Div([
#         html.Img(src=get_image_base64(path), style={
#             "width": "300px", "height": "400px", "borderRadius": "10px",
#             "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"
#         }),
#         html.H5(f"Letter: {name}", className="text-center mt-2 text-primary")
#     ])


@app.callback(
    Output("toggle-pause-btn", "children"),
    Input("toggle-pause-btn", "n_clicks"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def toggle_pause_resume(n_clicks, session_data):
    if n_clicks is None:
        raise PreventUpdate

    # Toggle pause state
    data_collection_control["pause"] = not data_collection_control["pause"]

    if data_collection_control["pause"]:
        print("[STATUS] Data collection paused.")
        return "▶️ Resume Data Collection"
    else:
        print("[STATUS] Data collection resumed.")
        return "⏸️ Pause Data Collection"

# Abort logic button
@app.callback(
    Output("train-status", "children", allow_duplicate=True),
    Input("abort-btn", "n_clicks"),
    prevent_initial_call=True
)
def abort_data_collection(n):
    data_collection_control["abort"] = True
    print("[STATUS] Data collection abort requested.")
    return "❌ Data collection aborted."

@app.callback(
    Output("live-image", "data", allow_duplicate=True),
    Input("interval-refresh", "n_intervals"),
    prevent_initial_call = True
)
def poll_live_image(n):
    return current_live_image_id["value"]

@app.callback(
    Output("train-status", "children", allow_duplicate=True),
    # Output("live-image", "data"),
    Input("collect-data-btn", "n_clicks"),
    State("session-store", "data"),
    State("current-letter", "data"),
    prevent_initial_call=True
)
def collect_train_data(n, session_data, letter_id):
    # if not n:
    #     raise PreventUpdate
    # thread = threading.Thread(target=collect_data, args=(session_data, letter_id,), daemon=True)
    # thread.start()
    if not n:
        raise PreventUpdate

        # Small helper function to run inside thread
    data_collection_control["data_collection_status"] = "⏳ Collecting data..."
    def run_data_collection():
        collect_data(session_data, letter_id)
        data_collection_control["data_collection_status"] = "Data Collection completed."
        return data_collection_control["data_collection_status"]

    # Start the collection in background
    thread = threading.Thread(target=run_data_collection, daemon=True)
    thread.start()

    # Immediately return the "Collecting..." status to UI
    return data_collection_control["data_collection_status"]

@app.callback(
    # Output("test-status", "children"),
    # Output("live-image", "data", allow_duplicate=True),
    Input("get-test-data", "n_clicks"),
    State("session-store", "data"),
    State("test-dropdown", "value"),
    prevent_initial_call=True
)
def collect_test_data(n, session_data, test_gesture_id):
    if not n:
        raise PreventUpdate
    thread = threading.Thread(target=collect_data, args=(session_data, test_gesture_id, "Test"), daemon=True)
    thread.start()
    # result = collect_data(session_data, test_gesture_id, Test="Test")
    # return result

@app.callback(
    # Output("train-status", "children", allow_duplicate=True),
    # Output("live-image", "data", allow_duplicate=True),
    Input("repeat-btn", "n_clicks"),
    State("session-store", "data"),
    State("repeat-gesture-dropdown", "value"),
    prevent_initial_call=True
)
def collect_train_drop_data(n, session_data, train_gesture_id,):
    if not n:
        raise PreventUpdate
    thread = threading.Thread(target=collect_data, args=(session_data, train_gesture_id,), daemon=True)
    thread.start()
    # result = collect_data(session_data, train_gesture_id, Test="")
    # return result

def load_training_data(session_data):
    gesture_map_for_train = {
        1: "A", 2: "B", 3: "C", 4: "D", 5: "F",
        6: "G", 7: "H", 8: "I", 9: "L", 10: "N", 11: "W",
        12: "Y", 13: "TR", 14: "TIM", 15: "TMR"
    }
    X, y = [], []
    for gid in gesture_map_for_train.keys():
        folder = os.path.join(TRAIN_ROOT, f"G_Alph_{session_data['T']}/Gesture {gid}")
        if not os.path.exists(folder):
            continue
        for i in range(1, 6):
            fname = os.path.join(folder, f"J{gid}.{i}.csv")
            if os.path.exists(fname):
                print(fname)
                df = pd.read_csv(fname)
                if df.shape[1] > 0:
                    X.append(df.values.flatten())
                    y.append(gid)
    return np.array(X), np.array(y)

def load_test_data(selected_gesture_id):
    X_test, y_test = [], []

    # Build test path only for the selected gesture
    test_path = os.path.join(TRAIN_ROOT, f"G_Alph_/Gesture {selected_gesture_id}Test", f"J{selected_gesture_id}.1.csv")

    if os.path.exists(test_path):
        df = pd.read_csv(test_path)
        if df.shape[1] > 0:
            X_test.append(df.values.flatten())
            y_test.append(selected_gesture_id)

    return np.array(X_test), np.array(y_test)

# Train models
def train_models(X, y, session_data):
    if session_data['T'] =="":
        print("[STATUS] Enter Folder Name First.")

    os.makedirs(os.path.join(TRAIN_ROOT, f"G_Alph_{session_data['T']}\model"), exist_ok=True)

    MODEL_PATHS = {
        "rf": os.path.join(TRAIN_ROOT, f"G_Alph_{session_data['T']}\model", "rf_model.pkl"),
    }

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, os.path.join(TRAIN_ROOT, f"G_Alph_{session_data['T']}\model", "label_encoder.pkl"))

    models = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    accs = {}

    for key, model in models.items():
        model.fit(X, y_encoded)
        joblib.dump(model, MODEL_PATHS[key])
        accs[key] = accuracy_score(y_encoded, model.predict(X))
    return accs

@app.callback(
    Output("train-status", "children", allow_duplicate=True),
    Input("train-btn", "n_clicks"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def train_model(n, session_data):
    try:
        X, y = load_training_data(session_data)
        if X.size == 0 or y.size == 0:
            return "❌ Training failed: No data found."
        accs = train_models(X, y, session_data)
        lines = [f"✅ {name.upper()} trained - Accuracy: {acc:.2f}" for name, acc in accs.items()]
        return html.Div([html.Div(line) for line in lines])
    except Exception as e:
        return f"❌ Training failed: {str(e)}"


@app.callback(
    Output("test-status", "children"),
    Input("test-btn", "n_clicks"),
    State("test-dropdown", "value"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def test_model(n_clicks, selected_gesture_id, session_data):
    if not n_clicks or selected_gesture_id is None:
        raise PreventUpdate

    try:
        # Load test data for the selected gesture
        X_test, y_test = load_test_data(selected_gesture_id)

        if X_test.size == 0:
            return "❌ No test data found for selected gesture."

        # Load Label Encoder
        le_path = os.path.join(TRAIN_ROOT, f"G_Alph_{session_data['T']}", "model", "label_encoder.pkl")
        if not os.path.exists(le_path):
            return "❌ Label encoder not found. Train first."
        le = joblib.load(le_path)

        # Load trained model (Random Forest here)
        model_path = os.path.join(TRAIN_ROOT, f"G_Alph_{session_data['T']}", "model", "rf_model.pkl")
        if not os.path.exists(model_path):
            return "❌ Trained model not found. Train first."
        model = joblib.load(model_path)

        # Predict
        y_pred_encoded = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)

        expected_gesture = gesture_map.get(y_test[0], "Unknown")
        predicted_gesture = gesture_map.get(y_pred[0], "Unknown")
        smile_path = "/gestures/smile.jpg"
        sad_path = "/gestures/sad.jpg"

        # If prediction correct
        if y_pred[0] == y_test[0]:
            return html.Div([
                html.H4(f"🎯 Correct! Predicted: {predicted_gesture}", style={"color": "#28a745", "marginTop": "10px"}),
                html.Img(src= get_image_base64(smile_path), style={"width": "150px", "marginTop": "20px"})
            ])
        else:
            return html.Div([
                html.H4(f"❌ Incorrect Prediction.", style={"color": "#dc3545", "marginTop": "10px"}),
                html.H5(f"Predicted: {predicted_gesture}", style={"color": "#007bff"}),
                html.H5(f"Expected: {expected_gesture}", style={"color": "#6c757d"}),
                html.Img(src=get_image_base64(sad_path), style={"width": "150px", "marginTop": "20px"})
            ])

    except Exception as e:
        return f"❌ Testing failed: {str(e)}"

# Run app
if __name__ == '__main__':
    app.run(debug=True)