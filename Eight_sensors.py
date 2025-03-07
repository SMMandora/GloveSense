from itertools import zip_longest
import joblib
import numpy as np
import pandas as pd
import serial
import streamlit as st
from PIL import Image
import os
import csv
import time  # For time.sleep()

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from streamlit import pyplot
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(page_title="Glove Sense", layout="wide")

# Title
st.title("Glove Sense")

# Initialize serial communication
try:
    ser = serial.Serial(port='COM3', baudrate=115200, timeout=1)
    ser.close()
except OSError:
    st.error("The glove is either not connected or the serial port is not available!")

# Initialize session state variables

def reset_session_state():
    # List of keys to reset
    keys_to_reset = [
        'T', 'cookies', 'SelectedOption', 'SelectedOptionRepeat',
        'a1', 'a2', 'a3', 'a4', 'a5', 'Energies', 'Gst', 'tFeature',
        'tFeature_Df', 'tTarget', 'tFeature1', 'tTarget1', 'll',
        'isFirstClick', 'Trial', 'model_scores', 'pause', 'abort', 'data_collection_complete'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize placeholders if needed
    st.session_state.image_placeholder = st.empty()
    st.session_state.letter_placeholder = st.empty()

    st.experimental_set_query_params(rerun=str(time.time()))

image_placeholder = st.empty()
letter_placeholder = st.empty()
if 'pause' not in st.session_state:
    st.session_state.pause = False
if 'abort' not in st.session_state:
    st.session_state.abort = False
if 'data_collection_complete' not in st.session_state:
    st.session_state.data_collection_complete = False
if 'T' not in st.session_state:
    st.session_state.T = ""
if 'cookies' not in st.session_state:
    st.session_state.cookies = 0
if 'SelectedOption' not in st.session_state:
    st.session_state.SelectedOption = 1
if 'SelectedOptionRepeat' not in st.session_state:
    st.session_state.SelectedOptionRepeat = 1
if 'a1' not in st.session_state:
    st.session_state.a1 = []
if 'a2' not in st.session_state:
    st.session_state.a2 = []
if 'a3' not in st.session_state:
    st.session_state.a3 = []
if 'a4' not in st.session_state:
    st.session_state.a4 = []
if 'a5' not in st.session_state:
    st.session_state.a5 = []
if 'a6' not in st.session_state:
    st.session_state.a6 = []
if 'a7' not in st.session_state:
    st.session_state.a7 = []
if 'a8' not in st.session_state:
    st.session_state.a8 = []
# Initialize placeholders for the image and letter
if 'image_placeholder' not in st.session_state:
    st.session_state.image_placeholder = st.empty()
if 'letter_placeholder' not in st.session_state:
    st.session_state.letter_placeholder = st.empty()
plot_placeholder = st.empty()


# Define functions
def ensure_string_columns(df):
    df.columns = df.columns.astype(str)
    return df


def CreateMainFolder(G):
    if G != "":
        pathFolder = "../GloveSense/G_Alph_" + str(G)
        st.session_state.T = G
        try:
            os.mkdir(pathFolder)
            st.success(f"Successfully created the directory {pathFolder}")
        except OSError:
            st.warning(f"The directory {pathFolder} already exists.")
    else:
        st.error("Please enter your name and create your folder")


def makeFig(a1, a2, a3, a4, a5):
    fig = plt.figure(figsize=(10, 8))

    # Plot A1
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(a1, marker='o')
    ax1.set_title('Thumb finger')

    # Plot A2
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(a2, marker='o')
    ax2.set_title('Index finger')

    # Plot A3
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(a3, marker='o')
    ax3.set_title('Middle finger')

    # Plot A4
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(a4, marker='o')
    ax4.set_title('Ring finger')

    # Plot A5
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(a5, marker='o')
    ax5.set_title('Pinky finger')

    # Adjust layout
    plt.tight_layout()

    return fig



def stop_function():
    # Reset the data arrays
    st.session_state.a1 = []
    st.session_state.a2 = []
    st.session_state.a3 = []
    st.session_state.a4 = []
    st.session_state.a5 = []
    st.session_state.a6 = []
    st.session_state.a7 = []
    st.session_state.a8 = []
    st.info("Please repeat the same gesture!")

def get_letters(image_number, letter_placeholder):
    # Map image numbers to letters
    letters = {
        0: "RS", 1: "A", 2: "B", 3: "C", 4: "D",
        5: "F", 6: "G", 7: "H", 8: "I", 9: "L",
        10: "N", 11: "W", 12: "Y", 13: "TR", 14: "TIM", 15: "TMR"
    }
    letter = letters.get(image_number, "Unknown")
    # Use the letter placeholder to update the letter in the same spot
    letter_placeholder.write(f"Letter: {letter}")

def get_image(image_number, image_placeholder, letter_placeholder):
    # Display the gesture image and corresponding letter
    try:
        img_path = f"../GloveSense/gestures/{image_number}.JPG"
        img = Image.open(img_path)
        img = img.resize((300, 400))
        # Use the image placeholder to display the image in the same spot
        image_placeholder.image(img)
        get_letters(image_number, letter_placeholder)
    except FileNotFoundError:
        st.warning(f"Image not found for gesture {image_number}")
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")

def CreatePathFolder(G, Test):
    pathFolder = f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {G}{Test}"
    try:
        os.mkdir(pathFolder)
        st.write(f"Successfully created the directory {pathFolder}")
    except OSError:
        st.warning(f"The directory {pathFolder} already exists.")

# def plot_energies(E1, E2, E3, E4, E5, gesture_number):
#     fig, axs = plt.subplots(3, 2, figsize=(10, 8))
#
#     # Plot Energy E1
#     axs[0, 0].plot(E1, marker='o')
#     axs[0, 0].set_title('Thumb Finger Energy')
#
#     # Plot Energy E2
#     axs[0, 1].plot(E2, marker='o')
#     axs[0, 1].set_title('Index Finger Energy')
#
#     # Plot Energy E3
#     axs[1, 0].plot(E3, marker='o')
#     axs[1, 0].set_title('Middle Finger Energy')
#
#     # Plot Energy E4
#     axs[1, 1].plot(E4, marker='o')
#     axs[1, 1].set_title('Ring Finger Energy')
#
#     # Plot Energy E5
#     axs[2, 0].plot(E5, marker='o')
#     axs[2, 0].set_title('Pinky Finger Energy')
#
#     # Remove the unused subplot (3,2)
#     fig.delaxes(axs[2, 1])
#
#     fig.suptitle(f'Gesture {gesture_number} Energy Readings', fontsize=16)
#     fig.tight_layout()
#     #sensor_plot_placeholder.pyplot(fig)
#     plt.close(fig)


# def plot_energies_after_data_collection(gesture_number, Test):
#     # Initialize data lists
#     A1_list = []
#     A2_list = []
#     A3_list = []
#     A4_list = []
#     A5_list = []
#
#     # Read the saved CSV files
#     for x in range(1, 6):  # Assuming 5 CSV files per gesture
#         file_path = f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {gesture_number} {Test}/J{gesture_number}.{x}.csv"
#         try:
#             mat = pd.read_csv(file_path)
#             # Append data from each column to the corresponding list
#             A1_list.extend(mat['A1'].tolist())
#             A2_list.extend(mat['A2'].tolist())
#             A3_list.extend(mat['A3'].tolist())
#             A4_list.extend(mat['A4'].tolist())
#             A5_list.extend(mat['A5'].tolist())
#         except FileNotFoundError:
#             st.error(f"File not found: {file_path}")
#             return  # Stop execution if file not found
#         except Exception as e:
#             st.error(f"An error occurred while reading {file_path}: {e}")
#             return  # Stop execution if any other error occurs
#
#     # Plot the raw data
#     plot_energies(A1_list, A2_list, A3_list, A4_list, A5_list, gesture_number)



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

def plot_multiple_csv(files, image_number, Test):
    plt.figure(figsize=(12, 6))

    lables = ['Thumb', 'index', 'middle', 'ring', 'pinky', 'Palm', 'Wrist', 'Ulnar']
    # Define a list of markers
    markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D', 'h', '>', '<']

    # Loop through each file path and its corresponding marker
    for file_path, marker, label in zip(files, markers, lables):
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

    save_path = os.path.join(f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}", "sensor_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def ReadAndSave_TrainData(image_number, Test):
    get_image(0, image_placeholder, letter_placeholder)  # Display the rest image
    CreatePathFolder(image_number, Test)
    # Ensure the serial port is open
    if not ser.is_open:
        ser.open()
    else:
        # If already open, flush the input buffer
        ser.reset_input_buffer()

    # Wait briefly to allow the device to initialize
    time.sleep(1)
    j = 1

    # Initialize the abort and pause flags if not already set
    if 'abort' not in st.session_state:
        st.session_state.abort = False
    if 'pause' not in st.session_state:
        st.session_state.pause = False

    while True:
        # Check if the abort flag is set
        if st.session_state.abort:
            # Reset the abort flag
            st.session_state.abort = False
            # Reset data arrays
            st.session_state.a1 = []
            st.session_state.a2 = []
            st.session_state.a3 = []
            st.session_state.a4 = []
            st.session_state.a5 = []
            st.session_state.a6 = []
            st.session_state.a7 = []
            st.session_state.a8 = []
            st.info("Data collection aborted. Restarting data collection for this gesture.")
            # Optionally, you can display the rest image again
            get_image(0, image_placeholder, letter_placeholder)  # Display the rest image
            # Continue to restart data collection
            continue

        # Check if the pause flag is set
        if st.session_state.pause:
            # Display a message that data collection is paused
            st.info("Data collection paused. Click 'Resume Data Collection' to continue.")
            # Wait and yield control to Streamlit
            time.sleep(0.1)
            continue  # Go back to the top of the loop

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

            continue  # Skip if cannot parse integer

            # Append to appropriate sensor data array
        if df[0] == 1:
            st.session_state.a1.append(sensor_value)
        elif df[0] == 2:
            st.session_state.a2.append(sensor_value)
        elif df[0] == 3:
            st.session_state.a3.append(sensor_value)
        elif df[0] == 4:
            st.session_state.a4.append(sensor_value)
        elif df[0] == 5:
            st.session_state.a5.append(sensor_value)
        elif df[0] == 6:
            st.session_state.a6.append(sensor_value)
        elif df[0] == 7:
            st.session_state.a7.append(sensor_value)
        elif df[0] == 8:
            st.session_state.a8.append(sensor_value)
        else:
            # st.write(f"Unknown sensor number: {df[0]}")
            continue
        j = j + 1

        # Determine if it's time to rest or beep
        if (len(st.session_state.a1) % 5 == 0 and len(st.session_state.a2) % 5 == 0 and
                len(st.session_state.a3) % 5 == 0 and len(st.session_state.a4) % 5 == 0 and
                len(st.session_state.a5) % 5 == 0 and len(st.session_state.a6) % 5 == 0 and len(st.session_state.a7) % 5 == 0 and len(st.session_state.a8) % 5 == 0):

            division = len(st.session_state.a1) / 5
            if division in [1, 3, 5, 7, 9]:
                # st.write('REST', division)
                get_image(0, image_placeholder, letter_placeholder)  # Display the rest image
                time.sleep(2)
            elif division in [2, 4, 6, 8, 10, 12]:
                # st.write('BEEP', division)
                get_image(int(image_number), image_placeholder, letter_placeholder)  # Display the gesture image
                time.sleep(2)
                # When sufficient data is collected
            min_length = min(len(st.session_state.a1), len(st.session_state.a2),
                             len(st.session_state.a3), len(st.session_state.a4),
                             len(st.session_state.a5), len(st.session_state.a6), len(st.session_state.a7),len(st.session_state.a8))
            if min_length == 55:
                # Convert lists to numpy arrays and subtract mean of first 5 samples
                print("A1, A2, A3, A4, A5, A6, A7, A8", st.session_state.a1[5:], st.session_state.a2[5:], st.session_state.a3[5:], st.session_state.a4[5:], st.session_state.a5[5:], st.session_state.a6[5:], st.session_state.a7[5:], st.session_state.a8[5:])
                log_filename = f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/sensor_values.log"
                with open(log_filename, "a") as log_file:
                    # Convert the arrays to strings for logging
                    a1_str = str(st.session_state.a1[5:])
                    a2_str = str(st.session_state.a2[5:])
                    a3_str = str(st.session_state.a3[5:])
                    a4_str = str(st.session_state.a4[5:])
                    a5_str = str(st.session_state.a5[5:])
                    a6_str = str(st.session_state.a6[5:])
                    a7_str = str(st.session_state.a7[5:])
                    a8_str = str(st.session_state.a8[5:])

                    # Create a single line of log output
                    log_line = (
                        "A1, A2, A3, A4, A5, A6, A7, A8: "
                        f"{a1_str} {a2_str} {a3_str} {a4_str} {a5_str} {a6_str} {a7_str} {a8_str}\n"
                    )

                    # Write the log line to the file
                    log_file.write(log_line)


                A1 = np.array(st.session_state.a1[5:], dtype=float)
                A2 = np.array(st.session_state.a2[5:], dtype=float)
                A3 = np.array(st.session_state.a3[5:], dtype=float)
                A4 = np.array(st.session_state.a4[5:], dtype=float)
                A5 = np.array(st.session_state.a5[5:], dtype=float)
                A6 = np.array(st.session_state.a6[5:], dtype=float)
                A7 = np.array(st.session_state.a7[5:], dtype=float)
                A8 = np.array(st.session_state.a8[5:], dtype=float)

                A1 -= np.mean(A1[:5])
                A2 -= np.mean(A2[:5])
                A3 -= np.mean(A3[:5])
                A4 -= np.mean(A4[:5])
                A5 -= np.mean(A5[:5])
                A6 -= np.mean(A6[:5])
                A7 -= np.mean(A7[:5])
                A8 -= np.mean(A8[:5])

                # Define VR and VM ranges
                VR = []
                VM = []
                for i in range(0, 50, 10):
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

                log_filename_processing = f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/sensor_values_after_processing.log"
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

                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A1.csv', A1, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A2.csv', A2, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A3.csv', A3, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A4.csv', A4, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A5.csv', A5, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A6.csv', A6, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A7.csv', A7, delimiter=',')
                np.savetxt(f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A8.csv', A8, delimiter=',')

                file_paths = [f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A1.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A2.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A3.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A4.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A5.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A6.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A7.csv', f'../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/A8.csv']

                plot_multiple_csv(file_paths, image_number, Test)


                E1 = [0, 0, 0, 0, 0]
                E2 = [0, 0, 0, 0, 0]
                E3 = [0, 0, 0, 0, 0]
                E4 = [0, 0, 0, 0, 0]
                E5 = [0, 0, 0, 0, 0]
                E6 = [0, 0, 0, 0, 0]
                E7 = [0, 0, 0, 0, 0]
                E8 = [0, 0, 0, 0, 0]
                for sa in range(1, 6):
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
                print("Q,J0,J1,J2,J3,J4,J5,J6,J7,J8",Q,J0,J1,J2,J3,J4)

                # Save the processed energy features into CSV files
                for k in range(5):
                    ii = str(image_number)
                    jj = str(k + 1)
                    Filename = f'J{ii}.{jj}.csv'
                    my_list = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
                    energy_values = locals()[f'J{k}']

                        # Prepare data to save
                    # Convert energy values to a list of lists to match CSV writing format
                    data_to_save = [energy_values.tolist()]

                    save_path = f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {image_number}{Test}/{Filename}"
                    with open(save_path, 'w', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow(my_list)
                        wr.writerows(data_to_save)

                # Reset data
                st.session_state.a1 = []
                st.session_state.a2 = []
                st.session_state.a3 = []
                st.session_state.a4 = []
                st.session_state.a5 = []
                st.session_state.a6 = []
                st.session_state.a7 = []
                st.session_state.a8 = []
                break

        # Yield control to Streamlit to update the UI
        time.sleep(0)

    ser.close()

def IncreaseGestureNumber():
    st.session_state.cookies += 1
    return st.session_state.cookies

def GetTrainData():
    try:
        if st.session_state.T != "":
            GNumber = st.session_state.cookies
            if GNumber < 13:
                GNumber = IncreaseGestureNumber()
                st.write(f"Gesture Number increased: {GNumber}")
                st.session_state.data_collection_complete = False
                ReadAndSave_TrainData(GNumber, "")
            else:
                st.write("Training completed.")
        else:
            st.warning("Please enter your name and create your folder.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def RepeatData():
    try:
        if st.session_state.T != "":
            GNumber = st.session_state.SelectedOptionRepeat
            st.session_state.data_collection_complete = False
            ReadAndSave_TrainData(GNumber, "")
        else:
            st.warning("Please enter your name and create your folder")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def GetDataGesture(gesture_number,test = ""):
    E1 = []
    E2 = []
    E3 = []
    E4 = []
    E5 = []
    E6 = []
    E7 = []
    E8 = []
    # Loop to read each CSV file
    for x in range(1, 11):
        # Load CSV file into DataFrame
        try:
            mat = pd.read_csv(f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {gesture_number}{test}/J{gesture_number}.{x}.csv")
            print("glovereadingpath"+f"../GloveSense/G_Alph_{st.session_state.T}/Gesture {gesture_number}{test}/J{gesture_number}.{x}.csv")
            # Read the energy values directly
            # Assuming the CSV has columns 'E1', 'E2', 'E3', 'E4', 'E5'
            # and only one data row
            E1_value = mat['E1'].iloc[0]
            E2_value = mat['E2'].iloc[0]
            E3_value = mat['E3'].iloc[0]
            E4_value = mat['E4'].iloc[0]
            E5_value = mat['E5'].iloc[0]
            E6_value = mat['E6'].iloc[0]
            E7_value = mat['E7'].iloc[0]
            E8_value = mat['E8'].iloc[0]
            # Append the energy values to the respective lists
            E1.append(E1_value)
            E2.append(E2_value)
            E3.append(E3_value)
            E4.append(E4_value)
            E5.append(E5_value)
            E6.append(E6_value)
            E7.append(E7_value)
            E8.append(E8_value)
            if test == "Test":
                break
        except FileNotFoundError:
            st.error(f"File not found. Please repeat Gesture {gesture_number} again.")


    return E1, E2, E3, E4, E5, E6, E7, E8

def callAllGesturesData():
    st.session_state.Energies = {}
    for gesture_number in range(1, 16):
        E1, E2, E3, E4, E5, E6, E7, E8 = GetDataGesture(gesture_number)
        st.session_state.Energies[gesture_number] = {
            'E1': E1, 'E2': E2, 'E3': E3, 'E4': E4, 'E5': E5, 'E6': E6, 'E7': E7, 'E8': E8
        }


def CombineAllEnergies():
    st.session_state.Gst = []
    for gesture_number in range(1, 16):
        energies = st.session_state.Energies[gesture_number]
        # Ensure there are at least 5 samples
        min_length = min(len(energies[key]) for key in energies)
        if min_length < 5:
            st.error(f"Not enough samples for gesture {gesture_number}. Expected at least 5, got {min_length}.")
            return  # Stop execution
        G = []
        for i in range(5):  # Use only the first 5 samples
            G.append([
                energies['E1'][i],
                energies['E2'][i],
                energies['E3'][i],
                energies['E4'][i],
                energies['E5'][i],
                energies['E6'][i],
                energies['E7'][i],
                energies['E8'][i]
            ])
        st.session_state.Gst.append(G)

def GetFeatures(data):
    Features = []
    for x in data:
        for i in x:
            Features.append(i)
    return Features


def GetTrialTargets(Trial):
    Target = []
    for x in range(1, 16):
        for j in range(1, (len(Trial) * 5) + 1):
            Target.append(x)
    return Target


def LoadFeaturesAndTargets():
    # fill_AllEsOutliers()
    callAllGesturesData()
    CombineAllEnergies()
    st.session_state.tFeature = GetFeatures(st.session_state.Gst)
    st.session_state.tFeature_Df = pd.DataFrame(st.session_state.tFeature)
    st.session_state.tTarget = GetTrialTargets(st.session_state.Trial)
    st.session_state.tFeature1 = st.session_state.tFeature_Df
    st.session_state.tTarget1 = st.session_state.tTarget
    st.session_state.ll = len(st.session_state.Trial)


def on_start_button_click():
    if 'isFirstClick' not in st.session_state:
        st.session_state.isFirstClick = True
    if st.session_state.isFirstClick:
        # Initialize Trial if not already set
        if 'Trial' not in st.session_state:
            st.session_state.Trial = [1]
        LoadFeaturesAndTargets()
        st.session_state.isFirstClick = False


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def findindex(features, feature):
    FeatureT = []
    for i in feature:
        FeatureT.append(features[int(i)])
    return FeatureT

def getAllTrainItems(data):
    TrainData = []
    for x in data:
        for i in x:
            TrainData.append(i)
    return TrainData


def GetTrainModel_FullTrained():
    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # Initialize AdaBoostClassifier with a DecisionTree base estimator
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada_model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Retrieve training features and targets from session state
    tFeature1 = st.session_state.tFeature1
    tTarget1 = st.session_state.tTarget1

    print("tFeature1, tTarget1", tFeature1, tTarget1)

    tFeature1 = ensure_string_columns(tFeature1)
    tFeature1.columns = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']

    # Fit the Random Forest model
    rf_model.fit(tFeature1, tTarget1)
    rf_model_filename = "model_FullTrained_RandomForest.joblib"
    joblib.dump(rf_model, rf_model_filename)

    # xgb_model.fit(tFeature1, tTarget1)
    # xgb_model_filename = "model_FullTrained_XGB.joblib"
    # joblib.dump(xgb_model, xgb_model_filename)

    # Hyperparameter tuning for AdaBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=ada_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(tFeature1, tTarget1)

    print("Best AdaBoost Parameters:", grid_search.best_params_)
    best_ada = grid_search.best_estimator_

    # Save the best AdaBoost model
    ada_model_filename = "model_FullTrained_AdaBoost.joblib"
    joblib.dump(best_ada, ada_model_filename)


def btn_Train_Model():
    if st.session_state.T != "":
        on_start_button_click()
        GetTrainModel_FullTrained()

        # Load the saved Random Forest model
        rf_model = joblib.load('model_FullTrained_RandomForest.joblib')
        # Load the saved AdaBoost model
        ada_model = joblib.load('model_FullTrained_AdaBoost.joblib')
        # xgb_model = joblib.load('model_FullTrained_XGB.joblib')
        # Retrieve the original (unnormalized) training features and targets
        tFeature1 = st.session_state.tFeature1  # Should be a DataFrame
        tTarget1 = st.session_state.tTarget1  # Should be a Series or 1D array

        tFeature1 = ensure_string_columns(tFeature1)
        # Rename columns to 'E1' to 'E8' to match testing data
        tFeature1.columns = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']

        # Predict using the Random Forest model on training data
        y_pred_rf = rf_model.predict(tFeature1)
        # Predict using the AdaBoost model on training data
        y_pred_ada = ada_model.predict(tFeature1)

        # y_pred_xgb = xgb_model.predict(tFeature1)

        # Evaluate the Random Forest model
        accuracy_rf = accuracy_score(tTarget1, y_pred_rf)
        # Evaluate the AdaBoost model
        accuracy_ada = accuracy_score(tTarget1, y_pred_ada)

        # accuracy_xgb = accuracy_score(tTarget1, y_pred_xgb)

        # Display the accuracies
        st.write(f"Random Forest Training Accuracy: {accuracy_rf:.2f}")
        st.write(f"AdaBoost Training Accuracy: {accuracy_ada:.2f}")
        # st.write(f"XGB Training Accuracy: {accuracy_xgb:.2f}")
    else:
        st.warning("Please enter your name and create your folder")


def Get_Test_GestureData():
    if st.session_state.T != "":
        GNumber = st.session_state.SelectedOption
        st.write(f"Collecting test data for Gesture {GNumber}")
        st.session_state.data_collection_complete = False
        ReadAndSave_TrainData(GNumber, "Test")
    else:
        st.warning("Please enter your name and create your folder")


def btn_Test_Model(SelectedOption):
    if st.session_state.T != "":
        on_start_button_click()

        # Define paths to both models
        rf_model_path = 'model_FullTrained_RandomForest.joblib'
        ada_model_path = 'model_FullTrained_AdaBoost.joblib'

        # Get test data
        E1_test, E2_test, E3_test, E4_test, E5_test, E6_test, E7_test, E8_test = GetDataGesture(SelectedOption, "Test")

        # Ensure we have data
        if not E1_test:
            st.error("No test data found. Please get Gesture test data.")
            return

        # Build the test feature set as a DataFrame
        FeatureGTest_Df = pd.DataFrame({
            'E1': E1_test,
            'E2': E2_test,
            'E3': E3_test,
            'E4': E4_test,
            'E5': E5_test,
            'E6': E6_test,
            'E7': E7_test,
            'E8': E8_test
        })

        FeatureGTest_Df = ensure_string_columns(FeatureGTest_Df)
        FeatureGTest_Df.columns = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']

        # Load the trained Random Forest model
        try:
            rf_model = joblib.load(rf_model_path)
        except FileNotFoundError:
            st.error(f"Random Forest model not found at {rf_model_path}. Please train the models first.")
            return

        # Load the trained AdaBoost model
        try:
            ada_model = joblib.load(ada_model_path)
        except FileNotFoundError:
            st.error(f"AdaBoost model not found at {ada_model_path}. Please train the models first.")
            return

        # Predict using the Random Forest model on test data
        y_pred_rf = rf_model.predict(FeatureGTest_Df)

        # Predict using the AdaBoost model on test data
        y_pred_ada = ada_model.predict(FeatureGTest_Df)

        # Map indices to gesture letters
        gesture_mapping = {
            1: "A", 2: "B", 3: "C", 4: "D",
            5: "F", 6: "G", 7: "H", 8: "I", 9: "L",
            10: "N", 11: "W", 12: "Y", 13: "TR", 14: "TIM", 15: "TMR"
        }

        predicted_letters_rf = [gesture_mapping.get(int(pred), "Unknown") for pred in y_pred_rf]
        predicted_letters_ada = [gesture_mapping.get(int(pred), "Unknown") for pred in y_pred_ada]

        # Display the predictions
        st.subheader("Test Gesture Predictions")

        # Create a DataFrame to neatly display predictions from both models
        predictions_df = pd.DataFrame({
            'E1': E1_test,
            'E2': E2_test,
            'E3': E3_test,
            'E4': E4_test,
            'E5': E5_test,
            'E6': E6_test,
            'E7': E7_test,
            'E8': E8_test,
            'Random Forest Prediction': predicted_letters_rf,
            'AdaBoost Prediction': predicted_letters_ada
        })

        st.dataframe(predictions_df)

        # Optionally, provide separate sections for each model
        st.write("### Random Forest Predictions")
        st.write(predicted_letters_rf)

        st.write("### AdaBoost Predictions")
        st.write(predicted_letters_ada)

        # If true labels are available, display evaluation metrics
        if 'tTarget_test' in st.session_state:
            tTarget_test = st.session_state.tTarget_test

            from sklearn.metrics import classification_report, confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.subheader("Evaluation Metrics")

            # Accuracy Scores
            accuracy_rf = accuracy_score(tTarget_test, y_pred_rf)
            accuracy_ada = accuracy_score(tTarget_test, y_pred_ada)

            st.write(f"**Random Forest Accuracy:** {accuracy_rf:.2f}")
            st.write(f"**AdaBoost Accuracy:** {accuracy_ada:.2f}")

            # Classification Reports
            st.write("#### Random Forest Classification Report")
            st.text(classification_report(tTarget_test, y_pred_rf,
                                          target_names=[gesture_mapping.get(i, "Unknown") for i in range(1, 16)]))

            st.write("#### AdaBoost Classification Report")
            st.text(classification_report(tTarget_test, y_pred_ada,
                                          target_names=[gesture_mapping.get(i, "Unknown") for i in range(1, 16)]))

            # Confusion Matrices
            st.write("#### Random Forest Confusion Matrix")
            cm_rf = confusion_matrix(tTarget_test, y_pred_rf)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[gesture_mapping.get(i, "Unknown") for i in range(1, 16)],
                        yticklabels=[gesture_mapping.get(i, "Unknown") for i in range(1, 16)])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)
            plt.clf()

            st.write("#### AdaBoost Confusion Matrix")
            cm_ada = confusion_matrix(tTarget_test, y_pred_ada)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Greens',
                        xticklabels=[gesture_mapping.get(i, "Unknown") for i in range(1, 16)],
                        yticklabels=[gesture_mapping.get(i, "Unknown") for i in range(1, 16)])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)
            plt.clf()
    else:
        st.warning("Please enter your name and create your folder")


# Now, the main code starts

# Sidebar for User Name and Folder Creation
with st.sidebar:
    st.header("User Setup")
    name = st.text_input("Enter Your Name", value=st.session_state.T)
    if st.button("Create Your Folder"):
        if name != "":
            # Check if the name has changed
            if 'T' in st.session_state and st.session_state.T != name:
                reset_session_state()
                st.session_state.T = name
            else:
                st.session_state.T = name
            # Create main folder
            CreateMainFolder(name)
        else:
            st.warning("Please enter your name to create your folder")

    # Add a "New User" button
    if st.button("New User"):
        reset_session_state()

# Display initial image
get_image(0, image_placeholder, letter_placeholder)

# Tabs for "Train" and "Test" sections
tab1, tab2 = st.tabs(["Train", "Test"])

### Training Section ###
with tab1:
    st.header("Training Section")
    col1, col2, col3 = st.columns([1, 1, 1])

    # Next Gesture Button
    with col1:
        if st.button("Next Gesture"):
            if st.session_state.T != "":
                if st.session_state.cookies < 15:
                    GetTrainData()
                else:
                    st.write("Training completed.")
            else:
                st.warning("Please enter your name and create your folder")

    # Repeat Gesture Dropdown and Button
    with col2:
        optionsRepeat = ["A", "B", "C", "D", "F", "G", "H", "I", "L", "N", "W", "Y", "TR", "TIM", "TMR"]
        selected_option_repeat = st.selectbox("Select Gesture to Repeat", optionsRepeat)
        st.session_state.SelectedOptionRepeat = optionsRepeat.index(selected_option_repeat) + 1
        if st.button("Repeat Gesture"):
            if st.session_state.T != "":
                st.write("Repeating Data for Gesture", st.session_state.SelectedOptionRepeat)
                RepeatData()
            else:
                st.warning("Please enter your name and create your folder")

    # Stop Training Button
    with col3:
        # # Determine the label of the pause button
        # pause_button_label = "Pause Data Collection" if not st.session_state.pause else "Resume Data Collection"
        # if st.button(pause_button_label):
        #     st.session_state.pause = not st.session_state.pause

        if st.button("Abort Data Collection"):
            st.session_state.abort = True

        if st.button("Stop Training"):
            stop_function()

    # Train Gestures Button
    if st.button("Train Gestures"):
        if st.session_state.T != "":
            st.write("Training Gestures...")
            btn_Train_Model()
        else:
            st.warning("Please enter your name and create your folder")


### Testing Section ###
with tab2:
    st.header("Testing Section")
    col4, col5, col6 = st.columns([1, 1, 1])

    # Select Gesture to Test Dropdown
    with col4:
        options = ["A", "B", "C", "D", "F", "G", "H", "I", "L", "N", "W", "Y", "TR", "TIM", "TMR"]
        selected_option_test = st.selectbox("Select Gesture to Test", options)
        st.session_state.SelectedOption = options.index(selected_option_test) + 1

    # Get Test Data Button
    with col5:
        if st.button("Get Test Data"):
            if st.session_state.T != "":
                st.write("Getting Test Data for Gesture", st.session_state.SelectedOption)
                Get_Test_GestureData()
            else:
                st.warning("Please enter your name and create your folder")

        # Determine the label of the pause button
        # pause_button_label = "Pause Data Collection" if not st.session_state.pause else "Resume Data Collection"
        # if st.button(pause_button_label):
        #     st.session_state.pause = not st.session_state.pause

        if st.button("Abort Data Collection in test"):
            st.session_state.abort = True

    # Test Accuracy Button
    with col6:
        if st.button("Test Accuracy"):
            if st.session_state.T != "":
                st.write("Testing Accuracy...")
                btn_Test_Model(st.session_state.SelectedOption)
            else:
                st.warning("Please enter your name and create your folder")




# Footer or any additional information
st.write("---")
st.write("Developed by Your Name or Organization")
