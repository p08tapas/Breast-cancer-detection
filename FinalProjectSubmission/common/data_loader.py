import pandas as pd
import os
import re

# Here I am getting the directory where this file is located (common/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Pointing to the common/archive folder
DATA_DIR = os.path.join(SCRIPT_DIR, 'archive')
DATA_DIR = os.path.abspath(DATA_DIR)  # This is converted to absolute path
JPEG_DIR = os.path.join(DATA_DIR, 'jpeg')
CSV_DIR = os.path.join(DATA_DIR, 'csv')


def extract_series_uid(dicom_path):
    """Now we are extracting SeriesInstanceUID from DICOM path."""
    if pd.isna(dicom_path) or not dicom_path:
        return None
    pattern = r'1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+'
    matches = re.findall(pattern, str(dicom_path))
    return matches[-1] if matches else None



def find_jpeg_file(series_uid):
    """I am trying to find the JPEG file for a given SeriesInstanceUID."""
    if not series_uid:
        return None
    series_dir = os.path.join(JPEG_DIR, series_uid)
    if not os.path.exists(series_dir):
        return None
    jpeg_files = [f for f in os.listdir(series_dir) if f.endswith('.jpg')]
    if not jpeg_files:
        return None


    # Now we are sorting files for deterministic selection
    jpeg_files = sorted(jpeg_files)


    # We are preferring cropped images (1-*.jpg) over masks (2-*.jpg)
    cropped = sorted([f for f in jpeg_files if f.startswith('1-')])
    if cropped:
        path = os.path.join(series_dir, cropped[0])  # I take the first after sorting
    else:
        path = os.path.join(series_dir, jpeg_files[0])  # I take the first after sorting


    # Now we are returning the absolute path to ensure flow_from_dataframe can find it
    return os.path.abspath(path)



def load_dataset():
    """Load and prepare dataset from CSV files."""
    mass_train = pd.read_csv(os.path.join(CSV_DIR, 'mass_case_description_train_set.csv'))
    calc_train = pd.read_csv(os.path.join(CSV_DIR, 'calc_case_description_train_set.csv'))
    mass_test = pd.read_csv(os.path.join(CSV_DIR, 'mass_case_description_test_set.csv'))
    calc_test = pd.read_csv(os.path.join(CSV_DIR, 'calc_case_description_test_set.csv'))
    
    train_df = pd.concat([mass_train, calc_train], ignore_index=True)
    test_df = pd.concat([mass_test, calc_test], ignore_index=True)
    
    train_data = []
    for idx, row in train_df.iterrows():
        dicom_path = row.get('cropped image file path', '')
        pathology = row.get('pathology', '')
        
        if pd.isna(pathology) or pathology == '':
            continue
        
        label = 1 if 'MALIGNANT' in str(pathology).upper() else 0
        series_uid = extract_series_uid(dicom_path)
        if not series_uid:
            continue
        
        jpeg_path = find_jpeg_file(series_uid)
        if not jpeg_path or not os.path.exists(jpeg_path):
            continue
        
        train_data.append({'image_path': jpeg_path, 'label': label})
    
    test_data = []
    for idx, row in test_df.iterrows():
        dicom_path = row.get('cropped image file path', '')
        pathology = row.get('pathology', '')
        
        if pd.isna(pathology) or pathology == '':
            continue
        
        label = 1 if 'MALIGNANT' in str(pathology).upper() else 0
        series_uid = extract_series_uid(dicom_path)
        if not series_uid:
            continue
        
        jpeg_path = find_jpeg_file(series_uid)
        if not jpeg_path or not os.path.exists(jpeg_path):
            continue
        
        test_data.append({'image_path': jpeg_path, 'label': label})
    
    train_df_processed = pd.DataFrame(train_data)
    test_df_processed = pd.DataFrame(test_data)
    
    train_df_processed['label'] = train_df_processed['label'].astype(str)
    test_df_processed['label'] = test_df_processed['label'].astype(str)
    
    return train_df_processed, test_df_processed
