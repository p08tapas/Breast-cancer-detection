import pandas as pd
import os
import re

DATA_DIR = 'archive'
JPEG_DIR = os.path.join(DATA_DIR, 'jpeg')
CSV_DIR = os.path.join(DATA_DIR, 'csv')


def extract_series_uid(dicom_path):
    """This helps to extract SeriesInstanceUID from DICOM path."""
    if pd.isna(dicom_path) or not dicom_path:
        return None
    pattern = r'1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+'
    matches = re.findall(pattern, str(dicom_path))
    return matches[-1] if matches else None



def find_jpeg_file(series_uid):
    """This is for finding JPEG file for given SeriesInstanceUID."""
    if not series_uid:
        return None
    series_dir = os.path.join(JPEG_DIR, series_uid)
    if not os.path.exists(series_dir):
        return None
    jpeg_files = [f for f in os.listdir(series_dir) if f.endswith('.jpg')]
    if not jpeg_files:
        return None
    # This is for preferring cropped images (1-*.jpg) over masks (2-*.jpg)
    cropped = [f for f in jpeg_files if f.startswith('1-')]
    if cropped:
        return os.path.join(series_dir, cropped[0])
    return os.path.join(series_dir, jpeg_files[0])



def load_dataset():
    """This loads and prepares the dataset from CSV files."""
    print("Loading dataset...")
    
    # Loading of the CSV files.
    mass_train = pd.read_csv(os.path.join(CSV_DIR, 'mass_case_description_train_set.csv'))
    calc_train = pd.read_csv(os.path.join(CSV_DIR, 'calc_case_description_train_set.csv'))
    mass_test = pd.read_csv(os.path.join(CSV_DIR, 'mass_case_description_test_set.csv'))
    calc_test = pd.read_csv(os.path.join(CSV_DIR, 'calc_case_description_test_set.csv'))
    
   
    # Combining the datasets for train and test.
    
    train_df = pd.concat([mass_train, calc_train], ignore_index=True)
    test_df = pd.concat([mass_test, calc_test], ignore_index=True)
    
    print(f"CSV records - Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Process training data
    
    
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
    
    # Process test data
    
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
    
    # This is for converting labels to strings for ImageDataGenerator

    train_df_processed['label'] = train_df_processed['label'].astype(str)
    test_df_processed['label'] = test_df_processed['label'].astype(str)
    
    print(f"\nLoaded images - Train: {len(train_df_processed)}, Test: {len(test_df_processed)}")
    
    # Class distribution
    train_mal = (train_df_processed['label'] == '1').sum()
    train_ben = (train_df_processed['label'] == '0').sum()
    test_mal = (test_df_processed['label'] == '1').sum()
    test_ben = (test_df_processed['label'] == '0').sum()
    
    print(f"\nClass Distribution:")
    print(f"  Train - MALIGNANT: {train_mal}, BENIGN: {train_ben}")
    print(f"  Test  - MALIGNANT: {test_mal}, BENIGN: {test_ben}")
    
    return train_df_processed, test_df_processed

