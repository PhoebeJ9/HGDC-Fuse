import pickle
import pandas as pd
from datetime import timedelta
import numpy as np

def loadmetadata_multi_cxr(cxr_data_dir, ehr_data_dir):
    """
    Modified metadata loader: keep all AP-view CXRs within 48 hours of ICU admission.
    """
    data_dir = cxr_data_dir
    cxr_metadata = pd.read_csv(f'{cxr_data_dir}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{ehr_data_dir}/root/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']

    # Keep subjects that appear in both ICU stays and CXR metadata
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns], how='inner', on='subject_id')

    # Combine study date and time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(
        cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str),
        format="%Y%m%d %H%M%S"
    )

    cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)
    
    end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)

    # Filter CXRs within 48 hours
    cxr_merged_icustays_during = cxr_merged_icustays.loc[
        (cxr_merged_icustays.StudyDateTime >= cxr_merged_icustays.intime) & 
        (cxr_merged_icustays.StudyDateTime <= end_time)
    ]
    print(f"ViewPosition types within 48h window: {cxr_merged_icustays_during['ViewPosition'].unique()}")

    # Keep AP view only
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']

    # Group by stay_id; keep all AP studies (do not take only the last one)
    groups = cxr_merged_icustays_AP.groupby('stay_id')
    
    groups_selected = []
    for stay_id, group in groups:
        # Sort by time; keep all AP studies
        selected = group.sort_values('StudyDateTime').reset_index()
        groups_selected.append(selected)
    
    groups = pd.concat(groups_selected, ignore_index=True)

    # Compute time offset (seconds)
    groups['cxr_time_offset_in_seconds'] = (groups['StudyDateTime'] - groups['intime']).dt.total_seconds()
    
    print(f"Total CXR records: {len(groups)}")
    print(f"Unique ICU stays: {groups['stay_id'].nunique()}")
    print(f"Avg CXRs per stay: {len(groups) / groups['stay_id'].nunique():.2f}")
    
    return groups

def create_time_stamp(time_offset_seconds, max_time=48*3600):
    """
    Create a normalized timestamp for DirMetaConv.
    DirMetaConv will handle time features itself, so just return the normalized timestamp here.
    """
    # Normalize to [0, 1] (or return raw seconds if desired)
    return time_offset_seconds / max_time  # alternatively: return time_offset_seconds



if __name__ == "__main__":
    cxr_dir = '..physionet.org/files/mimic-cxr-jpg/2.0.0/files'
    ehr_data_dir = '../mimic4extract/data'
    
    # Load metadata
    metadata = loadmetadata_multi_cxr(cxr_dir, ehr_data_dir)
    
    # Load label splits
    splits_labels_train = pd.read_csv(f'{ehr_data_dir}/phenotyping/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{ehr_data_dir}/phenotyping/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{ehr_data_dir}/phenotyping/test_listfile.csv')
    
    # Merge labels
    train_meta_with_labels = metadata.merge(splits_labels_train, how='inner', on='stay_id')
    val_meta_with_labels = metadata.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = metadata.merge(splits_labels_test, how='inner', on='stay_id')
    
    # Create simplified time encoding for each CXR record (compatible with DirMetaConv and existing dataloaders)
    for df in [train_meta_with_labels, val_meta_with_labels, test_meta_with_labels]:
        df['time_encoding'] = df['cxr_time_offset_in_seconds'].apply(lambda x: create_time_stamp(x))
    
    # Save metadata
    with open('../mimic4extract/data/metas_with_labels_multi_cxr.pkl', 'wb') as f:
        pickle.dump({
            'train': train_meta_with_labels,
            'val': val_meta_with_labels,
            'test': test_meta_with_labels
        }, f)
    
    print("Metadata processing complete!")
    print(f"Train: {len(train_meta_with_labels)} CXR records")
    print(f"Validation: {len(val_meta_with_labels)} CXR records")
    print(f"Test: {len(test_meta_with_labels)} CXR records")
