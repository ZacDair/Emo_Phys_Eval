import pytest
import pandas as pd
import numpy as np
from Modules import Feature_Operations as FO


def test_computeFeatures():
    windows = {0: {'ecg': [83, 116, 155, 147, 140, 80, 100, 157, 127, 133, 95, 128, 95, 98, 106, 125, 106, 130, 90, 145,
                           114, 147, 145, 119, 140, 82, 81, 93, 138, 97, 98, 148, 150, 152, 101, 114, 134, 154, 132,
                           120, 94, 110, 117, 82, 110, 144, 130, 151, 144, 82, 82, 126, 84, 122, 111, 101, 143, 111,
                           134, 153, 109, 151, 128, 106, 85, 151, 153, 124, 95, 86, 149, 131, 140, 91, 134, 114, 98,
                           148, 143, 144, 91, 88, 85, 151, 128, 151, 87, 100, 150, 156, 102, 142, 149, 136, 110, 136,
                           89, 149, 130, 89],
                   'Label': 0
                   }
               }
    sr = 100
    filtering = {
        'enabled': True,
        'method': {
            'signal_size': 'window',
            'name': 'heartpy_basic',
            'min cutoff': 0.1,
            'max cutoff': 0.8,
            'filter type': 'bandpass'}
        }

    # Incorrect feature mode - default to "HeartPy"
    df_incorrect = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [], mode="Incorrect")

    # Neurokit feature mode - in development message + redirect to heartpy
    df_nk = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [], mode="Neurokit")

    # Neurokit feature mode - in development message + redirect to heartpy
    df_heartpy = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [], mode="HeartPy")
    df_default = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [])

    # Assert all are equal
    pd.testing.assert_frame_equal(df_incorrect, df_default)
    pd.testing.assert_frame_equal(df_nk, df_default)
    pd.testing.assert_frame_equal(df_heartpy, df_default)

    # Test no window data given - empty dataframe
    df = FO.computeFeatures([], 'ecg', sr, filtering, "sub_1", "TEST", [])
    pd.testing.assert_frame_equal(df, pd.DataFrame())

    # Test data key not present at all
    df = FO.computeFeatures([], 'noKey', sr, filtering, "sub_1", "TEST", [])
    pd.testing.assert_frame_equal(df, pd.DataFrame())

    # Test data key not present after processing some data - returns DF with current work
    # Add window with a missing key
    windows[1] = {}
    df = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [])
    pd.testing.assert_frame_equal(df, df_heartpy)

    # Add Skipping label
    del(windows[1])
    df = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [0])
    pd.testing.assert_frame_equal(df, pd.DataFrame())

    # Default without filtering
    filtering['enabled'] = False
    df_default_no_filter = FO.computeFeatures(windows, 'ecg', sr, filtering, "sub_1", "TEST", [])
    assert len(df_default_no_filter.columns) == len(df_default.columns)
    assert len(df_default_no_filter) == len(df_default)

    # Ensure the label columns and og_window_index are added
    assert "label" in df_default.columns
    assert "og_window_index" in df_default.columns





