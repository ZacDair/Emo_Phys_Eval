from unittest import TestCase
import pytest
import pandas as pd
import numpy as np
import os

from Modules import Data_Operations as DO


# def test_case_alterations():
#     assert pytest.fail("Unimplemented Test for CASE alterations")


def test_wesad_alterations():
    # Test wrong file type
    ecg, ppg, label = DO.wesadAlterations("Tests/Test_Assets/wrong_file_type.csv")
    assert (ecg, ppg, label) == (None, None, None)

    # Test dummy.pkl with no keys
    ecg, ppg, label = DO.wesadAlterations("Tests/Test_Assets/dummy_datasets/data/WESAD/dummy_wesad_error.pkl")
    assert (ecg, ppg, label) == (None, None, None)

    # Test dummy.pkl actual
    ecg, ppg, label = DO.wesadAlterations("Tests/Test_Assets/dummy_datasets/data/WESAD/dummy_wesad.pkl")
    assert ecg is not None
    assert ppg is not None
    assert label is not None


def test_processRawData():
    sr = {'ecg': 100, 'ppg': 100, 'label': 100}
    sc = {'noise reduction': {
        'enabled': True,
        'method': {
            'signal_size': 'window',
            'name': 'heartpy_basic',
            'min cutoff': 0.5,
            'max cutoff': 2.5,
            'filter_type': 'bandpass'}
        }
    }

    dataLocation = "Tests/Test_Assets/dummy_datasets/data/CASE"
    outputLocation = "Tests/Test_Assets/dummy_datasets/features"

    # Test unsupported dataset name
    name = "error name"
    status = DO.processRawData(name, dataLocation, 10, sr, sc, [], outputLocation)
    assert not status

    # Test CASE with no valid files - returns True as .csv are ignored
    name = "CASE"
    status = DO.processRawData(name, dataLocation, 10, sr, sc, [], outputLocation)
    assert status

    # Test


def test_findAllFiles():
    # Find files in invalid location
    files = DO.findAllFiles("Tests/Test_Assets/Test_Data_Files")
    assert len(files) == 0

    # Find files in Test Assets
    files = DO.findAllFiles("Tests/Test_Assets/dummy_datasets/data/CASE")
    assert len(files) == 2


def test_combineDataFrames():
    # Empty File list - FAIL
    df = DO.combineDataFrames([], "ecg")
    assert df is None

    # Find all dummy feature files
    files = DO.findAllFiles("Tests/Test_Assets/dummy_datasets/features/")
    assert len(files) == 2

    # ECG - OK
    ecg_df = DO.combineDataFrames(files, "ecg")
    assert len(ecg_df) == 1

    # PPG - ok
    ppg_df = DO.combineDataFrames(files, "ppg")
    assert len(ecg_df) == 1

    # Wrong data key - returns None as no files were found
    df = DO.combineDataFrames(files, "none")
    assert df is None


def test_cleanDataFrame():
    # Not a dataframe
    df, inds = DO.cleanDataFrame("not_a_df")
    assert (df, inds) == (None, None)

    # Empty dataframe
    df = pd.DataFrame()
    df, inds = DO.cleanDataFrame(df)
    assert len(inds) == len(df.index)

    # Dataframe with no removable values
    # Validate indicies
    df = pd.read_csv("Tests/Test_Assets/dummy_datasets/features/dummy_1_features_ecg.csv")
    ogSize = len(df)
    df, inds = DO.cleanDataFrame(df)
    assert ogSize == len(df)
    assert len(inds) == len(df.index)

    # Dataframe with removable values
    # Add row of np.nan values to remove
    df = pd.DataFrame([np.nan] * len(df.columns))
    df, inds = DO.cleanDataFrame(df)
    assert 0 == len(df)
    assert len(inds) == len(df.index)


def test_dropColumns():
    # Not a dataframe
    df = DO.dropColumns("not_a_df", [])
    assert df is None

    # Empty dataframe
    df = pd.DataFrame()
    df1 = DO.dropColumns(df, [])
    pd.testing.assert_frame_equal(df1, df)

    # Empty dataframe and columns
    df = pd.DataFrame()
    df1 = DO.dropColumns(df, ['bpm'])
    pd.testing.assert_frame_equal(df1, df)

    # dataframe and columns
    df = pd.read_csv("Tests/Test_Assets/dummy_datasets/features/dummy_1_features_ecg.csv")
    df1 = DO.dropColumns(df, ['bpm'])
    assert len(df.columns)-1 == len(df1.columns)

    # dataframe and no columns
    df1 = DO.dropColumns(df, [])
    pd.testing.assert_frame_equal(df1, df)


def test_selectColumns():
    # Not a dataframe
    df = DO.selectColumns("not_a_df", [])
    assert df is None

    # Empty dataframe
    df = pd.DataFrame()
    df1 = DO.selectColumns(df, [])
    pd.testing.assert_frame_equal(df1, df)

    # dataframe and columns
    df = pd.read_csv("Tests/Test_Assets/dummy_datasets/features/dummy_1_features_ecg.csv")
    df1 = DO.selectColumns(df, ['bpm'])
    assert len(df1.columns) == 1

    # dataframe and no columns
    df1 = DO.selectColumns(df, [])
    pd.testing.assert_frame_equal(df1, df)


# def test_getFeaturesAndLabels():
#     # Not passing dataframes
#     #getFeaturesAndLabels(data, subjects, labels, removeCols, featureList, subjectCol='subject', labelCol='label'):
#
#     #


