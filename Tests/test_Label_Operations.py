# Third Party Imports
import pytest

# Project Level Imports
from Modules import Label_Operations as LO


def test_lowNeutralHigh():
    # Not int
    res = LO.lowNeutralHigh("string")
    assert res is None

    # int not between 0-10
    res = LO.lowNeutralHigh(-1)
    assert res is None
    res = LO.lowNeutralHigh(11)
    assert res is None

    # int between 0-3.5
    res = LO.lowNeutralHigh(2.2)
    assert res == 'L'

    # int between 3.5-7
    res = LO.lowNeutralHigh(4)
    assert res == 'N'

    # int between 7-10
    res = LO.lowNeutralHigh(8)
    assert res == 'H'


def test_stringLabelToNumber():
    # Key not in labels
    res_1 = LO.stringLabelToNumber("Wrong_Key")
    assert res_1 is None

    # Key not a string
    res_2 = LO.stringLabelToNumber(1)
    assert res_1 is None and res_1 == res_2

    # Get L-L value == 0 and H-H == 8
    res_1 = LO.stringLabelToNumber("L-L")
    res_2 = LO.stringLabelToNumber("H-H")
    assert res_1 == 0
    assert res_2 == 8


def test_convertArousalValence():
    arousalLabels = [0.5, 3.3, 9, 10]
    valenceLabels = [0.5, 3.3, 9, 10]

    # Ensure we get a label for each value
    labels = LO.convertArousalValence(arousalLabels, valenceLabels)
    assert len(labels) == len(arousalLabels) == len(valenceLabels)

    # Empty list
    labels = LO.convertArousalValence([], [])
    assert labels == []

    # Ensure we get one of each label
    goal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    labels = LO.convertArousalValence([1, 1, 1, 4, 4, 4, 8, 8, 8], [1, 4, 8, 1, 4, 8, 1, 4, 8])
    assert goal == labels
