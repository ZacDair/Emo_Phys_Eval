# Low - Neutral - High from numbers 0-10
def lowNeutralHigh(value):
    """
    Converts a numerical value between 0-10 into L, N or H based on 3 splits
    :param value: Integer value to convert into String
    :return: String "L" (0-3.5), "N" (3.5-7) or "H" (7-10)
    """
    try:
        assert isinstance(value, (int, float, complex))
        if value < 0 or value > 10:
            print("ERROR - Only integer values between 0-10 can converted into L, N, H values!")
            return None
        if value <= 3.5:
            return "L"
        elif 3.5 <= value <= 7:
            return "N"
        else:
            return "H"
    except AssertionError:
        print("ERROR - Only integer values can converted into L, N, H values!")


# String label to number
def stringLabelToNumber(labelString):
    """
    Converts a string into a numerical integer representing a combination Low, Neutral and High.
    Expected inputs ("*-*" where '*' = L or N or H)
    :param labelString: String to convert into a numerical label
    :return: Integer between 0 and 8
    """
    res = {"L-L": 0, "L-N": 1, "L-H": 2, "N-L": 3, "N-N": 4, "N-H": 5, "H-L": 6, "H-N": 7, "H-H": 8}
    try:
        return res[str(labelString)]
    except KeyError as e:
        print(f"ERROR - Attempted to use a non-existent key: {e}")
        return None


# Converts Arousal/Valence values into a single value
def convertArousalValence(arousalLabels, valenceLabels):
    """
    Combines a list of arousal and valence numerical values into low, neutral, high and then to integer representations
    of a combination Low, Neutral and High.
    :param arousalLabels: List of numerical values representing arousal axis
    :param valenceLabels: List of numerical values representing valence axis
    :return: List of numerical labels representing combinations of L/N/H arousal-valence
    """
    discreteLabels = []
    try:
        # Ensure the two label lists are the same length
        assert len(arousalLabels) == len(valenceLabels)
        assert len(arousalLabels) != 0 and len(valenceLabels) != 0

        for a, v in zip(arousalLabels, valenceLabels):
            discreteLabels.append(stringLabelToNumber(lowNeutralHigh(a)+"-"+lowNeutralHigh(v)))
        return discreteLabels
    except AssertionError:
        print("ERROR - Please ensure the arousal and valence value lists are the same length and not empty")
        return discreteLabels
