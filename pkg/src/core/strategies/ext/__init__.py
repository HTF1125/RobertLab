"""ROBERT"""


strategies = {
    "UsSectorEwVcv1m": {
        "universe": "UnitedStatesSectors",
        "benchmark": "UnitedStatesSectorsEw",
        "inception": "2003-1-1",
        "frequency": "M",
        "commission": 10,
        "optimizer": "EqualWeight",
        "min_window": 252,
        "factors": ("VolumeCoefficientOfVariation1M",),
        "allow_fractional_shares": False,
    },
    "UsSectorEwPm6M1M": {
        "name": "UsSectorEwPm6M1M",
        "universe": "UnitedStatesSectors",
        "benchmark": "UnitedStatesSectorsEw",
        "inception": "2003-1-1",
        "frequency": "M",
        "commission": 10,
        "optimizer": "EqualWeight",
        "min_window": 252,
        "factors": ("PriceMomentum6M1M",),
        "allow_fractional_shares": False,
    },
    "GaaEwPm1M": {
        "universe": "GlobalAssetAllocation",
        "benchmark": "UnitedStatesSectorsEw",
        "inception": "2003-1-1",
        "frequency": "M",
        "commission": 10,
        "optimizer": "EqualWeight",
        "min_window": 252,
        "factors": ("PriceMomentum1M",),
        "allow_fractional_shares": False,
    },
}
