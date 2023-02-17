from dumbifier.rtl_kerning import process_font
from pathlib import Path
from glyphsLib import load

from collections import OrderedDict


def test_rtl_kerning():
    test_font = load(Path(__file__).parent / "data" / "RTL_kerning_v3.glyphs")
    process_font(test_font)
    assert test_font.kerning["B0D53B35-34A4-475E-9EF4-52C3D10908C6"] == OrderedDict(
        {"@MMK_L_reh-ar": OrderedDict({"@MMK_R_hah-ar.init.swsh": -30})}
    )
    assert test_font.kerning["m01"] == OrderedDict(
        {
            "@MMK_L_leftAlef": OrderedDict({"@MMK_R_rightBet": -20, "he-hb": 4}),
            "@MMK_L_leftBet": OrderedDict({"@MMK_R_rightAlef": 20}),
            "@MMK_L_reh-ar": OrderedDict({"@MMK_R_hah-ar.init.swsh": -50}),
            "he-hb": OrderedDict({"@MMK_R_rightAlef": -2, "he-hb": -21}),
        }
    )
    assert not test_font.kerningRTL
