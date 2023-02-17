from dumbifier.corner_components import process_font
import pytest
from pathlib import Path
from glyphsLib import load

test_font = load(Path(__file__).parent / 'data' / 'CornerComponents.glyphs')
test_glyphs = [glyph.name[:-12] for glyph in test_font.glyphs if glyph.name.endswith(".expectation")]


@pytest.mark.parametrize("glyph", sorted(test_glyphs))
def test_corner_components(glyph):
  if "left_anchor" in glyph:
      pytest.xfail("left anchors not quite working yet")
  process_font(test_font, only_glyph=glyph)
  got = test_font.glyphs[glyph].layers[0]
  expected = test_font.glyphs[glyph+".expectation"].layers[0]
  for got_path, expected_path in zip(got.paths, expected.paths):
    assert str(got_path.nodes) == str(expected_path.nodes)
