from collections import OrderedDict

def flip(s):
  if "@MMK_L_" in s:
    return s.replace("@MMK_L_", "@MMK_R_")
  else:
    return s.replace("@MMK_R_", "@MMK_L_")

def process_font(font):
  for master, table in font.kerningRTL.items():
    master_kerning = font.kerning.setdefault(master, OrderedDict())
    for kern1, subtable in table.items():
      table_kerning = master_kerning.setdefault(flip(kern1), OrderedDict())
      for kern2, value in subtable.items():
        table_kerning[flip(kern2)] = value
  font.kerningRTL = {}
