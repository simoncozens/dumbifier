def process_font(f):
    for glyph in f.glyphs:
        for layer in glyph.layers:
            if layer._is_brace_layer():
                layer.attributes["coordinates"] = [int(x) for x in layer.attributes["coordinates"]]
                layer.name = f"{{{', '.join(map(str, layer.attributes['coordinates']))}}}"

    for glyph in f.glyphs:
        for layer in glyph.layers:
            if layer._is_bracket_layer():
                if layer.attributes["axisRules"][0].get("min"):
                    layer.attributes["axisRules"][0]["min"] = int(layer.attributes["axisRules"][0]["min"])
                if layer.attributes["axisRules"][0].get("max"):
                    layer.attributes["axisRules"][0]["max"] = int(layer.attributes["axisRules"][0]["max"])
