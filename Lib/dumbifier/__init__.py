import dumbifier.corner_components as corner_components
import dumbifier.rtl_kerning as rtl_kerning
import dumbifier.destringify_axis_values as destringify_axis_values

modules = [
    corner_components,
    rtl_kerning,
    destringify_axis_values,
]
module_names = [module.__name__.replace("dumbifier.", "") for module in modules]


def process_font(font, disable=set(), only=set()):
    for module, module_name in zip(modules, module_names):
        if only and module_name not in only:
            continue
        if disable and module_name in disable:
            continue
        module.process_font(font)
