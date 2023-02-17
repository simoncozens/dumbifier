import argparse
from dumbifier import module_names, process_font
from glyphsLib import load


def comma_separated_set(s):
    module = set(s.split(","))
    unknown = module - set(module_names)
    if unknown:
        raise ValueError(f"Unknown filter names: {', '.join(unknown)}")
    return module


parser = argparse.ArgumentParser(
    description="Remove smart stuff from a Glyphs font",
    epilog=f"Available filters include: {', '.join(module_names)}",
)
parser.add_argument(
    "--output", "-o", metavar="FILE", help="path to write dumb font source"
)
parser.add_argument(
    "--only",
    dest="only",
    type=comma_separated_set,
    help="only use the given filters (a comma separated list)",
)
parser.add_argument(
    "--disable",
    dest="disable",
    type=comma_separated_set,
    help="disable the given filters (a comma separated list)",
)
parser.add_argument("input", metavar="FILE", help="Glyphs source filter to process")
args = parser.parse_args()

if not args.output:
    args.output = args.input.replace(".glyphs", "-dumb.glyphs")

gsfont = load(args.input)
process_font(gsfont, only=args.only, disable=args.disable)
print(f"Saving on {args.output}")
gsfont.save(args.output)
