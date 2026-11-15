"""Parse a user instruction like 'pick a blue box and place it at B' into a `Command`."""
import re

from conveyor_sort.interfaces import BoxColor, Command

REGION_NAMES = {"a": "region_A", "b": "region_B", "c": "region_C", "bin": "bin_gray"}
_COLORS = {color.value: color for color in BoxColor} | {"grey": BoxColor.GRAY}
# the destination follows a preposition, so the article in "pick a blue box" is not read as region A
_REGION = re.compile(r"\b(?:at|to|in|into|on)\s+(?:the\s+)?(?:region\s+|gr[ae]y\s+)?(a|b|c|bin)\b")


def parse_command(text: str) -> Command:
    """Raises ValueError with a message for the user if the color or the destination is missing or ambiguous."""
    lowered = text.lower()
    regions = _REGION.findall(lowered)
    colors = [_COLORS[word] for word in re.findall(r"[a-z]+", _REGION.sub(" ", lowered)) if word in _COLORS]
    if len(set(colors)) != 1:
        problem = "no box color" if not colors else "more than one box color"
        raise ValueError(f"{problem} in '{text}': name one of red, green, blue, gray")
    if len(set(regions)) != 1:
        problem = "no destination" if not regions else "more than one destination"
        raise ValueError(f"{problem} in '{text}': say e.g. 'at A', 'to region B', 'at C' or 'into the bin'")
    return Command(colors[0], REGION_NAMES[regions[0]])
