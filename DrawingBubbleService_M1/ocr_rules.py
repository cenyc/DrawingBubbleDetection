import re
import math
import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List

try:
    from .diameter_ocr import normalize_diameter_symbols
except ImportError:
    from diameter_ocr import normalize_diameter_symbols

logger = logging.getLogger(__name__)

@dataclass
class OCRToken:
    text: str
    cx: float
    cy: float
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    original_text: str | None = None


@dataclass
class NormalizedToken:
    raw_text: str
    text: str
    cx: float
    cy: float
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    token_type: str
    semantic_type: str
    dual_use: bool = False
    # Colour-stream tag set by the detector after OCR:
    #   True  → the token's bbox sits on annotation-coloured pixels
    #           (red/maroon/pink/magenta/purple) — i.e. a balloon
    #           number, a leader-line label, or other CAD markup
    #   False → black text on white background — i.e. a dimension,
    #           callout, title-block entry, or table cell
    # Used to route tokens to the correct downstream consumer:
    # bubble identification only accepts True; dimension callout
    # grouping only accepts False. Defaults to False so this stays
    # backward-compatible with code paths that don't set it.
    is_maroon: bool = False


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_engineering_text(text: str) -> str:
    """Robust engineering text normalization for diverse OCR outputs."""
    if not text:
        return text
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = normalize_diameter_symbols(text)
    
    # Resolve U+FFFD (replacement character) using the surrounding notation.
    # Replace in context: digit × digit → × (U+00D7); digit at end/space → ° (U+00B0).
    if '\ufffd' in text:
        text = re.sub(r'(\d)\ufffd(\d)',
                      lambda m: m.group(1) + chr(0x00D7) + m.group(2),
                      text)
        text = re.sub(r'(\d)\ufffd(?=\s|$)',
                      lambda m: m.group(1) + chr(0x00B0),
                      text)
        # U+FFFD at the START of a token followed by digits is almost always
        # the Ø (diameter) symbol — RapidOCR cannot encode Ø in some fonts
        # and outputs U+FFFD instead.  Convert to "0" here (not directly to Ø)
        # so the existing ^0\d{2,}→Ø rule downstream handles it cleanly
        # while retaining the existing replacement-character recovery behavior.
        text = re.sub(r'^\ufffd(?=\d)', '0', text)
        # Same pattern after whitespace (multi-token strings: " \ufffd63.1")
        text = re.sub(r'(?<=\s)\ufffd(?=\d)', '0', text)
        text = text.replace('\ufffd', '')   # drop any remaining stray FFFD

    # OCR character-variant substitutions (before encoding normalization):
    #   Φ (Greek Phi, U+03A6)  → Ø (diameter) — OCR confuses these
    #   º (masculine ordinal, U+00BA) → ° (degree symbol)
    #   ˚ (ring above, U+02DA) → ° (degree symbol)
    #   O/ at start of digits → Ø (diameter misread as "O" + "/")
    text = text.replace('\u03A6', 'Ø')
    text = text.replace('\u00BA', '°')
    text = text.replace('\u02DA', '°')
    text = re.sub(r'^O/(?=\d)', 'Ø', text)
    text = re.sub(r'(?<=\s)O/(?=\d)', 'Ø', text)

    # Preserve Unicode engineering symbols and annotations. A lossy Latin-1
    # round trip deletes symbols such as ⌀, ≤ and perpendicularity outright.
    
    # Remove common OCR artifacts
    normalized = text.strip()
    
    # Enhanced encoding fixes for mixed-language documents
    try:
        # Handle specific encoding issues we've seen
        if 'Ã' in normalized and 'Â' in normalized:
            # Fix double-encoded UTF-8
            normalized = normalized.encode('latin1').decode('utf-8', errors='ignore')
        elif 'ä¸' in normalized:
            # Fix Chinese character encoding
            normalized = normalized.replace('ä¸', '+X')
        elif '±' not in normalized and '+' in normalized and '-' in normalized:
            # Fix tolerance symbols
            normalized = re.sub(r'\+\s*-', '±', normalized)
    except:
        pass
    
    # PRESERVE VALID NUMBERS FIRST - Don't corrupt good OCR results
    # Check if this is already a valid number
    if re.match(r'^\d+$', normalized):  # Pure integer
        return normalized
    if re.match(r'^\d+\.\d+$', normalized):  # Pure decimal
        # But NOT if it starts with "0" followed by 2+ digits — that's
        # likely Ø (diameter symbol) misread as "0" by OCR.
        if not re.match(r'^0\d{2,}', normalized):
            return normalized
    if re.match(r'^\.\d+$', normalized):  # Leading decimal
        return '0' + normalized
    if re.match(r'^\d+/\d+$', normalized):  # Fraction
        return normalized
    if re.match(r'^\d+\.\d+±\d+\.\d+$', normalized):  # Tolerance
        return normalized
    if re.match(r'^\d+°$', normalized):  # Angle
        return normalized
    
    # ── Guard: text is already a recognisable engineering pattern ────
    # Skip all confusion-matrix substitutions to avoid corrupting good OCR.
    _upper = normalized.upper()
    _is_clean = bool(
        re.search(r'\bDIA\b', _upper)                          # "4 DIA", "MAJOR DIA"
        or re.search(r'\bMAJOR\b', _upper)
        or re.fullmatch(r'MJ?\d+[Xx\*×]\d+(\.\d+)?.*', _upper)  # MJ5x0.8 ...
        or re.fullmatch(r'[A-Z]\d*\s+\d+[\.,]\d+[Xx×]\d+', _upper)  # B2 0.5x45
        or re.search(r'\d+[Hh]\d+[Hh]', _upper)               # 4h6h
    )
    if _is_clean:
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Still apply safe symbol fixes (commas→periods inside numbers)
        normalized = re.sub(r'(?<=\d),(?=\d)', '.', normalized)
        # Thread spec: comma between pitch and tolerance → space
        # "MJ5x0.8, 4h6h" → "MJ5x0.8 4h6h"
        normalized = re.sub(
            r'(MJ?\d+[xX\xd7]\d+[\.,]\d+)\s*[,]\s*(\d+[hH])',
            r'\1 \2', normalized,
        )
        return normalized

    # Fix specific Ø corruption patterns before confusion matrix.
    # IMPORTANT: only strip Ø when it is genuinely a corruption artefact,
    # NOT when it is a valid diameter prefix (e.g. "Ø52.28" must stay).
    if 'Ø' in normalized:
        normalized = re.sub(r'Ø(\d+)Ø', r'\1', normalized)          # Ø5Ø -> 5

    # Focused confusion matrix — only unambiguous single-character swaps in
    # purely numeric/symbolic contexts.  Removed entries that corrupt
    # engineering abbreviations:
    #   'D' removed from '0' (D→0 breaks "4DIA" → "4Ø1A")
    #   lowercase symbol group removed ('l','i','o' → breaks "MAJOR DIA")
    ocr_confusions = {
        # Pure-numeric contexts only
        '0': ['O', 'Q'],      # 0/O/Q confusion — NOT 'D'
        '1': ['I', 'L', 'T'],
        '6': ['G', 'b'],
        '8': ['B'],       # 'R' removed — it's the radius symbol in engineering
        '9': ['g', 'q'],
        # Symbol
        '一': ['1'],
    }

    # Context-aware correction
    def is_likely_number_context(text):
        """True when text is primarily numeric."""
        letters = sum(1 for c in text if c.isalpha())
        digits  = sum(1 for c in text if c.isdigit())
        return digits > 0 and digits >= letters

    # Apply corrections based on context
    for correct, wrong_list in ocr_confusions.items():
        for wrong in wrong_list:
            if wrong in normalized and correct not in normalized:
                if is_likely_number_context(normalized) and correct.isdigit():
                    normalized = normalized.replace(wrong, correct)
    
    # ── Engineering-specific OCR corruption repairs ──────────────
    # These fix known patterns where RapidOCR at 4× scale corrupts
    # specific engineering abbreviations and symbols.

    # Middle-dot / ø corruptions: OCR reads 'D' as '·' or 'ø' in certain fonts
    #   "4ø1A" → "4 DIA",  "4·1A" → "4 DIA",  "Ø1A" → "DIA" (full word only)
    #   Include ø (U+00F8 LATIN SMALL LETTER O WITH STROKE) and · (U+00B7 MIDDLE DOT)
    normalized = re.sub(r'[·∙•øØ][1I][Aa](?!\w)', ' DIA', normalized)
    # Middle-dot / ø as digit separator: "ø.8" or "·5" → "0.8" / ".5"
    normalized = re.sub(r'(?<=\d)[·∙•](?=\d)', '.', normalized)
    normalized = re.sub(r'^[·∙•](?=\d)', '0.', normalized)
    # Leading ø before decimal: "ø.8" → "0.8" (OCR of "0.8" where 0 misread as ø)
    normalized = re.sub(r'(?<![Øø\d])[ø]\.(?=\d)', '0.', normalized)

    # 'x' used as multiplication in engineering specs: normalise case
    #   "MJSX" → "MJS×" is wrong; keep as "MJ5x" after digit fix
    #   "MJS" is likely "MJ5" (S→5 in number context inside thread spec)
    normalized = re.sub(r'\bMJ[Ss]([Xx])', r'MJ5x', normalized)
    #   "4hGh" → "4h6h"  (G→6 in tolerance class context)
    normalized = re.sub(r'(\d[Hh])([Gg])([Hh])', lambda m: m.group(1) + '6' + m.group(3), normalized)

    # Strip leading/trailing bare dashes and square brackets (OCR artifacts).
    # IMPORTANT: do NOT include ( or ) here — they are part of valid engineering
    # notation like (REF), (3X4 RIBS), (BOSS) and must be preserved intact.
    normalized = re.sub(r'^[-–—\[\]]+\s*', '', normalized)
    normalized = re.sub(r'\s*[-–—\[\]]+$', '', normalized)

    # Comma in thread spec: "MJ5x0.8, 4h6h" → "MJ5x0.8 4h6h"
    normalized = re.sub(r'(MJ?\d+[xX×]\d+[\.,]\d+)\s*[,]\s*(\d+[hH])', r'\1 \2', normalized)

    # Strip leading "L" OCR artifact before angle or parenthesised content.
    # OCR often reads a leader-line junction or arrowhead touching the text
    # as a leading "L", giving e.g. "L9°(REF)" instead of "9°(REF)".
    # Only strip when the "L" is immediately followed by digits+degree or "("
    # so we do not corrupt valid tokens like "L19" part labels.
    normalized = re.sub(r'^L(\d+[°(])', r'\1', normalized)

    # Q misread as Ø (common OCR confusion)
    normalized = re.sub(r'^Q(\.\d)', r'Ø\1', normalized)
    # Note: "0.→Ø" tolerance class fix moved after symbol_patterns

    # Unicode replacement character (U+FFFD) in engineering contexts.
    # RapidOCR outputs U+FFFD when it cannot encode a symbol.  Restore the
    # most likely intended symbol based on surrounding context:
    #   digits \ufffd digits (mid-token) → multiplication sign × (U+00D7)
    normalized = re.sub(r'(\d)\ufffd(\d)',
                        lambda m: m.group(1) + chr(0x00D7) + m.group(2),
                        normalized)
    #   digit \ufffd at end or before whitespace → degree sign ° (U+00B0)
    normalized = re.sub(r'(\d)\ufffd(?=\s|$)',
                        lambda m: m.group(1) + chr(0x00B0),
                        normalized)

    # Stray leading/trailing middle dots from Ø OCR corruption
    normalized = re.sub(r'^[·∙•]\s*', '', normalized)
    normalized = re.sub(r'\s*[·∙•]$', '', normalized)

    # Enhanced symbol restoration with more patterns
    symbol_patterns = [
        # Diameter symbols
        (r'[Ø⌀ø]\s*([0-9.]+)', r'Ø\1'),
        # Convert letter 'O' before an INTEGER to Ø (e.g. "O10" → "Ø10").
        # Do NOT convert digit '0' — "0.5" must stay "0.5", not become "Ø.5".
        (r'\bO([0-9]+)\b', r'Ø\1'),

        # Tolerance symbols
        (r'\+\s*[-/]\s*', r'±'),
        (r'\+/-', r'±'),
        (r'\+\-', r'±'),

        # Decimal points (missing or extra) - only for clearly corrupted patterns
        (r'([0-9])\s+([0-9]{2,})', r'\1.\2'),  # 1 23 -> 1.23 (space separated)

        # Division symbols
        (r'([0-9])\s*/\s*([0-9])', r'\1/\2'),

        # Radius symbols ("F" is a common OCR misread of "R")
        (r'R\s*([0-9.]+)', r'R\1'),
        (r'r\s*([0-9.]+)', r'R\1'),
        (r'(\d+\.?\d*)\s*F\b', r'\1R'),  # "3.5F" → "3.5R" (trailing R misread)

        # Thread specifications
        (r'M\s*([0-9])', r'M\1'),
        (r'([0-9])\s*H', r'\1h'),

        # Reference markers — single pattern with negative lookarounds to
        # prevent double-wrapping: "9°(REF)" must NOT become "9°((REF))".
        # The two original entries (REF + ref) each applied re.IGNORECASE,
        # so they ran twice on the already-parenthesised text, tripling parens.
        (r'(?<!\()\bREF\b(?!\))', r'(REF)'),
    ]
    
    for pattern, replacement in symbol_patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    # Multiplication symbol normalisation (avoids source-encoding issues with ×)
    # "5x45" or "5X45" or "5×45" → "5×45" (U+00D7)
    _TIMES = chr(0x00D7)
    normalized = re.sub(
        r'([0-9])\s*[xX\xd7*]\s*([0-9])',
        lambda m: m.group(1) + _TIMES + m.group(2),
        normalized,
    )

    # Degree symbol normalisation (U+00B0)
    _DEG = chr(0x00B0)
    normalized = re.sub(
        r'([0-9])\s*[\xb0]\s*',
        lambda m: m.group(1) + _DEG,
        normalized,
    )

    # "0.40h8" → "Ø40 H8" — OCR reads "Ø" as "0." before tolerance class.
    normalized = re.sub(r'^0\.(\d+)\s*([hHgGfF]\d+)', r'Ø\1 \2', normalized)

    # "052.28" → "Ø52.28", "0204.98" → "Ø204.98"
    # DOMAIN PRINCIPLE: OCR reads the Ø (diameter) symbol as "0" because
    # they share a round shape. A leading "0" before a multi-digit number
    # (≥2 digits) that forms a plausible diameter value is likely Ø.
    # Only apply when the remaining digits form a valid decimal number.
    normalized = re.sub(r'^0(\d{2,}(?:\.\d+)?)\s*$', r'Ø\1', normalized)

    # Clean up spacing and artifacts
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized

def _is_valid_engineering_text(text: str) -> bool:
    """
    Check if text is already a valid engineering pattern.
    If so, do NOT normalize it.
    """
    t = text.strip().upper()
    
    # Valid thread specifications (enhanced patterns)
    if re.fullmatch(r"MJ?\d+[Xx]\d+(\.\d+)?\s*\d+[Hh]\d+[Hh]", t):
        return True
    if re.fullmatch(r"MJ?\d+[Xx]\d+(\.\d+)?", t):
        return True
    if re.fullmatch(r"\d+[Hh]\d+[Hh]", t):
        return True
    
    # Valid chamfer specifications (enhanced patterns)
    if re.fullmatch(r"[A-Z]\d*\s*\d+(\.\d+)?[Xx]\d+°", t):
        return True
    if re.fullmatch(r"[A-Z]\d*\s*\d+(\.\d+)?[Xx]\d+DEG", t):
        return True
    if re.fullmatch(r"\d+(\.\d+)?[Xx]\d+°", t):
        return True
    
    # Valid diameter specifications (enhanced patterns)
    if re.fullmatch(r"Ø?\d+(\.\d+)?±\d+(\.\d+)?", t):
        return True
    if re.fullmatch(r"Ø?\d+(\.\d+)?", t):
        return True
    if re.fullmatch(r"\d+\s*DIA", t):
        return True
    if re.fullmatch(r"\[\d+(\.\d+)?", t):  # Handle [10 → Ø10
        return True
    # IMPORTANT: Preserve Ø symbol in precision patterns
    if re.fullmatch(r"Ø\d+\.\d+", t):
        return True
    if re.fullmatch(r"Ø\d+", t):
        return True
    # ENHANCED: Preserve Ø in tolerance patterns
    if re.fullmatch(r"Ø\d+\.\d+±\d+\.\d+", t):
        return True
    if re.fullmatch(r"Ø\d+\.\d+±\d+", t):
        return True
    if re.fullmatch(r"Ø\d+±\d+", t):
        return True
    
    # Valid major diameter specifications (enhanced patterns)
    if re.fullmatch(r"MAJOR DIA\s*\d+(\.\d+)?\s*\d+(\.\d+)?", t):
        return True
    if re.fullmatch(r"MAJOR DIA\s*\d+(\.\d+)?", t):
        return True
    
    # Valid stacked pairs (already with slashes)
    if re.fullmatch(r"\d+(\.\d+)?/\d+(\.\d+)?", t):
        return True
    
    # Valid angle specifications (enhanced patterns)
    if re.fullmatch(r"-?\d+°\(REF\)", t):
        return True
    if re.fullmatch(r"-?\d+°", t):
        return True
    if re.fullmatch(r"\d+°\(REF\)", t):
        return True
    
    # Valid coordinates
    if re.fullmatch(r"\d+,\d+", t):
        return True
    if re.fullmatch(r"\d+\.\d+,\d+\.\d+", t):
        return True
    
    # Valid radius specifications
    if re.fullmatch(r"R\d+(\.\d+)?", t):
        return True
    if re.fullmatch(r"\d+(\.\d+)?R", t):
        return True
    
    return False


# ─────────────────────────────────────────────
# BUBBLE
# ─────────────────────────────────────────────

def is_bubble_token(text: str) -> bool:
    t = text.strip().upper()
    # Accept patterns:
    #   "1", "11", "82"            → plain integer (1-3 digits)
    #   "11A", "3B"                → digit + revision letter
    #   "(82)", "(45)"             → parenthesised legacy/sub-revision
    #
    # We deliberately do NOT accept "[A-Z]\d{1,2}" (e.g. "R1", "M60") —
    # those are radius/thread callouts, not bubbles. Real "A1"/"B2"-style
    # bubbles are still caught geometrically (Hough circle + tint).
    return bool(
        re.fullmatch(r"\d{1,2}", t)              # 1, 11, 82  (1-2 digits)
        # Revision-letter suffix: only A-G accepted. R/M/H/D are
        # physical-dim suffix letters (radius / thread / hole-class /
        # diameter) and must NOT be treated as bubble revision marks
        # — under warm color cast or low contrast, OCR promotes
        # "3R" radius text to a "3R" phantom bubble.
        or re.fullmatch(r"\d{1,2}[A-G]", t)      # 11A, 3B (revision)
        or re.fullmatch(r"\(\d{1,3}\)", t)       # (82), (45)  full parens
        or re.fullmatch(r"\(\d{1,2}", t)         # (82          OCR-split left paren
        or re.fullmatch(r"\d{1,2}\)", t)         # 82)          OCR-split right paren
    )


def normalize_bubble_value(text: str) -> str:
    """Strip leading zeros from bubble IDs (e.g. "07" -> "7").

    Never returns an empty string: if lstrip removes all characters
    (e.g. text="0" or "00"), returns "0" so callers can detect and
    discard zero-only tokens via the norm == "0" guard rather than
    receiving bubble_number="" which creates orphan rows in the UI.
    """
    # Strip parentheses around bubble IDs, e.g. "(82)" -> "82"
    s = text.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = s[1:-1].strip()
    stripped = s.lstrip("0")
    return stripped if stripped else "0"


# ─────────────────────────────────────────────
# DIMENSION DETECTION
# ─────────────────────────────────────────────

def is_dimension_token(text: str) -> bool:
    t = text.upper()

    patterns = [
        # Basic decimal numbers
        r"\d+\.\d+",
        r"\.\d+",  # .5 → 0.5
        
        # Stacked pairs and tolerances
        r"\d+(\.\d+)?/\d+(\.\d+)?",
        r"\d+(\.\d+)?±\d+(\.\d+)?",
        r"\d+(\.\d+)?\+\-\d+(\.\d+)?",
        
        # Thread specifications
        r"MJ?\d+x\d+(\.\d+)?",
        r"M\d+X\d+(\.\d+)?",  # Handle OCR variations
        r"\d+h\d+h",
        
        # Chamfer specifications
        r"\d+(\.\d+)?x\d+°",
        r"\d+(\.\d+)?X\d+°",
        r"\d+°\s*CHAMFER",
        
        # Radius specifications
        r"R\d+",
        r"\d+R",
        r"RADIUS\s*\d+",
        
        # Diameter specifications (enhanced)
        r"Ø?\d+(\.\d+)?",
        r"\d+\s*DIA",
        r"\d+\s*DIAM",
        r"DIAM\s*\d+",
        r"\[\d+(\.\d+)?",  # [10 → Ø10
        
        # Coordinate specifications
        r"\+X|\-Y",
        r"\d+,\d+",
        r"\d+\.\d+,\d+\.\d+",
        r"\d+,\d+\.\d+",
        r"\d+\.\d+,\d+",
        
        # Angle specifications
        r"\d+°",
        r"\d+DEG",
        r"\d+\s*DEG",
        r"\d+\s*DEGREES",
        
        # Reference specifications
        r"\d+\(REF\)",
        r"\d+\s*REF",
        r"REF\s*\d+",
        
        # Surface finish
        r"\d+\s*RA",
        r"RA\s*\d+",
        
        # Taper and slope
        r"\d+:\d+",
        r"\d+(\.\d+)?%",
    ]

    # Enhanced rejection of obvious noise
    if re.fullmatch(r"[A-Z]+\s*\d*", t) and "DIA" not in t and "DIAM" not in t and "REF" not in t:
        return False
    
    # Reject single letters that aren't dimensions
    if re.fullmatch(r"[A-Z]", t):
        return False
    
    # Reject common OCR noise patterns
    if re.fullmatch(r"[\\|/<>{}\[\]]+", t):
        return False

    return any(re.search(p, t) for p in patterns)


def is_dimension_primary_dual_use_token(text: str) -> bool:
    """Return True when a bubble-like token should enter callout grouping."""
    t = text.strip().upper()
    if not is_bubble_token(t) or not is_dimension_token(t):
        return False
    return bool(
        "." in t
        or re.search(r"[Ã˜ØR]", t)
        or re.search(r"\bMJ?\d+[XxÃ—]", t)
        or re.search(r"[Â±/]", t)
        or re.search(r"\d+[Hh]\d+[Hh]", t)
    )


# ─────────────────────────────────────────────
# SEMANTIC TYPE
# ─────────────────────────────────────────────

def classify_semantic_type(text: str) -> str:
    t = text.upper()

    # Enhanced semantic classification with more patterns
    
    # Stacked pairs and tolerances
    if re.fullmatch(r"\d+(\.\d+)?/\d+(\.\d+)?", t):
        return "numeric_pair"
    if re.search(r"±|\+\-|\+ -", t):
        return "tolerance"

    # Thread specifications (enhanced)
    if re.search(r"\d+h\d+h", t):
        return "thread"
    if re.search(r"MJ?\d+x\d", t):
        return "thread"
    if re.search(r"M\d+X\d", t):
        return "thread"

    # Chamfer specifications
    if re.search(r"\d+(\.\d+)?[xX]\d+°", t):
        return "chamfer"
    if re.search(r"\d+°\s*CHAMFER", t):
        return "chamfer"

    # Coordinate specifications
    if re.search(r"\+X|\-Y", t):
        return "coordinate"
    if re.search(r"\d+,\d+", t):
        return "coordinate"

    # Diameter specifications (enhanced)
    if re.search(r"Ø|\d+\s*DIA|\d+\s*DIAM|DIAM\s*\d+|\[\d+", t):
        return "diameter"

    # Radius specifications
    if re.search(r"R\d+|\d+R|RADIUS\s*\d+", t):
        return "radius"

    # Angle specifications
    if re.search(r"\d+°|\d+DEG|\d+\s*DEG|\d+\s*DEGREES", t):
        return "angle"

    # Reference specifications
    if re.search(r"\d+\(REF\)|\d+\s*REF|REF\s*\d+", t):
        return "reference"

    # Surface finish
    if re.search(r"\d+\s*RA|RA\s*\d+", t):
        return "surface_finish"

    # Taper and slope
    if re.search(r"\d+:\d+|\d+(\.\d+)?%", t):
        return "taper"

    # Numeric values (fallback)
    if re.search(r"\d+(\.\d+)?", t):
        return "numeric"

    return "unknown"


# ─────────────────────────────────────────────
# TOKEN PIPELINE
# ─────────────────────────────────────────────

def normalize_ocr_tokens(tokens: List[OCRToken], image_w: int, image_h: int):
    out = []

    for t in tokens:
        norm = normalize_engineering_text(t.text)
        if is_bubble_token(norm):
            token_type = "bubble"
        elif is_dimension_token(norm):
            token_type = "dimension"
        else:
            token_type = "keyword"

        semantic = classify_semantic_type(norm)

        out.append(
            NormalizedToken(
                raw_text=t.original_text if t.original_text is not None else t.text,
                text=norm,
                cx=t.cx,
                cy=t.cy,
                conf=t.conf,
                x1=t.x1,
                y1=t.y1,
                x2=t.x2,
                y2=t.y2,
                token_type=token_type,
                semantic_type=semantic,
            )
        )

    return out


# ─────────────────────────────────────────────
# HELPER FUNCTIONS FOR CALLOUT RULES
# ─────────────────────────────────────────────

def token_center_distance(token1: NormalizedToken, token2: NormalizedToken) -> float:
    """Calculate Euclidean distance between token centers."""
    return math.dist((token1.cx, token1.cy), (token2.cx, token2.cy))


def token_is_numeric_value(text: str) -> bool:
    """Check if text represents a numeric value."""
    return bool(re.fullmatch(r"\d+\.?\d*", text.strip()))


def token_is_chamfer(text: str) -> bool:
    """Check if text represents a chamfer specification."""
    t = text.upper()
    # Match patterns like "0.5x45", "0.5×45", "0.5x45°", "0.5X45DEG"
    if re.search(r"\d+(\.\d+)?[xX\xd7]\d+", t):
        return True
    if "CHAMFER" in t:
        return True
    return False


def token_is_thread(text: str) -> bool:
    """Check if text represents a thread specification."""
    t = text.upper()
    return bool(
        re.search(r"MJ?\d+[xX]\d+", t)        # metric w/ pitch: M12x1.5, MJ5x0.8
        or re.search(r"\d+[hH]\d+[hH]", t)    # fit class: 4h6h
        or re.fullmatch(r"M\d+", t)           # bare metric size: M60, M12
    )


def token_is_diameter(text: str) -> bool:
    """Check if text represents a diameter specification."""
    t = text.upper()
    return bool(re.search(r"Ø|\d+\s*DIA|\d+\s*DIAM|DIAM\s*\d+", t))


def token_is_radius(text: str) -> bool:
    """Check if text represents a radius specification."""
    t = text.upper()
    return bool(re.search(r"R\d+|\d+R|RADIUS\s*\d+", t))


def recover_precision_notes_local(roi: np.ndarray, offset_x: int, offset_y: int) -> List[NormalizedToken]:
    """Local OCR recovery for precision tolerance notes with explicit engine selection."""
    
    import cv2
    from .ocr_config import get_ocr_config, get_preferred_ocr_engine
    
    recovered_tokens = []
    
    # Get OCR configuration
    preferred_engine = get_preferred_ocr_engine()
    ocr_config = get_ocr_config(preferred_engine)
    
    # Select the actual engine to use
    selected_engine = ocr_config.select_engine()
    engine_info = ocr_config.get_engine_info()
    
    logger.info("Precision Recovery OCR Engine: %s", engine_info)
    
    # Enhanced OCR techniques for tolerance patterns
    techniques = [
        ('original', roi.copy()),
        ('enhanced', _enhance_for_precision(roi.copy())),
        ('inverted', _invert_for_precision(roi.copy())),
        ('sharpened', _sharpen_for_precision(roi.copy())),
        ('threshold', _adaptive_threshold_for_precision(roi.copy())),
        ('morphological', _morphological_cleanup_for_precision(roi.copy()))
    ]
    
    for technique_name, processed_roi in techniques:
        try:
            # Upscale for better OCR accuracy
            scale = 3
            h, w = processed_roi.shape[:2]
            upscaled = cv2.resize(processed_roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            
            # Run OCR with selected engine
            if selected_engine.value == "paddle":
                tokens = _run_paddle_precision_recovery(upscaled, scale, offset_x, offset_y)
            elif selected_engine.value == "rapid":
                tokens = _run_rapid_precision_recovery(upscaled, scale, offset_x, offset_y)
            
            # Filter for precision-relevant tokens
            for token in tokens:
                if _is_precision_token(token.text) and token.conf > 0.4:
                    recovered_tokens.append(token)
                    break  # Avoid duplicates across techniques
                
        except Exception as e:
            logger.debug("Precision recovery technique '%s' failed: %s", technique_name, e)
            continue  # Silently fail for individual techniques
    
    if recovered_tokens:
        logger.info("Precision recovery found %d tokens with %s", len(recovered_tokens), selected_engine.value.upper())
    
    return recovered_tokens

def _run_paddle_precision_recovery(upscaled: np.ndarray, scale: int, offset_x: int, offset_y: int) -> List[NormalizedToken]:
    """Run PaddleOCR for precision recovery."""
    try:
        from paddleocr import PaddleOCR
        
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en'
        )
        
        # Convert to RGB for PaddleOCR
        if len(upscaled.shape) == 3:
            upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        else:
            upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        
        # Run PaddleOCR
        result = ocr.ocr(upscaled_rgb, cls=True)
        
        tokens = []
        if result and result[0]:
            def normalize_engineering_text(text: str) -> str:
                """
                Normalize engineering text with enhanced pattern recognition.
                
                Handles:
                - Diameter symbols (Ø, O, 0)
                - Tolerance patterns (.800±.001)
                - Thread specifications
                - Chamfer angles
                - Radius symbols (R)
                """
                if not text or not text.strip():
                    return text
                
                original = text.strip()
                
                # Early exit for already valid patterns
                if re.match(r'^Ø[\d.]+±[\d.]+$', original) or re.match(r'^\d+\.?\d*/\d+\.?\d*$', original):
                    return original
                
                # Character-level fixes for common OCR confusions
                fixed = original
                
                # Enhanced OCR symbol confusion handling
                symbol_map = {
                    'O': '0',  # O to zero
                    'Q': '0',  # Q to zero  
                    '°': '°',  # Keep degree symbol
                    '±': '±',  # Keep tolerance symbol
                    '×': '×',  # Keep multiplication symbol
                }
                
                # Apply symbol mapping with context awareness
                result_chars = []
                for i, char in enumerate(fixed):
                    if char in symbol_map:
                        # Special handling for diameter symbols
                        if char in ['O', 'Q']:
                            # Check if this could be a diameter symbol
                            if i == 0 or (i > 0 and fixed[i-1] in [' ', 'Ø']):
                                result_chars.append('Ø')
                            elif i < len(fixed) - 1 and fixed[i+1].isdigit():
                                # O followed by digit could be diameter
                                result_chars.append('Ø')
                            else:
                                result_chars.append(symbol_map[char])
                        else:
                            result_chars.append(symbol_map[char])
                    else:
                        result_chars.append(char)
                
                fixed = ''.join(result_chars)
                
                # Handle missing leading zeros in decimal dimensions.
                # Do not invent a diameter prefix from a memorized value list;
                # diameter restoration must come from observed OCR symbols or
                # clear prefix context, not from exact numeric values.
                if re.match(r'^\.\d+', fixed):
                    fixed = '0' + fixed
                
                # Handle radius symbols
                if re.match(r'^[Rr]\s*\.?\d+', fixed):
                    fixed = 'R ' + re.sub(r'^[Rr]\s*', '', fixed)
                
                # Clean up spacing and artifacts
                fixed = re.sub(r'\s+', ' ', fixed).strip()
                
                # Final validation
                if fixed != original:
                    return fixed
                return original
            
            for line in result[0]:
                if len(line) >= 2:
                    bbox_points = line[0]
                    text_info = line[1]
                    
                    if len(text_info) >= 2:
                        text = text_info[0].strip()
                        confidence = float(text_info[1])
                        
                        if len(bbox_points) >= 4:
                            # Convert bbox points to (x1, y1, x2, y2) format
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            x1, y1 = min(x_coords), min(y_coords)
                            x2, y2 = max(x_coords), max(y_coords)
                            
                            # Scale coordinates back to original image space
                            x1, y1, x2, y2 = float(x1)/scale + offset_x, float(y1)/scale + offset_y, float(x2)/scale + offset_x, float(y2)/scale + offset_y
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            
                            # Create normalized token
                            token = NormalizedToken(
                                raw_text=text.strip(),
                                text=normalize_engineering_text(text.strip()),
                                cx=cx, cy=cy,
                                conf=min(confidence, 0.8),
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                token_type=_classify_precision_token(text),
                                semantic_type="dimension"
                            )
                            tokens.append(token)
        
        return tokens
        
    except Exception as e:
        logger.error("PaddleOCR precision recovery failed: %s", e)
        raise

def _run_rapid_precision_recovery(upscaled: np.ndarray, scale: int, offset_x: int, offset_y: int) -> List[NormalizedToken]:
    """Run RapidOCR for precision recovery."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        
        ocr = RapidOCR()
        result = ocr(upscaled)
        
        tokens = []
        if result and isinstance(result, tuple) and len(result) >= 2:
            # RapidOCR returns (detected_texts, detection_boxes)
            detected_texts, detection_boxes = result[:2]
            
            # STRICT VALIDATION - Check for None values
            if detected_texts is None or detection_boxes is None:
                logger.warning(f"RapidOCR returned None values: texts={detected_texts}, boxes={detection_boxes}")
                return tokens
        else:
            # STRICT VALIDATION - Log invalid result and return empty
            logger.warning(f"Invalid OCR result: {type(result)} - {result}")
            return tokens  # Return empty list if result is invalid
            
        for i, text_info in enumerate(detected_texts):
            if len(text_info) >= 2:
                bbox_points = text_info[0]  # Bounding box points
                text = text_info[1]        # Text content
                confidence = text_info[2] if len(text_info) > 2 else 0.9
                
                if len(bbox_points) >= 4:
                    # Convert bbox points to (x1, y1, x2, y2) format
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    confidence = text_info[2] if len(text_info) > 2 else 0.9
                    
                    if len(bbox_points) >= 4:
                        # Convert bbox points to (x1, y1, x2, y2) format
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        # Scale coordinates back to original image space
                        x1, y1, x2, y2 = float(x1)/scale + offset_x, float(y1)/scale + offset_y, float(x2)/scale + offset_x, float(y2)/scale + offset_y
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        # Create normalized token
                        token = NormalizedToken(
                            raw_text=text.strip(),
                            text=text.strip(),
                            cx=cx, cy=cy,
                            conf=min(confidence, 0.8),
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            token_type=_classify_precision_token(text),
                            semantic_type="dimension"
                        )
                        tokens.append(token)
        
        return tokens
        
    except Exception as e:
        logger.error("RapidOCR precision recovery failed: %s", e)
        raise

def _enhance_for_precision(image: np.ndarray) -> np.ndarray:
    """Enhance image for precision note OCR."""
    
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive threshold for better text separation
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return binary

def _invert_for_precision(image: np.ndarray) -> np.ndarray:
    """Invert image for precision note OCR (useful for light text on dark background)."""
    
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Invert the image
    inverted = cv2.bitwise_not(gray)
    
    return inverted


def _sharpen_for_precision(image: np.ndarray) -> np.ndarray:
    """Sharpen image for precision note OCR."""
    
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    return sharpened


def _is_precision_token(text: str) -> bool:
    """Check if text is likely a precision note token."""
    
    text = text.strip()
    
    # Check for tolerance patterns
    if re.search(r'±', text):
        return True
    
    # Check for diameter symbols
    if 'Ø' in text:
        return True
    
    # Check for decimal numbers (precision values)
    if re.fullmatch(r'\d+\.\d{2,3}', text):
        return True
    
    # Check for leading decimals
    if re.fullmatch(r'\.\d+', text):
        return True
    
    # Check for engineering symbols
    if any(symbol in text for symbol in ['°', 'x', '/']):
        return True
    
    return False


def _classify_precision_token(text: str) -> str:
    """Classify precision token type."""
    
    text = text.strip()
    
    if 'Ø' in text or re.search(r'±', text):
        return "dimension"
    elif re.fullmatch(r'\d+\.\d+', text) or re.fullmatch(r'\.\d+', text):
        return "dimension"
    elif any(symbol in text for symbol in ['°', 'x', '/']):
        return "dimension"
    else:
        return "keyword"


def deduplicate_recovered_tokens(recovered_tokens: List[NormalizedToken], existing_tokens: List[NormalizedToken]) -> List[NormalizedToken]:
    """SAFETY CONTROL 1: Deduplicate recovered tokens against existing OCR tokens."""
    
    deduplicated = []
    
    for recovered_token in recovered_tokens:
        is_duplicate = False
        
        # Check against existing OCR tokens
        for existing_token in existing_tokens:
            # Same text and close position (<20px)
            if (recovered_token.text.strip() == existing_token.text.strip()):
                dist = ((recovered_token.cx - existing_token.cx)**2 + (recovered_token.cy - existing_token.cy)**2)**0.5
                if dist < 20:
                    is_duplicate = True
                    break
        
        # Keep token if not duplicate
        if not is_duplicate:
            deduplicated.append(recovered_token)
    
    return deduplicated


def apply_confidence_hierarchy(original_tokens: List[NormalizedToken], recovered_tokens: List[NormalizedToken]) -> List[NormalizedToken]:
    """SAFETY CONTROL 2: Ensure original OCR > recovered OCR in confidence hierarchy."""
    
    # Lower confidence for recovered tokens to ensure original OCR is preferred
    for token in recovered_tokens:
        token.conf = min(token.conf, 0.75)  # Cap recovered tokens at 0.75 confidence
    
    # Keep original tokens at their original confidence
    all_tokens = original_tokens + recovered_tokens
    
    return all_tokens


def _adaptive_threshold_for_precision(image: np.ndarray) -> np.ndarray:
    """Apply adaptive threshold for precision note OCR."""
    
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive threshold with different parameters
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
    
    return binary


def _morphological_cleanup_for_precision(image: np.ndarray) -> np.ndarray:
    """Apply morphological operations for precision note OCR."""
    
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # Apply threshold
    _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary
