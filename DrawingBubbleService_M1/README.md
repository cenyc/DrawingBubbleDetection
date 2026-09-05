# Drawing Bubble Detection Service v4.0

Automatically detects numbered balloons (bubbles) in engineering drawings and maps each bubble number to its associated dimension or callout text.

Runs fully offline with no external API calls or data leakage.

## Architecture

### 10-Step Detection Pipeline

```
1.  Full-image OCR          RapidOCR + PaddleOCR + Florence-2 ensemble
2.  Token normalisation      Engineering text cleanup (symbols, units, prefixes)
3.  Circle detection         HoughCircles + HSV colour + evidence-based fallback
4.  Bubble identification    OCR inside circles, edge bubbles, uncircled numbers
5.  Callout grouping         Tolerance stacks, phrases, chamfers, threads, diameters
6.  Leader segment extract   HoughLinesP for dimension lines and leaders
7.  Leader seed detection    Radial exit at balloon rim boundary
8.  Leader path tracing      BFS + A* + ray-cast from seed to callout
9.  Global assignment        Hungarian algorithm (scipy.optimize.linear_sum_assignment)
10. Fallback linking         Proximity + direction scoring for unresolved bubbles
```

### OCR Ensemble

Three engines run in parallel for maximum text detection:

| Engine | Strength | Role |
|--------|----------|------|
| RapidOCR (ONNX) | Fast, reliable on medium text | Primary — always runs |
| PaddleOCR v5 | Better detection on small text | Augments bubble ID detection |
| Florence-2 (Microsoft) | Reads tiny text, understands layout | Augments bubble ID detection |

Results are deduplicated by IoU overlap and centre distance.

### Assignment Strategy

- **Path A (leader-based):** When a leader line exits the balloon, trace it to the callout it points at. Uses HoughLinesP rim-crossing detection, through-line filtering, and cone search.
- **Path B (proximity-based):** When no leader is found, use normalised distance with direction alignment as a tiebreaker.
- **Hungarian solver:** Builds a cost matrix across all bubbles and callouts, then finds the globally optimal one-to-one assignment.

### Key Modules

| File | Purpose |
|------|---------|
| `detector.py` | Main pipeline — orchestrates all 10 steps |
| `ocr_rules.py` | OCR text normalisation and token classification |
| `callout_rules.py` | Groups OCR tokens into callout candidates |
| `leader_geometry.py` | Radial exit detection, cone search, leader following |
| `leader_seed_rules.py` | Leader seed detection at balloon boundary |
| `trace_algorithms.py` | A* pathfinding, ray-casting, segment merging |
| `annotation_layer.py` | HSV-based annotation colour extraction |
| `ocr_ensemble.py` | Multi-engine OCR with deduplication |
| `linker_rules.py` | Fallback bubble-to-callout linking |
| `main.py` | FastAPI server with authentication |

## Setup

### Step 1 — Place the folder

```text
your_project/
└── DrawingBubbleService/
    ├── detector.py
    ├── main.py
    ├── requirements.txt
    ├── run.bat / start.sh
    └── ...
```

### Step 2 — Configure

```bash
cd DrawingBubbleService

# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` and set a strong API key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 3 — Install dependencies and start

```bash
pip install -r requirements.txt
```

**Optional OCR engines** (improve accuracy but not required):

```bash
# PaddleOCR v5 — better small-text detection
pip install paddlepaddle==3.0.0 paddleocr==3.4.0

# Florence-2 — tiny-text reading (requires ~1 GB disk)
pip install torch==2.4.0 torchvision==0.19.0 transformers==4.44.0
```

Start the server:

```bash
# Windows
run.bat

# Linux / macOS
bash start.sh
```

Server runs at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

## API

### `POST /api/detect`

Upload an engineering drawing and receive bubble-to-dimension assignments.

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | API key from `.env` |

**Form fields:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Drawing image (PNG, JPEG, TIFF, BMP; max 50 MB) |
| `include_annotated_image` | bool | Return annotated image as base64 PNG |
| `include_diagnostics` | bool | Return detailed diagnostics |

**Response example:**

```json
{
  "bubble_count": 10,
  "bubbles": [
    {
      "bubble_number": "1",
      "dimension": "4 DIA",
      "x": 520,
      "y": 340,
      "radius": 35,
      "confidence": 0.92,
      "needs_review": false
    }
  ],
  "processing_time_ms": 4200.5
}
```

### `GET /api/health`

Returns service status, uptime, memory usage, and request counts.

## Quick Test

```bash
python DrawingBubbleService/test_api.py drawing.png YOUR_API_KEY
```

## Offline Install

On a machine with internet access:

```bash
# Download all wheels
bash DrawingBubbleService/offline_install.sh download

# Windows
DrawingBubbleService\offline_install.bat download
```

Copy the entire `DrawingBubbleService/` folder to the offline machine:

```bash
bash DrawingBubbleService/offline_install.sh install

# Windows
DrawingBubbleService\offline_install.bat install
```

## Regression Tests

```bash
python -m pytest DrawingBubbleService/tests/test_regression.py -v
```

Tests run against 8 reference drawings with ground-truth annotations. Current performance: 100% bubble recall, ~70% value accuracy across all test images.

## Requirements

### Diameter symbol recovery

Both OCR paths preserve observed `⌀`, `Ø`, `ø`, `Φ`, and `φ` as `Ø`.
Suspicious readings such as `2-06`, `2-03.5`, or a split `06` trigger
local quadrilateral rectification and enlargement. Recovery accepts a matching
quantity/value when a second OCR read contains a diameter symbol and scores
at least 0.8, or when the original character image verifies a ring split
diagonally by a stroke. The pixel check requires character/component alignment
and rejects ordinary zeros and vertically stacked eight-shaped holes. Dimension
underlines and small edge noise are excluded from that alignment. Numeric zeros
are never replaced from the text heuristic alone; normal decimals such as
`0.6` and `2-0.5` are excluded.

Each OCR invocation retries at most 12 suspicious regions with two reads each.
The second read bypasses text detection to avoid splitting the symbol again.
Production OCR caches repeated region reads across preprocessing variants.
The auto-annotation endpoint also verifies rotated OCR tokens before grouping,
preserves their original text and uses the engine's actual confidence values.
Unresolved readings use the existing review flag with
`ambiguous_diameter_symbol`; V2 debug records also retain the raw OCR text.
This does not guarantee recovery when the model omits or fragments a callout.

Run the focused regression suite from this directory:

```bash
python -m pytest test_diameter_ocr.py -q
```

### Runtime dependencies

- Python 3.12
- OpenCV 4.10
- RapidOCR (ONNX Runtime 1.17)
- FastAPI + Uvicorn
- scipy (for Hungarian algorithm)
- Optional: PaddleOCR v5, Florence-2 (transformers + PyTorch)
