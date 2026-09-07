"use client";

import {
  AlertCircle,
  Box,
  Check,
  ChevronLeft,
  ChevronRight,
  CirclePlus,
  Download,
  FileImage,
  FileSpreadsheet,
  Focus,
  Frame,
  LoaderCircle,
  Maximize2,
  MousePointer2,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { ChangeEvent, CSSProperties, DragEvent, KeyboardEvent, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, useEffect, useLayoutEffect, useMemo, useRef, useState, WheelEvent as ReactWheelEvent } from "react";

type FeatureType = "直径" | "线性尺寸" | "半径" | "角度" | "形位公差" | "螺纹" | "表面粗糙度";

type Feature = {
  uid: string;
  id: number;
  anchorX: number;
  anchorY: number;
  bubbleX: number;
  bubbleY: number;
  type: FeatureType;
  nominal: string;
  tolerance: string;
  method: string;
  instrument: string;
  confidence: number | null;
  rawText?: string;
  source?: "pdf-vector" | "paddleocr" | "bubble-api" | "browser" | "manual";
  regionId?: string;
};

type RegionStatus = "pending" | "recognizing" | "succeeded" | "failed";

type RecognitionRegion = {
  id: string;
  label: number;
  x: number;
  y: number;
  width: number;
  height: number;
  status: RegionStatus;
  resultCount: number;
  error?: string;
};

type RegionDraft = Omit<RecognitionRegion, "id" | "label" | "status" | "resultCount"> & {
  startX: number;
  startY: number;
  invalid: boolean;
};

type DocumentPage = {
  pageNumber: number;
  width: number;
  height: number;
  imageSrc: string;
  features: Feature[];
  regions: RecognitionRegion[];
};

type BubbleApiResponse = {
  request_id: string;
  balloon_count: number;
  processing_time_ms: number;
  balloons: Array<{
    number: number;
    balloon: {
      cx: number;
      cy: number;
      radius: number;
    };
    dimension_text: string;
    dim_bbox: [number, number, number, number];
    leader_points: Array<[number, number]>;
  }>;
};

type PdfDocumentLike = {
  numPages: number;
  getPage: (page: number) => Promise<{
    getViewport: (options: { scale: number }) => { width: number; height: number; convertToViewportPoint: (x: number, y: number) => [number, number] };
    render: (options: { canvas: HTMLCanvasElement; viewport: unknown }) => { promise: Promise<void> };
    getTextContent: () => Promise<{ items: Array<{ str?: string; transform?: number[] }> }>;
  }>;
};

const TYPE_META: Record<FeatureType, { method: string; instrument: string }> = {
  直径: { method: "两点测量", instrument: "数显卡尺" },
  线性尺寸: { method: "直接测量", instrument: "数显卡尺" },
  半径: { method: "轮廓比对", instrument: "半径规" },
  角度: { method: "角度测量", instrument: "万能角度尺" },
  形位公差: { method: "基准拟合", instrument: "三坐标" },
  螺纹: { method: "通止检验", instrument: "螺纹塞规" },
  表面粗糙度: { method: "轮廓法", instrument: "粗糙度仪" },
};

const SAMPLE_FEATURES: Feature[] = [
  { uid: "sample-1", id: 1, anchorX: 905, anchorY: 90, bubbleX: 967, bubbleY: 90, type: "线性尺寸", nominal: "23", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "23", source: "bubble-api" },
  { uid: "sample-2", id: 2, anchorX: 820, anchorY: 52, bubbleX: 727, bubbleY: 145, type: "线性尺寸", nominal: "29.5", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "29.5", source: "bubble-api" },
  { uid: "sample-3", id: 3, anchorX: 964, anchorY: 154, bubbleX: 1026, bubbleY: 154, type: "线性尺寸", nominal: "14", tolerance: "±0.2", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "14±0.2", source: "bubble-api" },
  { uid: "sample-4", id: 4, anchorX: 261, anchorY: 307, bubbleX: 210, bubbleY: 307, type: "线性尺寸", nominal: "1.94", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "1.94", source: "bubble-api" },
  { uid: "sample-5", id: 5, anchorX: 155, anchorY: 493, bubbleX: 111, bubbleY: 452, type: "线性尺寸", nominal: "15.8", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "15.8", source: "bubble-api" },
  { uid: "sample-6", id: 6, anchorX: 1483, anchorY: 605, bubbleX: 1545, bubbleY: 605, type: "直径", nominal: "2-Ø6", tolerance: "", method: "两点测量", instrument: "数显卡尺", confidence: null, rawText: "2-Ø6", source: "bubble-api" },
  { uid: "sample-7", id: 7, anchorX: 109, anchorY: 611, bubbleX: 109, bubbleY: 636, type: "线性尺寸", nominal: "21.5", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "21.5", source: "bubble-api" },
  { uid: "sample-8", id: 8, anchorX: 1366, anchorY: 767, bubbleX: 1366, bubbleY: 759, type: "直径", nominal: "2-Ø3.5", tolerance: "", method: "两点测量", instrument: "数显卡尺", confidence: null, rawText: "2-Ø3.5", source: "bubble-api" },
  { uid: "sample-9", id: 9, anchorX: 299, anchorY: 812, bubbleX: 250, bubbleY: 812, type: "线性尺寸", nominal: "2.06", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "2.06", source: "bubble-api" },
  { uid: "sample-10", id: 10, anchorX: 907, anchorY: 899, bubbleX: 907, bubbleY: 852, type: "线性尺寸", nominal: "50.3", tolerance: "±0.3", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "50.3±0.3", source: "bubble-api" },
];

const SAMPLE_IMAGE_SRC = "/demo-engineering-drawing.png";
const SAMPLE_FILE_NAME = "ScreenShot_2026-07-23_161820_677.png";
const SAMPLE_IMAGE_SIZE = { width: 1658, height: 990 };
const PANEL_STATE_STORAGE_KEY = "bubbleiq:workspace-panels:v1";
const DEFAULT_TABLE_PANEL_WIDTH = 520;
const MIN_TABLE_PANEL_WIDTH = 360;
const MIN_DRAWING_PANEL_WIDTH = 430;
const PANEL_RESIZER_WIDTH = 10;
const PANEL_GAP_WIDTH = 10;
const MIN_VIEW_ZOOM = 0.6;
const MAX_VIEW_ZOOM = 2;
const REGION_STATUS_LABELS: Record<RegionStatus, string> = {
  pending: "待识别",
  recognizing: "识别中",
  succeeded: "已完成",
  failed: "识别失败",
};

let fallbackIdCounter = 0;

function createClientId(): string {
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi?.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  if (typeof cryptoApi?.getRandomValues === "function") {
    const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
  }

  fallbackIdCounter += 1;
  return `client-${Date.now().toString(36)}-${fallbackIdCounter.toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function createSampleDrawing(): string {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 800;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, 1200, 800);
  ctx.strokeStyle = "#17232e";
  ctx.lineWidth = 2;
  ctx.strokeRect(28, 28, 1144, 744);
  ctx.lineWidth = 1;
  ctx.strokeRect(900, 650, 272, 122);
  ctx.beginPath();
  ctx.moveTo(900, 692); ctx.lineTo(1172, 692);
  ctx.moveTo(900, 730); ctx.lineTo(1172, 730);
  ctx.moveTo(1020, 650); ctx.lineTo(1020, 772);
  ctx.stroke();
  ctx.fillStyle = "#17232e";
  ctx.font = "600 18px Arial";
  ctx.fillText("FLANGE HOUSING", 1035, 677);
  ctx.font = "13px Arial";
  ctx.fillText("PART NO.  FH-2407-A", 912, 716);
  ctx.fillText("SCALE  1:1", 1035, 716);
  ctx.fillText("MATERIAL  AL 6061-T6", 912, 752);
  ctx.fillText("REV.  B", 1035, 752);

  ctx.save();
  ctx.translate(600, 372);
  ctx.strokeStyle = "#101820";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(0, 0, 178, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(0, 0, 94, 0, Math.PI * 2);
  ctx.arc(0, 0, 46, 0, Math.PI * 2);
  ctx.stroke();
  [0, 90, 180, 270].forEach((deg) => {
    const angle = deg * Math.PI / 180;
    const x = Math.cos(angle) * 128;
    const y = Math.sin(angle) * 128;
    ctx.beginPath();
    ctx.arc(x, y, 18, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 28, y); ctx.lineTo(x + 28, y);
    ctx.moveTo(x, y - 28); ctx.lineTo(x, y + 28);
    ctx.stroke();
  });
  ctx.setLineDash([12, 7]);
  ctx.strokeStyle = "#64727e";
  ctx.beginPath();
  ctx.moveTo(-215, 0); ctx.lineTo(215, 0);
  ctx.moveTo(0, -215); ctx.lineTo(0, 215);
  ctx.stroke();
  ctx.restore();

  ctx.strokeStyle = "#17232e";
  ctx.setLineDash([]);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(422, 160); ctx.lineTo(778, 160);
  ctx.moveTo(422, 148); ctx.lineTo(422, 172);
  ctx.moveTo(778, 148); ctx.lineTo(778, 172);
  ctx.moveTo(422, 160); ctx.lineTo(438, 154);
  ctx.moveTo(422, 160); ctx.lineTo(438, 166);
  ctx.moveTo(778, 160); ctx.lineTo(762, 154);
  ctx.moveTo(778, 160); ctx.lineTo(762, 166);
  ctx.stroke();
  ctx.font = "17px Arial";
  ctx.fillStyle = "#17232e";
  ctx.fillText("⌀120 ±0.05", 548, 148);
  ctx.fillText("4 × ⌀18 THRU", 820, 330);
  ctx.fillText("⌀32 ±0.02", 625, 430);
  ctx.fillText("Ra 1.6", 710, 540);
  ctx.fillText("UNLESS OTHERWISE SPECIFIED", 70, 680);
  ctx.font = "13px Arial";
  ctx.fillText("DIMENSIONS ARE IN MILLIMETERS", 70, 705);
  ctx.fillText("GENERAL TOLERANCE: ISO 2768-mK", 70, 727);
  ctx.fillText("BREAK SHARP EDGES 0.2–0.5", 70, 749);
  return canvas.toDataURL("image/png");
}

function classifyToken(token: string): FeatureType {
  const text = token.toUpperCase();
  if (/RA|RZ/.test(text)) return "表面粗糙度";
  if (/M\s*\d/.test(text)) return "螺纹";
  if (/Ø|⌀|Φ|DIA/.test(text)) return "直径";
  if (/R\s*\d/.test(text)) return "半径";
  if (/°|DEG/.test(text)) return "角度";
  if (/POSITION|FLAT|PARALLEL|COAX/.test(text)) return "形位公差";
  return "线性尺寸";
}

function inferredTolerance(token: string, type: FeatureType): string {
  const plusMinus = token.match(/±\s*([0-9.]+)/);
  if (plusMinus) return `±${plusMinus[1]}`;
  if (type === "表面粗糙度") {
    const roughness = token.match(/R[az]\s*([0-9.]+)/i);
    return roughness ? `≤${roughness[1]} μm` : "";
  }
  return "";
}

function cleanToken(value: string): string {
  return value.replace(/\s+/g, " ").replace(/[|]/g, "").trim();
}

function downloadBlob(content: BlobPart, type: string, filename: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const workspaceRef = useRef<HTMLDivElement>(null);
  const drawingViewportRef = useRef<HTMLDivElement>(null);
  const drawingStageRef = useRef<HTMLDivElement>(null);
  const uploadPanelRef = useRef<HTMLElement>(null);
  const tablePanelRef = useRef<HTMLElement>(null);
  const pendingTableWidthRef = useRef<number | null>(null);
  const isTableResizingRef = useRef(false);
  const stopTableResizeListenersRef = useRef<(() => void) | null>(null);
  const zoomRef = useRef(1);
  const pendingZoomAnchorRef = useRef<{
    clientX: number;
    clientY: number;
    anchorX: number;
    anchorY: number;
  } | null>(null);
  const viewPanRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startScrollLeft: number;
    startScrollTop: number;
  } | null>(null);
  const suppressViewClickRef = useRef(false);
  const [imageSrc, setImageSrc] = useState(SAMPLE_IMAGE_SRC);
  const [fileName, setFileName] = useState(SAMPLE_FILE_NAME);
  const [imageSize, setImageSize] = useState(SAMPLE_IMAGE_SIZE);
  const [features, setFeatures] = useState<Feature[]>(SAMPLE_FEATURES);
  const [selectedId, setSelectedId] = useState<string | null>("sample-1");
  const [dragId, setDragId] = useState<string | null>(null);
  const [addMode, setAddMode] = useState(false);
  const [regionMode, setRegionMode] = useState(false);
  const [regions, setRegions] = useState<RecognitionRegion[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [regionDraft, setRegionDraft] = useState<RegionDraft | null>(null);
  const [regionNotice, setRegionNotice] = useState("");
  const [zoom, setZoom] = useState(1);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(100);
  const [stage, setStage] = useState("识别完成");
  const [errorMessage, setErrorMessage] = useState("");
  const [pageCount, setPageCount] = useState(1);
  const [pageNumber, setPageNumber] = useState(1);
  const [documentPages, setDocumentPages] = useState<DocumentPage[]>([]);
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("全部类型");
  const [dragOver, setDragOver] = useState(false);
  const [analysisEngine, setAnalysisEngine] = useState("DrawingBubbleDetection 示例结果");
  const [analysisWarning, setAnalysisWarning] = useState("");
  const [isUploadPanelCollapsed, setIsUploadPanelCollapsed] = useState(false);
  const [isTablePanelCollapsed, setIsTablePanelCollapsed] = useState(true);
  const [tablePanelWidth, setTablePanelWidth] = useState<number | null>(null);
  const [isTableResizing, setIsTableResizing] = useState(false);
  const [isPanelStateLoaded, setIsPanelStateLoaded] = useState(false);
  const [isViewPanning, setIsViewPanning] = useState(false);

  useEffect(() => () => stopTableResizeListenersRef.current?.(), []);

  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);

  useLayoutEffect(() => {
    const anchor = pendingZoomAnchorRef.current;
    const viewport = drawingViewportRef.current;
    const stage = drawingStageRef.current;
    if (!anchor || !viewport || !stage) return;

    const stageRect = stage.getBoundingClientRect();
    viewport.scrollLeft += stageRect.left + anchor.anchorX * stageRect.width - anchor.clientX;
    viewport.scrollTop += stageRect.top + anchor.anchorY * stageRect.height - anchor.clientY;
    pendingZoomAnchorRef.current = null;
  }, [zoom]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const savedState = window.localStorage.getItem(PANEL_STATE_STORAGE_KEY);
        if (savedState) {
          const parsedState: unknown = JSON.parse(savedState);
          if (typeof parsedState === "object" && parsedState !== null) {
            const panelState = parsedState as { uploadCollapsed?: unknown; tableCollapsed?: unknown; tableWidth?: unknown };
            if (typeof panelState.uploadCollapsed === "boolean") {
              setIsUploadPanelCollapsed(panelState.uploadCollapsed);
            }
            if (typeof panelState.tableCollapsed === "boolean") {
              setIsTablePanelCollapsed(panelState.tableCollapsed);
            }
            if (typeof panelState.tableWidth === "number" && Number.isFinite(panelState.tableWidth)) {
              const workspaceWidth = workspaceRef.current?.getBoundingClientRect().width ?? window.innerWidth;
              const uploadWidth = uploadPanelRef.current?.getBoundingClientRect().width ?? 220;
              const maxWidth = Math.max(
                MIN_TABLE_PANEL_WIDTH,
                workspaceWidth - uploadWidth - PANEL_GAP_WIDTH - MIN_DRAWING_PANEL_WIDTH - PANEL_RESIZER_WIDTH,
              );
              setTablePanelWidth(Math.min(Math.max(panelState.tableWidth, MIN_TABLE_PANEL_WIDTH), maxWidth));
            }
          }
        }
      } catch {
        // Keep the safe defaults when storage is unavailable or contains invalid data.
      } finally {
        setIsPanelStateLoaded(true);
      }
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!isPanelStateLoaded) return;

    try {
      window.localStorage.setItem(PANEL_STATE_STORAGE_KEY, JSON.stringify({
        uploadCollapsed: isUploadPanelCollapsed,
        tableCollapsed: isTablePanelCollapsed,
        tableWidth: tablePanelWidth,
      }));
    } catch {
      // The controls still work for this visit when persistent storage is unavailable.
    }
  }, [isPanelStateLoaded, isTablePanelCollapsed, isUploadPanelCollapsed, tablePanelWidth]);

  const filteredFeatures = useMemo(() => features.filter((feature) => {
    const matchesQuery = `${feature.id} ${feature.type} ${feature.nominal} ${feature.instrument}`.toLowerCase().includes(query.toLowerCase());
    return matchesQuery && (typeFilter === "全部类型" || feature.type === typeFilter);
  }), [features, query, typeFilter]);

  const scoredFeatures = features.filter(
    (feature): feature is Feature & { confidence: number } => feature.confidence !== null,
  );
  const averageConfidence = scoredFeatures.length
    ? Math.round(scoredFeatures.reduce((sum, item) => sum + item.confidence, 0) / scoredFeatures.length)
    : 0;
  const regionSelectionDisabled = !imageSrc || processing;
  const pageRecognitionDisabled = !imageSrc || processing;
  const regionRecognitionDisabled = processing || !regions.some(
    (region) => region.status === "pending" || region.status === "failed",
  );

  function resizeTablePanel(clientX: number): number | null {
    const workspace = workspaceRef.current;
    if (!workspace || isTablePanelCollapsed) return null;

    const workspaceRect = workspace.getBoundingClientRect();
    const uploadWidth = uploadPanelRef.current?.getBoundingClientRect().width ?? 0;
    const maxWidth = Math.max(
      MIN_TABLE_PANEL_WIDTH,
      workspaceRect.width - uploadWidth - PANEL_GAP_WIDTH - MIN_DRAWING_PANEL_WIDTH - PANEL_RESIZER_WIDTH,
    );
    const nextWidth = Math.min(Math.max(workspaceRect.right - clientX, MIN_TABLE_PANEL_WIDTH), maxWidth);
    workspace.style.setProperty("--right-panel-width", `${nextWidth}px`);
    pendingTableWidthRef.current = nextWidth;
    return nextWidth;
  }

  function handleTableResizeStart(event: ReactPointerEvent<HTMLDivElement>) {
    if (isTablePanelCollapsed) return;
    event.preventDefault();
    isTableResizingRef.current = true;
    pendingTableWidthRef.current = tablePanelRef.current?.getBoundingClientRect().width ?? tablePanelWidth;
    setIsTableResizing(true);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      if (moveEvent.buttons === 0) {
        finishResize();
        return;
      }
      resizeTablePanel(moveEvent.clientX);
    };
    const stopListening = () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", finishResize);
      window.removeEventListener("pointercancel", finishResize);
      if (stopTableResizeListenersRef.current === stopListening) {
        stopTableResizeListenersRef.current = null;
      }
    };
    const finishResize = () => {
      stopListening();
      commitTableResize();
    };

    stopTableResizeListenersRef.current?.();
    stopTableResizeListenersRef.current = stopListening;
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", finishResize);
    window.addEventListener("pointercancel", finishResize);
  }

  function commitTableResize() {
    if (!isTableResizingRef.current) return;
    isTableResizingRef.current = false;
    if (pendingTableWidthRef.current !== null) {
      setTablePanelWidth(Math.round(pendingTableWidthRef.current));
    }
    pendingTableWidthRef.current = null;
    setIsTableResizing(false);
  }

  function handleTableResizeKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (isTablePanelCollapsed || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) return;
    event.preventDefault();
    const currentWidth = tablePanelRef.current?.getBoundingClientRect().width ?? MIN_TABLE_PANEL_WIDTH;
    const direction = event.key === "ArrowLeft" ? 24 : -24;
    const workspaceRight = workspaceRef.current?.getBoundingClientRect().right ?? window.innerWidth;
    const nextWidth = resizeTablePanel(workspaceRight - currentWidth - direction);
    if (nextWidth !== null) {
      setTablePanelWidth(Math.round(nextWidth));
      pendingTableWidthRef.current = null;
    }
  }

  async function loadImageDimensions(src: string): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = reject;
      image.src = src;
    });
  }

  function applyDocumentPage(page: DocumentPage) {
    setImageSrc(page.imageSrc);
    setImageSize({ width: page.width, height: page.height });
    setFeatures(page.features);
    setRegions(page.regions);
    setSelectedId(page.features[0]?.uid ?? null);
    setSelectedRegionId(null);
    setRegionDraft(null);
    setRegionNotice("");
    setPageNumber(page.pageNumber);
  }

  function featuresFromBubbleApi(result: BubbleApiResponse, region?: RecognitionRegion): Feature[] {
    return result.balloons.map((item) => {
      const text = cleanToken(item.dimension_text);
      const type = classifyToken(text);
      const tolerance = inferredTolerance(text, type);
      const nominal = tolerance ? text.replace(tolerance, "").trim() : text;
      const [x0, y0, x1, y1] = item.dim_bbox;
      const leaderEnd = item.leader_points.at(-1);
      const meta = TYPE_META[type];
      const offsetX = region?.x ?? 0;
      const offsetY = region?.y ?? 0;
      return {
        uid: createClientId(),
        id: item.number,
        anchorX: offsetX + (leaderEnd?.[0] ?? (x0 + x1) / 2),
        anchorY: offsetY + (leaderEnd?.[1] ?? (y0 + y1) / 2),
        bubbleX: offsetX + item.balloon.cx,
        bubbleY: offsetY + item.balloon.cy,
        type,
        nominal,
        tolerance,
        method: meta.method,
        instrument: meta.instrument,
        confidence: null,
        rawText: text,
        source: "bubble-api",
        regionId: region?.id,
      };
    });
  }

  async function autoAnnotateWithBubbleService(file: File): Promise<BubbleApiResponse> {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/bubble/auto-annotate", {
      method: "POST",
      body,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.detail || `HTTP ${response.status}`);
    }
    return payload as BubbleApiResponse;
  }

  async function renderPdfPageAsImage(pdfDoc: PdfDocumentLike, targetPage: number, originalName: string) {
    const page = await pdfDoc.getPage(targetPage);
    const baseViewport = page.getViewport({ scale: 1 });
    const scale = Math.min(2.2, 1800 / baseViewport.width);
    const viewport = page.getViewport({ scale });
    const canvas = window.document.createElement("canvas");
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    await page.render({ canvas, viewport }).promise;
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => {
        if (value) resolve(value);
        else reject(new Error(`PDF 第 ${targetPage} 页转换图片失败`));
      }, "image/png");
    });
    const baseName = originalName.replace(/\.pdf$/i, "");
    return {
      imageSrc: canvas.toDataURL("image/png"),
      width: canvas.width,
      height: canvas.height,
      file: new File([blob], `${baseName}_第${targetPage}页.png`, { type: "image/png" }),
    };
  }

  async function processFile(file: File) {
    const lowerName = file.name.toLowerCase();
    const isPdf = file.type === "application/pdf" || lowerName.endsWith(".pdf");
    const isImage =
      ["image/png", "image/jpeg"].includes(file.type)
      || /\.(png|jpe?g)$/i.test(lowerName);
    if (!file || (!isImage && !isPdf)) {
      setErrorMessage("暂不支持该文件格式，请上传 PDF、PNG 或 JPG 文件。");
      setStage("文件格式不受支持");
      setProgress(0);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setErrorMessage("文件超过 20 MB，请压缩图纸后重试。");
      setStage("文件过大");
      setProgress(0);
      return;
    }
    setProcessing(true);
    setErrorMessage("");
    setFileName(file.name);
    setStage("读取工程图文件");
    setProgress(8);
    setZoom(1);
    setImageSrc("");
    setFeatures([]);
    setRegions([]);
    setSelectedId(null);
    setSelectedRegionId(null);
    setRegionDraft(null);
    setRegionMode(false);
    setAddMode(false);
    setDocumentPages([]);
    setAnalysisWarning("");
    try {
      if (isImage) {
        const src = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
        const size = await loadImageDimensions(src);
        const imagePage: DocumentPage = {
          pageNumber: 1,
          width: size.width,
          height: size.height,
          imageSrc: src,
          features: [],
          regions: [],
        };
        setPageCount(1);
        setDocumentPages([imagePage]);
        applyDocumentPage(imagePage);
        setAnalysisEngine("DrawingBubbleDetection 图纸识别");
        setAnalysisWarning("图纸已渲染，可直接识别当前页，或框选一个或多个区域进行精细识别。");
        setStage("等待选择识别范围");
        setProgress(25);
        setProcessing(false);
        return;
      }

      const pdfjs = await import("pdfjs-dist");
      pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
      const bytes = new Uint8Array(await file.arrayBuffer());
      const pdfDoc = await pdfjs.getDocument({ data: bytes }).promise as unknown as PdfDocumentLike;
      const pagesToProcess = Math.min(pdfDoc.numPages, 10);
      const detectedPages: DocumentPage[] = [];
      setPageCount(pagesToProcess);
      setAnalysisEngine("DrawingBubbleDetection PDF 图纸识别");

      for (let targetPage = 1; targetPage <= pagesToProcess; targetPage += 1) {
        setStage(`渲染 PDF 第 ${targetPage} / ${pagesToProcess} 页`);
        setProgress(8 + Math.round(((targetPage - 1) / pagesToProcess) * 17));
        const rendered = await renderPdfPageAsImage(pdfDoc, targetPage, file.name);
        detectedPages.push({
          pageNumber: targetPage,
          width: rendered.width,
          height: rendered.height,
          imageSrc: rendered.imageSrc,
          features: [],
          regions: [],
        });
      }

      setDocumentPages(detectedPages);
      applyDocumentPage(detectedPages[0]);
      const truncatedMessage = pdfDoc.numPages > pagesToProcess
        ? ` 为控制处理时间，本次只渲染前 ${pagesToProcess} 页。`
        : "";
      setAnalysisWarning(
        `PDF 已渲染，可逐页直接识别，或框选区域进行精细识别。${truncatedMessage}`,
      );
      setStage("等待选择识别范围");
      setProgress(25);
      setProcessing(false);
    } catch (error) {
      console.error("Drawing processing failed", error);
      setStage("气泡标注失败");
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "未能处理这份图纸，请检查本地气泡标注服务。",
      );
      setProgress(0);
      setProcessing(false);
    }
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void processFile(file);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void processFile(file);
  }

  function updateCurrentPageFeatures(updater: (current: Feature[]) => Feature[]) {
    setFeatures(updater);
    setDocumentPages((pages) => pages.map((page) => (
      page.pageNumber === pageNumber ? { ...page, features: updater(page.features) } : page
    )));
  }

  function updateCurrentPageRegions(updater: (current: RecognitionRegion[]) => RecognitionRegion[]) {
    setRegions(updater);
    setDocumentPages((pages) => pages.map((page) => (
      page.pageNumber === pageNumber ? { ...page, regions: updater(page.regions) } : page
    )));
  }

  function updateFeature(uid: string, key: keyof Feature, value: string | number) {
    updateCurrentPageFeatures((current) => current.map((feature) => (
      feature.uid === uid ? { ...feature, [key]: value } : feature
    )));
  }

  function removeFeature(uid: string) {
    updateCurrentPageFeatures((current) => current
      .filter((feature) => feature.uid !== uid)
      .map((feature, index) => ({ ...feature, id: index + 1 })));
    setSelectedId(null);
  }

  function pointIsInsideRegion(x: number, y: number, region: Pick<RecognitionRegion, "x" | "y" | "width" | "height">) {
    return x >= region.x
      && x <= region.x + region.width
      && y >= region.y
      && y <= region.y + region.height;
  }

  function featureIsInsideRegion(feature: Feature, region: RecognitionRegion) {
    return feature.regionId === region.id
      || pointIsInsideRegion(feature.bubbleX, feature.bubbleY, region);
  }

  function requestClearAllBubbles() {
    if (processing || (!features.length && !regions.length)) return;
    const confirmed = window.confirm(`确定清空当前页的 ${features.length} 个气泡和 ${regions.length} 个识别区域吗？此操作不可撤销。`);
    if (!confirmed) return;

    updateCurrentPageFeatures(() => []);
    updateCurrentPageRegions(() => []);
    setSelectedId(null);
    setSelectedRegionId(null);
    setRegionDraft(null);
    setRegionNotice("");
    setRegionMode(false);
    setProgress(25);
    setStage("已清空当前页标注");
    setAnalysisWarning(`当前页的 ${features.length} 个气泡和 ${regions.length} 个识别区域已清空。`);
  }

  function regionsOverlap(a: Pick<RecognitionRegion, "x" | "y" | "width" | "height">, b: Pick<RecognitionRegion, "x" | "y" | "width" | "height">) {
    return a.x < b.x + b.width
      && a.x + a.width > b.x
      && a.y < b.y + b.height
      && a.y + a.height > b.y;
  }

  function nextRegionLabel(current: RecognitionRegion[]) {
    const used = new Set(current.map((region) => region.label));
    let label = 1;
    while (used.has(label)) label += 1;
    return label;
  }

  function requestDeleteRegion(regionId: string) {
    if (processing) return;
    const region = regions.find((item) => item.id === regionId);
    if (!region) return;
    if (region.resultCount > 0) {
      const confirmed = window.confirm(`删除区域 ${region.label} 将同时删除它生成的 ${region.resultCount} 条检验特性，是否继续？`);
      if (!confirmed) return;
    }
    updateCurrentPageFeatures((current) => current
      .filter((feature) => feature.regionId !== regionId)
      .map((feature, index) => ({ ...feature, id: index + 1 })));
    updateCurrentPageRegions((current) => current.filter((item) => item.id !== regionId));
    const remainingRegions = regions.filter((item) => item.id !== regionId);
    setSelectedRegionId((current) => current === regionId ? null : current);
    setSelectedId(null);
    setRegionNotice(`区域 ${region.label} 已删除。`);
    setProgress(
      !remainingRegions.length
        ? 25
        : remainingRegions.every((item) => item.status === "succeeded") ? 100 : 35,
    );
    setStage(remainingRegions.length ? "区域已删除" : "等待选择识别范围");
    setAnalysisWarning(
      region.resultCount > 0
        ? `区域 ${region.label} 及其 ${region.resultCount} 项识别结果已删除。`
        : `区域 ${region.label} 已删除。`,
    );
  }

  async function createRecognitionFile(region?: RecognitionRegion): Promise<File> {
    const image = new Image();
    image.src = imageSrc;
    await image.decode();
    const bounds = region ?? {
      x: 0,
      y: 0,
      width: imageSize.width,
      height: imageSize.height,
    };
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bounds.width));
    canvas.height = Math.max(1, Math.round(bounds.height));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("无法创建识别画布。");
    context.drawImage(
      image,
      bounds.x,
      bounds.y,
      bounds.width,
      bounds.height,
      0,
      0,
      canvas.width,
      canvas.height,
    );
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => value ? resolve(value) : reject(new Error("识别图片生成失败。")), "image/png");
    });
    const scopeName = region ? `区域${region.label}` : `第${pageNumber}页_整页`;
    return new File([blob], `${fileName.replace(/\.[^.]+$/, "")}_${scopeName}.png`, { type: "image/png" });
  }

  function cropRegion(region: RecognitionRegion): Promise<File> {
    return createRecognitionFile(region);
  }

  async function recognizeCurrentPage() {
    if (pageRecognitionDisabled) return;

    setProcessing(true);
    setErrorMessage("");
    setRegionNotice("");
    setAnalysisWarning("");
    setSelectedId(null);
    setSelectedRegionId(null);
    setRegionDraft(null);
    setRegionMode(false);
    setStage(`正在识别第 ${pageNumber} 页`);
    setProgress(35);

    try {
      const pageFile = await createRecognitionFile();
      setProgress(65);
      const result = await autoAnnotateWithBubbleService(pageFile);
      const detected = featuresFromBubbleApi(result);
      updateCurrentPageFeatures((current) => [
        ...current.filter((feature) => feature.source === "manual"),
        ...detected,
      ]
        .sort((a, b) => a.anchorY - b.anchorY || a.anchorX - b.anchorX)
        .map((feature, index) => ({ ...feature, id: index + 1 })));
      updateCurrentPageRegions((current) => current.map((region) => ({
        ...region,
        status: "pending",
        resultCount: 0,
        error: undefined,
      })));
      setProgress(100);
      setStage("整页识别完成");
      setAnalysisWarning(
        `当前页识别出 ${detected.length} 项特性；可继续框选区域进行精细识别，区域结果会替换对应范围内的自动结果。`,
      );
    } catch (error) {
      console.error("Full-page recognition failed", error);
      setStage("整页识别失败");
      setErrorMessage(error instanceof Error ? error.message : "整页识别失败，请检查识别服务后重试。");
      setProgress(25);
    } finally {
      setProcessing(false);
    }
  }

  async function recognizeRegions(regionIds?: string[]) {
    const requestedIds = regionIds ? new Set(regionIds) : null;
    const targets = regions.filter((region) => (
      (requestedIds ? requestedIds.has(region.id) : region.status === "pending" || region.status === "failed")
      && region.status !== "recognizing"
    ));
    if (!targets.length || processing) return;

    setProcessing(true);
    setErrorMessage("");
    setRegionNotice("");
    setAnalysisWarning("");
    let succeeded = 0;
    let failed = 0;

    for (let index = 0; index < targets.length; index += 1) {
      const region = targets[index];
      setStage(`正在识别区域 ${region.label}（${index + 1}/${targets.length}）`);
      setProgress(35 + Math.round((index / targets.length) * 55));
      updateCurrentPageRegions((current) => current.map((item) => (
        item.id === region.id ? { ...item, status: "recognizing", error: undefined } : item
      )));
      setSelectedId(null);
      try {
        const croppedFile = await cropRegion(region);
        const result = await autoAnnotateWithBubbleService(croppedFile);
        const detected = featuresFromBubbleApi(result, region);
        updateCurrentPageFeatures((current) => [
          ...current.filter((feature) => (
            feature.source === "manual" || !featureIsInsideRegion(feature, region)
          )),
          ...detected,
        ]
          .sort((a, b) => a.anchorY - b.anchorY || a.anchorX - b.anchorX)
          .map((feature, featureIndex) => ({ ...feature, id: featureIndex + 1 })));
        updateCurrentPageRegions((current) => current.map((item) => (
          item.id === region.id
            ? { ...item, status: "succeeded", resultCount: detected.length, error: undefined }
            : item
        )));
        succeeded += 1;
      } catch (error) {
        const message = error instanceof Error ? error.message : "区域识别失败";
        updateCurrentPageRegions((current) => current.map((item) => (
          item.id === region.id ? { ...item, status: "failed", resultCount: 0, error: message } : item
        )));
        failed += 1;
      }
    }

    setProgress(100);
    setStage(failed ? "部分区域识别完成" : "区域识别完成");
    setAnalysisWarning(
      failed
        ? `${succeeded} 个区域识别成功，${failed} 个区域失败，可在区域列表中重试。`
        : `${succeeded} 个区域识别完成；所有结果已合并到检验特性表，仍需由工程师复核。`,
    );
    setProcessing(false);
  }

  function pointerCoordinates(event: ReactPointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.min(imageSize.width, Math.max(0, (event.clientX - rect.left) / rect.width * imageSize.width)),
      y: Math.min(imageSize.height, Math.max(0, (event.clientY - rect.top) / rect.height * imageSize.height)),
    };
  }

  function handleRegionPointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    if (!regionMode || processing || event.button !== 0 || event.target !== event.currentTarget) return;
    event.preventDefault();
    const point = pointerCoordinates(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelectedRegionId(null);
    setSelectedId(null);
    setRegionNotice("");
    setRegionDraft({
      startX: point.x,
      startY: point.y,
      x: point.x,
      y: point.y,
      width: 0,
      height: 0,
      invalid: false,
    });
  }

  function handleRegionPointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!regionDraft || !regionMode) return;
    const point = pointerCoordinates(event);
    const next = {
      x: Math.min(regionDraft.startX, point.x),
      y: Math.min(regionDraft.startY, point.y),
      width: Math.abs(point.x - regionDraft.startX),
      height: Math.abs(point.y - regionDraft.startY),
    };
    const overlapsExistingRegion = regions.some((region) => regionsOverlap(next, region));
    if (overlapsExistingRegion && !regionDraft.invalid) {
      setRegionNotice("所选区域与已有区域重叠，请调整框选范围。");
    } else if (!overlapsExistingRegion && regionDraft.invalid) {
      setRegionNotice("");
    }
    setRegionDraft({
      ...regionDraft,
      ...next,
      invalid: overlapsExistingRegion,
    });
  }

  function finishRegionSelection(event: ReactPointerEvent<SVGSVGElement>) {
    if (!regionDraft || !regionMode) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const minWidth = Math.max(24, imageSize.width * 0.01);
    const minHeight = Math.max(24, imageSize.height * 0.01);
    if (regionDraft.invalid) {
      setRegionNotice("识别区域不能与已有区域重叠，请重新框选。");
      setRegionDraft(null);
      return;
    }
    if (regionDraft.width < minWidth || regionDraft.height < minHeight) {
      setRegionNotice("选择范围过小，请拖动选择更大的识别区域。");
      setRegionDraft(null);
      return;
    }
    const region: RecognitionRegion = {
      id: createClientId(),
      label: nextRegionLabel(regions),
      x: regionDraft.x,
      y: regionDraft.y,
      width: regionDraft.width,
      height: regionDraft.height,
      status: "pending",
      resultCount: 0,
    };
    updateCurrentPageRegions((current) => [...current, region]);
    setSelectedRegionId(region.id);
    setRegionDraft(null);
    setRegionNotice(`区域 ${region.label} 已添加，可继续框选。`);
    setProgress((current) => Math.max(current, 35));
    setStage("已选择识别区域");
  }

  function cancelRegionSelection(event: ReactPointerEvent<SVGSVGElement>) {
    if (!regionDraft) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setRegionDraft(null);
  }

  function handleOverlayKeyDown(event: KeyboardEvent<SVGSVGElement>) {
    if ((event.key === "Delete" || event.key === "Backspace") && selectedRegionId) {
      event.preventDefault();
      requestDeleteRegion(selectedRegionId);
    }
  }

  function handleViewportWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (!event.shiftKey) return;

    event.preventDefault();
    const viewport = drawingViewportRef.current;
    const stage = drawingStageRef.current;
    if (!viewport || !stage) return;

    const wheelDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
    if (wheelDelta === 0) return;

    const currentZoom = zoomRef.current;
    const nextZoom = Math.min(
      MAX_VIEW_ZOOM,
      Math.max(MIN_VIEW_ZOOM, currentZoom * Math.exp(-wheelDelta * 0.0015)),
    );
    if (Math.abs(nextZoom - currentZoom) < 0.001) return;

    const oldStageRect = stage.getBoundingClientRect();
    const anchorX = Math.min(1, Math.max(0, (event.clientX - oldStageRect.left) / oldStageRect.width));
    const anchorY = Math.min(1, Math.max(0, (event.clientY - oldStageRect.top) / oldStageRect.height));
    pendingZoomAnchorRef.current = {
      clientX: event.clientX,
      clientY: event.clientY,
      anchorX,
      anchorY,
    };
    zoomRef.current = nextZoom;
    setZoom(nextZoom);
  }

  function handleViewportPointerDownCapture(event: ReactPointerEvent<HTMLDivElement>) {
    if (!event.shiftKey || event.button !== 0) return;

    const viewport = event.currentTarget;
    event.preventDefault();
    event.stopPropagation();
    suppressViewClickRef.current = true;
    viewPanRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startScrollLeft: viewport.scrollLeft,
      startScrollTop: viewport.scrollTop,
    };
    viewport.setPointerCapture(event.pointerId);
    setIsViewPanning(true);
  }

  function handleViewportPointerMoveCapture(event: ReactPointerEvent<HTMLDivElement>) {
    const pan = viewPanRef.current;
    if (!pan || pan.pointerId !== event.pointerId) return;

    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.scrollLeft = pan.startScrollLeft - (event.clientX - pan.startX);
    event.currentTarget.scrollTop = pan.startScrollTop - (event.clientY - pan.startY);
  }

  function finishViewportPan(event: ReactPointerEvent<HTMLDivElement>, suppressClick: boolean) {
    const pan = viewPanRef.current;
    if (!pan || pan.pointerId !== event.pointerId) return;

    event.preventDefault();
    event.stopPropagation();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    viewPanRef.current = null;
    suppressViewClickRef.current = suppressClick;
    setIsViewPanning(false);
  }

  function handleViewportClickCapture(event: ReactMouseEvent<HTMLDivElement>) {
    if (!event.shiftKey && !suppressViewClickRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    suppressViewClickRef.current = false;
  }

  function handleOverlayPointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (regionMode) {
      handleRegionPointerMove(event);
      return;
    }
    if (dragId === null) return;
    const point = pointerCoordinates(event);
    updateCurrentPageFeatures((current) => current.map((feature) => (
      feature.uid === dragId ? { ...feature, bubbleX: point.x, bubbleY: point.y } : feature
    )));
  }

  function handleOverlayClick(event: React.MouseEvent<SVGSVGElement>) {
    if (regionMode || !addMode || dragId !== null) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width * imageSize.width;
    const y = (event.clientY - rect.top) / rect.height * imageSize.height;
    const id = features.length + 1;
    const uid = createClientId();
    const meta = TYPE_META["线性尺寸"];
    updateCurrentPageFeatures((current) => [...current, {
      uid,
      id,
      anchorX: x,
      anchorY: y,
      bubbleX: Math.min(imageSize.width - 35, x + imageSize.width * 0.1),
      bubbleY: Math.max(35, y - imageSize.height * 0.1),
      type: "线性尺寸",
      nominal: "待确认",
      tolerance: "±0.10",
      method: meta.method,
      instrument: meta.instrument,
      confidence: 100,
      source: "manual",
    }]);
    setSelectedId(uid);
    setAddMode(false);
  }

  function resetDemo() {
    setImageSrc(SAMPLE_IMAGE_SRC);
    setImageSize(SAMPLE_IMAGE_SIZE);
    setFileName(SAMPLE_FILE_NAME);
    setFeatures(SAMPLE_FEATURES);
    setRegions([]);
    setSelectedId("sample-1");
    setSelectedRegionId(null);
    setRegionDraft(null);
    setRegionMode(false);
    setAddMode(false);
    setPageCount(1);
    setPageNumber(1);
    setDocumentPages([]);
    setAnalysisEngine("DrawingBubbleDetection 示例结果");
    setAnalysisWarning("");
    setStage("识别完成");
    setErrorMessage("");
    setProgress(100);
  }

  function exportCsv() {
    const rows = [
      ["序号", "特性类型", "标称值", "公差/要求", "检验方法", "检具", "置信度"],
      ...features.map((feature) => [
        feature.id,
        feature.type,
        feature.nominal,
        feature.tolerance,
        feature.method,
        feature.instrument,
        feature.confidence === null ? "未提供" : `${feature.confidence}%`,
      ]),
    ];
    const csv = `\uFEFF${rows.map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(",")).join("\n")}`;
    downloadBlob(csv, "text/csv;charset=utf-8", `${fileName.replace(/\.[^.]+$/, "")}_检验特性.csv`);
  }

  async function exportBubbleImage() {
    const image = new Image();
    image.src = imageSrc;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = imageSize.width;
    canvas.height = imageSize.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const bubbleRadius = Math.max(16, imageSize.width * 0.016);
    ctx.font = `700 ${Math.round(bubbleRadius * 1.05)}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    features.forEach((feature) => {
      ctx.strokeStyle = "#e4572e";
      ctx.fillStyle = "#fff8f3";
      ctx.lineWidth = Math.max(2, imageSize.width * 0.002);
      ctx.beginPath();
      ctx.moveTo(feature.anchorX, feature.anchorY);
      ctx.lineTo(feature.bubbleX, feature.bubbleY);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(feature.anchorX, feature.anchorY, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#e4572e";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(feature.bubbleX, feature.bubbleY, bubbleRadius, 0, Math.PI * 2);
      ctx.fillStyle = "#fff8f3";
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#c43c16";
      ctx.fillText(String(feature.id), feature.bubbleX, feature.bubbleY + 1);
    });
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, "image/png", `${fileName.replace(/\.[^.]+$/, "")}_气泡图.png`);
    }, "image/png");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Focus size={22} strokeWidth={2.4} /></div>
          <div>
            <div className="brand-name">BUBBLE<span>IQ</span></div>
            <div className="brand-subtitle">工程图纸智能气泡标注</div>
          </div>
        </div>
        <div className="topbar-center">
          <span className="workspace-chip" title={`识别引擎：${analysisEngine}`}><Box size={14} /> 零件检验工作台</span>
          <span className="divider" />
          <span className="file-title" title={fileName}>{fileName}</span>
        </div>
        <div className="topbar-actions">
          <button className="button ghost" onClick={resetDemo}><RotateCcw size={16} /> 重置示例</button>
          <button className="button primary" onClick={exportCsv}><FileSpreadsheet size={16} /> 导出检验表</button>
        </div>
      </header>

      <section className="summary-strip">
        <div className="summary-copy">
          <span className={`status-dot ${errorMessage ? "error" : ""}`} />
          <strong className={errorMessage ? "error-text" : ""}>{processing || errorMessage ? stage : "图纸解析完成"}</strong>
          <span>{processing ? `正在处理 · ${progress}%` : errorMessage || analysisWarning || `已识别 ${features.length} 项检验特性 · ${analysisEngine}`}</span>
        </div>
        <div className="summary-metrics">
          <div><b>{features.length}</b><span>检验特性</span></div>
          <div><b>{scoredFeatures.length ? `${averageConfidence}%` : "—"}</b><span>平均置信度</span></div>
          <div><b>{pageCount}</b><span>图纸页数</span></div>
        </div>
      </section>

      <div
        ref={workspaceRef}
        className={`workspace ${isUploadPanelCollapsed ? "left-collapsed" : ""} ${isTablePanelCollapsed ? "right-collapsed" : ""} ${isTableResizing ? "is-resizing" : ""}`}
        style={tablePanelWidth === null || isTablePanelCollapsed ? undefined : ({ "--right-panel-width": `${tablePanelWidth}px` } as CSSProperties)}
      >
        <aside ref={uploadPanelRef} id="upload-panel" className={`upload-panel side-panel ${isUploadPanelCollapsed ? "is-collapsed" : ""}`}>
          {isUploadPanelCollapsed ? (
            <div className="collapsed-panel-rail">
              <button
                className="panel-collapse-button"
                type="button"
                aria-label="展开工程导入面板"
                aria-controls="upload-panel"
                aria-expanded={false}
                title="展开工程导入"
                onClick={() => setIsUploadPanelCollapsed(false)}
              >
                <ChevronRight size={16} />
              </button>
              <span>工程导入</span>
            </div>
          ) : (
            <>
              <div className="side-panel-heading">
                <div className="panel-heading">
                  <span className="step-index">01</span>
                  <div><h2>导入工程图</h2><p>PDF / PNG / JPG</p></div>
                </div>
                <button
                  className="panel-collapse-button"
                  type="button"
                  aria-label="折叠工程导入面板"
                  aria-controls="upload-panel"
                  aria-expanded={true}
                  title="折叠工程导入"
                  onClick={() => setIsUploadPanelCollapsed(true)}
                >
                  <ChevronLeft size={16} />
                </button>
              </div>
          <input ref={fileInputRef} type="file" accept="application/pdf,image/png,image/jpeg" hidden onChange={handleFileInput} />
          <div
            className={`dropzone ${dragOver ? "is-dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInputRef.current?.click(); }}
          >
            <div className="upload-icon"><UploadCloud size={25} /></div>
            <strong>拖入零件图纸</strong>
            <span>或点击选择文件</span>
            <small>最大 20 MB · 支持扫描件</small>
          </div>

          <div className="region-card">
            <div className="region-card-head">
              <b>识别方式</b>
              <span className="recognition-page-chip">第 {pageNumber} 页</span>
            </div>
            <div className="recognition-scope-actions" aria-label="选择识别方式">
              <button
                type="button"
                className="recognition-scope-button full-page"
                disabled={pageRecognitionDisabled}
                onClick={() => void recognizeCurrentPage()}
                aria-label={`整页识别第 ${pageNumber} 页`}
              >
                <span><Sparkles size={16} /></span>
                <strong>整页识别</strong>
              </button>
              <button
                type="button"
                className={`recognition-scope-button selected-area ${regionMode ? "active" : ""}`}
                disabled={regionSelectionDisabled}
                aria-label={regionMode ? "结束区域框选" : "开始区域识别框选"}
                onClick={() => {
                  if (regionSelectionDisabled) return;
                  setRegionMode((value) => !value);
                  setAddMode(false);
                  setSelectedId(null);
                  setRegionNotice("");
                }}
              >
                <span><Frame size={16} /></span>
                <strong>区域识别</strong>
              </button>
            </div>
            {regionMode && regions.length === 0 ? (
              <p className="region-selection-hint">请在图纸上拖动框选区域</p>
            ) : null}
            {regions.length > 0 ? (
              <div className="region-list">
                {regions.map((region) => (
                  <div
                    key={region.id}
                    className={`region-list-item ${selectedRegionId === region.id ? "selected" : ""}`}
                    onClick={() => { setSelectedRegionId(region.id); setSelectedId(null); }}
                  >
                    <i>{region.label}</i>
                    <div>
                      <b>区域 {region.label}</b>
                      <span className={`region-status ${region.status}`}>
                        {REGION_STATUS_LABELS[region.status]}
                        {region.status === "succeeded" ? ` · ${region.resultCount} 项` : ""}
                      </span>
                    </div>
                    {region.status === "failed" ? (
                      <button
                        type="button"
                        className="region-retry"
                        title={region.error || "重试识别"}
                        onClick={(event) => { event.stopPropagation(); void recognizeRegions([region.id]); }}
                      >重试</button>
                    ) : null}
                    <button
                      type="button"
                      className="region-delete"
                      title={`删除区域 ${region.label}`}
                      aria-label={`删除区域 ${region.label}`}
                      disabled={processing}
                      onClick={(event) => { event.stopPropagation(); requestDeleteRegion(region.id); }}
                    ><Trash2 size={13} /></button>
                  </div>
                ))}
              </div>
            ) : null}
            {regionNotice ? <p className="region-notice" role="status">{regionNotice}</p> : null}
            {regions.length > 0 ? (
              <button
                type="button"
                className="button primary region-recognize-button"
                aria-disabled={regionRecognitionDisabled}
                onClick={() => void recognizeRegions()}
              ><Sparkles size={13} /> 识别 {regions.length} 个区域</button>
            ) : null}
          </div>

          <div className="process-card">
            <div className="process-head"><span>识别流程</span><b>{progress}%</b></div>
            <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
            <div className="process-steps">
              <div className={progress >= 18 ? "done" : ""}><span>{progress >= 18 ? <Check size={12} /> : "1"}</span>图纸渲染</div>
              <div className={progress >= 35 ? "done" : ""}><span>{progress >= 35 ? <Check size={12} /> : "2"}</span>选择识别范围</div>
              <div className={progress >= 90 ? "done" : ""}><span>{progress >= 90 ? <Check size={12} /> : "3"}</span>特征识别</div>
              <div className={progress === 100 ? "done" : ""}><span>{progress === 100 ? <Check size={12} /> : "4"}</span>结果合并</div>
            </div>
          </div>

          <div className="privacy-note">
            <ShieldCheck size={17} />
            <div><b>可控环境处理</b><span>图纸发送到您配置的本机或内网 Python 服务，不接入第三方云端。</span></div>
          </div>

          <div className="legend-card">
            <h3>图层说明</h3>
            <div><i className="legend-bubble">1</i><span>检验特性气泡</span></div>
            <div><i className="legend-anchor" /><span>尺寸锚点</span></div>
            <div><i className="legend-line" /><span>关联引线</span></div>
            <p>拖动气泡可调整位置；点击气泡可定位表格行。</p>
          </div>
            </>
          )}
        </aside>

        <section className="drawing-panel">
          <div className="panel-toolbar">
            <div className="toolbar-group">
              <button className={`tool-button ${!addMode && !regionMode ? "active" : ""}`} onClick={() => { setAddMode(false); setRegionMode(false); }} title="选择"><MousePointer2 size={16} /></button>
              <button className={`tool-button ${regionMode ? "active" : ""}`} onClick={() => { setRegionMode((value) => !value); setAddMode(false); }} title="选择识别区域"><Frame size={16} /></button>
              <button className={`tool-button ${addMode ? "active" : ""}`} onClick={() => { setAddMode((value) => !value); setRegionMode(false); }} title="添加气泡"><CirclePlus size={16} /></button>
              <span className="toolbar-separator" />
              <button className="tool-button" onClick={() => setZoom((value) => Math.max(MIN_VIEW_ZOOM, value - 0.1))} title="缩小"><ZoomOut size={16} /></button>
              <span className="zoom-value">{Math.round(zoom * 100)}%</span>
              <button className="tool-button" onClick={() => setZoom((value) => Math.min(MAX_VIEW_ZOOM, value + 0.1))} title="放大"><ZoomIn size={16} /></button>
              <button className="tool-button" onClick={() => setZoom(1)} title="适合窗口"><Maximize2 size={16} /></button>
            </div>
            <div className="toolbar-title">
              <span className="live-indicator" /> 气泡图预览
              {addMode && <em>点击图纸添加检验点</em>}
              {regionMode && <em>拖动框选多个区域</em>}
            </div>
            <div className="drawing-toolbar-actions">
              <button
                type="button"
                className="button outline danger drawing-action-button"
                disabled={processing || (!features.length && !regions.length)}
                onClick={requestClearAllBubbles}
                aria-label="清空气泡和识别区域"
                title="清空气泡和识别区域"
              ><Trash2 size={15} /> 清空</button>
              <button
                type="button"
                className="button outline drawing-action-button"
                onClick={exportBubbleImage}
                aria-label="下载气泡图"
                title="下载气泡图"
              ><Download size={15} /> 下载</button>
            </div>
          </div>

          {(regionDraft?.invalid || regionNotice.includes("重叠")) ? (
            <div className="drawing-overlap-alert" role="alert" aria-live="assertive">
              <AlertCircle size={17} />
              <span>{regionDraft?.invalid ? "当前框选与已有区域重叠，请调整范围。" : regionNotice}</span>
            </div>
          ) : null}

          <div
            ref={drawingViewportRef}
            className={`drawing-viewport ${isViewPanning ? "is-view-panning" : ""}`}
            onWheel={handleViewportWheel}
            onPointerDownCapture={handleViewportPointerDownCapture}
            onPointerMoveCapture={handleViewportPointerMoveCapture}
            onPointerUpCapture={(event) => finishViewportPan(event, true)}
            onPointerCancelCapture={(event) => finishViewportPan(event, false)}
            onClickCapture={handleViewportClickCapture}
            title="Shift + 鼠标滚轮缩放；Shift + 鼠标左键拖动平移"
          >
            {processing && (
              <div className="processing-overlay">
                <LoaderCircle className="spinner" size={28} />
                <strong>{stage}</strong>
                <span>正在分析图纸几何与尺寸标注</span>
              </div>
            )}
            {!processing && errorMessage && !imageSrc && (
              <div className="drawing-error" role="alert">
                <div><AlertCircle size={24} /></div>
                <strong>图纸未能显示</strong>
                <span>{errorMessage}</span>
                <button className="button outline" onClick={() => fileInputRef.current?.click()}><UploadCloud size={15} /> 重新选择文件</button>
              </div>
            )}
            <div ref={drawingStageRef} className="drawing-stage" style={{ width: `${zoom * 100}%` }}>
              {imageSrc && (
                // The source can be a PDF canvas data URL or user-selected blob;
                // a native image preserves the exact coordinate system used by the SVG overlay.
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={imageSrc}
                  alt="工业零件工程图"
                  draggable={false}
                  onError={() => setImageSrc(createSampleDrawing())}
                />
              )}
              <svg
                className={`annotation-layer ${addMode ? "add-mode" : ""} ${regionMode ? "region-mode" : ""}`}
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                onPointerDown={handleRegionPointerDown}
                onPointerMove={handleOverlayPointerMove}
                onPointerUp={(event) => {
                  if (regionDraft) finishRegionSelection(event);
                  setDragId(null);
                }}
                onPointerCancel={cancelRegionSelection}
                onPointerLeave={() => { if (!regionDraft) setDragId(null); }}
                onClick={handleOverlayClick}
                onKeyDown={handleOverlayKeyDown}
                tabIndex={0}
                aria-label="气泡标注图层"
              >
                {regions.map((region) => (
                  <g
                    key={region.id}
                    className={`recognition-region ${region.status} ${selectedRegionId === region.id ? "selected" : ""}`}
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      if (regionMode) {
                        setRegionNotice("框选起点位于已有区域内，识别区域不能重叠。");
                        event.currentTarget.ownerSVGElement?.focus();
                        return;
                      }
                      setSelectedRegionId(region.id);
                      setSelectedId(null);
                      event.currentTarget.ownerSVGElement?.focus();
                    }}
                  >
                    <rect x={region.x} y={region.y} width={region.width} height={region.height} />
                    <g className="region-label" transform={`translate(${region.x + 6} ${region.y + 6})`}>
                      <rect width="145" height="32" rx="4" />
                      <text x="10" y="17">区域 {region.label} · {REGION_STATUS_LABELS[region.status]}</text>
                    </g>
                  </g>
                ))}
                {regionDraft ? (
                  <rect
                    className={`region-draft ${regionDraft.invalid ? "invalid" : ""}`}
                    x={regionDraft.x}
                    y={regionDraft.y}
                    width={regionDraft.width}
                    height={regionDraft.height}
                  />
                ) : null}
                {features.map((feature) => {
                  const selected = feature.uid === selectedId;
                  const radius = Math.max(17, imageSize.width * 0.016);
                  return (
                    <g
                      key={feature.uid}
                      className={`annotation ${selected ? "selected" : ""}`}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        if (regionMode) return;
                        setDragId(feature.uid);
                        setSelectedId(feature.uid);
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                    >
                      <line x1={feature.anchorX} y1={feature.anchorY} x2={feature.bubbleX} y2={feature.bubbleY} />
                      <circle className="anchor" cx={feature.anchorX} cy={feature.anchorY} r={Math.max(3.5, imageSize.width * 0.003)} />
                      <circle className="bubble" cx={feature.bubbleX} cy={feature.bubbleY} r={radius} />
                      <text x={feature.bubbleX} y={feature.bubbleY + radius * 0.08} fontSize={radius * 1.05}>{feature.id}</text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          <div className="page-footer">
            <span><FileImage size={14} /> {imageSize.width} × {imageSize.height} px</span>
            <div className="pager">
              <button
                disabled={!documentPages.length || pageNumber <= 1 || processing}
                onClick={() => {
                  const previousPage = documentPages[pageNumber - 2];
                  if (previousPage) applyDocumentPage(previousPage);
                }}
              ><ChevronLeft size={15} /></button>
              <span>第 <b>{pageNumber}</b> / {pageCount} 页</span>
              <button
                disabled={!documentPages.length || pageNumber >= pageCount || processing}
                onClick={() => {
                  const nextPage = documentPages[pageNumber];
                  if (nextPage) applyDocumentPage(nextPage);
                }}
              ><ChevronRight size={15} /></button>
            </div>
            <span className="coordinate-note">单位：mm · 坐标已归一化</span>
          </div>
        </section>

        <div
          className={`panel-resizer ${isTablePanelCollapsed ? "is-disabled" : ""}`}
          role="separator"
          aria-label="调整工作区与检验特性表宽度"
          aria-orientation="vertical"
          aria-disabled={isTablePanelCollapsed}
          aria-valuemin={MIN_TABLE_PANEL_WIDTH}
          aria-valuenow={Math.round(tablePanelWidth ?? DEFAULT_TABLE_PANEL_WIDTH)}
          tabIndex={isTablePanelCollapsed ? -1 : 0}
          onPointerDown={handleTableResizeStart}
          onKeyDown={handleTableResizeKeyDown}
        />

        <section ref={tablePanelRef} id="table-panel" className={`table-panel side-panel ${isTablePanelCollapsed ? "is-collapsed" : ""}`}>
          {isTablePanelCollapsed ? (
            <div className="collapsed-panel-rail">
              <button
                className="panel-collapse-button"
                type="button"
                aria-label="展开检验特性表"
                aria-controls="table-panel"
                aria-expanded={false}
                title="展开检验特性表"
                onClick={() => setIsTablePanelCollapsed(false)}
              >
                <ChevronLeft size={16} />
              </button>
              <span>检验特性表</span>
              <b>{features.length}</b>
            </div>
          ) : (
            <>
              <div className="table-heading">
                <div className="panel-heading compact">
                  <span className="step-index">02</span>
                  <div><h2>检验特性表</h2><p>{features.length} 项特性</p></div>
                </div>
                <div className="table-heading-actions">
                  <button
                    className="panel-collapse-button"
                    type="button"
                    aria-label="折叠检验特性表"
                    aria-controls="table-panel"
                    aria-expanded={true}
                    title="折叠检验特性表"
                    onClick={() => setIsTablePanelCollapsed(true)}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>
          <div className="table-filters">
            <label className="search-field"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索特性..." /></label>
            <select aria-label="特性类型筛选" value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
              <option>全部类型</option>
              {Object.keys(TYPE_META).map((type) => <option key={type}>{type}</option>)}
            </select>
          </div>
          <div className="table-scroll">
            <table>
              <thead><tr><th>#</th><th>特性</th><th>标称值</th><th>公差 / 要求</th><th>检具</th><th>操作</th></tr></thead>
              <tbody>
                {filteredFeatures.map((feature) => (
                  <tr key={feature.uid} className={feature.uid === selectedId ? "selected" : ""} onClick={() => { setSelectedId(feature.uid); setSelectedRegionId(feature.regionId ?? null); }}>
                    <td><span className="row-number">{feature.id}</span></td>
                    <td>
                      <select
                        aria-label={`特性 ${feature.id} 类型`}
                        value={feature.type}
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) => {
                          const type = event.target.value as FeatureType;
                          updateFeature(feature.uid, "type", type);
                          updateFeature(feature.uid, "method", TYPE_META[type].method);
                          updateFeature(feature.uid, "instrument", TYPE_META[type].instrument);
                        }}
                      >
                        {Object.keys(TYPE_META).map((type) => <option key={type}>{type}</option>)}
                      </select>
                      <small>{feature.method}</small>
                    </td>
                    <td><input aria-label={`特性 ${feature.id} 标称值`} value={feature.nominal} onClick={(event) => event.stopPropagation()} onChange={(event) => updateFeature(feature.uid, "nominal", event.target.value)} /></td>
                    <td><input aria-label={`特性 ${feature.id} 公差`} value={feature.tolerance} onClick={(event) => event.stopPropagation()} onChange={(event) => updateFeature(feature.uid, "tolerance", event.target.value)} /></td>
                    <td><input aria-label={`特性 ${feature.id} 检具`} value={feature.instrument} onClick={(event) => event.stopPropagation()} onChange={(event) => updateFeature(feature.uid, "instrument", event.target.value)} /><span className={`confidence ${feature.confidence !== null && feature.confidence < 90 ? "warn" : ""}`}>{feature.confidence === null ? "—" : `${feature.confidence}%`}</span></td>
                    <td>
                      <button
                        className="remove-row"
                        title={`删除特性 ${feature.id}`}
                        aria-label={`删除特性 ${feature.id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          removeFeature(feature.uid);
                        }}
                      >
                        <Trash2 size={13} />
                        <span>删除</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filteredFeatures.length === 0 && <div className="empty-table"><AlertCircle size={22} /><span>没有匹配的检验特性</span></div>}
          </div>
          <div className="table-footer">
            <button className="add-row" onClick={() => { setAddMode(true); setRegionMode(false); }}><Plus size={15} /> 在图纸中添加特性</button>
            <div><Sparkles size={14} /><span>自动识别结果需由工程师复核</span></div>
          </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
