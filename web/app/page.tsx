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
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { ChangeEvent, DragEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";

type FeatureType = "直径" | "线性尺寸" | "半径" | "角度" | "形位公差" | "螺纹" | "表面粗糙度";

type Feature = {
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
};

type DocumentPage = {
  pageNumber: number;
  width: number;
  height: number;
  imageSrc: string;
  features: Feature[];
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
  { id: 1, anchorX: 905, anchorY: 90, bubbleX: 967, bubbleY: 90, type: "线性尺寸", nominal: "23", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "23", source: "bubble-api" },
  { id: 2, anchorX: 820, anchorY: 52, bubbleX: 727, bubbleY: 145, type: "线性尺寸", nominal: "29.5", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "29.5", source: "bubble-api" },
  { id: 3, anchorX: 964, anchorY: 154, bubbleX: 1026, bubbleY: 154, type: "线性尺寸", nominal: "14", tolerance: "±0.2", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "14±0.2", source: "bubble-api" },
  { id: 4, anchorX: 261, anchorY: 307, bubbleX: 210, bubbleY: 307, type: "线性尺寸", nominal: "1.94", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "1.94", source: "bubble-api" },
  { id: 5, anchorX: 155, anchorY: 493, bubbleX: 111, bubbleY: 452, type: "线性尺寸", nominal: "15.8", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "15.8", source: "bubble-api" },
  { id: 6, anchorX: 1483, anchorY: 605, bubbleX: 1545, bubbleY: 605, type: "线性尺寸", nominal: "2-06", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "2-06", source: "bubble-api" },
  { id: 7, anchorX: 109, anchorY: 611, bubbleX: 109, bubbleY: 636, type: "线性尺寸", nominal: "21.5", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "21.5", source: "bubble-api" },
  { id: 8, anchorX: 1366, anchorY: 767, bubbleX: 1366, bubbleY: 759, type: "线性尺寸", nominal: "2-03.5", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "2-03.5", source: "bubble-api" },
  { id: 9, anchorX: 299, anchorY: 812, bubbleX: 250, bubbleY: 812, type: "线性尺寸", nominal: "2.06", tolerance: "", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "2.06", source: "bubble-api" },
  { id: 10, anchorX: 907, anchorY: 899, bubbleX: 907, bubbleY: 852, type: "线性尺寸", nominal: "50.3", tolerance: "±0.3", method: "直接测量", instrument: "数显卡尺", confidence: null, rawText: "50.3±0.3", source: "bubble-api" },
];

const SAMPLE_IMAGE_SRC = "/demo-engineering-drawing.png";
const SAMPLE_FILE_NAME = "ScreenShot_2026-07-23_161820_677.png";
const SAMPLE_IMAGE_SIZE = { width: 1658, height: 990 };

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
  const [imageSrc, setImageSrc] = useState("");
  const [fileName, setFileName] = useState(SAMPLE_FILE_NAME);
  const [imageSize, setImageSize] = useState(SAMPLE_IMAGE_SIZE);
  const [features, setFeatures] = useState<Feature[]>(SAMPLE_FEATURES);
  const [selectedId, setSelectedId] = useState<number | null>(1);
  const [dragId, setDragId] = useState<number | null>(null);
  const [addMode, setAddMode] = useState(false);
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

  useEffect(() => {
    const timer = window.setTimeout(() => setImageSrc(SAMPLE_IMAGE_SRC), 0);
    return () => window.clearTimeout(timer);
  }, []);

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
    setSelectedId(page.features[0]?.id ?? null);
    setPageNumber(page.pageNumber);
  }

  function featuresFromBubbleApi(result: BubbleApiResponse): Feature[] {
    return result.balloons.map((item) => {
      const text = cleanToken(item.dimension_text);
      const type = classifyToken(text);
      const tolerance = inferredTolerance(text, type);
      const nominal = tolerance ? text.replace(tolerance, "").trim() : text;
      const [x0, y0, x1, y1] = item.dim_bbox;
      const leaderEnd = item.leader_points.at(-1);
      const meta = TYPE_META[type];
      return {
        id: item.number,
        anchorX: leaderEnd?.[0] ?? (x0 + x1) / 2,
        anchorY: leaderEnd?.[1] ?? (y0 + y1) / 2,
        bubbleX: item.balloon.cx,
        bubbleY: item.balloon.cy,
        type,
        nominal,
        tolerance,
        method: meta.method,
        instrument: meta.instrument,
        confidence: null,
        rawText: text,
        source: "bubble-api",
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
    setSelectedId(null);
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
        setStage("识别尺寸并生成气泡");
        setProgress(35);
        const bubbleResult = await autoAnnotateWithBubbleService(file);
        const detected = featuresFromBubbleApi(bubbleResult);
        setPageCount(1);
        setPageNumber(1);
        setImageSrc(src);
        setImageSize(size);
        setFeatures(detected);
        setSelectedId(detected[0]?.id ?? null);
        setAnalysisEngine("DrawingBubbleDetection Auto-Annotate");
        setAnalysisWarning(
          detected.length
            ? "自动标注服务未提供逐项置信度；所有识别结果仍需由工程师复核。"
            : "DrawingBubbleDetection 已完成分析，但没有识别到可自动标注的尺寸。",
        );
        setStage("气泡标注完成");
        setProgress(100);
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
      setAnalysisEngine("DrawingBubbleDetection PDF Auto-Annotate");

      for (let targetPage = 1; targetPage <= pagesToProcess; targetPage += 1) {
        setStage(`渲染 PDF 第 ${targetPage} / ${pagesToProcess} 页`);
        setProgress(8 + Math.round(((targetPage - 1) / pagesToProcess) * 84));
        const rendered = await renderPdfPageAsImage(pdfDoc, targetPage, file.name);
        setStage(`识别 PDF 第 ${targetPage} / ${pagesToProcess} 页`);
        setProgress(12 + Math.round(((targetPage - 0.5) / pagesToProcess) * 84));
        const bubbleResult = await autoAnnotateWithBubbleService(rendered.file);
        detectedPages.push({
          pageNumber: targetPage,
          width: rendered.width,
          height: rendered.height,
          imageSrc: rendered.imageSrc,
          features: featuresFromBubbleApi(bubbleResult),
        });
      }

      setDocumentPages(detectedPages);
      applyDocumentPage(detectedPages[0]);
      const detectedCount = detectedPages.reduce((sum, page) => sum + page.features.length, 0);
      const truncatedMessage = pdfDoc.numPages > pagesToProcess
        ? ` 为控制处理时间，本次只检测前 ${pagesToProcess} 页。`
        : "";
      setAnalysisWarning(
        `${pagesToProcess} 页共识别 ${detectedCount} 项；自动标注服务未提供逐项置信度，结果仍需由工程师复核。${truncatedMessage}`,
      );
      setStage("PDF 气泡标注完成");
      setProgress(100);
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

  function updateFeature(id: number, key: keyof Feature, value: string | number) {
    updateCurrentPageFeatures((current) => current.map((feature) => (
      feature.id === id ? { ...feature, [key]: value } : feature
    )));
  }

  function removeFeature(id: number) {
    updateCurrentPageFeatures((current) => current
      .filter((feature) => feature.id !== id)
      .map((feature, index) => ({ ...feature, id: index + 1 })));
    setSelectedId(null);
  }

  function pointerCoordinates(event: PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width * imageSize.width,
      y: (event.clientY - rect.top) / rect.height * imageSize.height,
    };
  }

  function handleOverlayPointerMove(event: PointerEvent<SVGSVGElement>) {
    if (dragId === null) return;
    const point = pointerCoordinates(event);
    updateCurrentPageFeatures((current) => current.map((feature) => (
      feature.id === dragId ? { ...feature, bubbleX: point.x, bubbleY: point.y } : feature
    )));
  }

  function handleOverlayClick(event: React.MouseEvent<SVGSVGElement>) {
    if (!addMode || dragId !== null) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width * imageSize.width;
    const y = (event.clientY - rect.top) / rect.height * imageSize.height;
    const id = features.length + 1;
    const meta = TYPE_META["线性尺寸"];
    updateCurrentPageFeatures((current) => [...current, {
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
    setSelectedId(id);
    setAddMode(false);
  }

  function resetDemo() {
    setImageSrc(SAMPLE_IMAGE_SRC);
    setImageSize(SAMPLE_IMAGE_SIZE);
    setFileName(SAMPLE_FILE_NAME);
    setFeatures(SAMPLE_FEATURES);
    setSelectedId(1);
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

      <div className="workspace">
        <aside className="upload-panel">
          <div className="panel-heading">
            <span className="step-index">01</span>
            <div><h2>导入工程图</h2><p>PDF / PNG / JPG</p></div>
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

          <div className="process-card">
            <div className="process-head"><span>识别流程</span><b>{progress}%</b></div>
            <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
            <div className="process-steps">
              <div className={progress >= 18 ? "done" : ""}><span>{progress >= 18 ? <Check size={12} /> : "1"}</span>图纸渲染</div>
              <div className={progress >= 48 ? "done" : ""}><span>{progress >= 48 ? <Check size={12} /> : "2"}</span>尺寸识别</div>
              <div className={progress >= 85 ? "done" : ""}><span>{progress >= 85 ? <Check size={12} /> : "3"}</span>特性分类</div>
              <div className={progress === 100 ? "done" : ""}><span>{progress === 100 ? <Check size={12} /> : "4"}</span>气泡生成</div>
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
        </aside>

        <section className="drawing-panel">
          <div className="panel-toolbar">
            <div className="toolbar-group">
              <button className={`tool-button ${!addMode ? "active" : ""}`} onClick={() => setAddMode(false)} title="选择"><MousePointer2 size={16} /></button>
              <button className={`tool-button ${addMode ? "active" : ""}`} onClick={() => setAddMode((value) => !value)} title="添加气泡"><CirclePlus size={16} /></button>
              <span className="toolbar-separator" />
              <button className="tool-button" onClick={() => setZoom((value) => Math.max(0.6, value - 0.1))} title="缩小"><ZoomOut size={16} /></button>
              <span className="zoom-value">{Math.round(zoom * 100)}%</span>
              <button className="tool-button" onClick={() => setZoom((value) => Math.min(2, value + 0.1))} title="放大"><ZoomIn size={16} /></button>
              <button className="tool-button" onClick={() => setZoom(1)} title="适合窗口"><Maximize2 size={16} /></button>
            </div>
            <div className="toolbar-title">
              <span className="live-indicator" /> 气泡图预览
              {addMode && <em>点击图纸添加检验点</em>}
            </div>
            <button className="button outline" onClick={exportBubbleImage}><Download size={15} /> 下载气泡图</button>
          </div>

          <div className="drawing-viewport">
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
            <div className="drawing-stage" style={{ width: `${zoom * 100}%` }}>
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
                className={`annotation-layer ${addMode ? "add-mode" : ""}`}
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                onPointerMove={handleOverlayPointerMove}
                onPointerUp={() => setDragId(null)}
                onPointerLeave={() => setDragId(null)}
                onClick={handleOverlayClick}
                aria-label="气泡标注图层"
              >
                {features.map((feature) => {
                  const selected = feature.id === selectedId;
                  const radius = Math.max(17, imageSize.width * 0.016);
                  return (
                    <g
                      key={feature.id}
                      className={`annotation ${selected ? "selected" : ""}`}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        setDragId(feature.id);
                        setSelectedId(feature.id);
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

        <section className="table-panel">
          <div className="table-heading">
            <div className="panel-heading compact">
              <span className="step-index">02</span>
              <div><h2>检验特性表</h2><p>{features.length} 项特性</p></div>
            </div>
            <button className="icon-close" title="清空筛选" onClick={() => { setQuery(""); setTypeFilter("全部类型"); }}><X size={16} /></button>
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
                  <tr key={feature.id} className={feature.id === selectedId ? "selected" : ""} onClick={() => setSelectedId(feature.id)}>
                    <td><span className="row-number">{feature.id}</span></td>
                    <td>
                      <select
                        aria-label={`特性 ${feature.id} 类型`}
                        value={feature.type}
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) => {
                          const type = event.target.value as FeatureType;
                          updateFeature(feature.id, "type", type);
                          updateFeature(feature.id, "method", TYPE_META[type].method);
                          updateFeature(feature.id, "instrument", TYPE_META[type].instrument);
                        }}
                      >
                        {Object.keys(TYPE_META).map((type) => <option key={type}>{type}</option>)}
                      </select>
                      <small>{feature.method}</small>
                    </td>
                    <td><input aria-label={`特性 ${feature.id} 标称值`} value={feature.nominal} onClick={(event) => event.stopPropagation()} onChange={(event) => updateFeature(feature.id, "nominal", event.target.value)} /></td>
                    <td><input aria-label={`特性 ${feature.id} 公差`} value={feature.tolerance} onClick={(event) => event.stopPropagation()} onChange={(event) => updateFeature(feature.id, "tolerance", event.target.value)} /></td>
                    <td><input aria-label={`特性 ${feature.id} 检具`} value={feature.instrument} onClick={(event) => event.stopPropagation()} onChange={(event) => updateFeature(feature.id, "instrument", event.target.value)} /><span className={`confidence ${feature.confidence !== null && feature.confidence < 90 ? "warn" : ""}`}>{feature.confidence === null ? "—" : `${feature.confidence}%`}</span></td>
                    <td>
                      <button
                        className="remove-row"
                        title={`删除特性 ${feature.id}`}
                        aria-label={`删除特性 ${feature.id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          removeFeature(feature.id);
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
            <button className="add-row" onClick={() => setAddMode(true)}><Plus size={15} /> 在图纸中添加特性</button>
            <div><Sparkles size={14} /><span>自动识别结果需由工程师复核</span></div>
          </div>
        </section>
      </div>
    </main>
  );
}
