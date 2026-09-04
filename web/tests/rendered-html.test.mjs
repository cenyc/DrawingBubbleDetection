import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

test("builds the standalone BubbleIQ application", async () => {
  const [page, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(layout, /BubbleIQ｜工业零件气泡图生成/);
  assert.match(page, /BUBBLE/);
  assert.match(page, /导入工程图/);
  assert.match(page, /气泡图预览/);
  assert.match(page, /检验特性表/);
  assert.match(page, /导出检验表/);
  assert.match(page, /折叠工程导入面板/);
  assert.match(page, /折叠检验特性表/);
  assert.match(page, /left-collapsed/);
  assert.match(page, /right-collapsed/);
  assert.match(page, /isTablePanelCollapsed, setIsTablePanelCollapsed\] = useState\(true\)/);
  assert.match(page, /bubbleiq:workspace-panels:v1/);
  assert.match(page, /window\.localStorage\.getItem\(PANEL_STATE_STORAGE_KEY\)/);
  assert.match(page, /window\.localStorage\.setItem\(PANEL_STATE_STORAGE_KEY/);
  assert.match(page, /tableWidth: tablePanelWidth/);
  assert.match(page, /role="separator"/);
  assert.match(page, /调整工作区与检验特性表宽度/);
  assert.doesNotMatch(page, /清空筛选/);
  assert.doesNotMatch(page, /codex-preview|Your site is taking shape/);
  await access(new URL("../.next/BUILD_ID", import.meta.url));
  await access(new URL("../.next/standalone/server.js", import.meta.url));
});

test("includes PDF rendering and editable annotation capabilities", async () => {
  const [page, layout, packageJson, styles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(page, /pdfjs-dist/);
  assert.match(page, /pdf\.worker\.min\.mjs/);
  assert.match(page, /renderPdfPageAsImage\(pdfDoc/);
  assert.match(page, /page\.render\(\{ canvas, viewport \}\)/);
  assert.match(page, /autoAnnotateWithBubbleService\(croppedFile\)/);
  assert.match(page, /setDocumentPages\(detectedPages\)/);
  assert.match(page, /exportBubbleImage/);
  assert.match(page, /exportCsv/);
  assert.match(page, /handleOverlayPointerMove/);
  assert.match(page, /removeFeature\(feature\.uid\)/);
  assert.match(page, /删除特性/);
  assert.match(page, /选择识别区域/);
  assert.match(page, /整页识别/);
  assert.match(page, /区域识别/);
  assert.match(page, /aria-label="选择识别方式"/);
  assert.doesNotMatch(page, /可一键识别整页，也可框选局部区域进行补充或精细识别/);
  assert.doesNotMatch(page, /在预览图中拖动鼠标，可连续选择多个互不重叠的区域/);
  assert.match(page, /recognizeCurrentPage/);
  assert.match(page, /createRecognitionFile/);
  assert.match(page, /feature\.source === "manual"/);
  assert.match(page, /区域结果会替换对应范围内的自动结果/);
  assert.match(page, /识别 \{regions\.length\} 个区域/);
  assert.match(page, /regionsOverlap/);
  assert.match(page, /识别区域不能与已有区域重叠/);
  assert.match(page, /cropRegion/);
  assert.match(page, /regionId/);
  assert.match(page, /删除区域.*将同时删除它生成的/);
  assert.match(page, /featuresFromBubbleApi\(result, region\)/);
  assert.match(page, /清空气泡和识别区域/);
  assert.match(page, /aria-label="清空气泡和识别区域"/);
  assert.match(page, /aria-label="下载气泡图"/);
  assert.match(page, /requestClearAllBubbles/);
  assert.match(page, /updateCurrentPageRegions\(\(\) => \[\]\)/);
  assert.match(page, /setRegionDraft\(null\)/);
  assert.match(page, /setRegionMode\(false\)/);
  assert.match(page, /featureIsInsideRegion/);
  assert.match(page, /feature\.source === "manual" \|\| !featureIsInsideRegion\(feature, region\)/);
  assert.match(page, /所选区域与已有区域重叠/);
  assert.match(page, /drawing-overlap-alert/);
  assert.match(styles, /\.drawing-toolbar-actions \{[^}]*justify-self: end;[^}]*gap: 10px;/);
  assert.match(styles, /\.drawing-overlap-alert \{[^}]*position: absolute;[^}]*pointer-events: none;/);
  assert.doesNotMatch(styles, /\.drawing-overlap-alert \{[^}]*position: sticky;/);
  assert.match(page, /application\/pdf,image\/png/);
  assert.match(page, /\/demo-engineering-drawing\.png/);
  assert.match(page, /ScreenShot_2026-07-23_161820_677\.png/);
  assert.match(page, /useState\(SAMPLE_IMAGE_SRC\)/);
  assert.doesNotMatch(page, /setTimeout\(\(\) => setImageSrc\(SAMPLE_IMAGE_SRC\)/);
  assert.match(page, /disabled=\{regionSelectionDisabled\}/);
  assert.match(page, /aria-disabled=\{regionRecognitionDisabled\}/);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(packageJson, /"pdfjs-dist"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await access(new URL("../public/pdf.worker.min.mjs", import.meta.url));
  await access(new URL("../public/demo-engineering-drawing.png", import.meta.url));
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});

test("proxies image annotation to DrawingBubbleDetection without exposing its API key", async () => {
  const [page, route, envExample] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/bubble/auto-annotate/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../.env.example", import.meta.url), "utf8"),
  ]);

  assert.match(page, /\/api\/bubble\/auto-annotate/);
  assert.match(page, /featuresFromBubbleApi/);
  assert.match(route, /\/api\/auto-annotate/);
  assert.match(route, /X-API-Key/);
  assert.match(route, /delete payload\.annotated_image_base64/);
  assert.match(envExample, /^BUBBLE_API_URL=/m);
  assert.match(envExample, /^BUBBLE_API_KEY=/m);
  assert.doesNotMatch(envExample, /NEXT_PUBLIC_BUBBLE/);
});
