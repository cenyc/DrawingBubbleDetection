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
  assert.doesNotMatch(page, /codex-preview|Your site is taking shape/);
  await access(new URL("../.next/BUILD_ID", import.meta.url));
  await access(new URL("../.next/standalone/server.js", import.meta.url));
});

test("includes PDF rendering and editable annotation capabilities", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /pdfjs-dist/);
  assert.match(page, /pdf\.worker\.min\.mjs/);
  assert.match(page, /renderPdfPageAsImage\(pdfDoc/);
  assert.match(page, /page\.render\(\{ canvas, viewport \}\)/);
  assert.match(page, /autoAnnotateWithBubbleService\(rendered\.file\)/);
  assert.match(page, /setDocumentPages\(detectedPages\)/);
  assert.match(page, /exportBubbleImage/);
  assert.match(page, /exportCsv/);
  assert.match(page, /handleOverlayPointerMove/);
  assert.match(page, /removeFeature\(feature\.id\)/);
  assert.match(page, /删除特性/);
  assert.match(page, /application\/pdf,image\/png/);
  assert.match(page, /\/demo-engineering-drawing\.png/);
  assert.match(page, /ScreenShot_2026-07-23_161820_677\.png/);
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
