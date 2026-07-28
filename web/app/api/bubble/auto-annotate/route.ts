const DEFAULT_BUBBLE_API_URL = "http://127.0.0.1:8001";
const MAX_FILE_BYTES = 20 * 1024 * 1024;

function apiError(message: string, status: number) {
  return Response.json({ detail: message }, { status });
}

export async function POST(request: Request) {
  const apiKey = process.env.BUBBLE_API_KEY?.trim();
  if (!apiKey) {
    return apiError("Web 服务未配置 BUBBLE_API_KEY。", 503);
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return apiError("无法读取上传内容。", 400);
  }

  const file = form.get("file");
  if (!(file instanceof File)) {
    return apiError("请选择需要标注的工程图。", 400);
  }
  if (file.size === 0) {
    return apiError("上传文件为空。", 400);
  }
  if (file.size > MAX_FILE_BYTES) {
    return apiError("文件不能超过 20 MB。", 413);
  }

  const lowerName = file.name.toLowerCase();
  const supported =
    ["image/png", "image/jpeg", "image/tiff", "image/bmp"].includes(file.type)
    || /\.(png|jpe?g|tiff?|bmp)$/i.test(lowerName);
  if (!supported) {
    return apiError("气泡标注服务仅支持 PNG、JPG、TIFF 和 BMP 图片。", 415);
  }

  const upstreamForm = new FormData();
  upstreamForm.append("file", file, file.name);
  const apiUrl = (process.env.BUBBLE_API_URL || DEFAULT_BUBBLE_API_URL).replace(/\/$/, "");

  try {
    const upstream = await fetch(
      `${apiUrl}/api/auto-annotate?balloon_radius=25&preserve_existing=false`,
      {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: upstreamForm,
      },
    );
    const payload = await upstream.json().catch(() => null) as Record<string, unknown> | null;
    if (!upstream.ok) {
      const detail =
        typeof payload?.detail === "string"
          ? payload.detail
          : `气泡标注服务返回 HTTP ${upstream.status}`;
      return apiError(detail, upstream.status);
    }
    if (!payload || !Array.isArray(payload.balloons)) {
      return apiError("气泡标注服务返回了无法识别的数据。", 502);
    }

    // The editable Web overlay only needs geometry and dimension text.
    // Avoid forwarding the large duplicate base64 rendering to the browser.
    delete payload.annotated_image_base64;
    return Response.json(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "连接失败";
    return apiError(`无法连接气泡标注服务：${message}`, 502);
  }
}
