# BubbleIQ 工业零件图气泡标注 Demo

BubbleIQ 是一个可本地运行的工程图识别框架：上传 PDF、PNG 或 JPG 后，系统显示原图、生成检验特性气泡，并给出可编辑和导出的检验表。

PNG/JPG 图片通过服务端代理调用相邻
`DrawingBubbleDetection/DrawingBubbleService_M1` 工程的气泡标注 API。PDF
会在浏览器中逐页渲染为高清 PNG，再调用同一个 API。API Key 只保存在服务端的
`.env.local` 中，不会发送到浏览器。

## 识别架构

```text
Web 工作台 (Next.js)
  ├─ PNG/JPG：直接上传
  └─ PDF：PDF.js 逐页渲染为 PNG（最多 10 页）
              │ 同源服务端代理 /api/bubble/auto-annotate
              ▼
DrawingBubbleDetection Python API (FastAPI)
  ├─ OCR 与尺寸文字识别
  ├─ 气泡位置和引线生成
  └─ 返回每页检验特性
              │
              ▼
Web 可编辑覆盖层 + 检验特性表 + CSV/气泡图导出
```

识别不到时返回空结果和告警，不会生成虚假的置信度。PDF 的每一页分别保存识别结果，
翻页后仍会保留用户对表格和气泡位置的修改。删除表格记录时，对应气泡和导出结果也会同步删除。

## 本地运行（推荐）

前端要求 Node.js `>=22.13.0`：

先在第一个终端启动 DrawingBubbleDetection API：

```bash
cd "../DrawingBubbleService_M1"
source .venv/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

确认 [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health) 可以访问。

然后在 `web/.env.local` 中配置 API 地址和
`DrawingBubbleService_M1/.env` 里的 Key：

```dotenv
BUBBLE_API_URL=http://127.0.0.1:8001
BUBBLE_API_KEY=请填写-DrawingBubbleService_M1-.env-中的-API-Key
```

最后在第二个终端启动前端：

```bash
cd "../web"
test -f .env.local || cp .env.example .env.local
npm install
npm run dev
```

如果已经创建过 `.env.local`，不要再次执行 `cp` 覆盖它。打开
[http://localhost:3000](http://localhost:3000)，上传 PDF、PNG 或 JPG 即可调用气泡标注 API。

## 配置

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `BUBBLE_API_URL` | `http://127.0.0.1:8001` | DrawingBubbleDetection API 地址，仅服务端读取 |
| `BUBBLE_API_KEY` | 无 | DrawingBubbleDetection API Key，仅服务端读取 |

Web 端单个文件限制为 20 MB，PDF 单次最多处理前 10 页。

## API

前端调用本工程的同源代理，避免把 API Key 暴露给浏览器：

```bash
curl -X POST http://localhost:3000/api/bubble/auto-annotate \
  -F "file=@/absolute/path/to/drawing.png"
```

## 验证

```bash
npm run lint
npm test
```

生产化前应使用来自目标工厂、图纸标准和扫描设备的数据集，分别统计字符准确率、字段准确率、气泡锚点命中率以及整图完全正确率，并对低置信度结果保留人工复核。
