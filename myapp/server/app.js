const path = require("path");
const fs = require("fs");
const express = require("express");
const fileUpload = require("express-fileupload");
const cors = require("cors");
const { PythonShell } = require("python-shell");

const app = express();
const port = process.env.PORT || 3000;
const serverDir = __dirname;
const publicDir = path.join(serverDir, "..");
const uploadDir = path.join(serverDir, "fotos");
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;
const ALLOWED_MIME = new Set([
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
]);

fs.mkdirSync(uploadDir, { recursive: true });

app.use(cors());
app.use(express.json());
app.use(
  fileUpload({
    limits: { fileSize: MAX_UPLOAD_BYTES },
    abortOnLimit: true,
  })
);
app.use(express.static(publicDir));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, service: "frame-lab" });
});

app.get("/api/classes", (_req, res) => {
  res.json({
    ok: true,
    classes: [
      { id: 1, label: "Attack on Titan" },
      { id: 2, label: "Death Note" },
      { id: 3, label: "Evangelion" },
      { id: 4, label: "Demon Slayer" },
      { id: 5, label: "The Lord of the Rings" },
      { id: 6, label: "NBA" },
      { id: 7, label: "One Piece" },
      { id: 8, label: "Sneakers" },
      { id: 9, label: "Star Wars" },
    ],
  });
});

function safeFilename(originalName) {
  const base = path.basename(originalName || "upload");
  const cleaned = base.replace(/[^\w.\-]+/g, "_").slice(0, 80) || "upload";
  return `${Date.now()}-${cleaned}`;
}

function runClassifier(imagePath) {
  return PythonShell.run("classify.py", {
    scriptPath: serverDir,
    pythonPath: process.platform === "win32" ? "python" : "python3",
    args: [imagePath],
    mode: "json",
  }).then((messages) => {
    const result = messages[messages.length - 1];
    if (!result || result.ok === false) {
      const error = new Error((result && result.error) || "Classifier failed");
      error.payload = result;
      throw error;
    }
    return result;
  });
}

app.post("/upload", async (req, res) => {
  try {
    if (!req.files || !req.files.file) {
      return res.status(400).json({ ok: false, error: "No file uploaded" });
    }

    const imageFile = Array.isArray(req.files.file)
      ? req.files.file[0]
      : req.files.file;

    if (imageFile.mimetype && !ALLOWED_MIME.has(imageFile.mimetype)) {
      return res.status(400).json({
        ok: false,
        error: "Please upload a JPEG, PNG, GIF, or WebP image",
      });
    }

    const dest = path.join(uploadDir, safeFilename(imageFile.name));
    await imageFile.mv(dest);

    const result = await runClassifier(dest);
    return res.json(result);
  } catch (err) {
    console.error("Upload/classify failed:", err);
    if (!res.headersSent) {
      return res.status(500).json({
        ok: false,
        error: err.message || "Could not classify image",
      });
    }
  }
});

app.listen(port, "0.0.0.0", () => {
  console.log(`Frame Lab listening on http://localhost:${port}`);
});
