const dropArea = document.querySelector(".drop-area");
const dragText = dropArea.querySelector("h2");
const button = dropArea.querySelector("button");
const input = dropArea.querySelector("#input-file");
const preview = document.querySelector("#preview");
const previewCount = document.querySelector("#preview-count");

const VALID_TYPES = new Set([
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
]);

const API_BASE = window.location.protocol === "file:" ? "http://localhost:3000" : "";

let processedCount = 0;

button.addEventListener("click", () => input.click());

dropArea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});

input.addEventListener("change", (event) => {
  showFiles(event.target.files);
  input.value = "";
});

["dragenter", "dragover", "dragleave", "drop"].forEach((name) => {
  document.addEventListener(name, (event) => event.preventDefault());
});

dropArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropArea.classList.add("active");
  dragText.textContent = "Drop to develop";
});

dropArea.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dropArea.classList.remove("active");
  dragText.textContent = "Drag images here";
});

dropArea.addEventListener("drop", (event) => {
  event.preventDefault();
  dropArea.classList.remove("active");
  dragText.textContent = "Drag images here";
  showFiles(event.dataTransfer.files);
});

function showFiles(fileList) {
  if (!fileList || fileList.length === 0) {
    return;
  }
  Array.from(fileList).forEach(processFile);
}

function processFile(file) {
  if (!VALID_TYPES.has(file.type)) {
    setPreviewMessage("That file is not a supported image.");
    return;
  }

  const id = `file-${crypto.randomUUID()}`;
  preview.classList.remove("empty");
  processedCount += 1;
  updateCount();

  const card = document.createElement("article");
  card.id = id;
  card.className = "frame-card";
  card.innerHTML = `
    <img alt="${escapeHtml(file.name)}" src="">
    <div class="frame-meta">
      <span class="frame-name">${escapeHtml(file.name)}</span>
      <h3 class="frame-label">Reading frame…</h3>
      <p class="status-text loading">Extracting color, edges, and histograms</p>
      <ul class="score-list" hidden></ul>
    </div>
  `;
  preview.prepend(card);

  const reader = new FileReader();
  reader.addEventListener("load", () => {
    const img = card.querySelector("img");
    if (img) {
      img.src = reader.result;
    }
  });
  reader.readAsDataURL(file);

  uploadFile(file, id);
}

async function uploadFile(file, id) {
  const card = document.getElementById(id);
  const status = card.querySelector(".status-text");
  const label = card.querySelector(".frame-label");
  const scores = card.querySelector(".score-list");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok || !payload || payload.ok === false) {
      throw new Error((payload && payload.error) || "Upload failed");
    }

    label.textContent = payload.label;
    status.className = "status-text success";
    status.textContent = `${Math.round(payload.confidence * 100)}% model confidence`;
    renderScores(scores, payload.scores);
    highlightClass(payload.classId);
  } catch (error) {
    label.textContent = "Could not classify";
    status.className = "status-text failure";
    status.textContent = error.message || "The lab could not read this frame.";
  }
}

function renderScores(list, scores) {
  if (!scores || !scores.length) {
    return;
  }
  list.hidden = false;
  list.innerHTML = scores
    .slice(0, 6)
    .map(
      (row) => `
      <li>
        <span>${escapeHtml(row.label)}</span>
        <span class="bar"><span style="width:${Math.max(3, Math.round(row.score * 100))}%"></span></span>
        <span>${Math.round(row.score * 100)}%</span>
      </li>`
    )
    .join("");
}

function highlightClass(classId) {
  document.querySelectorAll(".class-grid li").forEach((item) => {
    item.style.outline = item.getAttribute("data-id") === String(classId)
      ? "2px solid var(--signal)"
      : "";
  });
}

function updateCount() {
  previewCount.textContent =
    processedCount === 1 ? "1 frame in the bath" : `${processedCount} frames in the bath`;
}

function setPreviewMessage(message) {
  preview.classList.remove("empty");
  const note = document.createElement("p");
  note.className = "status-text failure";
  note.textContent = message;
  preview.prepend(note);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
