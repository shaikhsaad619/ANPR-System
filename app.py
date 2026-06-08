"""
Flask REST API — ANPR system with Vehicle Registration
"""

import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

from database import db
from detector import ANPRDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

detector = ANPRDetector(
    model_path=os.getenv("YOLO_MODEL", "yolov8n.pt"),
    conf_threshold=float(os.getenv("YOLO_CONF", "0.4")),
    ocr_conf_threshold=float(os.getenv("OCR_CONF", "0.3")),
)


def allowed_file(f):
    return "." in f and f.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Detection API
# ---------------------------------------------------------------------------

@app.post("/api/detect")
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    save_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(save_path)

    try:
        results = detector.detect_from_path(str(save_path))
    except Exception as exc:
        logger.exception("Detection failed")
        return jsonify({"error": str(exc)}), 500

    output = []
    for r in results:
        if not r.plate_text:
            continue
        det = db.save_detection(
            plate_text=r.plate_text,
            confidence=r.confidence,
            detection_score=r.detection_score,
            image_path=r.crop_path,
            source=filename,
        )
        d = det.to_dict()
        # Attach vehicle info if registered
        vehicle = db.get_vehicle(r.plate_text)
        d["vehicle"] = vehicle
        output.append(d)

    return jsonify({"detected": len(output), "plates": output, "source": filename})


@app.get("/api/detections")
def api_detections():
    limit = min(int(request.args.get("limit", 50)), 500)
    detections = db.get_recent(limit)
    # Enrich each detection with vehicle info
    for d in detections:
        d["vehicle"] = db.get_vehicle(d["plate_text"])
    return jsonify(detections)


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 100)), 500)
    if not q:
        return jsonify({"error": "Provide ?q= parameter"}), 400
    rows = db.search_plate(q, limit)
    for d in rows:
        d["vehicle"] = db.get_vehicle(d["plate_text"])
    return jsonify(rows)


@app.get("/api/stats")
def api_stats():
    recent = db.get_recent(1000)
    flagged = [r for r in recent if r["is_flagged"]]
    plates = {r["plate_text"] for r in recent}
    total_vehicles = len(db.get_all_vehicles())
    return jsonify({
        "total_detections": len(recent),
        "unique_plates": len(plates),
        "flagged_hits": len(flagged),
        "registered_vehicles": total_vehicles,
    })


# ---------------------------------------------------------------------------
# Vehicle Registration API
# ---------------------------------------------------------------------------

@app.post("/api/vehicles")
def api_register_vehicle():
    body = request.get_json(silent=True) or {}
    if not body.get("plate_text") or not body.get("owner_name"):
        return jsonify({"error": "plate_text and owner_name are required"}), 400
    v = db.register_vehicle(body)
    return jsonify(v.to_dict()), 201


@app.get("/api/vehicles")
def api_list_vehicles():
    return jsonify(db.get_all_vehicles())


@app.get("/api/vehicles/<plate>")
def api_get_vehicle(plate):
    v = db.get_vehicle(plate)
    if not v:
        return jsonify({"error": "Not found"}), 404
    return jsonify(v)


@app.delete("/api/vehicles/<plate>")
def api_delete_vehicle(plate):
    deleted = db.delete_vehicle(plate)
    if not deleted:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"deleted": plate.upper()})


@app.post("/api/watchlist")
def api_watchlist_add():
    body = request.get_json(silent=True) or {}
    plate = body.get("plate", "").strip().upper()
    if not plate:
        return jsonify({"error": "plate field required"}), 400
    db.add_to_watchlist(plate, body.get("reason", ""))
    return jsonify({"added": plate}), 201


@app.get("/crops/<path:filename>")
def serve_crop(filename):
    return send_from_directory("uploads/crops", filename)


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ANPR Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d0f1a;color:#e2e8f0;min-height:100vh}
nav{background:#13162a;border-bottom:1px solid #1e2240;padding:0 28px;display:flex;align-items:center;gap:0}
nav h1{font-size:1.1rem;color:#a78bfa;margin-right:32px;padding:18px 0}
nav button{background:none;border:none;color:#94a3b8;padding:18px 18px;cursor:pointer;font-size:0.88rem;border-bottom:2px solid transparent;transition:.2s}
nav button.active,nav button:hover{color:#a78bfa;border-bottom-color:#a78bfa}
.page{display:none;padding:28px}
.page.active{display:block}

/* Stats */
.stats{display:flex;gap:14px;margin-bottom:24px;flex-wrap:wrap}
.stat{background:#13162a;border:1px solid #1e2240;border-radius:10px;padding:18px 24px;min-width:150px}
.stat .num{font-size:1.9rem;font-weight:700;color:#a78bfa}
.stat .lbl{font-size:0.78rem;color:#64748b;margin-top:3px}

/* Upload */
.box{background:#13162a;border:1px solid #1e2240;border-radius:12px;padding:22px;margin-bottom:22px}
.box h2{font-size:1rem;margin-bottom:14px;color:#c4b5fd}
input[type=file]{color:#e2e8f0;font-size:0.9rem}
.btn{background:#7c3aed;color:#fff;border:none;border-radius:7px;padding:9px 20px;cursor:pointer;font-size:0.88rem;margin-top:10px;transition:.2s}
.btn:hover{background:#6d28d9}
.btn.red{background:#dc2626}.btn.red:hover{background:#b91c1c}
.btn.gray{background:#374151}.btn.gray:hover{background:#4b5563}
pre#result{margin-top:12px;background:#0d0f1a;border-radius:7px;padding:14px;font-size:0.82rem;white-space:pre-wrap;display:none;border:1px solid #1e2240}

/* Detection card popup */
.det-popup{background:#13162a;border:1px solid #7c3aed;border-radius:12px;padding:20px;margin-top:16px;display:none}
.det-popup h3{color:#a78bfa;margin-bottom:12px;font-size:1rem}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 20px}
.info-item .ilbl{font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.info-item .ival{font-size:0.95rem;color:#e2e8f0;font-weight:500}
.color-dot{display:inline-block;width:14px;height:14px;border-radius:50%;border:1px solid #fff3;vertical-align:middle;margin-right:6px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600}
.badge.reg{background:#14532d;color:#86efac}
.badge.unreg{background:#7f1d1d;color:#fca5a5}
.badge.flag{background:#78350f;color:#fcd34d}

/* Table */
.search-row{display:flex;gap:10px;margin-bottom:14px}
.search-row input{flex:1;background:#13162a;border:1px solid #1e2240;border-radius:7px;padding:8px 13px;color:#e2e8f0;font-size:0.88rem}
table{width:100%;border-collapse:collapse;background:#13162a;border-radius:12px;overflow:hidden}
th{background:#1a1d35;padding:10px 14px;text-align:left;font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
td{padding:10px 14px;border-top:1px solid #1a1d35;font-size:0.85rem;vertical-align:middle}
tr:hover td{background:#1a1d35}
.plate-badge{background:#1e1b4b;color:#a78bfa;font-weight:700;padding:3px 10px;border-radius:6px;font-family:monospace;font-size:0.9rem}

/* Registration form */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.form-group{display:flex;flex-direction:column;gap:5px}
.form-group label{font-size:0.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.04em}
.form-group input,.form-group select{background:#0d0f1a;border:1px solid #1e2240;border-radius:7px;padding:9px 12px;color:#e2e8f0;font-size:0.9rem;width:100%}
.form-group input:focus,.form-group select:focus{outline:none;border-color:#7c3aed}
.form-actions{display:flex;gap:10px;margin-top:8px}
.msg{padding:10px 14px;border-radius:7px;font-size:0.88rem;margin-top:12px;display:none}
.msg.ok{background:#14532d;color:#86efac;display:block}
.msg.err{background:#7f1d1d;color:#fca5a5;display:block}

/* Vehicle list */
.veh-card{background:#13162a;border:1px solid #1e2240;border-radius:10px;padding:16px 20px;margin-bottom:12px;display:flex;align-items:center;gap:16px}
.veh-avatar{width:48px;height:48px;border-radius:50%;background:#1e1b4b;display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0}
.veh-info{flex:1}
.veh-plate{font-family:monospace;font-size:1rem;font-weight:700;color:#a78bfa}
.veh-owner{font-size:0.9rem;color:#e2e8f0;margin-top:2px}
.veh-meta{font-size:0.78rem;color:#64748b;margin-top:3px}
.veh-actions{display:flex;gap:8px}
</style>
</head>
<body>

<nav>
  <h1>🚗 ANPR System</h1>
  <button class="active" onclick="showPage('detect',this)">Detection</button>
  <button onclick="showPage('register',this)">Register Vehicle</button>
  <button onclick="showPage('vehicles',this)">Vehicle Database</button>
  <button onclick="showPage('history',this)">Detection History</button>
</nav>

<!-- ==================== DETECTION PAGE ==================== -->
<div id="page-detect" class="page active">
  <div class="stats">
    <div class="stat"><div class="num" id="s-total">—</div><div class="lbl">Total detections</div></div>
    <div class="stat"><div class="num" id="s-unique">—</div><div class="lbl">Unique plates</div></div>
    <div class="stat"><div class="num" id="s-flagged">—</div><div class="lbl">Flagged hits</div></div>
    <div class="stat"><div class="num" id="s-reg">—</div><div class="lbl">Registered vehicles</div></div>
  </div>

  <div class="box">
    <h2>Upload plate image</h2>
    <input type="file" id="imgFile" accept="image/*">
    <button class="btn" onclick="uploadImage()">Run detection</button>
    <div class="det-popup" id="detPopup">
      <h3 id="popPlate"></h3>
      <div id="popContent"></div>
    </div>
    <pre id="result"></pre>
  </div>
</div>

<!-- ==================== REGISTER PAGE ==================== -->
<div id="page-register" class="page">
  <div class="box">
    <h2>Register a vehicle</h2>
    <div class="form-grid">
      <div class="form-group">
        <label>Number Plate *</label>
        <input id="f-plate" placeholder="e.g. ABC-123" style="text-transform:uppercase">
      </div>
      <div class="form-group">
        <label>Owner Full Name *</label>
        <input id="f-owner" placeholder="Muhammad Ali">
      </div>
      <div class="form-group">
        <label>Owner CNIC</label>
        <input id="f-cnic" placeholder="42201-1234567-8">
      </div>
      <div class="form-group">
        <label>Phone Number</label>
        <input id="f-phone" placeholder="0300-1234567">
      </div>
      <div class="form-group">
        <label>Car Brand</label>
        <select id="f-brand">
          <option value="">— Select brand —</option>
          <option>Toyota</option><option>Honda</option><option>Suzuki</option>
          <option>Hyundai</option><option>Kia</option><option>Daihatsu</option>
          <option>Mitsubishi</option><option>Nissan</option><option>BMW</option>
          <option>Mercedes-Benz</option><option>Audi</option><option>Ford</option>
          <option>Chevrolet</option><option>Other</option>
        </select>
      </div>
      <div class="form-group">
        <label>Model</label>
        <input id="f-model" placeholder="e.g. Corolla, Civic, Alto">
      </div>
      <div class="form-group">
        <label>Model Year</label>
        <input id="f-year" type="number" placeholder="2020" min="1960" max="2026">
      </div>
      <div class="form-group">
        <label>Car Colour</label>
        <input id="f-color" placeholder="e.g. White, Silver, Black">
      </div>
      <div class="form-group">
        <label>Engine (CC)</label>
        <input id="f-engine" placeholder="e.g. 1300cc, 1800cc">
      </div>
      <div class="form-group">
        <label>Notes</label>
        <input id="f-notes" placeholder="Any additional info">
      </div>
    </div>
    <div class="form-actions">
      <button class="btn" onclick="registerVehicle()">Save Vehicle</button>
      <button class="btn gray" onclick="clearForm()">Clear</button>
    </div>
    <div id="reg-msg" class="msg"></div>
  </div>
</div>

<!-- ==================== VEHICLE DATABASE PAGE ==================== -->
<div id="page-vehicles" class="page">
  <div class="search-row" style="margin-bottom:16px">
    <input id="veh-search" placeholder="Search by plate or owner name…" oninput="filterVehicles()">
    <button class="btn" onclick="loadVehicles()">Refresh</button>
  </div>
  <div id="veh-list"></div>
</div>

<!-- ==================== HISTORY PAGE ==================== -->
<div id="page-history" class="page">
  <div class="search-row">
    <input id="searchQ" placeholder="Search plate…">
    <button class="btn" onclick="searchPlate()">Search</button>
    <button class="btn gray" onclick="loadHistory()">Refresh</button>
  </div>
  <table>
    <thead><tr>
      <th>#</th><th>Plate</th><th>Owner</th><th>Vehicle</th>
      <th>Confidence</th><th>Time</th><th>Status</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<script>
// ── Navigation ──────────────────────────────────────────────────────────────
function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  btn.classList.add('active');
  if(name==='vehicles') loadVehicles();
  if(name==='history')  loadHistory();
  if(name==='detect')   loadStats();
}

// ── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  const r = await fetch('/api/stats').then(r=>r.json());
  document.getElementById('s-total').textContent   = r.total_detections;
  document.getElementById('s-unique').textContent  = r.unique_plates;
  document.getElementById('s-flagged').textContent = r.flagged_hits;
  document.getElementById('s-reg').textContent     = r.registered_vehicles;
}

// ── Detection ────────────────────────────────────────────────────────────────
async function uploadImage() {
  const file = document.getElementById('imgFile').files[0];
  if (!file) return alert('Please select an image first');
  const fd = new FormData();
  fd.append('image', file);
  const res = document.getElementById('result');
  const popup = document.getElementById('detPopup');
  popup.style.display = 'none';
  res.style.display = 'block';
  res.textContent = '🔍 Processing…';
  const data = await fetch('/api/detect',{method:'POST',body:fd}).then(r=>r.json());
  res.style.display = 'none';

  if (!data.plates || data.plates.length === 0) {
    res.style.display = 'block';
    res.textContent = '❌ No plate detected. Try a clearer image.';
    return;
  }

  const p = data.plates[0];
  const v = p.vehicle;
  popup.style.display = 'block';
  document.getElementById('popPlate').innerHTML =
    `<span class="plate-badge">${p.plate_text}</span>
     &nbsp; ${v ? '<span class="badge reg">✓ Registered</span>' : '<span class="badge unreg">Unregistered</span>'}
     ${p.is_flagged ? '&nbsp;<span class="badge flag">⚠ FLAGGED</span>' : ''}`;

  let html = '<div class="info-grid">';
  if (v) {
    const colorDot = v.color ? `<span class="color-dot" style="background:${cssColor(v.color)}"></span>` : '';
    html += infoItem('Owner', v.owner_name);
    html += infoItem('CNIC', v.owner_cnic || '—');
    html += infoItem('Phone', v.owner_phone || '—');
    html += infoItem('Brand', v.brand || '—');
    html += infoItem('Model', v.model || '—');
    html += infoItem('Year', v.model_year || '—');
    html += infoItem('Colour', colorDot + (v.color || '—'));
    html += infoItem('Engine', v.engine_cc || '—');
    if (v.notes) html += infoItem('Notes', v.notes);
  } else {
    html += `<div style="grid-column:1/-1;color:#94a3b8;font-size:0.9rem">
      This vehicle is not registered in the database.
      <a href="#" onclick="goRegister('${p.plate_text}')" style="color:#a78bfa;margin-left:8px">Register now →</a>
    </div>`;
  }
  html += `</div>`;
  html += `<div style="margin-top:14px;font-size:0.78rem;color:#64748b">
    OCR confidence: ${(p.confidence*100).toFixed(1)}% &nbsp;|&nbsp;
    YOLO: ${(p.detection_score*100).toFixed(1)}% &nbsp;|&nbsp;
    Detected: ${new Date(p.timestamp).toLocaleTimeString()}
  </div>`;
  document.getElementById('popContent').innerHTML = html;
  loadStats();
}

function infoItem(label, val) {
  return `<div class="info-item"><div class="ilbl">${label}</div><div class="ival">${val}</div></div>`;
}

function cssColor(name) {
  const map = {white:'#f8fafc',black:'#111',silver:'#c0c0c0',gray:'#6b7280',
    grey:'#6b7280',red:'#ef4444',blue:'#3b82f6',green:'#22c55e',
    yellow:'#eab308',orange:'#f97316',brown:'#92400e',maroon:'#7f1d1d',
    gold:'#d97706',beige:'#d4b483',purple:'#7c3aed',pink:'#ec4899'};
  return map[name.toLowerCase()] || '#a78bfa';
}

function goRegister(plate) {
  document.getElementById('f-plate').value = plate.toUpperCase();
  showPage('register', document.querySelectorAll('nav button')[1]);
}

// ── Registration ─────────────────────────────────────────────────────────────
async function registerVehicle() {
  const plate = document.getElementById('f-plate').value.trim().toUpperCase();
  const owner = document.getElementById('f-owner').value.trim();
  if (!plate || !owner) { showMsg('Plate and owner name are required','err'); return; }

  const body = {
    plate_text:  plate,
    owner_name:  owner,
    owner_cnic:  document.getElementById('f-cnic').value.trim() || null,
    owner_phone: document.getElementById('f-phone').value.trim() || null,
    brand:       document.getElementById('f-brand').value || null,
    model:       document.getElementById('f-model').value.trim() || null,
    model_year:  parseInt(document.getElementById('f-year').value) || null,
    color:       document.getElementById('f-color').value.trim() || null,
    engine_cc:   document.getElementById('f-engine').value.trim() || null,
    notes:       document.getElementById('f-notes').value.trim() || null,
  };

  const res = await fetch('/api/vehicles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if (res.ok) {
    showMsg(`✓ Vehicle ${plate} registered successfully!`, 'ok');
    clearForm();
  } else {
    const e = await res.json();
    showMsg('Error: ' + (e.error || 'Unknown error'), 'err');
  }
}

function showMsg(text, type) {
  const m = document.getElementById('reg-msg');
  m.textContent = text;
  m.className = 'msg ' + type;
  setTimeout(() => m.className = 'msg', 4000);
}

function clearForm() {
  ['f-plate','f-owner','f-cnic','f-phone','f-brand','f-model','f-year','f-color','f-engine','f-notes']
    .forEach(id => { const el=document.getElementById(id); if(el) el.value=''; });
}

// ── Vehicle Database ──────────────────────────────────────────────────────────
let allVehicles = [];
async function loadVehicles() {
  allVehicles = await fetch('/api/vehicles').then(r=>r.json());
  renderVehicles(allVehicles);
}

function filterVehicles() {
  const q = document.getElementById('veh-search').value.toLowerCase();
  renderVehicles(allVehicles.filter(v =>
    v.plate_text.toLowerCase().includes(q) ||
    (v.owner_name||'').toLowerCase().includes(q) ||
    (v.brand||'').toLowerCase().includes(q)
  ));
}

function renderVehicles(list) {
  const div = document.getElementById('veh-list');
  if (!list.length) { div.innerHTML='<p style="color:#64748b;padding:20px">No vehicles registered yet.</p>'; return; }
  div.innerHTML = list.map(v => `
    <div class="veh-card">
      <div class="veh-avatar">🚗</div>
      <div class="veh-info">
        <div class="veh-plate">${v.plate_text}</div>
        <div class="veh-owner">${v.owner_name} ${v.owner_phone ? '· '+v.owner_phone : ''}</div>
        <div class="veh-meta">
          ${[v.brand, v.model, v.model_year, v.color ? v.color+' colour' : '', v.engine_cc].filter(Boolean).join(' · ')}
        </div>
      </div>
      <div class="veh-actions">
        <button class="btn gray" onclick="editVehicle('${v.plate_text}')">Edit</button>
        <button class="btn red" onclick="deleteVehicle('${v.plate_text}')">Delete</button>
      </div>
    </div>
  `).join('');
}

async function deleteVehicle(plate) {
  if (!confirm(`Delete vehicle ${plate}?`)) return;
  await fetch(`/api/vehicles/${plate}`,{method:'DELETE'});
  loadVehicles();
}

async function editVehicle(plate) {
  const v = await fetch(`/api/vehicles/${plate}`).then(r=>r.json());
  document.getElementById('f-plate').value  = v.plate_text;
  document.getElementById('f-owner').value  = v.owner_name;
  document.getElementById('f-cnic').value   = v.owner_cnic || '';
  document.getElementById('f-phone').value  = v.owner_phone || '';
  document.getElementById('f-brand').value  = v.brand || '';
  document.getElementById('f-model').value  = v.model || '';
  document.getElementById('f-year').value   = v.model_year || '';
  document.getElementById('f-color').value  = v.color || '';
  document.getElementById('f-engine').value = v.engine_cc || '';
  document.getElementById('f-notes').value  = v.notes || '';
  showPage('register', document.querySelectorAll('nav button')[1]);
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory() {
  const data = await fetch('/api/detections?limit=100').then(r=>r.json());
  renderHistory(data);
}

async function searchPlate() {
  const q = document.getElementById('searchQ').value.trim();
  if (!q) return loadHistory();
  const data = await fetch(`/api/search?q=${encodeURIComponent(q)}`).then(r=>r.json());
  renderHistory(Array.isArray(data) ? data : []);
}

function renderHistory(data) {
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  data.forEach(d => {
    const v = d.vehicle;
    const row = tbody.insertRow();
    row.innerHTML = `
      <td>${d.id}</td>
      <td><span class="plate-badge">${d.plate_text}</span></td>
      <td>${v ? v.owner_name : '<span style="color:#64748b">Unknown</span>'}</td>
      <td style="font-size:0.8rem">${v ? [v.brand,v.model,v.model_year].filter(Boolean).join(' ') : '—'}</td>
      <td>${(d.confidence*100).toFixed(1)}%</td>
      <td>${new Date(d.timestamp).toLocaleTimeString()}</td>
      <td>${d.is_flagged
        ? '<span class="badge flag">⚠ Flagged</span>'
        : v
          ? '<span class="badge reg">✓ Registered</span>'
          : '<span class="badge unreg">Unregistered</span>'}</td>
    `;
  });
}

// Init
loadStats();
setInterval(loadStats, 15000);
</script>
</body>
</html>"""


@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
