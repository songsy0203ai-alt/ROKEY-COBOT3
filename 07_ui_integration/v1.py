#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request, send_file, Response

STATUS_PATH = Path(os.environ.get("ACS_STATUS_PATH", "/tmp/acs_monitor/status.json"))
COMMAND_PATH = Path(os.environ.get("ACS_COMMAND_PATH", "/tmp/acs_monitor/command.json"))
PREVIEW_DIR = Path(os.environ.get("ACS_PREVIEW_DIR", "/tmp/acs_monitor/previews"))
CURRENT_PREVIEW_PATH = PREVIEW_DIR / "current.png"
POLL_MS = int(os.environ.get("ACS_POLL_MS", "400"))
HOST = os.environ.get("ACS_HOST", "0.0.0.0")
PORT = int(os.environ.get("ACS_PORT", "5000"))

app = Flask(__name__)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def preview_placeholder_svg() -> str:
    return """<svg xmlns='http://www.w3.org/2000/svg' width='960' height='540' viewBox='0 0 960 540'>
    <defs>
      <linearGradient id='g' x1='0' x2='0' y1='0' y2='1'>
        <stop offset='0%' stop-color='#111722'/>
        <stop offset='100%' stop-color='#1e2633'/>
      </linearGradient>
    </defs>
    <rect width='960' height='540' fill='url(#g)'/>
    <rect x='40' y='40' width='880' height='460' rx='18' ry='18' fill='none' stroke='#4d5a6d' stroke-width='3'/>
    <text x='480' y='245' text-anchor='middle' fill='#dce3ea' font-size='34' font-family='Arial, sans-serif'>Camera Preview 대기 중</text>
    <text x='480' y='290' text-anchor='middle' fill='#93a2b3' font-size='20' font-family='Arial, sans-serif'>Isaac Sim이 현재 활성 카메라 뷰를 캡처하면 여기에 표시됩니다.</text>
    </svg>"""


def read_status() -> Dict[str, Any]:
    if not STATUS_PATH.exists():
        return {
            "connected": False,
            "timestamp": None,
            "message": f"status file not found: {STATUS_PATH}",
            "summary": {
                "sim_playing": False,
                "oht_total": 0,
                "carrying_count": 0,
                "blocked_count": 0,
                "pickup_stock": 0,
                "bridge_queue": 0,
                "bridge_state": "-",
                "placed_wafer_count": 0,
                "ur10_phase_text": "-",
            },
            "layout": None,
            "graph": {"nodes": {}, "edges": []},
            "ohts": [],
            "bridge": {},
            "ur10": {},
            "camera": {
                "selected_path": "/World/ScriptCamera",
                "selected_label": "Overview",
                "preview": {
                    "exists": False,
                    "path": str(CURRENT_PREVIEW_PATH),
                    "mtime": None,
                    "size": None,
                },
            },
            "jobs": [],
            "alarms": [["INFO", "Isaac Sim status 대기 중"]],
        }

    try:
        with STATUS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "connected": False,
            "timestamp": None,
            "message": f"status read error: {e}",
            "summary": {
                "sim_playing": False,
                "oht_total": 0,
                "carrying_count": 0,
                "blocked_count": 0,
                "pickup_stock": 0,
                "bridge_queue": 0,
                "bridge_state": "-",
                "placed_wafer_count": 0,
                "ur10_phase_text": "-",
            },
            "layout": None,
            "graph": {"nodes": {}, "edges": []},
            "ohts": [],
            "bridge": {},
            "ur10": {},
            "camera": {
                "selected_path": "/World/ScriptCamera",
                "selected_label": "Overview",
                "preview": {
                    "exists": False,
                    "path": str(CURRENT_PREVIEW_PATH),
                    "mtime": None,
                    "size": None,
                },
            },
            "jobs": [],
            "alarms": [["WARN", "status.json 파싱 실패"]],
        }

    data["connected"] = True

    # 카메라 정보가 status.json에 아직 없더라도 웹이 죽지 않도록 기본값 보강
    if "camera" not in data:
        data["camera"] = {
            "selected_path": "/World/ScriptCamera",
            "selected_label": "Overview",
        }
    if "preview" not in data["camera"]:
        data["camera"]["preview"] = {}

    preview = data["camera"]["preview"]
    preview["path"] = str(CURRENT_PREVIEW_PATH)
    if CURRENT_PREVIEW_PATH.exists():
        st = CURRENT_PREVIEW_PATH.stat()
        preview.setdefault("exists", True)
        preview.setdefault("mtime", st.st_mtime)
        preview.setdefault("size", st.st_size)
    else:
        preview.setdefault("exists", False)
        preview.setdefault("mtime", None)
        preview.setdefault("size", None)

    return data


INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OHT Monitoring &amp; Control System</title>
  <style>
    :root {
      --bg: #1d2128;
      --panel: #2c313a;
      --panel2: #353c46;
      --border: #576171;
      --text: #ebeff4;
      --muted: #aeb7c3;
      --green: #27d45e;
      --cyan: #2bc0de;
      --warn: #f0b400;
      --red: #ff6767;
      --orange: #ff9c35;
      --ok: #57d07a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #1a1d24 0%, #1e232b 100%);
      color: var(--text);
      font-family: "Noto Sans KR", "Malgun Gothic", Arial, sans-serif;
    }
    .app {
      min-height: 100vh;
      padding: 16px;
    }
    .header {
      background: #f5f7fa;
      color: #2e2f33;
      font-size: 28px;
      font-weight: 800;
      padding: 18px 20px;
      border-radius: 4px;
      margin-bottom: 14px;
    }
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .kpi-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px 12px;
      min-height: 84px;
      position: relative;
      overflow: hidden;
    }
    .kpi-bar {
      width: 42px;
      height: 4px;
      background: var(--cyan);
      border-radius: 8px;
      margin-bottom: 8px;
    }
    .kpi-title { color: var(--muted); font-size: 12px; }
    .kpi-value { font-size: 24px; font-weight: 800; line-height: 1.1; margin-top: 4px; }
    .kpi-sub { color: var(--muted); font-size: 11px; margin-top: 4px; }

    .main {
      display: grid;
      grid-template-columns: 80px minmax(680px, 1fr) 420px;
      gap: 14px;
      min-height: calc(100vh - 190px);
    }
    .nav, .center, .right {
      background: rgba(44, 49, 58, 0.96);
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    .nav {
      padding: 12px 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      align-items: center;
    }
    .nav-item {
      width: 62px;
      height: 56px;
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      font-size: 12px;
      color: var(--text);
      user-select: none;
    }
    .nav-item.active { background: #3f4a57; }

    .center {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .badge-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .badge {
      background: var(--panel2);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 12px;
      min-width: 84px;
      text-align: center;
    }
    .badge.run { background: #2e6d42; }
    .badge.stop { background: #7e3a3a; }
    .controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    button.ctrl {
      background: #475362;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 700;
    }
    button.ctrl:hover { filter: brightness(1.08); }
    #mapCanvas {
      width: 100%;
      height: clamp(320px, 45vh, 500px);
      min-height: 320px;
      background: radial-gradient(circle at 50% 50%, #232831 0%, #1b2027 100%);
      border: 1px solid var(--border);
      border-radius: 6px;
      display: block;
    }
    .center-subtitle {
      font-size: 18px;
      font-weight: 800;
      margin-top: 2px;
      margin-bottom: 2px;
    }
    #cameraPanel {
      width: 100%;
    }
    #cameraPanel .card {
      margin-bottom: 0;
    }
    #cameraPanel .preview-wrap {
      width: 100%;
      height: clamp(240px, 34vh, 380px);
      border-radius: 6px;
      overflow: hidden;
      background: #111722;
      border: 1px solid var(--border);
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    #cameraPanel .preview-img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #111722;
    }

    .right {
      padding: 14px;
      overflow: auto;
    }
    .section-title {
      font-size: 22px;
      font-weight: 800;
      margin-bottom: 8px;
    }
    .section-rule {
      border-top: 1px solid var(--border);
      margin-bottom: 10px;
    }
    .desc { color: var(--text); font-size: 15px; line-height: 1.9; margin-bottom: 14px; }
    .card {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .card-title { font-size: 15px; font-weight: 800; margin-bottom: 6px; }
    .card-sub { color: var(--muted); font-size: 12px; line-height: 1.6; word-break: break-word; }
    .card.clickable { cursor: pointer; }
    .card.clickable:hover { border-color: var(--cyan); box-shadow: 0 0 0 1px rgba(43,192,222,0.35) inset; }
    .alarm {
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 8px;
      color: #101318;
      font-weight: 800;
      font-size: 12px;
    }
    .alarm.INFO { background: #bfc7d3; }
    .alarm.WARN { background: var(--warn); }
    .alarm.READY { background: var(--ok); }
    .alarm.RUN { background: var(--cyan); }
    .alarm.OK { background: var(--ok); }
    .status-line { font-size: 12px; color: var(--muted); margin-top: 8px; }
    .small-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }
    .mini-card {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
    }
    .mini-label { color: var(--muted); font-size: 12px; }
    .mini-value { font-size: 18px; font-weight: 800; margin-top: 4px; }
    .preview-wrap {
      width: 100%;
      border-radius: 6px;
      overflow: hidden;
      background: #1b2027;
      border: 1px solid var(--border);
      margin-top: 8px;
    }
    .preview-img {
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #1b2027;
    }
    .preview-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-top: 8px;
    }

    @media (max-width: 1450px) {
      .main { grid-template-columns: 80px 1fr; }
      .right { grid-column: 1 / -1; }
      #mapCanvas { min-height: 480px; height: 60vh; }
    }
    @media (max-width: 980px) {
      .kpi-row { grid-template-columns: repeat(2, 1fr); }
      .main { grid-template-columns: 1fr; }
      .nav { flex-direction: row; justify-content: center; }
      .right { grid-column: auto; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">• ACS :: LINE OVERVIEW Monitoring</div>

    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-bar"></div><div class="kpi-title">SIM STATUS</div><div class="kpi-value" id="kpiSim">-</div><div class="kpi-sub" id="kpiSource">source -</div></div>
      <div class="kpi-card"><div class="kpi-bar"></div><div class="kpi-title">ACTIVE OHT</div><div class="kpi-value" id="kpiOht">0</div><div class="kpi-sub">fleet total</div></div>
      <div class="kpi-card"><div class="kpi-bar"></div><div class="kpi-title">CARRYING</div><div class="kpi-value" id="kpiCarry">0</div><div class="kpi-sub">loaded carriers</div></div>
      <div class="kpi-card"><div class="kpi-bar"></div><div class="kpi-title">PICKUP STOCK</div><div class="kpi-value" id="kpiPickup">0</div><div class="kpi-sub">pickup pod stock</div></div>
      <div class="kpi-card"><div class="kpi-bar"></div><div class="kpi-title">BRIDGE</div><div class="kpi-value" id="kpiBridge">-</div><div class="kpi-sub" id="kpiQueue">queue=0</div></div>
      <div class="kpi-card"><div class="kpi-bar"></div><div class="kpi-title">UR10</div><div class="kpi-value" id="kpiUr10">-</div><div class="kpi-sub" id="kpiPlaced">placed=0</div></div>
    </div>

    <div class="main">
      <div class="nav">
        <div class="nav-item active">LIST</div>
        <div class="nav-item">JOB</div>
        <div class="nav-item">ALARM</div>
        <div class="nav-item">REPORT</div>
        <div class="nav-item">CONFIG</div>
      </div>

      <div class="center">
        <div class="toolbar">
          <div class="badge-row">
            <div class="badge" id="badgeSim">SIM -</div>
            <div class="badge" id="badgeOht">OHT 0</div>
            <div class="badge" id="badgeUr10">UR10 -</div>
            <div class="badge" id="badgeQueue">QUEUE 0</div>
            <div class="badge" id="badgePod">POD 0</div>
            <div class="badge" id="badgeCam">CAM Overview</div>
          </div>
          <div class="controls">
            <button class="ctrl" onclick="sendCommand('play')">PLAY</button>
            <button class="ctrl" onclick="sendCommand('pause')">PAUSE</button>
            <button class="ctrl" onclick="sendCommand('reset')">RESET</button>
            <button class="ctrl" onclick="sendCommand('stop')">STOP</button>
            <button class="ctrl" onclick="sendOverviewCamera()">OVERVIEW CAM</button>
          </div>
        </div>
        <canvas id="mapCanvas" width="1100" height="640"></canvas>
        <div class="center-subtitle">CAMERA PREVIEW</div>
        <div id="cameraPanel"></div>
        <div class="status-line" id="statusLine">status: loading...</div>
      </div>

      <div class="right">
        <div class="section-title">[ 전체 레이아웃 ]</div>
        <div class="section-rule"></div>
        <div class="desc">
          1) OHT Moving 상태 실시간 모니터링<br>
          2) JOB 단위 무인 반송 작업 상태 관리<br>
          3) UR10 웨이퍼 픽업 / 배치 상태 확인<br>
          4) Drop Queue / Alarm / 이력 누적 확인
        </div>

        <div class="small-grid">
          <div class="mini-card"><div class="mini-label">Blocked OHT</div><div class="mini-value" id="miniBlocked">0</div></div>
          <div class="mini-card"><div class="mini-label">Placed Wafer</div><div class="mini-value" id="miniPlaced">0</div></div>
        </div>

        <div class="section-title" style="font-size:20px;">OHT STATUS</div>
        <div class="section-rule"></div>
        <div id="ohtList"></div>

        <div class="section-title" style="font-size:20px; margin-top:16px;">UR10 / BRIDGE</div>
        <div class="section-rule"></div>
        <div id="ur10Bridge"></div>

        <div class="section-title" style="font-size:20px; margin-top:16px;">JOB</div>
        <div class="section-rule"></div>
        <div id="jobList"></div>

        <div class="section-title" style="font-size:20px; margin-top:16px;">ALARM</div>
        <div class="section-rule"></div>
        <div id="alarmList"></div>
      </div>
    </div>
  </div>

  <script>
    const POLL_MS = {{ poll_ms }};
    let lastData = null;
    let ohtHitboxes = [];

    function fmtPos(p) {
      if (!p || p.length < 3) return '-';
      return `(${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)})`;
    }

    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }

    function previewSrc(data) {
      const cam = (data && data.camera) ? data.camera : {};
      const preview = cam.preview || {};
      const stamp = preview.mtime ? Math.floor(Number(preview.mtime) * 1000) : Date.now();
      return `/preview/current?ts=${stamp}`;
    }

    function worldToCanvas(pos, layout, w, h) {
      const padX = 70;
      const padY = 60;
      const usableW = w - padX * 2;
      const usableH = h - padY * 2;
      const u = (pos[0] - layout.left_x) / (layout.right_x - layout.left_x || 1.0);
      const v = (layout.top_y - pos[1]) / (layout.top_y - layout.bottom_y || 1.0);
      return {
        x: padX + Math.max(0, Math.min(1, u)) * usableW,
        y: padY + Math.max(0, Math.min(1, v)) * usableH,
      };
    }

    function drawRoundedRect(ctx, x, y, w, h, r, fill, stroke) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    function drawMap(data) {
      const canvas = document.getElementById('mapCanvas');
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(300, Math.floor(rect.width * ratio));
      const height = Math.max(300, Math.floor(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext('2d');
      ohtHitboxes = [];
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#1b2027';
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = '#20252d';
      ctx.fillRect(18 * ratio, 14 * ratio, 620 * ratio, 34 * ratio);
      ctx.strokeStyle = '#576171';
      ctx.strokeRect(18 * ratio, 14 * ratio, 620 * ratio, 34 * ratio);
      ctx.fillStyle = '#ebeff4';
      ctx.font = `${13 * ratio}px sans-serif`;
      ctx.fillText('REALTIME OVERVIEW', 28 * ratio, 36 * ratio);

      if (!data.layout || !data.graph) {
        ctx.fillStyle = '#aeb7c3';
        ctx.font = `${18 * ratio}px sans-serif`;
        ctx.fillText('layout / graph 데이터 대기 중', 60 * ratio, 100 * ratio);
        return;
      }

      const layout = data.layout;
      const graph = data.graph;

      (graph.edges || []).forEach(edge => {
        const a = graph.nodes[edge.start];
        const b = graph.nodes[edge.end];
        if (!a || !b) return;
        const p0 = worldToCanvas(a.pos, layout, width, height);
        const p1 = worldToCanvas(b.pos, layout, width, height);
        ctx.strokeStyle = '#28d562';
        ctx.lineWidth = 6 * ratio;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      });

      Object.entries(graph.nodes || {}).forEach(([name, node]) => {
        const p = worldToCanvas(node.pos, layout, width, height);
        ctx.fillStyle = '#ef9528';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5 * ratio, 0, Math.PI * 2);
        ctx.fill();
      });

      function drawTag(text, x, y, fill) {
        ctx.font = `${12 * ratio}px sans-serif`;
        const tw = ctx.measureText(text).width + 18 * ratio;
        const th = 24 * ratio;
        drawRoundedRect(ctx, x - tw / 2, y - th / 2, tw, th, 5 * ratio, fill, '#576171');
        ctx.fillStyle = '#f4f7fa';
        ctx.fillText(text, x - tw / 2 + 9 * ratio, y + 5 * ratio);
      }

      const loadP = worldToCanvas(layout.load_pos, layout, width, height);
      const unloadP = worldToCanvas(layout.unload_pos, layout, width, height);
      drawTag('LOAD / PICKUP', loadP.x, loadP.y - 28 * ratio, '#2bc0de');
      drawTag('UNLOAD / DROP', unloadP.x, unloadP.y + 28 * ratio, '#ff6767');

      if (data.bridge && data.bridge.pick_world) {
        const bp = worldToCanvas(data.bridge.pick_world, layout, width, height);
        drawTag('TRANSFER', bp.x, bp.y + 30 * ratio, '#f0b400');
      }

      if (data.ur10 && data.ur10.place_position) {
        const up = worldToCanvas(data.ur10.place_position, layout, width, height);
        drawTag('UR10 PLACE', up.x, up.y - 30 * ratio, '#475362');
      }

      (data.ohts || []).forEach(oht => {
        if (!oht.pos) return;
        const p = worldToCanvas(oht.pos, layout, width, height);
        let fill = '#3f4a57';
        if (oht.blocked) fill = '#f0b400';
        else if (oht.carrying) fill = '#2bc0de';
        else if (oht.state === 'LOWER_PICK' || oht.state === 'PICK_WAIT') fill = '#57d07a';
        else if (oht.state === 'LOWER_DROP' || oht.state === 'DROP_WAIT') fill = '#ff6767';

        const label = String(oht.id || '?').replace('OHT_', '');
        const rx = p.x - 24 * ratio;
        const ry = p.y - 14 * ratio;
        const rw = 48 * ratio;
        const rh = 28 * ratio;
        drawRoundedRect(ctx, rx, ry, rw, rh, 6 * ratio, fill, '#d8e0e8');
        ctx.fillStyle = '#ffffff';
        ctx.font = `bold ${12 * ratio}px sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillText(label, p.x - tw / 2, p.y + 4 * ratio);
        ohtHitboxes.push({
          x: rx, y: ry, w: rw, h: rh,
          ohtId: oht.id,
          cameraPath: oht.camera_path || null,
          label: `${oht.id} CAM`,
        });
      });

      ctx.fillStyle = '#aeb7c3';
      ctx.font = `${12 * ratio}px sans-serif`;
      const placed = data.bridge && typeof data.bridge.placed_wafer_count === 'number' ? data.bridge.placed_wafer_count : 0;
      ctx.fillText(`Placed Wafer : ${placed}`, 20 * ratio, height - 18 * ratio);
    }

    function renderLists(data) {
      const ohtList = document.getElementById('ohtList');
      const ur10Bridge = document.getElementById('ur10Bridge');
      const cameraPanel = document.getElementById('cameraPanel');
      const jobList = document.getElementById('jobList');
      const alarmList = document.getElementById('alarmList');

      ohtList.innerHTML = '';
      (data.ohts || []).forEach(oht => {
        const div = document.createElement('div');
        div.className = 'card clickable';
        if (oht.camera_path) {
          div.onclick = () => sendCameraCommand(oht.camera_path, `${oht.id} CAM`);
          div.title = `${oht.id} 카메라 보기`;
        }
        div.innerHTML = `
          <div class="card-title">${oht.id} | ${oht.state}</div>
          <div class="card-sub">node=${oht.current_node || '-'} / edge=${oht.current_edge || '-'}</div>
          <div class="card-sub">carry=${oht.carrying} / blocked=${oht.blocked} / wafer=${oht.wafer_state || '-'}</div>
          <div class="card-sub">pos=${fmtPos(oht.pos)}</div>
          <div class="card-sub">camera=${oht.camera_path || '-'}</div>
        `;
        ohtList.appendChild(div);
      });

      const ur10 = data.ur10 || {};
      const bridge = data.bridge || {};
      ur10Bridge.innerHTML = `
        <div class="card">
          <div class="card-title">UR10 Phase : ${ur10.phase_text || '-'}</div>
          <div class="card-sub">Bridge State : ${bridge.state || '-'} / Queue : ${bridge.queue_len || 0}</div>
          <div class="card-sub">Place Pos : ${fmtPos(ur10.place_position)}</div>
          <div class="card-sub">Bridge Pick : ${fmtPos(bridge.pick_world)}</div>
        </div>
      `;

      const cam = data.camera || {};
      const preview = cam.preview || {};
      const previewInfo = preview.exists ? `mtime=${preview.mtime || '-'} / size=${preview.size || '-'}` : '미리보기 파일 대기 중';
      cameraPanel.innerHTML = `
        <div class="card">
          <div class="card-title">Selected Camera : ${cam.selected_label || 'Overview'}</div>
          <div class="card-sub">path=${cam.selected_path || '-'}</div>
          <div class="preview-wrap">
            <img class="preview-img" src="${previewSrc(data)}" alt="camera preview" />
          </div>
          <div class="preview-note">
            현재 Isaac Sim 활성 뷰포트를 주기적으로 캡처한 프리뷰입니다.<br>
            ${previewInfo}
          </div>
        </div>
      `;

      jobList.innerHTML = '';
      (data.jobs || []).forEach(job => {
        const div = document.createElement('div');
        div.className = 'card';
        div.innerHTML = `<div class="card-sub" style="color:#ebeff4; font-size:12px;">${job}</div>`;
        jobList.appendChild(div);
      });

      alarmList.innerHTML = '';
      (data.alarms || []).forEach(item => {
        const level = item[0] || 'INFO';
        const msg = item[1] || '-';
        const div = document.createElement('div');
        div.className = `alarm ${level}`;
        div.textContent = `[${level}] ${msg}`;
        alarmList.appendChild(div);
      });
    }

    function renderKpis(data) {
      const s = data.summary || {};
      setText('kpiSim', s.sim_playing ? 'RUN' : 'STOP');
      setText('kpiSource', data.connected ? 'source connected' : 'source disconnected');
      setText('kpiOht', s.oht_total || 0);
      setText('kpiCarry', s.carrying_count || 0);
      setText('kpiPickup', s.pickup_stock || 0);
      setText('kpiBridge', s.bridge_state || '-');
      setText('kpiQueue', `queue=${s.bridge_queue || 0}`);
      setText('kpiUr10', s.ur10_phase_text || '-');
      setText('kpiPlaced', `placed=${s.placed_wafer_count || 0}`);

      setText('badgeSim', `SIM ${s.sim_playing ? 'RUN' : 'STOP'}`);
      const simBadge = document.getElementById('badgeSim');
      simBadge.className = `badge ${s.sim_playing ? 'run' : 'stop'}`;
      setText('badgeOht', `OHT ${s.oht_total || 0}`);
      setText('badgeUr10', `UR10 ${s.ur10_phase_text || '-'}`);
      setText('badgeQueue', `QUEUE ${s.bridge_queue || 0}`);
      setText('badgePod', `POD ${s.pickup_stock || 0}`);
      setText('badgeCam', `CAM ${(data.camera && data.camera.selected_label) ? data.camera.selected_label : 'Overview'}`);

      setText('miniBlocked', s.blocked_count || 0);
      setText('miniPlaced', s.placed_wafer_count || 0);
    }

    async function sendCommand(action, extra = {}) {
      try {
        const res = await fetch('/api/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action, ...extra })
        });
        const out = await res.json();
        setText('statusLine', `command sent: ${out.action} / id=${out.id}`);
      } catch (e) {
        setText('statusLine', `command error: ${e}`);
      }
    }

    async function sendCameraCommand(cameraPath, label) {
      if (!cameraPath) return;
      await sendCommand('view_camera', { camera_path: cameraPath, label: label || cameraPath });
    }

    async function sendOverviewCamera() {
      await sendCommand('view_overview_camera', { label: 'Overview' });
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        const data = await res.json();
        lastData = data;
        renderKpis(data);
        renderLists(data);
        drawMap(data);

        const ts = data.timestamp ? new Date(data.timestamp * 1000) : null;
        const age = data.timestamp ? Math.max(0, (Date.now() / 1000) - data.timestamp) : null;
        const tsText = ts ? ts.toLocaleTimeString() : '-';
        const ageText = age !== null ? `${age.toFixed(2)}s` : '-';
        setText('statusLine', `status update: ${tsText} / age=${ageText} / status_path={{ status_path }}`);
      } catch (e) {
        setText('statusLine', `status poll failed: ${e}`);
      }
    }

    document.getElementById('mapCanvas').addEventListener('click', (ev) => {
      const canvas = ev.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const x = (ev.clientX - rect.left) * ratio;
      const y = (ev.clientY - rect.top) * ratio;

      for (const hit of ohtHitboxes) {
        if (x >= hit.x && x <= hit.x + hit.w && y >= hit.y && y <= hit.y + hit.h) {
          sendCameraCommand(hit.cameraPath, hit.label);
          break;
        }
      }
    });

    window.addEventListener('resize', () => {
      if (lastData) drawMap(lastData);
    });

    pollStatus();
    setInterval(pollStatus, POLL_MS);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML,
        poll_ms=POLL_MS,
        status_path=str(STATUS_PATH),
    )


@app.route("/api/status")
def api_status():
    return jsonify(read_status())


@app.route("/api/command", methods=["POST"])
def api_command():
    body = request.get_json(silent=True) or {}
    action = str(body.get("action", "")).strip().lower()
    if action not in {"play", "pause", "reset", "stop", "view_camera", "view_overview_camera"}:
        return jsonify({"ok": False, "error": "unsupported action"}), 400

    payload = {
        "id": uuid.uuid4().hex,
        "action": action,
        "created_at": time.time(),
    }
    if "camera_path" in body:
        payload["camera_path"] = body.get("camera_path")
    if "label" in body:
        payload["label"] = body.get("label")
    atomic_write_json(COMMAND_PATH, payload)
    return jsonify({"ok": True, **payload})


@app.route("/preview/current")
def preview_current():
    if CURRENT_PREVIEW_PATH.exists():
        resp = send_file(str(CURRENT_PREVIEW_PATH), mimetype="image/png", max_age=0)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    return Response(preview_placeholder_svg(), mimetype="image/svg+xml", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
    })


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "status_path": str(STATUS_PATH),
        "command_path": str(COMMAND_PATH),
        "preview_path": str(CURRENT_PREVIEW_PATH),
    })


if __name__ == "__main__":
    ensure_parent(STATUS_PATH)
    ensure_parent(COMMAND_PATH)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=HOST, port=PORT, debug=False)