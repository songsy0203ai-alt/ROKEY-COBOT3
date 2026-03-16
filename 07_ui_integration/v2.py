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

# rosbridge WebSocket URL (클라이언트 브라우저에서 접속하는 주소)
ROSBRIDGE_URL = os.environ.get("ACS_ROSBRIDGE_URL", "ws://localhost:9090")

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

    /* 웨이퍼/검사 전용 버튼 강조 스타일 */
    button.ctrl.wafer-btn {
      background: #1b4060;
      border-color: var(--cyan);
      color: var(--cyan);
    }
    button.ctrl.wafer-btn:hover { background: #1f506e; }
    button.ctrl.defect-btn {
      background: #3d2b10;
      border-color: var(--warn);
      color: var(--warn);
    }
    button.ctrl.defect-btn:hover { background: #4e360f; }

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

    /* =============================================
       MODAL OVERLAY
    ============================================= */
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.72);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }
    .modal-overlay.open { display: flex; }

    .modal-box {
      background: #22282f;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0;
      width: min(960px, 96vw);
      max-height: 92vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 8px 48px rgba(0,0,0,0.7);
      overflow: hidden;
    }

    .modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      background: #1b2027;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .modal-title {
      font-size: 17px;
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .modal-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: var(--red);
      display: inline-block;
    }
    .modal-dot.live { background: var(--green); animation: pulse 1.2s ease-in-out infinite; }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }
    .modal-close {
      background: #3f4a57;
      border: 1px solid var(--border);
      color: var(--text);
      width: 32px; height: 32px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 18px;
      display: flex; align-items: center; justify-content: center;
    }
    .modal-close:hover { background: #576171; }

    .modal-body {
      padding: 16px 18px;
      overflow-y: auto;
      flex: 1;
    }

    /* =============================================
       WAFER CAMERA MODAL
    ============================================= */
    #waferCamModal .feed-wrap {
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    #waferCamModal .feed-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    #waferCamModal .feed-offline {
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: var(--muted);
      font-size: 16px;
      font-weight: 700;
      text-align: center;
      position: absolute;
      inset: 0;
      background: #000;
    }
    #waferCamModal .feed-offline svg { opacity: 0.35; }
    #waferCamModal .feed-meta {
      display: flex;
      gap: 16px;
      margin-top: 12px;
      font-size: 12px;
      color: var(--muted);
      flex-wrap: wrap;
    }
    #waferCamModal .feed-meta span { background: var(--panel2); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); }

    /* =============================================
       DEFECT LOG MODAL
    ============================================= */
    #defectLogModal .log-controls {
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    #defectLogModal .log-filter {
      display: flex;
      gap: 6px;
    }
    .filter-btn {
      background: var(--panel2);
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 5px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }
    .filter-btn.active-all  { border-color: var(--cyan); color: var(--cyan); background: #1a3848; }
    .filter-btn.active-ok   { border-color: var(--ok);   color: var(--ok);   background: #1b3022; }
    .filter-btn.active-ng   { border-color: var(--red);  color: var(--red);  background: #3a1818; }
    .log-count-badge {
      margin-left: auto;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      gap: 10px;
    }
    .log-count-badge span { background: var(--panel2); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); }
    .log-count-badge .cnt-ok  { color: var(--ok); }
    .log-count-badge .cnt-ng  { color: var(--red); }

    #defectLogModal .log-table-wrap {
      width: 100%;
      max-height: 55vh;
      overflow-y: auto;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    #defectLogModal table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    #defectLogModal thead th {
      background: #1b2027;
      color: var(--muted);
      font-weight: 700;
      padding: 10px 14px;
      text-align: left;
      position: sticky;
      top: 0;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    #defectLogModal tbody tr {
      border-bottom: 1px solid #3a4250;
      transition: background 0.1s;
    }
    #defectLogModal tbody tr:hover { background: #2c313a; }
    #defectLogModal tbody td {
      padding: 9px 14px;
      vertical-align: middle;
    }
    .result-ok {
      display: inline-block;
      background: #1b3022;
      color: var(--ok);
      border: 1px solid var(--ok);
      border-radius: 4px;
      padding: 2px 10px;
      font-weight: 800;
      font-size: 12px;
    }
    .result-ng {
      display: inline-block;
      background: #3a1818;
      color: var(--red);
      border: 1px solid var(--red);
      border-radius: 4px;
      padding: 2px 10px;
      font-weight: 800;
      font-size: 12px;
    }
    #defectLogModal .log-status-bar {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 6px;
    }
    .ws-indicator {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
    }
    .ws-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--muted);
      flex-shrink: 0;
    }
    .ws-dot.connected { background: var(--green); animation: pulse 1.2s ease-in-out infinite; }
    .ws-dot.error { background: var(--red); }

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
            <!-- ★ 추가된 두 버튼 -->
            <button class="ctrl wafer-btn" onclick="openWaferCam()">📷 WAFER CAM</button>
            <button class="ctrl defect-btn" onclick="openDefectLog()">🔍 DEFECT LOG</button>
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

  <!-- =============================================
       MODAL 1: WAFER CAMERA LIVE FEED
  ============================================= -->
  <div class="modal-overlay" id="waferCamModal">
    <div class="modal-box">
      <div class="modal-header">
        <div class="modal-title">
          <span class="modal-dot" id="waferCamDot"></span>
          📷 Wafer Detection Camera &nbsp;
          <span style="font-size:13px; color:var(--muted); font-weight:400;">/wafer_camera/Compressed (sensor_msgs/Image)</span>
        </div>
        <button class="modal-close" onclick="closeWaferCam()">✕</button>
      </div>
      <div class="modal-body">
        <div class="feed-wrap">
          <!-- sensor_msgs/Image raw 데이터를 Canvas에 직접 렌더링 -->
          <canvas id="waferFeedCanvas" class="feed-img" style="display:none;"></canvas>
          <div class="feed-offline" id="waferOffline" style="display:flex;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#aeb7c3" stroke-width="1.5">
              <circle cx="12" cy="12" r="9"/><line x1="3" y1="3" x2="21" y2="21"/>
            </svg>
            <div>현재 모니터링이 꺼져 있습니다</div>
            <div style="font-size:12px; font-weight:400; color:#576171;">rosbridge 연결 대기 중 또는 토픽 수신 없음</div>
          </div>
        </div>
        <div class="feed-meta">
          <span>Topic: /wafer_camera/Compressed (raw Image)</span>
          <span id="waferFps">FPS: -</span>
          <span id="waferResolution">해상도: -</span>
          <span id="waferLastRecv">마지막 수신: -</span>
          <span class="ws-indicator">
            <span class="ws-dot" id="waferWsDot"></span>
            <span id="waferWsStatus">rosbridge 연결 중...</span>
          </span>
        </div>
      </div>
    </div>
  </div>

  <!-- =============================================
       MODAL 2: DEFECT DETECTION LOG
  ============================================= -->
  <div class="modal-overlay" id="defectLogModal">
    <div class="modal-box">
      <div class="modal-header">
        <div class="modal-title">
          <span class="modal-dot" id="defectLogDot"></span>
          🔍 Defect Detection Log &nbsp;
          <span style="font-size:13px; color:var(--muted); font-weight:400;">/def_det_result</span>
        </div>
        <button class="modal-close" onclick="closeDefectLog()">✕</button>
      </div>
      <div class="modal-body">
        <div class="log-controls">
          <div class="log-filter">
            <button class="filter-btn active-all" id="fAll"  onclick="setFilter('ALL')">전체</button>
            <button class="filter-btn"             id="fOk"   onclick="setFilter('OK')">정상</button>
            <button class="filter-btn"             id="fNg"   onclick="setFilter('NG')">불량</button>
          </div>
          <div class="log-count-badge">
            <span>전체 <b id="cntTotal">0</b></span>
            <span class="cnt-ok">정상 <b id="cntOk">0</b></span>
            <span class="cnt-ng">불량 <b id="cntNg">0</b></span>
          </div>
          <button class="filter-btn" style="margin-left:auto;" onclick="clearDefectLog()">🗑 로그 초기화</button>
        </div>

        <div class="log-table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>송신 시간</th>
                <th>경과</th>
                <th>결과</th>
                <th>원본 메시지</th>
              </tr>
            </thead>
            <tbody id="defectLogBody">
              <tr>
                <td colspan="5" style="color:var(--muted); text-align:center; padding:24px;">
                  데이터 대기 중... (rosbridge 연결 필요)
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="log-status-bar">
          <span class="ws-indicator">
            <span class="ws-dot" id="defectWsDot"></span>
            <span id="defectWsStatus">rosbridge 연결 중...</span>
          </span>
          <span id="defectLastMsg">마지막 수신: -</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    const POLL_MS = {{ poll_ms }};
    const ROSBRIDGE_URL = "{{ rosbridge_url }}";
    let lastData = null;
    let ohtHitboxes = [];

    // ── rosbridge WebSocket 인스턴스 (모달별 독립 관리) ──────────────────
    let waferWs = null;
    let defectWs = null;

    // ── 웨이퍼 카메라 상태 ────────────────────────────────────────────────
    let waferFrameCount = 0;
    let waferFpsTimer = null;
    let waferLastRecvTime = null;
    let waferTimeoutTimer = null;
    const WAFER_TIMEOUT_MS = 3000; // 3초 이상 프레임 없으면 오프라인 처리

    // ── 불량 검출 로그 상태 ───────────────────────────────────────────────
    let defectLogs = [];          // { seq, time, elapsed, result, raw }
    let defectFilter = 'ALL';     // 'ALL' | 'OK' | 'NG'
    let defectSeq = 0;

    // ─────────────────────────────────────────────────────────────────────
    // 공통 유틸
    // ─────────────────────────────────────────────────────────────────────
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
    function nowStr() {
      return new Date().toLocaleTimeString('ko-KR', { hour12: false }) + '.' +
             String(new Date().getMilliseconds()).padStart(3, '0');
    }
    function elapsedStr(ts) {
      const s = ((Date.now() - ts) / 1000).toFixed(1);
      return `${s}s 전`;
    }

    // ─────────────────────────────────────────────────────────────────────
    // rosbridge 구독 헬퍼
    //   rosbridgeLib가 없으면 직접 JSON 프로토콜 사용
    // ─────────────────────────────────────────────────────────────────────
    function rosbridgeSubscribe(ws, topic, msgType, callback) {
      const subMsg = JSON.stringify({
        op: 'subscribe',
        topic: topic,
        type: msgType,
      });
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(subMsg);
      } else {
        ws.addEventListener('open', () => ws.send(subMsg), { once: true });
      }
      ws.addEventListener('message', (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.topic === topic && data.msg) {
            callback(data.msg);
          }
        } catch (_) {}
      });
    }

    function rosbridgeUnsubscribe(ws, topic) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ op: 'unsubscribe', topic: topic }));
      }
    }

    // ─────────────────────────────────────────────────────────────────────
    // MODAL 1: WAFER CAMERA
    // ─────────────────────────────────────────────────────────────────────
    function openWaferCam() {
      document.getElementById('waferCamModal').classList.add('open');
      startWaferCam();
    }

    function closeWaferCam() {
      document.getElementById('waferCamModal').classList.remove('open');
      stopWaferCam();
    }

    function startWaferCam() {
      stopWaferCam(); // 기존 연결 정리

      setWaferWsStatus('connecting');
      try {
        waferWs = new WebSocket(ROSBRIDGE_URL);
      } catch (e) {
        setWaferWsStatus('error');
        showWaferOffline();
        return;
      }

      waferWs.onopen = () => {
        setWaferWsStatus('connected');
        rosbridgeSubscribe(waferWs, '/wafer_camera/Compressed', 'sensor_msgs/Image', onWaferFrame);
      };

      waferWs.onerror = () => {
        setWaferWsStatus('error');
        showWaferOffline();
      };

      waferWs.onclose = () => {
        setWaferWsStatus('disconnected');
        showWaferOffline();
      };

      // FPS 카운터 (1초마다 갱신)
      waferFpsTimer = setInterval(() => {
        setText('waferFps', `FPS: ${waferFrameCount}`);
        waferFrameCount = 0;
      }, 1000);

      // 타임아웃 워치독 (첫 프레임 대기)
      resetWaferTimeout();
    }

    function stopWaferCam() {
      if (waferWs) {
        rosbridgeUnsubscribe(waferWs, '/wafer_camera/Compressed');
        waferWs.close();
        waferWs = null;
      }
      if (waferFpsTimer) { clearInterval(waferFpsTimer); waferFpsTimer = null; }
      if (waferTimeoutTimer) { clearTimeout(waferTimeoutTimer); waferTimeoutTimer = null; }
      waferFrameCount = 0;
    }

    function onWaferFrame(msg) {
      // sensor_msgs/Image 필드: width, height, encoding, is_bigendian, step, data(base64)
      waferFrameCount++;
      waferLastRecvTime = Date.now();
      resetWaferTimeout();

      const w = msg.width;
      const h = msg.height;
      const encoding = (msg.encoding || 'rgb8').toLowerCase();

      if (!w || !h || !msg.data) return;

      // base64 → Uint8Array
      let raw;
      try {
        const bin = atob(msg.data);
        raw = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) raw[i] = bin.charCodeAt(i);
      } catch (e) {
        console.warn('wafer frame decode error', e);
        return;
      }

      const canvas = document.getElementById('waferFeedCanvas');
      canvas.width  = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(w, h);
      const px = imgData.data; // RGBA flat array

      const n = w * h;

      if (encoding === 'rgb8') {
        // 3 bytes/pixel: R G B
        for (let i = 0; i < n; i++) {
          px[i * 4]     = raw[i * 3];
          px[i * 4 + 1] = raw[i * 3 + 1];
          px[i * 4 + 2] = raw[i * 3 + 2];
          px[i * 4 + 3] = 255;
        }
      } else if (encoding === 'bgr8') {
        // 3 bytes/pixel: B G R  → swap to R G B
        for (let i = 0; i < n; i++) {
          px[i * 4]     = raw[i * 3 + 2];
          px[i * 4 + 1] = raw[i * 3 + 1];
          px[i * 4 + 2] = raw[i * 3];
          px[i * 4 + 3] = 255;
        }
      } else if (encoding === 'rgba8') {
        // 4 bytes/pixel: R G B A
        for (let i = 0; i < n; i++) {
          px[i * 4]     = raw[i * 4];
          px[i * 4 + 1] = raw[i * 4 + 1];
          px[i * 4 + 2] = raw[i * 4 + 2];
          px[i * 4 + 3] = raw[i * 4 + 3];
        }
      } else if (encoding === 'bgra8') {
        // 4 bytes/pixel: B G R A
        for (let i = 0; i < n; i++) {
          px[i * 4]     = raw[i * 4 + 2];
          px[i * 4 + 1] = raw[i * 4 + 1];
          px[i * 4 + 2] = raw[i * 4];
          px[i * 4 + 3] = raw[i * 4 + 3];
        }
      } else if (encoding === 'mono8' || encoding === '8uc1') {
        // 1 byte/pixel: grayscale
        for (let i = 0; i < n; i++) {
          const v = raw[i];
          px[i * 4]     = v;
          px[i * 4 + 1] = v;
          px[i * 4 + 2] = v;
          px[i * 4 + 3] = 255;
        }
      } else if (encoding === 'mono16' || encoding === '16uc1') {
        // 2 bytes/pixel: 16-bit grayscale → scale to 8-bit
        for (let i = 0; i < n; i++) {
          const v = Math.min(255, (raw[i * 2] | (raw[i * 2 + 1] << 8)) >> 8);
          px[i * 4]     = v;
          px[i * 4 + 1] = v;
          px[i * 4 + 2] = v;
          px[i * 4 + 3] = 255;
        }
      } else {
        // 알 수 없는 인코딩: 첫 채널만 그레이스케일로 표시
        const step = msg.step || w * 3;
        const bytesPerPixel = Math.floor(step / w);
        for (let i = 0; i < n; i++) {
          const v = raw[i * bytesPerPixel] || 0;
          px[i * 4]     = v;
          px[i * 4 + 1] = v;
          px[i * 4 + 2] = v;
          px[i * 4 + 3] = 255;
        }
      }

      ctx.putImageData(imgData, 0, 0);

      canvas.style.display = 'block';
      document.getElementById('waferOffline').style.display = 'none';
      document.getElementById('waferCamDot').classList.add('live');

      setText('waferResolution', `해상도: ${w}×${h} (${encoding})`);
      setText('waferLastRecv', `마지막 수신: ${nowStr()}`);
    }

    function resetWaferTimeout() {
      if (waferTimeoutTimer) clearTimeout(waferTimeoutTimer);
      waferTimeoutTimer = setTimeout(() => {
        showWaferOffline();
      }, WAFER_TIMEOUT_MS);
    }

    function showWaferOffline() {
      document.getElementById('waferFeedCanvas').style.display = 'none';
      document.getElementById('waferOffline').style.display = 'flex';
      document.getElementById('waferCamDot').classList.remove('live');
    }

    function setWaferWsStatus(state) {
      const dot = document.getElementById('waferWsDot');
      const txt = document.getElementById('waferWsStatus');
      dot.className = 'ws-dot';
      if (state === 'connected')    { dot.classList.add('connected'); txt.textContent = 'rosbridge 연결됨'; }
      else if (state === 'error')   { dot.classList.add('error');     txt.textContent = 'rosbridge 연결 오류'; }
      else if (state === 'disconnected') { txt.textContent = 'rosbridge 연결 끊김'; }
      else                          { txt.textContent = 'rosbridge 연결 중...'; }
    }

    // ─────────────────────────────────────────────────────────────────────
    // MODAL 2: DEFECT LOG
    // ─────────────────────────────────────────────────────────────────────
    function openDefectLog() {
      document.getElementById('defectLogModal').classList.add('open');
      startDefectLog();
    }

    function closeDefectLog() {
      document.getElementById('defectLogModal').classList.remove('open');
      stopDefectLog();
    }

    function startDefectLog() {
      stopDefectLog();

      setDefectWsStatus('connecting');
      try {
        defectWs = new WebSocket(ROSBRIDGE_URL);
      } catch (e) {
        setDefectWsStatus('error');
        return;
      }

      defectWs.onopen = () => {
        setDefectWsStatus('connected');
        rosbridgeSubscribe(defectWs, '/def_det_result', 'std_msgs/String', onDefectMsg);
      };

      defectWs.onerror = () => setDefectWsStatus('error');
      defectWs.onclose = () => setDefectWsStatus('disconnected');
    }

    function stopDefectLog() {
      if (defectWs) {
        rosbridgeUnsubscribe(defectWs, '/def_det_result');
        defectWs.close();
        defectWs = null;
      }
    }

    function onDefectMsg(msg) {
      // msg.data: "정상" or "불량" (std_msgs/String)
      const raw = (msg.data || '').trim();
      const isOk = raw === '정상';
      const isNg = raw === '불량';
      const result = isOk ? 'OK' : isNg ? 'NG' : 'UNKNOWN';

      defectSeq++;
      const entry = {
        seq: defectSeq,
        time: new Date(),
        tsMs: Date.now(),
        result,
        raw,
      };
      defectLogs.unshift(entry);        // 최신 항목이 위에 오도록
      if (defectLogs.length > 500) defectLogs.pop();  // 최대 500건 보관

      document.getElementById('defectLogDot').classList.toggle('live', true);
      setText('defectLastMsg', `마지막 수신: ${nowStr()}`);

      renderDefectLog();
    }

    function renderDefectLog() {
      const filtered = defectFilter === 'ALL'
        ? defectLogs
        : defectLogs.filter(e => e.result === defectFilter);

      const okCount = defectLogs.filter(e => e.result === 'OK').length;
      const ngCount = defectLogs.filter(e => e.result === 'NG').length;

      setText('cntTotal', defectLogs.length);
      setText('cntOk', okCount);
      setText('cntNg', ngCount);

      const tbody = document.getElementById('defectLogBody');
      if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="color:var(--muted);text-align:center;padding:24px;">
          ${defectLogs.length === 0 ? '데이터 대기 중... (rosbridge 연결 필요)' : '해당 필터에 해당하는 항목 없음'}
        </td></tr>`;
        return;
      }

      tbody.innerHTML = filtered.map(e => {
        const timeStr = e.time.toLocaleTimeString('ko-KR', { hour12: false }) + '.' +
                        String(e.time.getMilliseconds()).padStart(3, '0');
        const elapsed = elapsedStr(e.tsMs);
        const badge = e.result === 'OK'
          ? `<span class="result-ok">정상</span>`
          : e.result === 'NG'
          ? `<span class="result-ng">불량</span>`
          : `<span style="color:var(--muted)">UNKNOWN</span>`;
        return `<tr>
          <td style="color:var(--muted); width:50px;">${e.seq}</td>
          <td style="font-family:monospace; white-space:nowrap;">${timeStr}</td>
          <td style="color:var(--muted); white-space:nowrap;">${elapsed}</td>
          <td>${badge}</td>
          <td style="color:var(--muted); font-size:12px;">${escHtml(e.raw)}</td>
        </tr>`;
      }).join('');
    }

    function escHtml(str) {
      return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function setFilter(f) {
      defectFilter = f;
      document.getElementById('fAll').className = 'filter-btn' + (f === 'ALL' ? ' active-all' : '');
      document.getElementById('fOk').className  = 'filter-btn' + (f === 'OK'  ? ' active-ok'  : '');
      document.getElementById('fNg').className  = 'filter-btn' + (f === 'NG'  ? ' active-ng'  : '');
      renderDefectLog();
    }

    function clearDefectLog() {
      defectLogs = [];
      defectSeq = 0;
      document.getElementById('defectLogDot').classList.remove('live');
      renderDefectLog();
    }

    function setDefectWsStatus(state) {
      const dot = document.getElementById('defectWsDot');
      const txt = document.getElementById('defectWsStatus');
      dot.className = 'ws-dot';
      if (state === 'connected')         { dot.classList.add('connected'); txt.textContent = 'rosbridge 연결됨'; }
      else if (state === 'error')        { dot.classList.add('error');     txt.textContent = 'rosbridge 연결 오류'; }
      else if (state === 'disconnected') {                                 txt.textContent = 'rosbridge 연결 끊김'; }
      else                               {                                 txt.textContent = 'rosbridge 연결 중...'; }
    }

    // 모달 외부 클릭 시 닫기
    document.getElementById('waferCamModal').addEventListener('click', function(e) {
      if (e.target === this) closeWaferCam();
    });
    document.getElementById('defectLogModal').addEventListener('click', function(e) {
      if (e.target === this) closeDefectLog();
    });

    // ─────────────────────────────────────────────────────────────────────
    // 기존 맵 / 상태 폴링 코드 (변경 없음)
    // ─────────────────────────────────────────────────────────────────────
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
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
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
        rosbridge_url=ROSBRIDGE_URL,
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
        "rosbridge_url": ROSBRIDGE_URL,
    })


if __name__ == "__main__":
    ensure_parent(STATUS_PATH)
    ensure_parent(COMMAND_PATH)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=HOST, port=PORT, debug=False)