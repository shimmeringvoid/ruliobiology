"""
ruliobiology.viewer3d -- an interactive, rotatable 3-D view of a 2-D
rulioorganism's space-time volume (t, y, x), written as a single
self-contained HTML file.

    import ruliobiology as rb
    ev = rb.evolve2d(steps=20000, seed=1)
    rb.write_viewer(rb.grow2d(ev.rule), "organism.html", title="Lineage 1")

Open the file in any browser (three.js is loaded from cdnjs).  Drag to
rotate, wheel / pinch to zoom.  Each cell is drawn as a small sphere at the
centre of its lattice cube (or as a cube, or as a shaded dot sprite, which
is the fastest for very large volumes), with a size slider; each colour has
its own opacity slider and visibility toggle; there is a cut plane along x
or y, a time window, a "surface cells only" mode that keeps just the
boundaries between colour regions, and a time-stretch slider -- so you can
look inside the organism in whatever way helps.  Two lattice overlays help
to read the 3-D structure: the empty lattice sites can be drawn as faint
beads (an Indra's net the organism's cells are the jewels of), and "world
lines" join each cell to its successor in time while its colour persists.
Long-lived organisms make tall, thin volumes: the "stretch space" slider
spreads the two spatial axes out and "stretch time" compresses the other,
so any aspect ratio can be had.

Any (T, H, W) uint8 volume with 0 = empty works, e.g. an age-coloured
Game of Life history from ``cadyn.ca2d``.
"""
from __future__ import annotations

import base64
import json
from typing import Optional, Sequence

import numpy as np

from .ca1d import PALETTE

THREE_JS = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def rle_encode(volume: np.ndarray) -> str:
    """Run-length encode a uint8 volume as base64 of (value, run) records:
    1 byte value + 2 bytes little-endian run length."""
    flat = np.ascontiguousarray(volume, dtype=np.uint8).ravel()
    if flat.size == 0:
        return ""
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [flat.size]])
    out = bytearray()
    for s, e in zip(starts, ends):
        v = int(flat[s])
        n = int(e - s)
        while n > 0:
            run = min(n, 65535)
            out += bytes((v, run & 0xFF, run >> 8))
            n -= run
    return base64.b64encode(bytes(out)).decode("ascii")


def write_viewer(volume: np.ndarray, path: Optional[str] = None, title: str = "Rulioorganism Viewer",
                 palette: Sequence[str] = PALETTE, names: Optional[Sequence[str]] = None,
                 caption: str = "", standalone: bool = True) -> str:
    """Build the viewer page for a (T, H, W) volume; write it to ``path`` if
    given; return the HTML.  ``standalone=False`` omits the document
    skeleton (for hosts that add their own)."""
    vol = np.asarray(volume, dtype=np.uint8)
    if vol.ndim != 3:
        raise ValueError("volume must be (T, H, W)")
    T, H, W = vol.shape
    k = int(vol.max()) + 1
    palette = list(palette)[:max(k, 2)]
    names = list(names) if names else ["empty", "red", "yellow", "green", "purple", "blue",
                                       "orange", "teal", "brown", "pink"][:len(palette)]
    cells = int((vol > 0).sum())
    data = {
        "T": T, "H": H, "W": W, "k": len(palette), "palette": palette, "names": names,
        "cells": cells, "title": title, "caption": caption, "rle": rle_encode(vol),
    }
    head = _HEAD.replace("__TITLE__", _esc(title))
    body = (_BODY.replace("__TITLE__", _esc(title))
                 .replace("__CAPTION__", _esc(caption))
                 .replace("__STATS__", f"lifetime {T} · {W} × {H} cells across · {cells:,} cells in all")
                 .replace("__THREE__", THREE_JS)
                 .replace("__DATA__", json.dumps(data, separators=(",", ":")))
                 .replace("__SCRIPT__", _SCRIPT))
    if standalone:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                f'{head}</head>\n<body>\n{body}</body>\n</html>\n')
    else:
        html = head + body
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


_HEAD = """<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
  /* layout: the space-time volume fills the screen; a slim instrument panel is docked at the right
     (it becomes a bottom sheet on phones).  A single committed dark look, like a lab monitor. */
  :root {
    --bg: #0c0f15; --panel: #151a24; --edge: #262d3c; --fg: #e8ebf1; --muted: #8e97ab;
    --accent: #f2c21b; --track: #2b3344; --thumb: #e8ebf1;
    --font-ui: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
    --font-data: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    color-scheme: dark;
  }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 var(--font-ui); overflow: hidden; }
  #stage { position: fixed; inset: 0 292px 0 0; }
  #stage canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
  #stage canvas:active { cursor: grabbing; }
  #panel {
    position: fixed; top: 0; right: 0; bottom: 0; width: 292px; box-sizing: border-box;
    padding-top: calc(16px + env(safe-area-inset-top, 0px));
    padding-bottom: calc(16px + env(safe-area-inset-bottom, 0px));
    padding-left: 16px; padding-right: max(16px, env(safe-area-inset-right, 0px));
    background: var(--panel); border-left: 1px solid var(--edge); overflow-y: auto;
    display: flex; flex-direction: column; gap: 18px;
  }
  @media (max-width: 640px) {
    #stage { inset: 0 0 46% 0; }
    #panel { top: auto; left: 0; width: auto; max-height: 46%; border-left: 0; border-top: 1px solid var(--edge);
             padding-left: max(16px, env(safe-area-inset-left, 0px)); gap: 14px; }
  }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 2px; text-wrap: balance; }
  .stats { font: 12px/1.4 var(--font-data); color: var(--muted); font-variant-numeric: tabular-nums; }
  .caption { color: var(--muted); font-size: 13px; margin: 4px 0 0; }
  .group { display: flex; flex-direction: column; gap: 8px; }
  .label { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
  .row { display: grid; grid-template-columns: 14px 56px 1fr 38px auto; align-items: center; gap: 8px; }
  .row.two { grid-template-columns: 1fr 1fr; }
  .row.wide { grid-template-columns: 1fr auto; }
  .swatch { width: 14px; height: 14px; border-radius: 3px; }
  .name { font: 13px var(--font-data); }
  .val { font: 12px var(--font-data); color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; min-width: 0; accent-color: var(--accent); margin: 0; }
  input[type=checkbox] { accent-color: var(--accent); width: 16px; height: 16px; margin: 0; }
  select, button {
    font: 13px var(--font-ui); color: var(--fg); background: var(--track); border: 1px solid var(--edge);
    border-radius: 6px; padding: 5px 8px;
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--muted); }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .hint { color: var(--muted); font-size: 12px; margin-top: auto; }
  .check { display: flex; align-items: center; gap: 8px; }
</style>
"""

_BODY = """<div id="stage"></div>
<aside id="panel">
  <div>
    <h1>__TITLE__</h1>
    <div class="stats">__STATS__</div>
    <div class="stats" id="rendered"></div>
    <p class="caption">__CAPTION__</p>
  </div>
  <div class="group">
    <div class="label">Colours · opacity</div>
    <div id="colours"></div>
  </div>
  <div class="group">
    <div class="label">Cells</div>
    <div class="row wide"><span>Shape</span>
      <select id="shape"><option value="sphere">sphere</option><option value="cube">cube</option><option value="dot">dot (fastest)</option></select></div>
    <div class="row wide"><span>Size</span><span class="val" id="sizeVal">80%</span></div>
    <input type="range" id="size" min="15" max="100" value="80">
  </div>
  <div class="group">
    <div class="label">Lattice</div>
    <label class="check"><input type="checkbox" id="net"> empty sites as beads</label>
    <div class="row two"><input type="range" id="netSize" min="4" max="40" value="14" aria-label="bead size"><input type="range" id="netOp" min="2" max="60" value="14" aria-label="bead opacity"></div>
    <label class="check"><input type="checkbox" id="lines"> world lines (cell keeps its colour)</label>
    <input type="range" id="lineOp" min="5" max="100" value="60" aria-label="world line opacity">
  </div>
  <div class="group">
    <div class="label">Look inside</div>
    <label class="check"><input type="checkbox" id="shell"> surface cells only</label>
    <div class="row wide"><span>Cut along</span>
      <select id="cutAxis"><option value="none">none</option><option value="x">x</option><option value="y">y</option></select></div>
    <div class="row wide"><input type="range" id="cut" min="0" max="1000" value="500"><span class="val" id="cutVal"></span></div>
  </div>
  <div class="group">
    <div class="label">Time window</div>
    <div class="row wide"><input type="range" id="tmin" min="0" value="0"><span class="val" id="tminVal">0</span></div>
    <div class="row wide"><input type="range" id="tmax" min="0" value="0"><span class="val" id="tmaxVal">0</span></div>
    <div class="row wide"><span>Stretch time</span><span class="val" id="stretchVal">1.0×</span></div>
    <input type="range" id="stretch" min="25" max="400" value="100">
    <div class="row wide"><span>Stretch space (x, y)</span><span class="val" id="sxVal">1.0×</span></div>
    <input type="range" id="sx" min="100" max="800" value="100">
  </div>
  <div class="group">
    <div class="row two">
      <label class="check"><input type="checkbox" id="spin"> auto-rotate</label>
      <button id="reset" type="button">Reset view</button>
    </div>
  </div>
  <div class="hint">Drag to rotate · wheel or pinch to zoom · time runs downward</div>
</aside>
<script src="__THREE__"></script>
<script>
const DATA = __DATA__;
__SCRIPT__
</script>
"""

_SCRIPT = r"""
(function () {
  if (typeof THREE === 'undefined') {
    const el = document.getElementById('stage');
    el.innerHTML = '<p style="margin:24px;color:#8e97ab;font:14px/1.5 system-ui">three.js could not be loaded from ' +
      'cdnjs.cloudflare.com, so the 3-D view cannot be drawn. Check the internet connection and reload.</p>';
    return;
  }
  const D = DATA, T = D.T, H = D.H, W = D.W, HW = H * W, N = T * HW;
  // ---- decode the run-length-encoded volume --------------------------------
  const vol = new Uint8Array(N);
  (function () {
    const bin = atob(D.rle); let pos = 0;
    for (let i = 0; i + 2 < bin.length; i += 3) {
      const v = bin.charCodeAt(i), len = bin.charCodeAt(i + 1) | (bin.charCodeAt(i + 2) << 8);
      if (v) vol.fill(v, pos, pos + len);
      pos += len;
    }
  })();
  // surface cells: any of the 6 face neighbours is empty, outside, or a different colour,
  // so the boundaries between colour regions ("organs") survive when the interior is dropped
  const shell = new Uint8Array(N);
  let nShell = 0;
  for (let t = 0; t < T; t++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = t * HW + y * W + x, v = vol[i];
    if (!v) continue;
    if (t === 0 || t === T - 1 || y === 0 || y === H - 1 || x === 0 || x === W - 1 ||
        vol[i - HW] !== v || vol[i + HW] !== v || vol[i - W] !== v || vol[i + W] !== v ||
        vol[i - 1] !== v || vol[i + 1] !== v) { shell[i] = 1; nShell++; }
  }
  const byColour = []; for (let c = 0; c < D.k; c++) byColour.push([]);
  for (let i = 0; i < N; i++) if (vol[i]) byColour[vol[i]].push(i);
  let dirty = true;

  // ---- scene -----------------------------------------------------------------
  const stage = document.getElementById('stage');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#0c0f15');
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 20000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.localClippingEnabled = true;
  stage.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(3, 5, 2); scene.add(sun);
  const sun2 = new THREE.DirectionalLight(0xffffff, 0.3); sun2.position.set(-4, -2, -3); scene.add(sun2);

  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const state = { stretch: 1, sx: 1, shell: D.cells > 150000 && nShell < D.cells, cutAxis: 'none', cut: 0.5,
                  tmin: 0, tmax: T - 1, spin: !reduced, shape: D.cells > 100000 ? 'dot' : 'sphere', size: 0.8,
                  net: false, netSize: 0.14, netOp: 0.14, lines: false, lineOp: 0.6,
                  opacity: new Array(D.k).fill(1), visible: new Array(D.k).fill(true) };
  const planes = { cut: new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e9),
                   top: new THREE.Plane(new THREE.Vector3(0, -1, 0), 1e9),
                   bot: new THREE.Plane(new THREE.Vector3(0, 1, 0), 1e9) };
  const clip = [planes.cut, planes.top, planes.bot];
  const meshes = [];
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
                                     new THREE.LineBasicMaterial({ color: 0x3a4357 }));
  scene.add(box);

  function worldY(t) { return -t * state.stretch; }
  function placeBox() {
    box.scale.set(W * state.sx, Math.max(T - 1, 0) * state.stretch + 1, H * state.sx);
    box.position.set(0, -(T - 1) * state.stretch / 2, 0);
  }
  function cellXYZ(i) {
    const t = Math.floor(i / HW), r = i % HW;
    return [(r % W - (W - 1) / 2) * state.sx, worldY(t), (Math.floor(r / W) - (H - 1) / 2) * state.sx];
  }
  function setMatrices(mesh, list) {
    const m = new THREE.Matrix4();
    for (let j = 0; j < list.length; j++) { const p = cellXYZ(list[j]); m.makeTranslation(p[0], p[1], p[2]); mesh.setMatrixAt(j, m); }
    mesh.instanceMatrix.needsUpdate = true;
  }
  function setPositions(points, list) {
    const a = points.geometry.getAttribute('position').array;
    for (let j = 0; j < list.length; j++) { const p = cellXYZ(list[j]); a[3 * j] = p[0]; a[3 * j + 1] = p[1]; a[3 * j + 2] = p[2]; }
    points.geometry.getAttribute('position').needsUpdate = true;
    points.geometry.computeBoundingSphere();
  }
  // a pre-shaded ball sprite for 'dot' mode; the material colour tints it
  const sprite = (function () {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'), grad = g.createRadialGradient(24, 22, 2, 32, 32, 30);
    grad.addColorStop(0, '#ffffff'); grad.addColorStop(0.3, '#d8d8d8'); grad.addColorStop(1, '#505050');
    g.fillStyle = grad; g.beginPath(); g.arc(32, 32, 30, 0, Math.PI * 2); g.fill();
    return new THREE.CanvasTexture(c);
  })();
  function dotSize() { return state.size / Math.tan(camera.fov * Math.PI / 360); }   // world size -> PointsMaterial.size
  function build() {
    for (const e of meshes) if (e) { scene.remove(e.mesh); e.mesh.geometry.dispose(); e.mesh.material.dispose(); }
    meshes.length = 0;
    const s = state.size;
    const geo = state.shape === 'cube' ? new THREE.BoxGeometry(s, s, s)
              : state.shape === 'sphere' ? new THREE.IcosahedronGeometry(s / 2, 1) : null;
    for (let c = 1; c < D.k; c++) {
      const list = state.shell ? byColour[c].filter(i => shell[i]) : byColour[c];
      if (!list.length) { meshes[c] = null; continue; }
      let mesh;
      if (state.shape === 'dot') {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(list.length * 3), 3));
        const mat = new THREE.PointsMaterial({ color: new THREE.Color(D.palette[c]), map: sprite, size: dotSize(),
                                              sizeAttenuation: true, transparent: true, alphaTest: 0.02,
                                              opacity: state.opacity[c], clippingPlanes: clip });
        mesh = new THREE.Points(g, mat);
        setPositions(mesh, list);
      } else {
        const mat = new THREE.MeshLambertMaterial({ color: new THREE.Color(D.palette[c]), transparent: true,
                                                    opacity: state.opacity[c], clippingPlanes: clip });
        mesh = new THREE.InstancedMesh(geo, mat, list.length);
        setMatrices(mesh, list);
      }
      meshes[c] = { mesh, list };
      scene.add(mesh);
    }
    applyMaterials(); placeBox(); applyPlanes(); report();
  }
  function report() {
    let n = 0; for (const e of meshes) if (e) n += e.list.length;
    document.getElementById('rendered').textContent = `${n.toLocaleString()} ${state.shape}s drawn` + (state.shell ? ' (surfaces only)' : '')
      + (net ? `, ${(net.geometry.getAttribute('position').count).toLocaleString()} beads` : '')
      + (lines ? `, ${(lines.geometry.getAttribute('position').count / 2).toLocaleString()} world lines` : '');
  }
  // Indra's net: every empty lattice site in the bounding box as a faint bead
  let net = null, lines = null;
  function buildNet() {
    if (net) { scene.remove(net); net.geometry.dispose(); net.material.dispose(); net = null; }
    if (!state.net) return;
    let n = 0; for (let i = 0; i < N; i++) if (!vol[i]) n++;
    const a = new Float32Array(n * 3); let j = 0;
    for (let i = 0; i < N; i++) if (!vol[i]) { const p = cellXYZ(i); a[j++] = p[0]; a[j++] = p[1]; a[j++] = p[2]; }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(a, 3));
    net = new THREE.Points(g, new THREE.PointsMaterial({ color: new THREE.Color('#aeb6c8'), map: sprite,
          size: state.netSize / Math.tan(camera.fov * Math.PI / 360), sizeAttenuation: true, transparent: true,
          alphaTest: 0.02, opacity: state.netOp, depthWrite: false, clippingPlanes: clip }));
    net.renderOrder = 2;
    scene.add(net);
  }
  // world lines: a thread from each cell to its successor when the colour persists
  function buildLines() {
    if (lines) { scene.remove(lines); lines.geometry.dispose(); lines.material.dispose(); lines = null; }
    if (!state.lines) return;
    const pos = [], col = [], tmp = new THREE.Color();
    for (let i = 0; i + HW < N; i++) {
      const v = vol[i];
      if (!v || vol[i + HW] !== v) continue;
      const p = cellXYZ(i), q = cellXYZ(i + HW);
      pos.push(p[0], p[1], p[2], q[0], q[1], q[2]);
      tmp.set(D.palette[v]); col.push(tmp.r, tmp.g, tmp.b, tmp.r, tmp.g, tmp.b);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
             opacity: state.lineOp, clippingPlanes: clip }));
    scene.add(lines);
  }
  function applyMaterials() {
    for (let c = 1; c < D.k; c++) {
      const e = meshes[c]; if (!e) continue;
      const o = state.opacity[c];
      e.mesh.visible = state.visible[c] && o > 0;
      e.mesh.material.opacity = o;
      e.mesh.material.depthWrite = o > 0.98;
      e.mesh.renderOrder = o > 0.98 ? 0 : 1;
    }
    dirty = true;
  }
  function applyPlanes() {
    planes.top.constant = worldY(state.tmin) + 0.5;    // keep y <= top of the window
    planes.bot.constant = -worldY(state.tmax) + 0.5;   // keep y >= bottom of the window
    if (state.cutAxis === 'none') { planes.cut.constant = 1e9; }
    else {
      const n = state.cutAxis === 'x' ? W : H;
      const at = (-n / 2 + state.cut * n) * state.sx;
      planes.cut.normal.set(state.cutAxis === 'x' ? -1 : 0, 0, state.cutAxis === 'y' ? -1 : 0);
      planes.cut.constant = at;
    }
    dirty = true;
  }
  function restretch() {
    for (let c = 1; c < D.k; c++) if (meshes[c]) {
      if (meshes[c].mesh.isPoints) setPositions(meshes[c].mesh, meshes[c].list); else setMatrices(meshes[c].mesh, meshes[c].list);
    }
    buildNet(); buildLines();
    placeBox(); applyPlanes(); target.y = -(T - 1) * state.stretch / 2;
  }

  // ---- camera: simple orbit --------------------------------------------------
  const target = new THREE.Vector3(0, -(T - 1) / 2, 0);
  const home = { theta: 0.7, phi: 1.15 };
  let theta = home.theta, phi = home.phi, radius = 1;
  function fit() { radius = 1.5 * Math.max(W * state.sx, H * state.sx, (T - 1) * state.stretch + 1, 4); }
  function updateCamera() {
    camera.position.set(target.x + radius * Math.sin(phi) * Math.sin(theta),
                        target.y + radius * Math.cos(phi),
                        target.z + radius * Math.sin(phi) * Math.cos(theta));
    camera.lookAt(target);
  }
  const canvas = renderer.domElement, pointers = new Map(); let lastPinch = 0;
  canvas.addEventListener('pointerdown', e => { pointers.set(e.pointerId, { x: e.clientX, y: e.clientY }); canvas.setPointerCapture(e.pointerId); state.spin = false; spinBox.checked = false; });
  canvas.addEventListener('pointermove', e => {
    if (!pointers.has(e.pointerId)) return;
    const p = pointers.get(e.pointerId), dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    dirty = true;
    if (pointers.size === 1) { theta -= dx * 0.006; phi = Math.min(Math.PI - 0.05, Math.max(0.05, phi - dy * 0.006)); }
    else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (lastPinch) radius *= lastPinch / Math.max(d, 1);
      lastPinch = d;
    }
  });
  const endPointer = e => { pointers.delete(e.pointerId); if (pointers.size < 2) lastPinch = 0; };
  canvas.addEventListener('pointerup', endPointer); canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('wheel', e => { e.preventDefault(); radius *= Math.exp(e.deltaY * 0.0012); dirty = true; }, { passive: false });
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
  }
  window.addEventListener('resize', resize);
  function animate() {
    requestAnimationFrame(animate);
    if (state.spin) { theta += 0.004; dirty = true; }
    if (!dirty) return;
    dirty = false;
    radius = Math.min(Math.max(radius, 2), 50000);
    updateCamera(); renderer.render(scene, camera);
  }

  // ---- controls --------------------------------------------------------------
  const $ = id => document.getElementById(id);
  const colours = $('colours');
  for (let c = 1; c < D.k; c++) {
    if (!byColour[c].length) continue;
    const row = document.createElement('div'); row.className = 'row';
    row.innerHTML = `<span class="swatch" style="background:${D.palette[c]}"></span>` +
      `<span class="name">${D.names[c]}</span>` +
      `<input type="range" id="op${c}" min="0" max="100" value="100" aria-label="opacity of ${D.names[c]}">` +
      `<span class="val" id="opv${c}">100%</span>` +
      `<input type="checkbox" id="vis${c}" checked aria-label="show ${D.names[c]}">`;
    colours.appendChild(row);
    $('op' + c).addEventListener('input', e => { state.opacity[c] = e.target.value / 100; $('opv' + c).textContent = `${e.target.value}%`; applyMaterials(); });
    $('vis' + c).addEventListener('change', e => { state.visible[c] = e.target.checked; applyMaterials(); });
  }
  $('shape').value = state.shape;
  $('shape').addEventListener('change', e => { state.shape = e.target.value; build(); });
  $('size').addEventListener('input', e => {
    state.size = e.target.value / 100; $('sizeVal').textContent = `${e.target.value}%`;
    if (state.shape === 'dot') { for (let c = 1; c < D.k; c++) if (meshes[c]) meshes[c].mesh.material.size = dotSize(); dirty = true; }
    else build();
  });
  $('net').addEventListener('change', e => { state.net = e.target.checked; buildNet(); report(); dirty = true; });
  $('netSize').addEventListener('input', e => { state.netSize = e.target.value / 100; if (net) { net.material.size = state.netSize / Math.tan(camera.fov * Math.PI / 360); dirty = true; } });
  $('netOp').addEventListener('input', e => { state.netOp = e.target.value / 100; if (net) { net.material.opacity = state.netOp; dirty = true; } });
  $('lines').addEventListener('change', e => { state.lines = e.target.checked; buildLines(); report(); dirty = true; });
  $('lineOp').addEventListener('input', e => { state.lineOp = e.target.value / 100; if (lines) { lines.material.opacity = state.lineOp; dirty = true; } });
  $('shell').checked = state.shell;
  $('shell').addEventListener('change', e => { state.shell = e.target.checked; build(); });
  const cutVal = $('cutVal');
  function showCut() { cutVal.textContent = state.cutAxis === 'none' ? '' : `${state.cutAxis} ≤ ${Math.round(state.cut * (state.cutAxis === 'x' ? W : H))}`; }
  $('cutAxis').addEventListener('change', e => { state.cutAxis = e.target.value; applyPlanes(); showCut(); });
  $('cut').addEventListener('input', e => { state.cut = e.target.value / 1000; applyPlanes(); showCut(); });
  const tmin = $('tmin'), tmax = $('tmax');
  tmin.max = tmax.max = T - 1; tmax.value = T - 1; $('tmaxVal').textContent = T - 1;
  tmin.addEventListener('input', e => { state.tmin = +e.target.value; if (state.tmin > state.tmax) { state.tmax = state.tmin; tmax.value = state.tmax; $('tmaxVal').textContent = state.tmax; } $('tminVal').textContent = state.tmin; applyPlanes(); });
  tmax.addEventListener('input', e => { state.tmax = +e.target.value; if (state.tmax < state.tmin) { state.tmin = state.tmax; tmin.value = state.tmin; $('tminVal').textContent = state.tmin; } $('tmaxVal').textContent = state.tmax; applyPlanes(); });
  $('stretch').addEventListener('input', e => { state.stretch = e.target.value / 100; $('stretchVal').textContent = state.stretch.toFixed(2).replace(/0$/, '') + '×'; restretch(); });
  $('sx').addEventListener('input', e => { state.sx = e.target.value / 100; $('sxVal').textContent = state.sx.toFixed(1) + '×'; restretch(); });
  const spinBox = $('spin'); spinBox.checked = state.spin;
  spinBox.addEventListener('change', e => { state.spin = e.target.checked; dirty = true; });
  $('reset').addEventListener('click', () => { theta = home.theta; phi = home.phi; fit(); dirty = true; });

  build(); fit(); resize(); showCut(); animate();
})();
"""
