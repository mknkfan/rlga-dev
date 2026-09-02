"""
A standalone, actually-rotatable viewer for a solved cycle.

Why this exists: a GIF can only ever show the viewpoints it was rendered from,
and a single fixed isometric view is exactly what makes boxes at different
depths read as though they intersect.  This writes the scene -- machine boxes,
tool path, and the robot itself -- into a self-contained HTML page that
projects and depth-sorts it in the browser, so the viewpoint belongs to
whoever is reading it.

The robot is a **solid model**, not a polyline.  :mod:`arm_study.armmesh`
sends its ~170 polygons once, in link-local coordinates, together with the
four joint angles of every sampled instant; the page rebuilds the five link
transforms itself (the same ones
:meth:`arm_study.robot.Puma560Arm.link_frames` computes) and lights each
polygon from its world normal.  That is what lets the arm re-shade as the
camera turns, and it keeps the page a few tens of kilobytes rather than a few
megabytes.

Every polygon in the scene -- machine faces and robot facets alike -- goes
into one list that is sorted back to front each frame (painter's algorithm),
so the arm occludes the machines it is in front of and is hidden by the ones
it is behind, from *any* angle rather than only the one it was baked at.

Playback runs off ``requestAnimationFrame`` with a speed control: a retracted
cycle is a few hundred sampled instants and a third of a minute to watch
through once at 1x, which is longer than anyone spends on it.  The control
scales the milliseconds per instant, so 0.25x is as available as 16x.

No libraries and no network access: the geometry is inlined as JSON and the
renderer is about a hundred lines of canvas drawing, so the file opens
straight off the filesystem.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>TITLE_TEXT</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; overflow:hidden;
         font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;
         background:#101317; color:#e8ecf1; }
  #wrap { display:flex; flex-direction:column; height:100vh; }
  header { padding:9px 14px; border-bottom:1px solid #2a3038; }
  header b { font-size:14px; }
  header span { color:#9aa5b1; margin-left:10px; }
  /* The canvas is absolutely positioned inside a relative box so that CSS
     alone decides its layout size.  A canvas sized only by flex still reports
     its backing-store dimensions as intrinsic size, and since the backing
     store is scaled by devicePixelRatio that feeds back into layout and pushes
     the controls off the bottom of the window. */
  #stage { flex:1; min-height:0; position:relative; }
  canvas { position:absolute; inset:0; width:100%; height:100%;
           display:block; cursor:grab; touch-action:none; }
  canvas:active { cursor:grabbing; }
  footer { padding:9px 14px; border-top:1px solid #2a3038; display:flex;
           gap:14px; align-items:center; flex-wrap:wrap; }
  button { background:#2b3a4f; color:#e8ecf1; border:1px solid #3d5474;
           border-radius:5px; padding:5px 12px; cursor:pointer; font:inherit; }
  button:hover { background:#36495f; }
  input[type=range] { flex:1; min-width:150px; }
  .hint { color:#8a94a0; font-size:12px; }
</style>
<div id="wrap">
  <header><b id="title"></b><span id="sub"></span></header>
  <div id="stage"><canvas id="c"></canvas></div>
  <footer>
    <button id="play">Pause</button>
    <button id="speed" title="playback speed - click to go faster, shift-click to slow down">&#x23E9; 1&times;</button>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span id="clock" class="hint"></span>
    <button id="top">Top</button>
    <button id="reset">Reset view</button>
    <span class="hint">drag = orbit, scroll = zoom, shift-drag = pan, [ ] = speed</span>
  </footer>
</div>
<script>
var S = SCENE_JSON;
var cv = document.getElementById('c'), cx = cv.getContext('2d');
document.getElementById('title').textContent =
  S.name + ' \\u2014 cycle ' + S.cycle.toFixed(3) + ' s';
document.getElementById('sub').textContent = 'posture ' + S.posture;

var az = -58*Math.PI/180, el = 22*Math.PI/180, zoom = 1, panX = 0, panY = 0;
var frame = 0, playing = true, dragging = false, lastX = 0, lastY = 0, shift = false;
var scrub = document.getElementById('scrub');
scrub.max = S.q.length - 1;

/* ---- the robot's link transforms -------------------------------------
   These are the same five frames robot.py's link_frames() builds, in the
   same order: pedestal, waist turned by q1, shoulder at (0, -d3, d1) pitched
   by q2, elbow a2 further along pitched by q3, wrist L3 further along pitched
   by q4.  A frame is twelve numbers, row-major [R | t].  Rebuilding them here
   is what lets the page ship the robot's polygons once instead of once per
   animation frame. */
function matMul(a, b) {
  var o = new Array(12), r, c;
  for (r = 0; r < 3; r++) {
    for (c = 0; c < 3; c++) {
      o[r*4+c] = a[r*4]*b[c] + a[r*4+1]*b[4+c] + a[r*4+2]*b[8+c];
    }
    o[r*4+3] = a[r*4]*b[3] + a[r*4+1]*b[7] + a[r*4+2]*b[11] + a[r*4+3];
  }
  return o;
}
function rotZ(q) { var c = Math.cos(q), s = Math.sin(q);
                   return [c,-s,0,0,  s,c,0,0,  0,0,1,0]; }
function rotY(q) { var c = Math.cos(q), s = Math.sin(q);
                   return [c,0,s,0,  0,1,0,0,  -s,0,c,0]; }
function trans(x, y, z) { return [1,0,0,x,  0,1,0,y,  0,0,1,z]; }
function applyM(m, p) {
  return [m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3],
          m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7],
          m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11]];
}
function linkFrames(q) {
  var g = S.arm.geom;
  var f0 = trans(g.base[0], g.base[1], 0);
  var f1 = matMul(f0, rotZ(q[0]));
  var f2 = matMul(matMul(f1, trans(0, -g.d3, g.d1)), rotY(-q[1]));
  var f3 = matMul(matMul(f2, trans(g.a2, 0, 0)), rotY(-q[2]));
  var f4 = matMul(matMul(f3, trans(g.L3, 0, 0)), rotY(-q[3]));
  return [f0, f1, f2, f3, f4];
}

/* One directional light plus ambient, from each polygon's world normal --
   the same formula armmesh.shade() uses, so the static figures and this page
   light the robot identically. */
var LIGHT = S.arm.light, AMBIENT = S.arm.ambient;
var MACHINE_RGB = [76, 114, 176];
function litFill(rgb, a, b, c) {
  var ux = b[0]-a[0], uy = b[1]-a[1], uz = b[2]-a[2];
  var vx = c[0]-a[0], vy = c[1]-a[1], vz = c[2]-a[2];
  var nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx;
  var len = Math.sqrt(nx*nx + ny*ny + nz*nz);
  var lam = len < 1e-12 ? 1 :
    Math.abs((nx*LIGHT[0] + ny*LIGHT[1] + nz*LIGHT[2]) / len);
  var k = AMBIENT + (1 - AMBIENT)*lam;
  return 'rgb(' + Math.round(rgb[0]*k) + ',' + Math.round(rgb[1]*k) + ',' +
         Math.round(rgb[2]*k) + ')';
}

function resize() {
  var r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width*d); cv.height = Math.round(r.height*d);
  cx.setTransform(d,0,0,d,0,0);
  draw();
}
window.addEventListener('resize', resize);

/* Orthographic: spin about z by the azimuth, tilt by the elevation, drop the
   depth axis but keep it for sorting. */
function project(p) {
  var ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
  var x = p[0]*ca + p[1]*sa;
  var y = -p[0]*sa + p[1]*ca;
  return [x, -(y*se) + p[2]*ce, y*ce + p[2]*se];
}
function view() {
  var r = cv.getBoundingClientRect();
  /* The robot is taller than the cell is wide once the elbow is up, so the
     ceiling has to be in the span or the arm walks off the top of the frame
     at low elevations. */
  var span = Math.max(S.bounds[1]-S.bounds[0], S.bounds[3]-S.bounds[2], S.ceiling);
  var s = Math.min(r.width, r.height) / (span*1.45) * zoom;
  /* World z = 0 projects onto the horizon, so a scene that is entirely above
     the floor sits in the top half of the frame.  Dropping the origin by half
     the ceiling centres what is actually there. */
  var lift = S.ceiling*0.5*Math.cos(el)*s;
  return { cx: r.width/2 + panX, cy: r.height/2 + panY + lift, s: s };
}
function toScreen(p, v) {
  var q = project(p);
  return [v.cx + q[0]*v.s, v.cy - q[1]*v.s, q[2]];
}

function draw() {
  var v = view(), r = cv.getBoundingClientRect(), i;
  cx.clearRect(0,0,r.width,r.height);

  cx.strokeStyle = '#232a33'; cx.lineWidth = 1;
  var x0 = S.bounds[0], x1 = S.bounds[1], y0 = S.bounds[2], y1 = S.bounds[3];
  for (i = 0; i <= 8; i++) {
    var fx = x0 + (x1-x0)*i/8, fy = y0 + (y1-y0)*i/8, a, b;
    a = toScreen([fx,y0,0],v); b = toScreen([fx,y1,0],v);
    cx.beginPath(); cx.moveTo(a[0],a[1]); cx.lineTo(b[0],b[1]); cx.stroke();
    a = toScreen([x0,fy,0],v); b = toScreen([x1,fy,0],v);
    cx.beginPath(); cx.moveTo(a[0],a[1]); cx.lineTo(b[0],b[1]); cx.stroke();
  }

  cx.strokeStyle = '#4a5563'; cx.lineWidth = 1.2; cx.beginPath();
  for (i = 0; i <= 48; i++) {
    var th = i/48*Math.PI*2;
    var p = toScreen([S.base[0]+S.keepOut*Math.cos(th),
                      S.base[1]+S.keepOut*Math.sin(th), 0], v);
    if (i) { cx.lineTo(p[0],p[1]); } else { cx.moveTo(p[0],p[1]); }
  }
  cx.stroke();

  /* Painter's algorithm over EVERY drawable -- machine faces, the robot's
     facets, each segment of the tool path and each port marker go into one
     list and are sorted back to front together.  Sorting them separately is
     what used to make the arm float in front of a machine it was behind. */
  var items = [], j, k;

  function pushPoly(world, fill, stroke, label) {
    var pts = [], d = 0;
    for (var m = 0; m < world.length; m++) {
      var sp = toScreen(world[m], v); pts.push(sp); d += sp[2];
    }
    items.push({kind: 0, pts: pts, depth: d/pts.length,
                fill: fill, stroke: stroke, label: label});
  }

  for (i = 0; i < S.boxes.length; i++) {
    var boxi = S.boxes[i];
    for (var f = 0; f < boxi.faces.length; f++) {
      var face = boxi.faces[f];
      pushPoly(face, litFill(MACHINE_RGB, face[0], face[1], face[2]), '#1b2734',
               f === 0 ? (boxi.name + ' #' + boxi.order) : null);
    }
  }

  var frames = linkFrames(S.q[frame]);
  for (i = 0; i < S.arm.parts.length; i++) {
    var part = S.arm.parts[i], mat = frames[part.f];
    for (j = 0; j < part.p.length; j++) {
      var local = part.p[j], world = [];
      for (k = 0; k < local.length; k++) { world.push(applyM(mat, local[k])); }
      /* Stroked in its own fill colour: that closes the hairline seams
         antialiasing leaves between neighbouring facets, without striping
         each cylinder into staves the way a contrasting edge would. */
      var lit = litFill(part.c, world[0], world[1], world[2]);
      pushPoly(world, lit, lit, null);
    }
  }

  /* The tool path is cut into per-segment drawables so that the part of it
     running behind a machine is actually hidden by that machine. */
  for (i = 1; i <= frame; i++) {
    var p0 = toScreen(S.tool[i-1], v), p1 = toScreen(S.tool[i], v);
    items.push({kind: 1, pts: [p0, p1], depth: (p0[2] + p1[2])/2});
  }
  S.ports.forEach(function (pt) {
    var sp = toScreen(pt, v);
    items.push({kind: 2, pts: [sp], depth: sp[2]});
  });

  items.sort(function (a, b) { return a.depth - b.depth; });
  items.forEach(function (it) {
    if (it.kind === 1) {
      cx.strokeStyle = '#e05252'; cx.lineWidth = 1.8; cx.lineCap = 'round';
      cx.beginPath(); cx.moveTo(it.pts[0][0], it.pts[0][1]);
      cx.lineTo(it.pts[1][0], it.pts[1][1]); cx.stroke();
      return;
    }
    if (it.kind === 2) {
      cx.fillStyle = '#e05252';
      cx.beginPath(); cx.arc(it.pts[0][0], it.pts[0][1], 3.4, 0, Math.PI*2); cx.fill();
      return;
    }
    cx.fillStyle = it.fill;
    cx.beginPath(); cx.moveTo(it.pts[0][0], it.pts[0][1]);
    for (var m = 1; m < it.pts.length; m++) { cx.lineTo(it.pts[m][0], it.pts[m][1]); }
    cx.closePath(); cx.fill();
    if (it.stroke) { cx.strokeStyle = it.stroke; cx.lineWidth = 1; cx.stroke(); }
    if (it.label) {
      var mx = 0, my = 0;
      it.pts.forEach(function (p) { mx += p[0]/it.pts.length; my += p[1]/it.pts.length; });
      cx.fillStyle = '#10203a'; cx.font = '11px system-ui'; cx.textAlign = 'center';
      cx.fillText(it.label, mx, my + 4);
    }
  });

  document.getElementById('clock').textContent =
    't = ' + S.t[frame].toFixed(3) + ' s   |   az ' + (az*180/Math.PI).toFixed(0) +
    '\\u00b0, elev ' + (el*180/Math.PI).toFixed(0) + '\\u00b0';
}

cv.addEventListener('pointerdown', function (e) {
  dragging = true; shift = e.shiftKey; lastX = e.clientX; lastY = e.clientY;
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointerup', function () { dragging = false; });
cv.addEventListener('pointermove', function (e) {
  if (!dragging) { return; }
  var dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (shift) { panX += dx; panY += dy; }
  else {
    az += dx*0.008;
    el = Math.max(-Math.PI/2+0.05, Math.min(Math.PI/2-0.05, el + dy*0.006));
  }
  draw();
});
cv.addEventListener('wheel', function (e) {
  e.preventDefault();
  zoom = Math.max(0.35, Math.min(6, zoom*(e.deltaY < 0 ? 1.1 : 1/1.1)));
  draw();
}, {passive:false});

document.getElementById('play').addEventListener('click', function (e) {
  playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';
});
scrub.addEventListener('input', function (e) {
  playing = false; document.getElementById('play').textContent = 'Play';
  frame = +e.target.value; draw();
});
document.getElementById('reset').addEventListener('click', function () {
  az = -58*Math.PI/180; el = 22*Math.PI/180; zoom = 1; panX = 0; panY = 0; draw();
});
document.getElementById('top').addEventListener('click', function () {
  az = -90*Math.PI/180; el = 89*Math.PI/180; draw();
});

/* A retracted cycle is a few hundred sampled instants, and at one instant per
   FRAME_MS that is a third of a minute to watch through once.  The control
   scales the milliseconds per instant rather than the instants per tick, so
   that slowing down works as well as speeding up.  Past about 2.5x there are
   fewer display refreshes than instants, and the loop then advances several
   at once rather than quietly falling behind the clock. */
var FRAME_MS = 40;
var SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16];
var speedIndex = 2;
var speedButton = document.getElementById('speed');

function setSpeed(index) {
  speedIndex = Math.max(0, Math.min(SPEEDS.length - 1, index));
  var s = SPEEDS[speedIndex];
  speedButton.textContent = '\\u23e9 ' + (s < 1 ? s : s.toFixed(0)) + '\\u00d7';
}
speedButton.addEventListener('click', function (e) {
  setSpeed(speedIndex + (e.shiftKey ? -1 : 1));
});
window.addEventListener('keydown', function (e) {
  /* Not while a control has focus: the browser already gives space and the
     arrow keys to a focused button or to the scrubber. */
  var tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'button' || tag === 'input') { return; }
  if (e.key === ']') { setSpeed(speedIndex + 1); }
  else if (e.key === '[') { setSpeed(speedIndex - 1); }
  else if (e.key === ' ') {
    e.preventDefault();
    playing = !playing;
    document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  }
});
setSpeed(speedIndex);

var last = 0;
function tick(now) {
  if (!last || !playing) {
    last = now;
  }
  if (playing) {
    var step = FRAME_MS / SPEEDS[speedIndex];
    var advance = Math.floor((now - last) / step);
    if (advance > 0) {
      /* Advance `last` by whole steps, not to `now`: the leftover fraction of
         a step carries into the next tick, so the playback rate stays exactly
         the requested multiple instead of drifting with the frame rate. */
      last += advance * step;
      frame = (frame + advance) % S.q.length;
      scrub.value = frame;
      draw();
    }
  }
  requestAnimationFrame(tick);
}
resize();
requestAnimationFrame(tick);
</script>
"""


def write_viewer(scene: Dict, path: str, title: str) -> None:
    """Inline ``scene`` into the template and write the page."""
    html = _TEMPLATE.replace("SCENE_JSON", json.dumps(scene, separators=(",", ":")))
    html = html.replace("TITLE_TEXT", title)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
