"use strict";

/* ---------- Tiny helpers ---------- */
const clamp01 = x => Math.max(0, Math.min(1, x));
const add = (a,b)=>[a[0]+b[0], a[1]+b[1]];
const sub = (a,b)=>[a[0]-b[0], a[1]-b[1]];
const dot = (a,b)=>a[0]*b[0]+a[1]*b[1];
const norm = v => Math.hypot(v[0], v[1]);

/* Safe DOM lookup (throws early if an id is missing) */
function qs(id){
  const el = document.getElementById(id);
  if(!el) throw new Error(`Missing DOM element #${id}. Ensure your HTML id matches the JS.`);
  return el;
}

/* ---------- Core LA ops ---------- */
function centroid(pts){ let sx=0,sy=0; for(const [x,y] of pts){sx+=x;sy+=y;} const n=pts.length||1; return [sx/n, sy/n]; }
function translate(pts, c){ return pts.map(([x,y])=>[x-c[0], y-c[1]]); }
function rmsRadius(pts){ let s=0,n=pts.length||1; for(const [x,y] of pts) s+=x*x+y*y; return Math.sqrt(s/n); }
function toUnit(pts){
  if(!pts.length) return [];
  const c=centroid(pts), T=translate(pts,c), r=Math.max(rmsRadius(T), 1e-8);
  return T.map(([x,y])=>[x/r, y/r]);
}
function rot2(theta){ const c=Math.cos(theta), s=Math.sin(theta); return [[c,-s],[s,c]]; }
function mulRot(pts, R){ const [[c,m],[s,n]]=R; return pts.map(([x,y])=>[c*x+m*y, s*x+n*y]); }

/* ---------- Resampling ---------- */
function polyLength(poly){ let L=0; for(let i=1;i<poly.length;i++) L+=Math.hypot(poly[i][0]-poly[i-1][0], poly[i][1]-poly[i-1][1]); return L; }
function resample(poly, N){
  if(!poly.length) return [];
  const d=[0]; for(let i=1;i<poly.length;i++) d.push(d[i-1]+Math.hypot(poly[i][0]-poly[i-1][0], poly[i][1]-poly[i-1][1]));
  const total=d[d.length-1]; if(total<1e-6) return Array(N).fill(poly[0]);
  const step=total/(N-1), out=[]; let j=0;
  for(let k=0;k<N;k++){
    const t=k*step; while(j<d.length-2 && d[j+1]<t) j++;
    const u=(t-d[j])/(d[j+1]-d[j]||1);
    out.push([ poly[j][0]+u*(poly[j+1][0]-poly[j][0]), poly[j][1]+u*(poly[j+1][1]-poly[j][1]) ]);
  }
  return out;
}
function resampleMulti(strokes, Ntotal){
  const Ls=strokes.map(polyLength), Lsum=Ls.reduce((a,b)=>a+b,0)||1;
  const Ns=Ls.map(L=>Math.max(4, Math.round(Ntotal*(L/Lsum))));
  let diff=Ntotal-Ns.reduce((a,b)=>a+b,0);
  for(let i=0; diff!==0 && i<Ns.length; i=(i+1)%Ns.length){ Ns[i]+=(diff>0?1:-1); diff+=(diff>0?-1:1); }
  return strokes.map((s,i)=>resample(s, Ns[i])).flat();
}

/* ---------- Built-in templates ---------- */
function houseBasic(){
  const base=[[-0.8,-0.6],[0.8,-0.6],[0.8,0.2],[-0.8,0.2],[-0.8,-0.6]];
  const roof=[[-0.9,0.2],[0,0.85],[0.9,0.2]];
  return {kind:"builtin", id:"house-basic", name:"House • Basic", strokes:[base, roof], corners:[base[0],base[1],base[2],base[3], roof[0],roof[1],roof[2]]};
}
function houseDoorWindow(){
  const h=houseBasic();
  const door=[[-0.15,-0.6],[-0.15,0.05],[0.15,0.05],[0.15,-0.6]];
  const win=[[0.35,-0.1],[0.65,-0.1],[0.65,0.15],[0.35,0.15],[0.35,-0.1]];
  const cross1=[[0.5,-0.1],[0.5,0.15]], cross2=[[0.35,0.025],[0.65,0.025]];
  return {kind:"builtin", id:"house-door-window", name:"House • Details", strokes:[...h.strokes, door, win, cross1, cross2], corners:[...h.corners, door[0],door[1],door[2],door[3], win[0],win[1],win[2],win[3]]};
}
function polySquare(){ const s=0.8; return {kind:"builtin", id:"square", name:"Square", strokes:[[[-s,-s],[s,-s],[s,s],[-s,s],[-s,-s]]], corners:[[-s,-s],[s,-s],[s,s],[-s,s]]}; }
function polyTriangle(){ return {kind:"builtin", id:"triangle", name:"Triangle", strokes:[[[0,0.9],[-0.9,-0.5],[0.9,-0.5],[0,0.9]]], corners:[[0,0.9],[-0.9,-0.5],[0.9,-0.5]]}; }
function polyStar(){
  const pts=[], outer=0.9, inner=0.36;
  for(let i=0;i<10;i++){ const r=i%2===0?outer:inner, a=-Math.PI/2+i*Math.PI/5; pts.push([r*Math.cos(a), r*Math.sin(a)]); }
  pts.push(pts[0]);
  return {kind:"builtin", id:"star", name:"Star", strokes:[pts], corners:pts.filter((_,i)=>i%2===0).slice(0,5)};
}
const builtinIndex = {
  "house-basic": houseBasic(),
  "house-door-window": houseDoorWindow(),
  "triangle": polyTriangle(),
  "square": polySquare(),
  "star": polyStar()
};

/* ---------- Spatial hash + weights ---------- */
function SpatialHash(points, cell=0.05){
  const map=new Map(), key=(i,j)=>i+"_"+j, cellIdx=([x,y])=>[Math.floor(x/cell), Math.floor(y/cell)];
  for(const p of points){ const [i,j]=cellIdx(p), k=key(i,j); if(!map.has(k)) map.set(k, []); map.get(k).push(p); }
  function neighbors(p){ const [i,j]=cellIdx(p), out=[]; for(let di=-1;di<=1;di++) for(let dj=-1;dj<=1;dj++){ const a=map.get(key(i+di,j+dj)); if(a) out.push(...a); } return out; }
  return {
    nearest(p){
      let best=Infinity, bestp=null; const N=neighbors(p);
      if(N.length) for(const q of N){ const dx=p[0]-q[0],dy=p[1]-q[1], d=dx*dx+dy*dy; if(d<best){best=d; bestp=q;} }
      if(bestp===null){ for(const q of points){ const dx=p[0]-q[0],dy=p[1]-q[1], d=dx*dx+dy*dy; if(d<best){best=d; bestp=q;} } }
      return {p:bestp, dist:Math.sqrt(best)};
    }
  };
}
function curvatureWeights(pts){
  const n=pts.length, w=new Array(n).fill(1);
  for(let i=1;i<n-1;i++){
    const a=sub(pts[i], pts[i-1]), b=sub(pts[i+1], pts[i]);
    const na=norm(a)||1e-9, nb=norm(b)||1e-9;
    let cos = (dot(a,b)/(na*nb)); cos = Math.max(-1, Math.min(1, cos));
    const k = 1 - (cos+1)/2; // 0 straight, 1 sharp
    w[i] = 0.8 + 1.2*k;
  }
  const mean = w.reduce((s,x)=>s+x,0)/n;
  return w.map(x=>x/mean);
}

/* ---------- Metrics ---------- */
function rmse(A,B){ let s=0,n=Math.min(A.length,B.length)||1; for(let i=0;i<n;i++){ const dx=A[i][0]-B[i][0], dy=A[i][1]-B[i][1]; s+=dx*dx+dy*dy; } return Math.sqrt(s/n); }
function weightedChamfer(A, B, tol){
  const wA = curvatureWeights(A); const hashB = SpatialHash(B, 0.05);
  let sum=0, within=0, wsum=0;
  for(let i=0;i<A.length;i++){ const wa=wA[i], {dist}=hashB.nearest(A[i]); sum += wa*dist; wsum += wa; if(dist<=tol) within += wa; }
  return {frac: clamp01(within/wsum), avg: sum/wsum};
}
function cornerQuality(corners, B, tol){
  if(!corners.length||!B.length) return {q:0, avg:Infinity};
  const hashB = SpatialHash(B, 0.05);
  let sum=0; for(const c of corners){ sum += hashB.nearest(c).dist; }
  const avg=sum/corners.length;
  return {q: clamp01(1 - avg/(tol*0.8)), avg};
}

/* ---------- Alignment models ---------- */
function crossCov2(X0, Y0){
  let a=0,b=0,c=0,d=0, xx=0;
  for(let i=0;i<X0.length;i++){
    const [x,y]=X0[i], [u,v]=Y0[i];
    a+=x*u; b+=x*v; c+=y*u; d+=y*v;
    xx+=x*x+y*y;
  }
  return {M:[a,b,c,d], xx};
}
function R_from_M(a,b,c,d){ const theta = Math.atan2(b-c, a+d); return {R: rot2(theta), theta}; }
function traceRTM(R, M){ const a=M[0], b=M[1], c=M[2], d=M[3]; return R[0][0]*a + R[0][1]*c + R[1][0]*b + R[1][1]*d; }

function alignOrthogonal(X, Y){
  const cx=centroid(X), cy=centroid(Y);
  const X0=translate(X, cx), Y0=translate(Y, cy);
  const {M} = crossCov2(X0, Y0);
  const {R, theta} = R_from_M(...M);
  const s=1, t=sub(cy, mulRot([cx], R)[0].map(v=>v*s));
  const aligned = X.map(p=>add(mulRot([p], R)[0].map(v=>v*s), t));
  return {aligned, R, s, t, theta, method:"Orthogonal (R,t)"};
}
function alignSimilarity(X, Y){
  const cx=centroid(X), cy=centroid(Y);
  const X0=translate(X, cx), Y0=translate(Y, cy);
  const {M, xx} = crossCov2(X0, Y0);
  const {R, theta} = R_from_M(...M);
  const tr = traceRTM(R, M);
  const s = (xx>1e-12) ? tr/xx : 1;
  const t = sub(cy, mulRot([cx], R)[0].map(v=>v*s));
  const aligned = X.map(p=>add(mulRot([p], R)[0].map(v=>v*s), t));
  return {aligned, R, s, t, theta, method:"Similarity (sR,t)"};
}
function solveLinear(A, b){
  const n=A.length, M=A.map((row,i)=>row.concat([b[i]]));
  for(let col=0; col<n; col++){
    let piv=col; for(let r=col+1;r<n;r++) if(Math.abs(M[r][col])>Math.abs(M[piv][col])) piv=r;
    if(Math.abs(M[piv][col])<1e-12) continue;
    if(piv!==col) [M[col],M[piv]]=[M[piv],M[col]];
    const fac=M[col][col];
    for(let c=col;c<=n;c++) M[col][c]/=fac;
    for(let r=0;r<n;r++){
      if(r===col) continue;
      const f=M[r][col]; if(Math.abs(f)<1e-16) continue;
      for(let c=col;c<=n;c++) M[r][c]-=f*M[col][c];
    }
  }
  return M.map(row=>row[n]||0);
}
function alignAffine(X, Y){
  const AtA = Array.from({length:6}, ()=>Array(6).fill(0));
  const Atb = Array(6).fill(0);
  for(let i=0;i<X.length;i++){
    const x=X[i][0], y=X[i][1], u=Y[i][0], v=Y[i][1];
    const urow=[x,y,1,0,0,0], vrow=[0,0,0,x,y,1];
    for(let r=0;r<6;r++){
      for(let c=r;c<6;c++){ AtA[r][c] += urow[r]*urow[c] + vrow[r]*vrow[c]; }
    }
    for(let r=0;r<6;r++) Atb[r] += urow[r]*u + vrow[r]*v;
  }
  for(let r=0;r<6;r++) for(let c=0;c<r;c++) AtA[r][c]=AtA[c][r];
  const p = solveLinear(AtA, Atb);
  const A = [[p[0], p[1]],[p[3], p[4]]], t=[p[2], p[5]];
  const aligned = X.map(([x,y])=>[A[0][0]*x + A[0][1]*y + t[0], A[1][0]*x + A[1][1]*y + t[1]]);
  const theta = Math.atan2(A[1][0], A[0][0]);
  const s = Math.sqrt((A[0][0]*A[0][0]+A[1][0]*A[1][0] + A[0][1]*A[0][1]+A[1][1]*A[1][1]) / 2);
  return {aligned, A, t, theta, s, method:"Affine (A,t)"};
}
function eig2(C){
  const a=C[0][0], b=C[0][1], d=C[1][1];
  const tr=a+d, det=a*d - b*b;
  const disc=Math.max(0, tr*tr/4 - det);
  const l1=tr/2 + Math.sqrt(disc), l2=tr/2 - Math.sqrt(disc);
  const v1 = (Math.abs(b)>1e-12)? [b, l1-a] : [1,0];
  const n1 = Math.hypot(v1[0], v1[1])||1; const e1=[v1[0]/n1, v1[1]/n1];
  const e2=[-e1[1], e1[0]];
  return {vals:[l1,l2], vecs:[e1,e2]};
}
function cov2(pts){
  const c=centroid(pts), T=translate(pts,c);
  let xx=0,xy=0,yy=0; const n=T.length||1;
  for(const [x,y] of T){ xx+=x*x; xy+=x*y; yy+=y*y; }
  return {C:[[xx/n, xy/n],[xy/n, yy/n]], center:c, T};
}
function alignPCA(X, Y){
  const cx=cov2(X), cy=cov2(Y);
  const ex=eig2(cx.C), ey=eig2(cy.C);
  const RX=[[ex.vecs[0][0], ex.vecs[1][0]],[ex.vecs[0][1], ex.vecs[1][1]]];
  const RY=[[ey.vecs[0][0], ey.vecs[1][0]],[ey.vecs[0][1], ey.vecs[1][1]]];
  const RT=[[RX[0][0], RX[1][0]],[RX[0][1], RX[1][1]]];
  const R=[
    [RY[0][0]*RT[0][0]+RY[0][1]*RT[1][0], RY[0][0]*RT[0][1]+RY[0][1]*RT[1][1]],
    [RY[1][0]*RT[0][0]+RY[1][1]*RT[1][0], RY[1][0]*RT[0][1]+RY[1][1]*RT[1][1]],
  ];
  const s = (rmsRadius(cy.T)/Math.max(rmsRadius(cx.T),1e-12));
  const t = sub(cy.center, mulRot([cx.center], R)[0].map(v=>v*s));
  const aligned = X.map(p=>add(mulRot([p], R)[0].map(v=>v*s), t));
  const theta = Math.atan2(R[1][0], R[0][0]);
  return {aligned, R, s, t, theta, PCA:{RX,RY,ex,ey}, method:"PCA Align (sR,t)"};
}

/* ---------- ICP refine (optional) ---------- */
function icpAlign(Xin, Yin, iters=5){
  const X=Xin.slice(), Y=Yin.slice();
  let s=1, theta=0, t=[0,0];
  for(let k=0;k<iters;k++){
    const R=rot2(theta);
    const Xc = X.map(p=>add(mulRot([p], R)[0].map(v=>v*s), t));
    const sh=SpatialHash(Y,0.05), pairs=[];
    for(const p of Xc){ const {p:qy}=sh.nearest(p); if(qy) pairs.push([p,qy]); }
    if(pairs.length<6) break;
    const PX=pairs.map(([px,_])=>px), PY=pairs.map(([_,py])=>py);
    const cx=centroid(PX), cy=centroid(PY);
    const X0=translate(PX,cx), Y0=translate(PY,cy);
    const {M, xx} = crossCov2(X0, Y0);
    const {R:Rd, theta:dt} = R_from_M(...M);
    const tr = traceRTM(Rd, M);
    const ds = (xx>1e-12)? tr/xx : 1;
    const dtv = sub(cy, mulRot([cx], Rd)[0].map(v=>v*ds));
    theta+=dt; s*=ds; t=add(mulRot([t], Rd)[0].map(v=>v*ds), dtv);
  }
  const Rf=rot2(theta);
  const aligned = Xin.map(p=>add(mulRot([p], Rf)[0].map(v=>v*s), t));
  return {aligned, theta, s, t};
}

/* ---------- Scoring (0..100) ---------- */
function scoreLA(userStrokes, template, baseTol, diff, allowMirror, model, useICP){
  if(!userStrokes.length) return {ok:false, msg:"Draw first.", score:0};

  const N=288;
  const tplPts = toUnit(resampleMulti(template.strokes, N));
  const tplCorners = toUnit(template.corners || []);
  const userPts0 = resampleMulti(userStrokes, N);
  const userPts = toUnit(userPts0);

  const alignCore = (U,T)=>{
    if(model==="orthogonal") return alignOrthogonal(U,T);
    if(model==="affine")     return alignAffine(U,T);
    if(model==="pca")        return alignPCA(U,T);
    return alignSimilarity(U,T);
  };
  const tryOne = (U,T) => {
    let out = alignCore(U,T);
    if(useICP && model!=="affine"){
      const icp = icpAlign(out.aligned, T, 5);
      out = {...out, aligned: icp.aligned, s: icp.s ?? out.s, theta: icp.theta ?? out.theta, t: icp.t ?? out.t, method: out.method + " + ICP"};
    }
    return out;
  };

  let best = tryOne(userPts, tplPts), mirrored=false;
  if(allowMirror){
    const mU = userPts.map(([x,y])=>[-x,y]);
    const cand = tryOne(mU, tplPts);
    if(rmse(cand.aligned, tplPts) < rmse(best.aligned, tplPts)){ best=cand; mirrored=true; }
  }

  const tol = baseTol * ({easy:1.4, normal:1.0, hard:0.8, insane:0.65}[diff]||1);
  const err = rmse(best.aligned, tplPts);
  const cov = weightedChamfer(tplPts, best.aligned, tol);
  const corn= cornerQuality(tplCorners, best.aligned, tol);
  const qRMSE = clamp01(1 - err/tol);
  const qCov  = cov.frac;
  const qCorn = corn.q;

  let raw = 100*(0.5*qRMSE + 0.3*qCov + 0.2*qCorn);
  if(!isFinite(raw)) raw=0;
  const score = Math.max(0, Math.min(100, raw));

  return {
    ok:true, score, rmse:err, coverage:qCov, cornerQ:qCorn, tol,
    templatePts:tplPts, aligned:best.aligned, theta:best.theta, s:best.s, t:best.t, R:best.R, A:best.A, PCA:best.PCA,
    method:best.method, mirrored
  };
}

/* ---------- Canvas & DPR ---------- */
const padWrap = qs('padWrap');
const canvas = qs('pad');
const ctx = canvas.getContext('2d');
let dpr=1, cssW=0, cssH=0;
let bg = document.createElement('canvas'), bgCtx = bg.getContext('2d');

function setVHVar(){ document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`); }
setVHVar();
window.addEventListener('resize', setVHVar);
window.addEventListener('orientationchange', setVHVar);

function resizeCanvas(){
  const rect = padWrap.getBoundingClientRect();
  cssW = Math.max(1, Math.round(rect.width));
  cssH = Math.max(1, Math.round(rect.height));
  dpr = Math.max(1, window.devicePixelRatio || 1);

  canvas.width  = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);

  bg.width  = Math.round(cssW * dpr);
  bg.height = Math.round(cssH * dpr);
  bgCtx.setTransform(dpr,0,0,dpr,0,0);

  rebuildBackground();
}
new ResizeObserver(resizeCanvas).observe(padWrap);

/* Unit <-> canvas */
function unitToCanvas([ux,uy]){ const s=Math.min(cssW,cssH)/2; return [ux*s + cssW/2, uy*s + cssH/2]; }
function canvasToUnit(x,y){ const s=Math.min(cssW,cssH)/2; return [(x - cssW/2)/s, (y - cssH/2)/s]; }

/* ---------- Background ---------- */
function drawGuideTo(ctx2d, template){
  ctx2d.save();
  ctx2d.lineWidth=5; ctx2d.strokeStyle='rgba(255,255,255,0.07)';
  for(const poly of template.strokes){
    ctx2d.beginPath();
    for(let i=0;i<poly.length;i++){ const [x,y]=unitToCanvas(poly[i]); if(i===0) ctx2d.moveTo(x,y); else ctx2d.lineTo(x,y); }
    ctx2d.stroke();
  }
  ctx2d.restore();
}
function rebuildBackground(){
  const tpl = currentTemplate();
  bgCtx.clearRect(0,0,cssW,cssH);
  bgCtx.strokeStyle='#2a2f3a'; bgCtx.lineWidth=1;
  bgCtx.beginPath();
  bgCtx.moveTo(cssW/2,0); bgCtx.lineTo(cssW/2,cssH);
  bgCtx.moveTo(0,cssH/2); bgCtx.lineTo(cssW,cssH/2);
  bgCtx.stroke();
  if(tpl && tpl.strokes) drawGuideTo(bgCtx, tpl);
  needsRedraw = true;
}

/* ---------- UI refs ---------- */
const topicSel = qs('topic');
const difficultySel = qs('difficulty');
const strict = qs('strict');
const strictVal = qs('strictVal');
const mirrorOK = qs('mirrorOK');
const liveJudge = qs('liveJudge');
const soundFX = qs('soundFX');

const methodSel = qs('methodSel');
const academicMode = qs('academicMode');
const vizResiduals = qs('vizResiduals');
const vizFrames = qs('vizFrames');
const vizPCA = qs('vizPCA');

const scoreEl = qs('score');
const rmseEl  = qs('rmse');
const covEl   = qs('cov');
const cornEl  = qs('corn');
const methodUsedEl = qs('methodUsed');
const detailsEl = qs('details');
const starsEl = qs('stars');
const timeEl  = qs('time');
const mathDump = qs('mathDump');

/* ---------- Template registry ---------- */
const STORAGE_KEY = 'lal_custom_templates_v1';
let customTemplates = new Map();
let customOrder = [];

function loadCustomTemplates(){
  try{
    const raw = localStorage.getItem(STORAGE_KEY);
    if(!raw) return;
    const arr = JSON.parse(raw);
    if(Array.isArray(arr)){
      for(const obj of arr){
        if(validateTemplateObject(obj)){
          customTemplates.set(obj.id, obj);
          customOrder.push(obj.id);
        }
      }
    }
  }catch(e){
    console.warn('Failed to load templates, clearing store', e);
    try{ localStorage.removeItem(STORAGE_KEY); }catch(_){}
  }
}
function saveCustomTemplates(){
  const arr = customOrder.map(id => customTemplates.get(id)).filter(Boolean);
  try{ localStorage.setItem(STORAGE_KEY, JSON.stringify(arr)); }catch(e){ console.warn('Failed to save templates', e); }
}

function topicValueFromId(id){ return `custom:${id}`; }
function isCustomValue(v){ return typeof v === 'string' && v.startsWith('custom:'); }
function idFromValue(v){ return v.split(':')[1]; }

function refreshTopicOptions(){
  const prev = topicSel.value;
  topicSel.innerHTML = '';
  const addOpt = (value, label, disabled=false) => {
    const opt = document.createElement('option');
    opt.value = value; opt.textContent = label; opt.disabled = disabled;
    topicSel.appendChild(opt);
  };
  addOpt('house-basic', 'House • Basic');
  addOpt('house-door-window', 'House • Details');
  addOpt('triangle', 'Triangle');
  addOpt('square', 'Square');
  addOpt('star', 'Star');

  if(customOrder.length){
    addOpt('sep', '────────', true);
    for(const id of customOrder){
      const t = customTemplates.get(id);
      if(!t) continue;
      addOpt(topicValueFromId(id), `Custom • ${t.name || 'Untitled'}`);
    }
  }
  // Try to restore previous selection if still present
  const options = Array.from(topicSel.options).map(o=>o.value);
  if(options.includes(prev)) topicSel.value = prev;
}

function currentTemplate(){
  const val = topicSel.value;
  if(isCustomValue(val)){
    const id = idFromValue(val);
    const t = customTemplates.get(id);
    if(t) return t;
  }
  return builtinIndex[val] || houseBasic();
}

/* ---------- State ---------- */
let strokes=[], current=null, undoStack=[], redoStack=[];
let startTime=null, timerId=null;
let lastLiveEval=null;
let needsRedraw=true;

/* ---------- Timer ---------- */
function formatTime(ms){ const s=Math.floor(ms/1000), m=Math.floor(s/60), r=s%60; return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`; }
function startTimer(){ startTime=Date.now(); if(timerId) cancelAnimationFrame(timerId); const tick=()=>{ timeEl.textContent=formatTime(Date.now()-startTime); timerId=requestAnimationFrame(tick); }; tick(); }
function stopTimer(){ if(timerId) cancelAnimationFrame(timerId); timerId=null; }

/* ---------- Input ---------- */
function pushUndo(){ undoStack.push(JSON.stringify(strokes)); if(undoStack.length>60) undoStack.shift(); redoStack.length=0; }
function pointerToUnit(ev){
  const r=canvas.getBoundingClientRect();
  const x=(ev.clientX ?? ev.pageX) - r.left;
  const y=(ev.clientY ?? ev.pageY) - r.top;
  return canvasToUnit(x,y);
}
function handlePointerDown(e){
  canvas.setPointerCapture(e.pointerId);
  if(!startTime) startTimer();
  current=[]; strokes.push(current); pushUndo();
  current.push(pointerToUnit(e));
  needsRedraw = true;
}
function handlePointerMove(e){
  if(!current) return;
  const evs = (e.getCoalescedEvents && e.getCoalescedEvents().length) ? e.getCoalescedEvents() : [e];
  for(const ev of evs){ current.push(pointerToUnit(ev)); }
  needsRedraw = true;
  if(liveJudge.checked) throttleLiveCheck();
}
function handleRawUpdate(e){ if(!current) return; current.push(pointerToUnit(e)); needsRedraw=true; }
function handlePointerUp(e){ if(!current) return; current.push(pointerToUnit(e)); current=null; needsRedraw=true; }

/* ---------- Academic viz helpers ---------- */
function drawArrow(p, q, color='#ff6', w=1.5){
  ctx.save();
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=w; ctx.lineCap='round';
  const [x1,y1]=unitToCanvas(p), [x2,y2]=unitToCanvas(q);
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  const ang=Math.atan2(y2-y1, x2-x1), len=10;
  ctx.beginPath();
  ctx.moveTo(x2,y2);
  ctx.lineTo(x2-len*Math.cos(ang-0.4), y2-len*Math.sin(ang-0.4));
  ctx.lineTo(x2-len*Math.cos(ang+0.4), y2-len*Math.sin(ang+0.4));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}
function drawFrame(origin=[0,0], R=[[1,0],[0,1]], scale=0.3){
  const i=[R[0][0]*scale, R[1][0]*scale], j=[R[0][1]*scale, R[1][1]*scale];
  drawArrow(origin, add(origin,i), '#4ade80', 2.2);
  drawArrow(origin, add(origin,j), '#60a5fa', 2.2);
}

/* ---------- Render loop ---------- */
function drawScene(){
  ctx.clearRect(0,0,cssW,cssH);
  ctx.drawImage(bg, 0, 0, bg.width/dpr, bg.height/dpr);

  // strokes
  ctx.lineCap='round'; ctx.lineJoin='round'; ctx.strokeStyle='#e8eaf0'; ctx.lineWidth=3;
  for(const s of strokes){ if(s.length<2) continue;
    ctx.beginPath(); let [x0,y0]=unitToCanvas(s[0]); ctx.moveTo(x0,y0);
    for(let i=1;i<s.length;i++){ const [x,y]=unitToCanvas(s[i]); ctx.lineTo(x,y); }
    ctx.stroke();
  }

  // overlays
  if(lastLiveEval && lastLiveEval.ok){
    ctx.lineWidth=2; ctx.strokeStyle='#2dd4bf';
    ctx.beginPath();
    for(let i=0;i<lastLiveEval.templatePts.length;i++){
      const [x,y]=unitToCanvas(lastLiveEval.templatePts[i]); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();

    ctx.strokeStyle='#ffd166';
    ctx.beginPath();
    for(let i=0;i<lastLiveEval.aligned.length;i++){
      const [x,y]=unitToCanvas(lastLiveEval.aligned[i]); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();

    if(academicMode.checked){
      if(vizFrames.checked){
        drawFrame([0,0], [[1,0],[0,1]], 0.25);
        const R = lastLiveEval.R || [[Math.cos(lastLiveEval.theta||0), -Math.sin(lastLiveEval.theta||0)],[Math.sin(lastLiveEval.theta||0), Math.cos(lastLiveEval.theta||0)]];
        drawFrame([0,0], R, 0.25);
      }
      if(vizResiduals.checked){
        const hash=SpatialHash(lastLiveEval.aligned, 0.05);
        for(let i=0;i<lastLiveEval.templatePts.length; i+=16){
          const p=lastLiveEval.templatePts[i];
          const {p:q}=hash.nearest(p);
          if(q) drawArrow(p, q, 'rgba(255,255,0,0.7)', 1.2);
        }
      }
      if(vizPCA.checked && lastLiveEval.PCA){
        const s=0.35, ex=lastLiveEval.PCA.ex.vecs, ey=lastLiveEval.PCA.ey.vecs;
        const origin=[0,0];
        drawArrow(origin, add(origin,[ex[0][0]*s, ex[0][1]*s]), '#f59e0b', 2.0);
        drawArrow(origin, add(origin,[ex[1][0]*s, ex[1][1]*s]), '#f59e0b', 2.0);
        drawArrow(origin, add(origin,[ey[0][0]*s, ey[0][1]*s]), '#f97316', 2.0);
        drawArrow(origin, add(origin,[ey[1][0]*s, ey[1][1]*s]), '#f97316', 2.0);
      }
    }
  }
}
function rafLoop(){ if(needsRedraw){ drawScene(); needsRedraw=false; } requestAnimationFrame(rafLoop); }

/* ---------- Live judge (throttled) ---------- */
let liveGate=0;
function throttleLiveCheck(){
  const now=performance.now();
  if(now - liveGate < 90) return; liveGate=now;
  doEvaluate(false);
}

/* ---------- HUD ---------- */
function showMetrics(res){
  if(!res || !res.ok){
    scoreEl.textContent='—'; rmseEl.textContent='—'; covEl.textContent='—'; cornEl.textContent='—'; methodUsedEl.textContent='—'; detailsEl.textContent=''; mathDump.textContent='—';
    return;
  }
  scoreEl.textContent = Math.round(res.score);
  rmseEl.textContent  = res.rmse.toFixed(3);
  covEl.textContent   = Math.round(res.coverage*100)+'%';
  cornEl.textContent  = Math.round(res.cornerQ*100)+'%';
  methodUsedEl.textContent = res.method + (res.mirrored ? ' (mirrored)' : '');
  detailsEl.textContent = `θ=${((res.theta||0)*180/Math.PI).toFixed(1)}° | s=${res.s?res.s.toFixed(3):'—'} | tol=${res.tol.toFixed(2)}`;

  const M = m => `[${m.map(r=>r.map(v=>(v>=0?' ':'')+v.toFixed(3)).join('  ')).join('\n ')}]`;
  const V = v => `[${v.map(x=>(x>=0?' ':'')+x.toFixed(3)).join(', ')}]`;
  let txt = '';
  if(res.R)  txt += `R (2×2):\n ${M(res.R)}\n\n`;
  if(res.A)  txt += `A (2×2, affine):\n ${M(res.A)}\n\n`;
  if(res.t)  txt += `t (translation): ${V(res.t)}\n\n`;
  if(res.PCA){
    txt += `PCA X basis (RX columns):\n ${M(res.PCA.RX)}\n\n`;
    txt += `PCA Y basis (RY columns):\n ${M(res.PCA.RY)}\n\n`;
    txt += `eig(X): ${V(res.PCA.ex.vals)}\n`;
    txt += `eig(Y): ${V(res.PCA.ey.vals)}\n`;
  }
  mathDump.textContent = txt || '—';
}

function beep(freq=660, ms=120){
  if(!soundFX.checked) return;
  const ac=new (window.AudioContext||window.webkitAudioContext)();
  const o=ac.createOscillator(), g=ac.createGain(); o.type='sine'; o.frequency.value=freq;
  o.connect(g); g.connect(ac.destination); g.gain.value=0.06; o.start();
  setTimeout(()=>{ o.stop(); ac.close(); }, ms);
}
function starsFromScore(score, elapsedMs, diff){
  const t = {easy:90000, normal:60000, hard:45000, insane:35000}[diff] || 60000;
  let s = score; if(elapsedMs < t) s+=5; if(elapsedMs < 0.6*t) s+=5;
  if(s>=90) return "★★★"; if(s>=75) return "★★☆"; if(s>=60) return "★☆☆"; return "☆☆☆";
}

/* ---------- Buttons & wiring ---------- */
function clearAll(){
  strokes=[]; undoStack.length=0; redoStack.length=0; current=null; startTime=null; stopTimer();
  lastLiveEval=null; showMetrics(null); starsEl.textContent='☆☆☆'; needsRedraw=true;
}
function undo(){ if(!undoStack.length) return; redoStack.push(JSON.stringify(strokes)); strokes = JSON.parse(undoStack.pop()); needsRedraw=true; }
function redo(){ if(!redoStack.length) return; undoStack.push(JSON.stringify(strokes)); strokes = JSON.parse(redoStack.pop()); needsRedraw=true; }

qs('clearBtn').addEventListener('click', clearAll);
qs('checkBtn').addEventListener('click', ()=>doEvaluate(true));
qs('undoBtn').addEventListener('click', undo);
qs('redoBtn').addEventListener('click', redo);

strict.addEventListener('input', ()=>{ strictVal.textContent=(Number(strict.value)/100).toFixed(2); if(liveJudge.checked) throttleLiveCheck(); });
[topicSel, difficultySel, methodSel].forEach(el=>el.addEventListener('change', ()=>{ rebuildBackground(); if(liveJudge.checked) throttleLiveCheck(); }));
[academicMode, vizResiduals, vizFrames, vizPCA].forEach(el=>el.addEventListener('change', ()=>{ needsRedraw=true; }));

canvas.addEventListener('pointerdown', handlePointerDown);
canvas.addEventListener('pointermove', handlePointerMove, {passive:true});
canvas.addEventListener('pointerup', handlePointerUp);
canvas.addEventListener('pointercancel', handlePointerUp);
canvas.addEventListener('pointerout', handlePointerUp);
canvas.addEventListener('pointerrawupdate', handleRawUpdate, {passive:true});

window.addEventListener('keydown', (e)=>{
  if((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='z'){ e.preventDefault(); undo(); }
  if((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='y'){ e.preventDefault(); redo(); }
});

/* ---------- Evaluate ---------- */
function doEvaluate(useICP){
  const tol = Number(strict.value)/100;
  const tpl = currentTemplate();
  const res = scoreLA(strokes, tpl, tol, difficultySel.value, mirrorOK.checked, methodSel.value, useICP);
  lastLiveEval = res;
  showMetrics(res);
  needsRedraw = true;
  if(res.ok){
    const elapsed = startTime ? (Date.now()-startTime) : 0;
    const starText = starsFromScore(res.score, elapsed, difficultySel.value);
    starsEl.textContent = starText;
    if(starText==="★★★") beep(880,140);
    else if(starText==="★★☆") beep(740,120);
  }
}

/* ---------- Template sharing ---------- */
const tplNameEl = qs('tplName');
const btnUseAsTemplate = qs('btnUseAsTemplate');
const btnExportTemplate = qs('btnExportTemplate');
const btnImportTemplate = qs('btnImportTemplate');
const importFileEl = qs('importFile');
const btnResetTemplates = qs('btnResetTemplates');

function flatten(arr){ return arr.reduce((a,b)=>a.concat(b), []); }
function uniquePoints(pts, minDist=0.05){
  const out=[];
  for(const p of pts){
    let ok=true;
    for(const q of out){
      if(Math.hypot(p[0]-q[0], p[1]-q[1]) < minDist){ ok=false; break; }
    }
    if(ok) out.push(p);
  }
  return out;
}
function detectCornersFromStrokes(strokes, maxPerStroke=6){
  const corners=[];
  for(const s of strokes){
    if(s.length<=2){
      if(s[0]) corners.push(s[0]);
      if(s[s.length-1]) corners.push(s[s.length-1]);
      continue;
    }
    const cand=[];
    for(let i=1;i<s.length-1;i++){
      const a=sub(s[i], s[i-1]), b=sub(s[i+1], s[i]);
      const na=norm(a)||1e-9, nb=norm(b)||1e-9;
      let cos=(dot(a,b)/(na*nb)); cos=Math.max(-1, Math.min(1, cos));
      const k = 1 - (cos+1)/2;
      cand.push({idx:i, k, p:s[i]});
    }
    cand.sort((u,v)=>v.k-u.k);
    const take = cand.slice(0, maxPerStroke).map(x=>x.p);
    corners.push(s[0], ...take, s[s.length-1]);
  }
  return uniquePoints(corners, 0.06).slice(0, 24);
}

function normalizeTemplateObj(obj){
  const allPts = flatten(obj.strokes);
  const c = centroid(allPts), r = Math.max(rmsRadius(translate(allPts, c)), 1e-8);
  const normPoint = ([x,y]) => [(x-c[0])/r, (y-c[1])/r];
  const strokes = obj.strokes.map(poly => poly.map(normPoint));
  const corners = (obj.corners||[]).map(normPoint);
  return {...obj, strokes, corners, meta:{...(obj.meta||{}), normalized:true}};
}
function validateTemplateObject(o){
  if(!o || typeof o!=='object') return false;
  if(!Array.isArray(o.strokes) || !o.strokes.length) return false;
  for(const poly of o.strokes){
    if(!Array.isArray(poly) || !poly.length) return false;
    for(const p of poly){
      if(!Array.isArray(p) || p.length!==2 || !isFinite(p[0]) || !isFinite(p[1])) return false;
    }
  }
  if(o.corners){
    if(!Array.isArray(o.corners)) return false;
    for(const p of o.corners){
      if(!Array.isArray(p) || p.length!==2 || !isFinite(p[0]) || !isFinite(p[1])) return false;
    }
  }
  return true;
}
function makeTemplateFromCurrentDrawing(name){
  const nameSafe = (name && name.trim()) ? name.trim() : 'Untitled';
  const unitStrokes = strokes.map(poly => poly.slice());
  if(!unitStrokes.length || !unitStrokes[0] || unitStrokes[0].length===0){
    alert('Draw something first, then try again.'); return null;
  }
  const corners = detectCornersFromStrokes(unitStrokes);
  const t = {
    kind:"custom",
    id: `t${Date.now().toString(36)}${Math.random().toString(36).slice(2,6)}`,
    name: nameSafe,
    strokes: unitStrokes,
    corners,
    meta: { version:1, createdAt: new Date().toISOString(), normalized:true }
  };
  return t;
}
function addCustomTemplate(t){
  if(!validateTemplateObject(t)) { alert('Invalid template.'); return; }
  const norm = t.meta && t.meta.normalized ? t : normalizeTemplateObj(t);
  customTemplates.set(norm.id, norm);
  if(!customOrder.includes(norm.id)) customOrder.push(norm.id);
  saveCustomTemplates();
  refreshTopicOptions();
  topicSel.value = topicValueFromId(norm.id);
  rebuildBackground();
  clearAll();
}
function exportTemplate(t){
  const payload = {
    $schema: "https://example.com/lal-template.schema.json",
    type: "LinearAlgebraLabTemplate",
    version: 1,
    name: t.name || 'Template',
    strokes: t.strokes,
    corners: t.corners || [],
    meta: { ...(t.meta||{}), exportedAt: new Date().toISOString() }
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const base = (t.name||'template').toLowerCase().replace(/[^a-z0-9-_]+/g,'-').replace(/-+/g,'-').replace(/^-|-$/g,'');
  a.download = `${base || 'template'}.lal.json`;
  document.body.appendChild(a);
  a.click();
  setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

/* Share UI */
btnUseAsTemplate.addEventListener('click', ()=>{
  const t = makeTemplateFromCurrentDrawing(tplNameEl.value);
  if(t) addCustomTemplate(t);
});
btnExportTemplate.addEventListener('click', ()=>{
  const t = currentTemplate();
  exportTemplate(t);
});
btnImportTemplate.addEventListener('click', ()=>importFileEl.click());
importFileEl.addEventListener('change', ()=>{
  const f = importFileEl.files && importFileEl.files[0];
  if(!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    try{
      const obj = JSON.parse(reader.result);
      if(obj && obj.type==="LinearAlgebraLabTemplate"){
        const t = {
          kind:"custom",
          id: `t${Date.now().toString(36)}${Math.random().toString(36).slice(2,6)}`,
          name: obj.name || 'Imported',
          strokes: obj.strokes,
          corners: obj.corners || [],
          meta: { ...(obj.meta||{}), importedAt: new Date().toISOString() }
        };
        addCustomTemplate(t);
      }else if(validateTemplateObject(obj)){
        const t = {
          kind:"custom",
          id: `t${Date.now().toString(36)}${Math.random().toString(36).slice(2,6)}`,
          name: obj.name || 'Imported',
          strokes: obj.strokes,
          corners: obj.corners || [],
          meta: { importedAt: new Date().toISOString() }
        };
        addCustomTemplate(t);
      }else{
        alert('File is not a valid template.');
      }
    }catch(e){
      alert('Failed to read template file.');
    }finally{
      importFileEl.value = '';
    }
  };
  reader.readAsText(f);
});
btnResetTemplates.addEventListener('click', ()=>{
  if(confirm('Remove all custom templates?')){
    customTemplates.clear();
    customOrder = [];
    try{ localStorage.removeItem(STORAGE_KEY); }catch(_){}
    refreshTopicOptions();
    topicSel.value = 'house-basic';
    rebuildBackground();
  }
});

/* ---------- Boot ---------- */
loadCustomTemplates();
refreshTopicOptions();
requestAnimationFrame(()=>{ resizeCanvas(); requestAnimationFrame(rafLoop); });

/* Optional: surface JS errors in the console and alert once (helps debugging) */
window.addEventListener('error', (e)=>{
  console.error('Runtime error:', e.message, e.error);
});
