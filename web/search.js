// 検索の中身。一節から探す(事前計算ベクトル)と 自由語検索(BM25)。
// どちらも**クエリ時のモデル推論を持たない**(N-01)。重いデータは使うときだけ取りに行く。

const DATA = "data/";
const MAX_PER_WORK = 2; // 同一作品からの結果の上限(F-16 の多様性制限)

const state = {
  works: null, byWork: null, index: null, vectors: null, dims: 0,
  skip: null, bm: null, textCache: new Map(),
};

async function loadJSON(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(name + " が読めない");
  return r.json();
}

async function loadBin(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(name + " が読めない");
  return r.arrayBuffer();
}

// float16 → float32。Float16Array があれば使い、無ければ手で展開する。
function decodeF16(buf, n, dims) {
  if (typeof Float16Array !== "undefined") {
    return Float32Array.from(new Float16Array(buf, 0, n * dims));
  }
  const u = new Uint16Array(buf, 0, n * dims);
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) out[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) out[i] = f ? NaN : (s ? -Infinity : Infinity);
    else out[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

// 索引は並行配列で来る(配信量のため)。1 チャンク分を組み立てて返す。
function chunkAt(doc) {
  const x = state.index;
  if (doc == null || doc < 0 || doc >= x.n) return null;
  return { w: x.works[x.w[doc]], p: x.p[doc], q: x.q[doc], n: x.len[doc] };
}

async function ensureBase() {
  if (state.works) return;
  const [w, idx, meta] = await Promise.all([
    loadJSON("works.json"), loadJSON("chunk_index.json"), loadJSON("bm25_meta.json"),
  ]);
  state.works = w.works;
  state.byWork = Object.fromEntries(w.works.map((r) => [r.id, r]));
  state.index = idx;
  state.dims = idx.dims;
  state.skip = meta.skipped || {};
  state.bmMeta = meta;
}

async function ensureVectors(onProgress) {
  await ensureBase();
  if (state.vectors) return;
  onProgress && onProgress("ベクトルを読み込み中(約 6 MB)…");
  const buf = await loadBin("chunk_vectors.f16");
  state.vectors = decodeF16(buf, state.index.n, state.dims);
  onProgress && onProgress("");
}

async function ensureBm25(onProgress) {
  await ensureBase();
  if (state.bm) return;
  onProgress && onProgress("索引を読み込み中(約 2 MB)…");
  const [terms, off, post, docs] = await Promise.all([
    fetch(DATA + "bm25_terms.txt").then((r) => r.text()),
    loadBin("bm25_offsets.bin"), loadBin("bm25_postings.bin"), loadBin("bm25_docs.bin"),
  ]);
  const list = terms.split("\n");
  const map = new Map();
  list.forEach((t, i) => map.set(t, i));
  state.bm = {
    terms: list, map,
    offsets: new Uint32Array(off),
    postings: new Uint8Array(post),
    doclen: new Uint16Array(docs),
    maxlen: list.reduce((a, t) => Math.max(a, t.length), 1),
  };
  onProgress && onProgress("");
}

// --- BM25 ------------------------------------------------------------------
function readPostings(bm, term) {
  const i = bm.map.get(term);
  if (i === undefined) return null;
  let pos = bm.offsets[i];
  const buf = bm.postings;
  const rd = () => {
    let n = 0, shift = 0;
    for (;;) {
      const b = buf[pos++];
      n |= (b & 0x7f) << shift;
      if (!(b & 0x80)) return n >>> 0;
      shift += 7;
    }
  };
  const df = rd();
  const out = new Array(df);
  let doc = 0;
  for (let k = 0; k < df; k++) {
    doc += rd();
    out[k] = [doc, rd()];
  }
  return out;
}

function tokenize(bm, query) {
  const out = [];
  let i = 0;
  while (i < query.length) {
    let hit = null;
    for (let n = Math.min(bm.maxlen, query.length - i); n >= 1; n--) {
      const cand = query.substr(i, n);
      if (bm.map.has(cand)) { hit = cand; break; }
    }
    if (hit) { out.push(hit); i += hit.length; } else { i += 1; }
  }
  return out;
}

function bm25Search(query, top) {
  const bm = state.bm, meta = state.bmMeta;
  const { n_docs: N, avgdl, k1, b } = meta;
  const scores = new Map();
  const used = [];
  for (const term of tokenize(bm, query)) {
    const plist = readPostings(bm, term);
    if (!plist || !plist.length) continue;
    used.push(term);
    const idf = Math.log(1 + (N - plist.length + 0.5) / (plist.length + 0.5));
    for (const [doc, tf] of plist) {
      const dl = bm.doclen[doc] || 1;
      const s = (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avgdl));
      scores.set(doc, (scores.get(doc) || 0) + s);
    }
  }
  const rows = [...scores.entries()].sort((x, y) => y[1] - x[1] || x[0] - y[0]);
  return { terms: used, hits: diversify(rows, top) };
}

// --- ベクトル近傍 ----------------------------------------------------------
function nearest(row, top) {
  const V = state.vectors, d = state.dims, n = state.index.n;
  const q = V.subarray(row * d, row * d + d);
  const out = [];
  for (let i = 0; i < n; i++) {
    if (i === row) continue;
    let s = 0;
    const base = i * d;
    for (let k = 0; k < d; k++) s += V[base + k] * q[k];
    out.push([i, s]);
  }
  out.sort((a, b) => b[1] - a[1]);
  return diversify(out, top);
}

// 同一作品からの結果を絞り、二重版で除外された作品を落とす(F-16 / F-18)
function diversify(rows, top) {
  const per = new Map();
  const out = [];
  for (const [doc, score] of rows) {
    const c = chunkAt(doc);
    if (!c || state.skip[c.w]) continue;
    const k = per.get(c.w) || 0;
    if (k >= MAX_PER_WORK) continue;
    per.set(c.w, k + 1);
    out.push({ doc, score, chunk: c });
    if (out.length >= top) break;
  }
  return out;
}

// --- 本文の取り出し --------------------------------------------------------
async function chunkText(c) {
  if (!state.textCache.has(c.w)) {
    state.textCache.set(c.w, loadJSON("texts/" + c.w + ".json"));
  }
  const d = await state.textCache.get(c.w);
  const paras = d.paras.slice(c.p, c.q + 1).map((p) =>
    p.map((x) => (x[0] === "t" ? x[1] : x[0] === "r" ? x[1] : "")).join("")
  );
  return paras.join("");
}

window.AngoSearch = {
  state, ensureBase, ensureVectors, ensureBm25, bm25Search, nearest,
  chunkText, tokenize, chunkAt,
};
