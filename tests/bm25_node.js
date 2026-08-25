// web/search.js の**実物**を node で動かし、問いごとの上位を JSON で出す。
// Python の参照実装(pipeline/bm25_ref.py)と突き合わせて二実装照合する。
// 使い方: node tests/bm25_node.js "呉清源" "桜の花と山賊" …
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const WEB = path.join(ROOT, "web");

const ctx = {
  console,
  Math, Object, Array, String, Number, JSON, Map, Set, Promise,
  Uint8Array, Uint16Array, Uint32Array, Float32Array,
  Float16Array: typeof Float16Array !== "undefined" ? Float16Array : undefined,
  URLSearchParams,
  window: {},
  fetch: (u) => {
    const p = path.join(WEB, u);
    if (!fs.existsSync(p)) return Promise.reject(new Error("missing " + u));
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(JSON.parse(fs.readFileSync(p, "utf-8"))),
      text: () => Promise.resolve(fs.readFileSync(p, "utf-8")),
      arrayBuffer: () => {
        const b = fs.readFileSync(p);
        return Promise.resolve(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
      },
    });
  },
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(WEB, "search.js"), "utf-8"), ctx);

// 引数が "#123" の形なら、そのチャンクを起点にしたベクトル近傍を返す
(async () => {
  const S = ctx.window.AngoSearch;
  const args = process.argv.slice(2);
  const out = {};
  if (args.some((a) => a.startsWith("#"))) await S.ensureVectors(null);
  if (args.some((a) => !a.startsWith("#"))) await S.ensureBm25(null);
  if (args.includes("--time")) {
    await S.ensureVectors(null);
    const t0 = Date.now();
    const reps = 20;
    for (let k = 0; k < reps; k++) S.nearest(7282 + k, 12);
    out["--time"] = { ms_per_query: (Date.now() - t0) / reps };
    console.log(JSON.stringify(out));
    return;
  }
  for (const a of args) {
    if (a.startsWith("#")) {
      const hits = S.nearest(Number(a.slice(1)), 10);
      out[a] = { hits: hits.map((h) => [h.doc, Number(h.score.toFixed(4))]) };
    } else {
      const { terms, hits } = S.bm25Search(a, 10);
      out[a] = { terms, hits: hits.map((h) => [h.doc, Number(h.score.toFixed(4))]) };
    }
  }
  console.log(JSON.stringify(out));
})().catch((e) => {
  console.error("ERR", e.message);
  process.exit(1);
});
