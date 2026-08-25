// 各ページのスクリプトを最小の DOM スタブで実行し、実データで例外が出ないかを見る。
// ブラウザを開けない環境での代替。node tests/smoke_pages.js
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const WEB = path.join(ROOT, "web");

function makeEl(tag) {
  const el = {
    tagName: tag, dataset: {}, style: {}, classList: { toggle() {}, add() {}, remove() {} },
    children: [], value: "", checked: false, textContent: "", _html: "",
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
    insertAdjacentHTML(_pos, html) { this._html += html; },
    addEventListener() {}, appendChild(c) { this.children.push(c); },
    querySelector() { return makeEl("div"); },
    querySelectorAll() { return []; },
    setAttribute() {}, getAttribute() { return null; },
  };
  return el;
}

function run(page) {
  const html = fs.readFileSync(path.join(WEB, page), "utf-8");
  // 外部スクリプト(src 付き)も同じ文脈で読み込む
  const externals = [...html.matchAll(/<script src="([^"]+)"><\/script>/g)]
    .map((m) => m[1])
    .filter((n) => n !== "nav.js");
  const scripts = externals.map((n) => fs.readFileSync(path.join(WEB, n), "utf-8"))
    .concat([...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]));
  const els = {};
  const doc = {
    body: makeEl("body"),
    title: "",
    getElementById: (id) => (els[id] = els[id] || makeEl("div")),
    querySelector: (sel) => {
      els["_sel_" + sel] = els["_sel_" + sel] || makeEl("div");
      return els["_sel_" + sel];
    },
    querySelectorAll: () => [],
  };
  doc.body.dataset.page = page.replace(".html", "");
  els.q = makeEl("input"); els.q.value = "呉清源";
  els.work = makeEl("select"); els.chunk = makeEl("select");
  const errors = [];
  const ctx = {
    document: doc,
    location: { search: "?w=42620" },
    URLSearchParams: global.URLSearchParams,
    Math, Object, Array, String, Number, JSON, console, Map, Set,
    Uint8Array, Uint16Array, Uint32Array, Float32Array,
    Float16Array: typeof Float16Array !== "undefined" ? Float16Array : undefined,
    window: {},
    fetch: (u) => {
      const f = path.join(WEB, u);
      if (!fs.existsSync(f)) { errors.push("missing " + u); return Promise.reject(new Error(u)); }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, "utf-8"))),
        text: () => Promise.resolve(fs.readFileSync(f, "utf-8")),
        arrayBuffer: () => {
          const b = fs.readFileSync(f);
          return Promise.resolve(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
        },
      });
    },
    Promise,
  };
  const vm = require("vm");
  vm.createContext(ctx);
  for (const s of scripts) {
    try { vm.runInContext(s, ctx); } catch (e) { errors.push(page + ": " + e.message); }
  }
  return new Promise((res) => setTimeout(() => res({ page, errors, els }), 400));
}

(async () => {
  let bad = 0;
  for (const p of ["index.html", "lens.html", "topic.html", "reader.html", "search.html"]) {
    const r = await run(p);
    const rendered = Object.entries(r.els)
      .filter(([, e]) => e.innerHTML && e.innerHTML.length > 40)
      .map(([k, e]) => k + "(" + e.innerHTML.length + ")");
    if (r.errors.length) { console.log("NG", p, r.errors); bad++; }
    else console.log("OK", p, "描画:", rendered.join(" ") || "(なし)");
    if (!r.errors.length && rendered.length === 0) { console.log("   ⚠ 何も描画されていない"); bad++; }
  }
  process.exit(bad ? 1 : 0);
})();
