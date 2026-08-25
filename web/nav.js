// 全ページ共通のヘッダ・フッタ。ページごとに aria-current を立てる。
(function () {
  const page = document.body.dataset.page || "";
  const items = [
    ["index.html", "年表・作品一覧"],
    ["lens.html", "文体マップ"],
    ["topic.html", "主題マップ"],
    ["search.html", "一節から探す"],
    ["reader.html", "リーダー"],
  ];
  const nav = items
    .map(([h, t]) => `<a href="${h}"${h.startsWith(page) && page ? ' aria-current="page"' : ""}>${t}</a>`)
    .join("");
  const head = document.querySelector("header");
  if (head) head.insertAdjacentHTML("beforeend", `<nav>${nav}</nav>`);
  document.body.insertAdjacentHTML(
    "beforeend",
    '<footer><span>&copy; 2026 twill3c</span>' +
      '<a href="https://github.com/twill3c/ango-atlas">GitHub</a>' +
      '<a id="link-howto" href="https://claude.ai/code/artifact/f036d15f-ee37-4ef5-b6bf-d9389dfd1acc">歩き方</a>' +
      '<a id="link-design" href="https://claude.ai/code/artifact/074115c0-6153-4e31-bb1e-516e0987c7f2">設計図</a>' +
      '<a href="https://app-menu-amber.vercel.app/">App Menu</a></footer>'
  );
})();
