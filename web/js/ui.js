// 通用 DOM 工具：全部通过 createElement + textContent 构建，禁止 innerHTML 拼接用户数据。
export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([key, value]) => {
    if (value === false || value === null || value === undefined) return;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "checked") node.checked = Boolean(value);
    else if (key === "value") node.value = value;
    else if (key === "disabled") node.disabled = Boolean(value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      node.setAttribute(key, value);
    }
  });
  children.flat().forEach((child) => {
    if (child === null || child === undefined || child === false) return;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  });
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function toast(message, kind = "") {
  const box = document.getElementById("toasts");
  const node = h("div", { class: `toast ${kind}`.trim(), text: message });
  box.append(node);
  setTimeout(() => node.remove(), kind === "err" ? 8000 : 4000);
}

// 选项弹窗：单选列表（默认第一项），点“确定”返回选中值，取消/遮罩/Esc 返回 null
let modalSeq = 0;

export function choose({
  title,
  message = "",
  note = "",
  options = [],
  defaultValue,
  confirmText = "确定",
  cancelText = "取消",
}) {
  return new Promise((resolve) => {
    if (!options.length) {
      resolve(null);
      return;
    }
    const group = `modal-choice-${++modalSeq}`;
    const initial = defaultValue ?? options[0].value;
    const mask = h("div", { class: "modal-mask" });
    const finish = (value) => {
      document.removeEventListener("keydown", onKey);
      mask.remove();
      resolve(value);
    };
    const confirm = () => {
      const checked = mask.querySelector(`input[name="${group}"]:checked`);
      finish(checked ? checked.value : initial);
    };
    const onKey = (event) => {
      if (event.key === "Escape") finish(null);
    };

    const choices = options.map((option) =>
      h(
        "label",
        { class: `modal-choice ${option.kind || ""}`.trim() },
        h("input", {
          type: "radio",
          name: group,
          value: option.value,
          checked: option.value === initial,
        }),
        h("span", { text: option.label })
      )
    );
    const sync = () => {
      choices.forEach((label) =>
        label.classList.toggle("selected", label.querySelector("input").checked)
      );
    };
    choices.forEach((label) =>
      label.querySelector("input").addEventListener("change", sync)
    );
    sync();

    mask.append(
      h(
        "div",
        { class: "modal", role: "dialog", "aria-modal": "true" },
        h("h3", { text: title }),
        message ? h("p", { class: "modal-message", text: message }) : null,
        h("div", { class: "modal-choices" }, choices),
        note ? h("div", { class: "modal-note", text: note }) : null,
        h(
          "div",
          { class: "modal-actions" },
          h("button", { class: "primary", text: confirmText, onclick: confirm }),
          h("button", { text: cancelText, onclick: () => finish(null) })
        )
      )
    );
    mask.addEventListener("click", (event) => {
      if (event.target === mask) finish(null);
    });
    document.addEventListener("keydown", onKey);
    document.body.append(mask);
  });
}

// 多选弹窗：确定返回选中的 value 数组，取消/遮罩/Esc 返回 null（不保存）
let multiSeq = 0;

export function pickMany({
  title,
  message = "",
  note = "",
  options = [],
  selected = [],
  confirmText = "确定",
  cancelText = "取消",
  searchPlaceholder = "搜索…",
  thumbUrl = null,
}) {
  return new Promise((resolve) => {
    if (!options.length) {
      resolve(null);
      return;
    }
    const group = `modal-many-${++multiSeq}`;
    const chosen = new Set(selected);
    const mask = h("div", { class: "modal-mask" });
    const finish = (value) => {
      document.removeEventListener("keydown", onKey);
      mask.remove();
      resolve(value);
    };
    const confirm = () => {
      finish(options.filter((option) => chosen.has(option.value)).map((option) => option.value));
    };
    const onKey = (event) => {
      if (event.key === "Escape") finish(null);
    };

    const count = h("span", { class: "modal-count" });
    const sync = () => {
      count.textContent = `已选 ${chosen.size} / ${options.length}`;
    };

    const rows = options.map((option) => {
      const input = h("input", {
        type: "checkbox",
        name: group,
        value: option.value,
        checked: chosen.has(option.value),
      });
      const row = h(
        "label",
        { class: `modal-item ${chosen.has(option.value) ? "checked" : ""}`.trim() },
        input,
        thumbUrl
          ? h("img", {
              class: "modal-thumb",
              src: thumbUrl(option.value),
              alt: "",
              loading: "lazy",
              onerror: (event) => event.target.remove(),
            })
          : null,
        h("span", { class: "modal-label", text: option.label }),
        option.hint ? h("span", { class: "modal-hint", text: option.hint }) : null
      );
      input.addEventListener("change", () => {
        if (input.checked) chosen.add(option.value);
        else chosen.delete(option.value);
        row.classList.toggle("checked", input.checked);
        sync();
      });
      return { row, input, text: `${option.label} ${option.value}`.toLowerCase() };
    });

    const search = h("input", { type: "text", placeholder: searchPlaceholder });
    search.addEventListener("input", () => {
      const keyword = search.value.trim().toLowerCase();
      rows.forEach(({ row, text }) => {
        row.hidden = Boolean(keyword) && !text.includes(keyword);
      });
    });

    // 全选/清空只作用于勾选框本身；“全选”针对当前筛选出的可见行
    const selectAll = () => {
      rows.forEach(({ row, input }) => {
        if (row.hidden) return;
        input.checked = true;
        chosen.add(input.value);
        row.classList.add("checked");
      });
      sync();
    };
    const clearAll = () => {
      rows.forEach(({ row, input }) => {
        input.checked = false;
        chosen.delete(input.value);
        row.classList.remove("checked");
      });
      sync();
    };

    mask.append(
      h(
        "div",
        { class: "modal modal-wide", role: "dialog", "aria-modal": "true" },
        h("h3", { text: title }),
        message ? h("p", { class: "modal-message", text: message }) : null,
        h(
          "div",
          { class: "modal-toolbar" },
          search,
          h("button", { type: "button", text: "全选", onclick: selectAll }),
          h("button", { type: "button", text: "清空", onclick: clearAll }),
          count
        ),
        h("div", { class: "modal-list" }, rows.map((item) => item.row)),
        note ? h("div", { class: "modal-note", text: note }) : null,
        h(
          "div",
          { class: "modal-actions" },
          h("button", { class: "primary", text: confirmText, onclick: confirm }),
          h("button", { text: cancelText, onclick: () => finish(null) })
        )
      )
    );
    mask.addEventListener("click", (event) => {
      if (event.target === mask) finish(null);
    });
    document.addEventListener("keydown", onKey);
    document.body.append(mask);
    sync();
  });
}

export function badge(text, kind = "") {
  return h("span", { class: `badge ${kind}`.trim(), text });
}

export function dot(kind) {
  return h("span", { class: `dot ${kind}` });
}

export function section(title, ...children) {
  return h("div", { class: "card" }, h("h2", { text: title }), ...children);
}

export function download(filename, text) {
  const blob = new Blob(["\ufeff" + text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
