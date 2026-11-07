// POST + SSE：EventSource 只支持 GET，长任务必须用 fetch + ReadableStream 手动解析。
//
// 用法：
//   const task = await openTask("/api/train", body, { onEvent(name, data) {...} });
//   task.abort();
export async function openTask(url, body, handlers = {}) {
  const controller = new AbortController();
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
    signal: controller.signal,
  });

  if (!response.ok || !response.body) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload?.error?.message || JSON.stringify(payload);
    } catch (err) {
      detail = `HTTP ${response.status}`;
    }
    throw new Error(detail || `请求失败（HTTP ${response.status}）`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let settled = false;

  (async () => {
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let index = buffer.indexOf("\n\n");
        while (index >= 0) {
          const chunk = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          dispatch(chunk, {
            onEvent(name, data) {
              if (name === "done" || name === "error") settled = true;
              if (handlers.onEvent) handlers.onEvent(name, data);
            },
          });
          index = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      settled = true;
      if (err.name !== "AbortError" && handlers.onError) {
        handlers.onError(err);
      }
    } finally {
      // 正常结束一定会先收到 done/error；没有收到说明连接中断，
      // 必须回调 onError，否则前端会永久停在“任务运行中”。
      if (!settled && !controller.signal.aborted && handlers.onError) {
        handlers.onError(new Error("任务连接已断开，请刷新确认运行状态"));
      }
    }
  })();

  return {
    abort() {
      controller.abort();
    },
  };
}

function dispatch(chunk, handlers) {
  let event = "message";
  const dataLines = [];
  chunk.split("\n").forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  });
  if (!dataLines.length) return; // 心跳

  let data = {};
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch (err) {
    data = { text: dataLines.join("\n"), level: "info" };
  }
  if (handlers.onEvent) handlers.onEvent(event, data);
}
