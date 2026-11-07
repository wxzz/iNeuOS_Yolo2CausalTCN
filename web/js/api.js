// 统一的后端请求封装：始终解开 {ok, data} / {ok:false, error} 信封
export class ApiError extends Error {
  constructor(code, message, detail) {
    super(message);
    this.code = code;
    this.detail = detail || "";
  }
}

async function parse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch (err) {
    throw new ApiError("E_BAD_RESPONSE", `响应不是 JSON（HTTP ${response.status}）`, String(err));
  }
  if (!payload || payload.ok !== true) {
    const error = (payload && payload.error) || {};
    throw new ApiError(
      error.code || "E_UNKNOWN",
      error.message || `请求失败（HTTP ${response.status}）`,
      error.detail || ""
    );
  }
  return payload.data;
}

export async function get(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    if (Array.isArray(value)) {
      value.forEach((item) => url.searchParams.append(key, item));
    } else {
      url.searchParams.set(key, value);
    }
  });
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  return parse(response);
}

export async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parse(response);
}

export async function put(path, body = {}) {
  const response = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parse(response);
}
