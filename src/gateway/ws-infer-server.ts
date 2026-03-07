/**
 * WebSocket Inference Server
 *
 * Hosts a standalone WebSocket endpoint for local inference via Ollama.
 * Runs in the foreground, accepting connections at ws://<host>:<port>/infer.
 *
 * Protocol (JSON messages):
 *
 * Client → Server:
 *   { id, model, messages, stream?, system?, options? }
 *
 * Server → Client (streaming token delta):
 *   { id, type: "delta", text }
 *
 * Server → Client (inference complete):
 *   { id, type: "done", usage: { input, output } }
 *
 * Server → Client (error):
 *   { id, type: "error", error }
 *
 * Server → Client (on connect):
 *   { type: "ready", version: "1" }
 */

import { createServer as createHttpServer, type IncomingMessage } from "node:http";
import WebSocket, { WebSocketServer } from "ws";
import { OLLAMA_NATIVE_BASE_URL, parseNdjsonStream } from "../agents/ollama-stream.js";

// ── Protocol types ────────────────────────────────────────────────────────────

interface InferRequest {
  id: string;
  model: string;
  messages: Array<{ role: string; content: string }>;
  stream?: boolean;
  system?: string;
  options?: {
    temperature?: number;
    maxTokens?: number;
    numCtx?: number;
  };
}

type ServerMessage =
  | { type: "ready"; version: string }
  | { id: string; type: "delta"; text: string }
  | { id: string; type: "done"; usage: { input: number; output: number } }
  | { id: string; type: "error"; error: string };

// ── Validation ────────────────────────────────────────────────────────────────

function parseInferRequest(raw: unknown): { ok: true; value: InferRequest } | { ok: false; error: string } {
  if (typeof raw !== "object" || raw === null) {
    return { ok: false, error: "message must be a JSON object" };
  }
  const obj = raw as Record<string, unknown>;

  if (typeof obj.id !== "string" || !obj.id) {
    return { ok: false, error: "missing required field: id (string)" };
  }
  if (typeof obj.model !== "string" || !obj.model) {
    return { ok: false, error: "missing required field: model (string)" };
  }
  if (!Array.isArray(obj.messages)) {
    return { ok: false, error: "missing required field: messages (array)" };
  }
  for (const [i, msg] of obj.messages.entries()) {
    if (typeof msg !== "object" || msg === null) {
      return { ok: false, error: `messages[${i}] must be an object` };
    }
    const m = msg as Record<string, unknown>;
    if (typeof m.role !== "string") {
      return { ok: false, error: `messages[${i}].role must be a string` };
    }
    if (typeof m.content !== "string") {
      return { ok: false, error: `messages[${i}].content must be a string` };
    }
  }

  const options =
    typeof obj.options === "object" && obj.options !== null
      ? (obj.options as Record<string, unknown>)
      : {};

  return {
    ok: true,
    value: {
      id: obj.id,
      model: obj.model,
      messages: obj.messages as Array<{ role: string; content: string }>,
      stream: obj.stream !== false,
      system: typeof obj.system === "string" ? obj.system : undefined,
      options: {
        temperature: typeof options.temperature === "number" ? options.temperature : undefined,
        maxTokens: typeof options.maxTokens === "number" ? options.maxTokens : undefined,
        numCtx: typeof options.numCtx === "number" ? options.numCtx : undefined,
      },
    },
  };
}

// ── Ollama streaming inference ────────────────────────────────────────────────

function resolveOllamaChatUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  const normalized = trimmed.replace(/\/v1$/i, "");
  return `${normalized || OLLAMA_NATIVE_BASE_URL}/api/chat`;
}

async function* streamOllamaInference(
  chatUrl: string,
  req: InferRequest,
  signal: AbortSignal,
): AsyncGenerator<{ text?: string; reasoning?: string; done: boolean; inputTokens: number; outputTokens: number }> {
  // Build message list, prepending system prompt if provided
  const messages: Array<{ role: string; content: string }> = [];
  if (req.system) {
    messages.push({ role: "system", content: req.system });
  }
  messages.push(...req.messages);

  const ollamaOptions: Record<string, unknown> = {
    num_ctx: req.options?.numCtx ?? 65536,
  };
  if (req.options?.temperature !== undefined) {
    ollamaOptions.temperature = req.options.temperature;
  }
  if (req.options?.maxTokens !== undefined) {
    ollamaOptions.num_predict = req.options.maxTokens;
  }

  const body = {
    model: req.model,
    messages,
    stream: true,
    options: ollamaOptions,
  };

  const response = await fetch(chatUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "unknown error");
    throw new Error(`Ollama API error ${response.status}: ${errorText}`);
  }

  if (!response.body) {
    throw new Error("Ollama API returned empty response body");
  }

  const reader = response.body.getReader();
  for await (const chunk of parseNdjsonStream(reader)) {
    yield {
      text: chunk.message?.content || undefined,
      reasoning: chunk.message?.reasoning || undefined,
      done: chunk.done,
      inputTokens: chunk.prompt_eval_count ?? 0,
      outputTokens: chunk.eval_count ?? 0,
    };
    if (chunk.done) {
      break;
    }
  }
}

// ── Per-connection inference handler ─────────────────────────────────────────

function send(ws: WebSocket, msg: ServerMessage): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

async function handleInferRequest(
  ws: WebSocket,
  req: InferRequest,
  chatUrl: string,
  abortControllers: Map<string, AbortController>,
): Promise<void> {
  // Cancel any prior in-flight request with the same ID
  abortControllers.get(req.id)?.abort();
  const ac = new AbortController();
  abortControllers.set(req.id, ac);

  try {
    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of streamOllamaInference(chatUrl, req, ac.signal)) {
      if (ac.signal.aborted) {
        break;
      }
      const delta = chunk.text ?? chunk.reasoning ?? "";
      if (delta) {
        send(ws, { id: req.id, type: "delta", text: delta });
      }
      if (chunk.done) {
        inputTokens = chunk.inputTokens;
        outputTokens = chunk.outputTokens;
      }
    }

    if (!ac.signal.aborted) {
      send(ws, { id: req.id, type: "done", usage: { input: inputTokens, output: outputTokens } });
    }
  } catch (err) {
    if (ac.signal.aborted) {
      return;
    }
    const msg = err instanceof Error ? err.message : String(err);
    send(ws, { id: req.id, type: "error", error: msg });
  } finally {
    abortControllers.delete(req.id);
  }
}

// ── WebSocket connection handler ──────────────────────────────────────────────

function handleConnection(ws: WebSocket, chatUrl: string, log: (msg: string) => void): void {
  const abortControllers = new Map<string, AbortController>();

  // Announce readiness on connect
  send(ws, { type: "ready", version: "1" });

  ws.on("message", (data) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(String(data));
    } catch {
      // Ignore non-JSON frames
      return;
    }

    const result = parseInferRequest(parsed);
    if (!result.ok) {
      // Best-effort: if we have an id send a typed error, else drop
      const id = typeof (parsed as Record<string, unknown>).id === "string"
        ? ((parsed as Record<string, unknown>).id as string)
        : undefined;
      if (id) {
        send(ws, { id, type: "error", error: result.error });
      }
      return;
    }

    log(`infer request id=${result.value.id} model=${result.value.model}`);
    void handleInferRequest(ws, result.value, chatUrl, abortControllers);
  });

  ws.on("close", () => {
    // Abort any in-flight requests when client disconnects
    for (const ac of abortControllers.values()) {
      ac.abort();
    }
    abortControllers.clear();
  });
}

// ── Public server API ─────────────────────────────────────────────────────────

export interface WsInferServerOptions {
  /** TCP port to bind (default: 18790) */
  port?: number;
  /** Bind host (default: "127.0.0.1") */
  host?: string;
  /** Ollama base URL (default: http://127.0.0.1:11434) */
  ollamaUrl?: string;
  /** Logger function for info messages */
  log?: (msg: string) => void;
}

export interface WsInferServer {
  /** Effective port the server is listening on */
  port: number;
  /** Effective host */
  host: string;
  /** WebSocket URL clients should connect to */
  url: string;
  /** Gracefully close the server */
  close: () => Promise<void>;
}

export function startWsInferServer(opts: WsInferServerOptions = {}): Promise<WsInferServer> {
  const port = opts.port ?? 18790;
  const host = opts.host ?? "127.0.0.1";
  const ollamaBaseUrl = opts.ollamaUrl ?? OLLAMA_NATIVE_BASE_URL;
  const log = opts.log ?? ((msg: string) => process.stderr.write(`[ws-infer] ${msg}\n`));

  const chatUrl = resolveOllamaChatUrl(ollamaBaseUrl);

  return new Promise((resolve, reject) => {
    const httpServer = createHttpServer((req, res) => {
      // Health probe for non-WS clients
      if (req.url === "/health" || req.url === "/healthz") {
        res.statusCode = 200;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      res.statusCode = 404;
      res.end("Not Found");
    });

    const wss = new WebSocketServer({ noServer: true });

    httpServer.on("upgrade", (req: IncomingMessage, socket, head) => {
      const url = new URL(req.url ?? "/", "http://localhost");
      if (url.pathname !== "/infer") {
        socket.write("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n");
        socket.destroy();
        return;
      }
      wss.handleUpgrade(req, socket, head, (ws) => {
        wss.emit("connection", ws, req);
      });
    });

    wss.on("connection", (ws: WebSocket) => {
      handleConnection(ws, chatUrl, log);
    });

    httpServer.on("error", (err) => {
      reject(err);
    });

    httpServer.listen(port, host, () => {
      const addr = httpServer.address();
      const effectivePort = typeof addr === "object" && addr !== null ? addr.port : port;
      const wsUrl = `ws://${host}:${effectivePort}/infer`;
      log(`listening on ${wsUrl}`);
      log(`proxying inference to ${chatUrl}`);

      resolve({
        port: effectivePort,
        host,
        url: wsUrl,
        close: () =>
          new Promise<void>((res, rej) => {
            wss.close(() => {
              httpServer.close((err) => {
                if (err) {
                  rej(err);
                } else {
                  res();
                }
              });
            });
          }),
      });
    });
  });
}
