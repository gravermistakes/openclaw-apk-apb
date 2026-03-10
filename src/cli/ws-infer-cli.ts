/**
 * CLI for the WebSocket inference server.
 *
 * Usage:
 *   openclaw ws-infer serve [--port 18790] [--host 127.0.0.1] [--ollama-url http://127.0.0.1:11434]
 *   openclaw ws-infer connect [--url ws://127.0.0.1:18790/infer] --model <model> [--message <text>]
 */

import type { Command } from "commander";
import { defaultRuntime } from "../runtime.js";

export function registerWsInferCli(program: Command) {
  const cmd = program
    .command("ws-infer")
    .description("WebSocket inference server and client for local models");

  // ── serve sub-command ────────────────────────────────────────────────────

  cmd
    .command("serve")
    .description("Start a foreground WebSocket inference server (proxies to Ollama)")
    .option("--port <port>", "TCP port to listen on", "18790")
    .option("--host <host>", "Bind host", "127.0.0.1")
    .option("--ollama-url <url>", "Ollama base URL", "http://127.0.0.1:11434")
    .action(async (opts) => {
      const port = Number.parseInt(String(opts.port ?? "18790"), 10);
      if (Number.isNaN(port) || port < 1 || port > 65535) {
        defaultRuntime.error(`Invalid --port value: ${String(opts.port)}`);
        defaultRuntime.exit(1);
        return;
      }

      const { startWsInferServer } = await import("../gateway/ws-infer-server.js");
      let server: Awaited<ReturnType<typeof startWsInferServer>>;
      try {
        server = await startWsInferServer({
          port,
          host: opts.host as string | undefined,
          ollamaUrl: opts.ollamaUrl as string | undefined,
          log: (msg) => defaultRuntime.log(msg),
        });
      } catch (err) {
        defaultRuntime.error(`Failed to start server: ${String(err)}`);
        defaultRuntime.exit(1);
        return;
      }

      defaultRuntime.log(`ws-infer server ready`);
      defaultRuntime.log(`  WebSocket : ${server.url}`);
      defaultRuntime.log(`  Health    : http://${server.host}:${server.port}/health`);
      defaultRuntime.log(`Press Ctrl+C to stop.`);

      // Run in the foreground until signal
      const shutdown = async () => {
        defaultRuntime.log("Shutting down ws-infer server…");
        await server.close().catch(() => {});
        defaultRuntime.exit(0);
      };

      process.once("SIGINT", () => void shutdown());
      process.once("SIGTERM", () => void shutdown());

      // Keep the process alive indefinitely (server runs in foreground)
      await new Promise<void>(() => {});
    });

  // ── connect sub-command ──────────────────────────────────────────────────

  cmd
    .command("connect")
    .description("Connect to a ws-infer server and run a single inference request")
    .option("--url <url>", "WebSocket server URL", "ws://127.0.0.1:18790/infer")
    .requiredOption("--model <model>", "Model name to use for inference")
    .option("--message <text>", "User message text (reads from stdin when omitted)")
    .option("--system <text>", "System prompt")
    .option("--temperature <n>", "Temperature (0.0–2.0)")
    .option("--max-tokens <n>", "Maximum output tokens")
    .action(async (opts) => {
      // Resolve user message: flag or stdin
      let userMessage: string;
      if (opts.message) {
        userMessage = opts.message as string;
      } else {
        // Read from stdin
        const chunks: Buffer[] = [];
        for await (const chunk of process.stdin) {
          chunks.push(chunk as Buffer);
        }
        userMessage = Buffer.concat(chunks).toString("utf8").trim();
        if (!userMessage) {
          defaultRuntime.error("Provide --message <text> or pipe text via stdin.");
          defaultRuntime.exit(1);
          return;
        }
      }

      const temperature =
        opts.temperature !== undefined ? Number.parseFloat(String(opts.temperature)) : undefined;
      const maxTokens =
        opts.maxTokens !== undefined ? Number.parseInt(String(opts.maxTokens), 10) : undefined;

      const { WebSocket } = await import("ws");
      const wsUrl = opts.url as string;
      const model = opts.model as string;

      const ws = new WebSocket(wsUrl);

      await new Promise<void>((resolve, reject) => {
        const requestId = `cli-${Date.now()}`;
        let ready = false;

        ws.on("error", (err) => reject(err));

        ws.on("open", () => {
          // Server will send { type: "ready" } first; we send the request then
        });

        ws.on("message", (data) => {
          let msg: Record<string, unknown>;
          try {
            msg = JSON.parse(String(data)) as Record<string, unknown>;
          } catch {
            return;
          }

          if (msg.type === "ready" && !ready) {
            ready = true;
            const request = {
              id: requestId,
              model,
              messages: [{ role: "user", content: userMessage }],
              stream: true,
              ...(opts.system ? { system: opts.system as string } : {}),
              options: {
                ...(temperature !== undefined ? { temperature } : {}),
                ...(maxTokens !== undefined && !Number.isNaN(maxTokens) ? { maxTokens } : {}),
              },
            };
            ws.send(JSON.stringify(request));
            return;
          }

          if (msg.id !== requestId) {
            return;
          }

          if (msg.type === "delta") {
            process.stdout.write(String(msg.text ?? ""));
          } else if (msg.type === "done") {
            process.stdout.write("\n");
            const usage = msg.usage as { input: number; output: number } | undefined;
            if (usage) {
              defaultRuntime.log(
                `[done] input_tokens=${usage.input} output_tokens=${usage.output}`,
              );
            }
            ws.close();
            resolve();
          } else if (msg.type === "error") {
            ws.close();
            reject(new Error(String(msg.error ?? "unknown inference error")));
          }
        });

        ws.on("close", () => resolve());
      }).catch((err) => {
        defaultRuntime.error(`ws-infer connect error: ${String(err)}`);
        defaultRuntime.exit(1);
      });
    });
}
