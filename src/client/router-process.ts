import { spawn, execFile, ChildProcess } from 'child_process';
import path from 'path';
import * as fs from 'fs';
import type { LlamaConfig, ModelLoadConfig } from "../config/types.js";
import { LlamaAPI } from "./llama-api.js";
import { logger } from '../logger.js';
import type { ConsolaInstance } from 'consola';

type Instance = {
    url: string;
    proc: ChildProcess;
    exited: Promise<void>;
    stderr_tail: StderrTail;
}

const STARTUP_STDERR_LINES = 20;

const log: ConsolaInstance = logger.withTag('router-process');

/**
 * Manages llama-server's router process. Launches the router at startup and
 * kills it on shutdown.
 */
export class RouterProcess {
    
    config: LlamaConfig;
    model_load_cfg: ModelLoadConfig;


    instance: Instance | undefined;

    constructor(cfg: LlamaConfig, model_load_cfg: ModelLoadConfig) {
        this.config = cfg;
        this.model_load_cfg = model_load_cfg;
    }

    // Spawn router and resolve once it's listening
    async start(preset_path: string): Promise<LlamaAPI> {
        log.info(`start`);

        const [log_path, log_stream] = createLogStream(this.config.llama_log_dir);
        const stderr_tail = new StderrTail(STARTUP_STDERR_LINES);

        const [host, port] = this.config.listen.replace(/^https?:\/\//, '').split(':');

        let argv = [
            '--models-preset', preset_path,
            '--no-models-autoload',
            '--host', host,
            '--port', port,
        ];

        log.info(`spawning llama-server process`);
        const proc = spawn(this.config.bin, argv, {
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: true,
            windowsHide: true,
        });

        log.info(`piping logs to ${log_path}`);
        proc.stdout?.pipe(log_stream);
        proc.stderr?.pipe(log_stream);
        proc.stderr?.on('data', (chunk: Buffer) => stderr_tail.push(chunk.toString()));

        this.instance = {
            url: this.config.listen,
            proc: proc,
            exited: new Promise<void>((resolve) => {
                proc.once('exit', (code, signal) => {
                    log.info(`llama-server exited pid=${proc.pid} code=${code} signal=${signal}`);
                    log_stream.end();
                    resolve();
                });
            }),
            stderr_tail,
        };

        // 'error' is emitted in a lot of different scenarios. we just log here and handle
        // process errors/exits elsewhere.
        proc.once('error', (err) => {
            log.error(`llama-server error: ${err}`);
            log_stream.end();
        });

        const base_url = `http://${host}:${port}`;
        const client = new LlamaAPI(base_url, this.model_load_cfg);

        log.info(`polling router`);

        const poll_start = performance.now();
        await this.#pollRouter(client);
        const poll_end = performance.now();
        const seconds = (poll_end - poll_start) / 1000;

        log.info(`router is active (pid: ${proc.pid}) [elapsed: ${seconds.toFixed(2)}s]`);
        return client;
    }

    // Gracefully shut down the router (SIGTERM). Use SIGKILL if process does not exit in time.
    async shutdown(): Promise<void> {
        if (!this.instance) return;

        log.debug(`shutdown`);
        const exited = this.instance.exited;
        const pid = this.instance.proc.pid!;
        const grace_period_ms = this.config.shutdown_grace_period_ms;

        if (process.platform === 'win32') {
            if (this.instance.proc.exitCode !== null || this.instance.proc.signalCode !== null) return;
            // Windows does not support negative-PID POSIX process-group signals.
            // Include model workers in the forced termination of the router tree.
            log.info(`terminating Windows router process tree (pid ${pid})...`);
            await new Promise<void>((resolve, reject) => {
                execFile('taskkill', ['/PID', String(pid), '/T', '/F'],
                    { windowsHide: true, timeout: grace_period_ms }, (err) => {
                        if (err && this.instance?.proc.exitCode === null && this.instance?.proc.signalCode === null) {
                            reject(new Error(`failed to terminate router process tree (pid ${pid}): ${err.message}`));
                        } else {
                            resolve();
                        }
                    });
            });
            await new Promise<void>((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error(`router process ${pid} did not exit`)), grace_period_ms);
                exited.then(() => { clearTimeout(timeout); resolve(); });
            });
            log.info('done! (Windows process tree terminated)');
            return;
        }

        // Send a kill signal to the process group, then wait for a grace period.
        const kill = async (
            signal: string,
            grace_period_ms: number,
        ): Promise<boolean> => {
            try { process.kill(-pid, signal) } catch { }

            return Promise.race([
                exited.then(() => true),
                new Promise<boolean>((resolve) => {
                    setTimeout(() => resolve(false), grace_period_ms);
                }),
            ]);
        };

        // try a graceful shutdown first (SIGTERM)
        log.info(`killing router process (pid ${pid}) (SIGTERM)...`);
        if (await kill('SIGTERM', grace_period_ms)) {
            log.info('done! (graceful shutdown)');
            return;
        }

        // grace period's up, now it's business (SIGKILL)
        //
        // *teleports behind you* "nothin personnel, kid"
        log.warn(`killing router process (pid ${pid}) (SIGKILL)...`);
        if (await kill('SIGKILL', grace_period_ms)) {
            log.warn(`done! (forced shutdown)`);
            return;
        }

        // if we don't get a shutdown, burn it all to the ground
        throw new Error(`failed to kill router process (pid ${pid})`);
    }

    async #pollRouter(client: LlamaAPI): Promise<void> {
        const deadline = Date.now() + this.config.poll_timeout_ms;

        while (Date.now() < deadline) {
            if (!this.instance) {
                throw new Error(`router died while polling`);
            }

            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 100);

            let exited = false;
            try {
                const healthy = await Promise.race([
                    this.instance.exited.then(() => { exited = true; return false; }),
                    client.getHealth(controller.signal),
                ]);

                if (healthy) return;
            } catch (err) {
                // Router not ready yet - continue polling while the process is alive
                log.debug(`health poll failed: ${err}`);
            } finally {
                clearTimeout(timeout);
            }

            if (exited || !this.isAlive()) {
                throw new Error(`router exited while polling${this.#stderrSuffix()}`);
            }

            await new Promise((r) => setTimeout(r, this.config.poll_interval_ms));
        }

        throw new Error(`router failed to start within ${this.config.poll_timeout_ms}ms${this.#stderrSuffix()}`);
    }

    #stderrSuffix(): string {
        const tail = this.instance?.stderr_tail.tail() ?? '';
        return tail ? `\n${tail}` : '';
    }

    isAlive(): boolean {
        const proc = this.instance?.proc;
        return proc !== undefined && proc.exitCode === null && proc.signalCode === null;
    }
}

function createLogStream(dir: string): [string, fs.WriteStream] {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    const file_name = `${timestamp}_router.log`;
    const log_path = path.join(dir, file_name);

    const stream = fs.createWriteStream(log_path, { flags: 'a' });

    return [log_path, stream];
}

// keep the last `max_lines` of the router's stderr to surface on startup fail
class StderrTail {

    lines: string[] = [];
    pending: string = '';
    max_lines: number;

    constructor(max_lines: number) {
        this.max_lines = max_lines;
    }

    push(text: string): void {
        this.pending += text;

        const parts = this.pending.split('\n');
        this.pending = parts.pop() ?? '';

        for (const line of parts) {
            this.lines.push(line);
            if (this.lines.length > this.max_lines) {
                this.lines.shift();
            }
        }
    }

    tail(): string {
        return [...this.lines, this.pending]
            .filter(l => l.trim() !== '')
            .slice(-this.max_lines)
            .join('\n');
    }
}
