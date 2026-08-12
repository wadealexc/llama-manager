import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import * as fs from 'fs';
import chalk from 'chalk';
import type { LlamaConfig } from "../config/types.js";
import { LlamaAPI } from "./llama-api.js";
import { logger } from '../logger.js';
import type { ConsolaInstance } from 'consola';

type Instance = {
    url: string;
    proc: ChildProcess;
    exited: Promise<void>;
}

const log: ConsolaInstance = logger.withTag('router-process');

/**
 * Manages llama-server's router process. Launches the router at startup and
 * kills it on shutdown.
 */
export class RouterProcess {
    
    config: LlamaConfig;
    instance: Instance | undefined;

    constructor(cfg: LlamaConfig) {
        this.config = cfg;
    }

    // Spawn router and resolve once it's listening
    async start(preset_path: string): Promise<LlamaAPI> {
        log.info(`start`);

        const [log_path, log_stream] = createLogStream(this.config.llama_log_dir);

        const [host, port] = this.config.listen.split(':');

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
        });

        log.info(`piping logs to ${log_path}`);
        proc.stdout?.pipe(log_stream);
        proc.stderr?.pipe(log_stream);

        this.instance = {
            url: this.config.listen,
            proc: proc,
            exited: new Promise<void>((resolve) => {
                proc.once('exit', (code, signal) => {
                    log.info(`llama-server exited code=${code} signal=${signal}`);
                    log_stream.end();
                    resolve();
                });
            }),
        };

        // 'error' is emitted in a lot of different scenarios. we just log here and handle
        // process errors/exits elsewhere.
        proc.once('error', (err) => {
            log.error(`llama-server error: ${err}`);
            log_stream.end();
        });

        const api = new LlamaAPI(this.config.listen);

        log.info(`polling router`);

        const poll_start = performance.now();
        await this.#pollRouter(api);
        const poll_end = performance.now();
        const seconds = (poll_end - poll_start) / 1000;

        log.info(`router is active (pid: ${proc.pid}) [elapsed: ${seconds.toFixed(2)}s]`);
        return api;
    }

    // Gracefully shut down the router (SIGTERM). Use SIGKILL if process does not exit in time.
    async shutdown(): Promise<void> {
        if (!this.instance) return;

        const exited = this.instance.exited;
        const pid = this.instance.proc.pid!;
        const grace_period_ms = this.config.shutdown_grace_period_ms;

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

    async #pollRouter(api: LlamaAPI): Promise<void> {
        const deadline = Date.now() + this.config.poll_timeout_ms;

        while (Date.now() < deadline) {
            if (!this.instance) {
                throw new Error(`router died while polling`);
            }

            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 100);

            try {
                const healthy = await Promise.race([
                    this.instance.exited.then(() => false),
                    api.getHealth(controller.signal),
                ]);

                if (healthy) return;
            } catch {
                // Router not ready yet — continue polling
            } finally {
                clearTimeout(timeout);
            }

            await new Promise((r) => setTimeout(r, this.config.poll_interval_ms));
        }

        throw new Error(`router failed to start within ${this.config.poll_timeout_ms}ms`);
    }
}

function createLogStream(dir: string): [string, fs.WriteStream] {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    const file_name = `${timestamp}_router.log`;
    const log_path = path.join(dir, file_name);

    const stream = fs.createWriteStream(log_path, { flags: 'a' });

    return [log_path, stream];
}