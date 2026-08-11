import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import * as fs from 'fs';
import * as readline from 'readline';
import chalk from 'chalk';
import type { LlamaConfig } from "../config/types.js";
import { LlamaAPI } from "./llama-api.js";

type Instance = {
    url: string;
    proc: ChildProcess;
    exited: Promise<void>;
}

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
    async start(): Promise<LlamaAPI> {
        const [log_path, log_stream] = createLogStream(this.config.llama_log_dir);

        const [host, port] = this.config.listen.split(':');

        let argv = [
            '--models-preset', this.config.preset,
            '--no-models-autoload',
            '--host', host,
            '--port', port,
        ];

        // Spawn llama-server
        const proc = spawn(this.config.bin, argv, {
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: true,
        });

        proc.stdout?.pipe(log_stream);
        proc.stderr?.pipe(log_stream);

        this.instance = {
            url: this.config.listen,
            proc: proc,
            exited: new Promise<void>((resolve) => {
                proc.once('exit', (code, signal) => {
                    console.log(`llama-server exited code=${code} signal=${signal}`);
                    log_stream.write(`exited: code=${code} signal=${signal}`);
                    log_stream.end();
                    resolve();
                });
            }),
        };

        // 'error' is emitted in a lot of different scenarios. we just log here and handle
        // process errors/exits elsewhere.
        proc.once('error', (err) => {
            console.error(`llama-server error: ${err}`);
        });

        return new Promise<LlamaAPI>(async (resolve) => {
            const api = new LlamaAPI(this.config.listen);

            const poll_start = performance.now();
            await this.#pollRouter(api);
            const poll_end = performance.now();
            const seconds = (poll_end - poll_start) / 1000;

            console.log(chalk.dim.green(`router is active (pid: ${proc.pid}) [elapsed: ${seconds.toFixed(2)}s]`));
            resolve(api);
        });
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
        console.log(chalk.dim(`killing router process (pid ${pid}) (${chalk.yellow('SIGTERM')})...`));
        if (await kill('SIGTERM', grace_period_ms)) {
            console.log(chalk.green('done! (graceful shutdown)'));
            return;
        }

        // grace period's up, now it's business (SIGKILL)
        //
        // *teleports behind you* "nothin personnel, kid"
        console.log(`failed to stop router process gracefully, sending ${chalk.red('SIGKILL')}...`);
        if (await kill('SIGKILL', grace_period_ms)) {
            console.log(chalk.yellow(`done! (forced shutdown)`));
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