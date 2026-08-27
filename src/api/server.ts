import express, { type Express } from "express";
import type { Server } from "node:http";
import type { ManagerConfig } from "../config/types.js";
import type { Planner } from "../planner/planner.js";
import { registerRoutes } from "./routes.js";
import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";

const log: ConsolaInstance = logger.withTag('api-server');

export class ApiServer {

    planner: Planner;
    config: ManagerConfig;

    app: Express;
    server?: Server;

    constructor(planner: Planner, config: ManagerConfig) {
        this.planner = planner;
        this.config = config;
        this.app = express();
        this.app.use(express.json());
        registerRoutes(this.app, this);
    }

    async start(): Promise<void> {
        const [host, port] = this.config.listen.replace(/^https?:\/\//, '').split(':');
        this.server = this.app.listen(Number(port), host);
    }

    async shutdown(): Promise<void> {
        log.debug(`shutdown`);
        return new Promise(resolve => {
            if (!this.server) {
                resolve();
            }

            this.server?.close(() => resolve());
        });
    }
}
