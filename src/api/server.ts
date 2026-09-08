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
        this.app.use(express.json({ limit: '50mb' }));
        registerRoutes(this.app, this);
    }

    async start(): Promise<void> {
        this.server = this.app.listen(this.config.port, this.config.host);
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
