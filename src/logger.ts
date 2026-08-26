import { createConsola } from "consola";

export const logger = createConsola({
    level: process.env.DEBUG ? 4 : 3,
});