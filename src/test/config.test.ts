import assert from 'node:assert/strict';
import { after, describe, test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { buildConfig } from '../config/build.js';
import { parseArgs } from '../config/parser.js';
import type { ManagerConfig } from '../config/types.js';
import { DEFAULT_FIT_OVERHEAD_MIB, MIN_ALLOWED_CTX } from '../llama-cpp-constants.js';

const directories: string[] = [];
after(async () => {
    await Promise.all(directories.map(directory => rm(directory, { recursive: true, force: true })));
});

type Built = { config: ManagerConfig; sections: Map<string, Map<string, string>> };

function parseIni(text: string): Map<string, Map<string, string>> {
    const sections = new Map<string, Map<string, string>>();
    let current: Map<string, string> | undefined;
    for (const line of text.split('\n')) {
        const header = line.match(/^\[(.+)\]$/);
        if (header) {
            current = new Map();
            sections.set(header[1], current);
            continue;
        }
        const eq = line.indexOf(' = ');
        if (current && eq > 0) current.set(line.slice(0, eq), line.slice(eq + 3));
    }
    return sections;
}

async function workspace(): Promise<string> {
    const directory = await mkdtemp(join(tmpdir(), 'llama-manager-config-'));
    directories.push(directory);
    return directory;
}

function routerYaml(directory: string, models: string): string {
    return [
        'router:',
        `    llama-log-dir: ${join(directory, 'logs')}`,
        `    slot-save-path: ${join(directory, 'slots')}`,
        '    listen: 127.0.0.1:0',
        'models:',
        models,
    ].join('\n');
}

async function buildFromYaml(models: string): Promise<Built> {
    const directory = await workspace();
    const config_path = join(directory, 'config.yaml');
    const preset_path = join(directory, 'preset.ini');
    await writeFile(config_path, routerYaml(directory, models), 'utf8');

    const config = await buildConfig(parseArgs(['--config', config_path, '--bin', process.execPath]), directory, preset_path, config_path);
    return { config, sections: parseIni(await readFile(preset_path, 'utf8')) };
}

async function buildFromArgs(argv: string[]): Promise<Built> {
    const directory = await workspace();
    const preset_path = join(directory, 'preset.ini');
    const argv_full = [...argv, '--bin', process.execPath, '--slot-save-path', join(directory, 'slots')];

    const config = await buildConfig(parseArgs(argv_full), directory, preset_path, join(directory, 'missing.yaml'));
    return { config, sections: parseIni(await readFile(preset_path, 'utf8')) };
}

describe('min-baseline-ctx', () => {
    test('defaults the preset ctx-size to the minimum allowed ctx', async () => {
        const { sections } = await buildFromYaml([
            '    plain:',
            '        model: /models/plain.gguf',
        ].join('\n'));

        assert.equal(sections.get('plain')!.get('ctx-size'), String(MIN_ALLOWED_CTX));
    });

    test('sets the preset ctx-size and is not passed through as a llama-server arg', async () => {
        const { sections } = await buildFromYaml([
            '    moe:',
            '        model: /models/moe.gguf',
            '        min-baseline-ctx: 100000',
        ].join('\n'));

        const section = sections.get('moe')!;
        assert.equal(section.get('ctx-size'), '100000');
        assert.equal(section.has('min-baseline-ctx'), false);
    });

    test('accepts the snake_case spelling', async () => {
        const { sections } = await buildFromYaml([
            '    moe:',
            '        model: /models/moe.gguf',
            '        min_baseline_ctx: 65536',
        ].join('\n'));

        assert.equal(sections.get('moe')!.get('ctx-size'), '65536');
    });

    test('applies to every model variant', async () => {
        const { sections } = await buildFromYaml([
            '    moe:',
            '        model: /models/moe-q6.gguf',
            '        min-baseline-ctx: 65536',
            '        model-variants:',
            '            q4: /models/moe-q4.gguf',
            '        ladder:',
            '            - swap-model:q4',
        ].join('\n'));

        assert.equal(sections.get('moe')!.get('ctx-size'), '65536');
        const variant = sections.get('moe__q4')!;
        assert.equal(variant.get('model'), '/models/moe-q4.gguf');
        assert.equal(variant.get('ctx-size'), '65536');
        assert.equal(variant.has('min-baseline-ctx'), false);
    });

    test('is accepted as a server-mode CLI flag', async () => {
        const { sections } = await buildFromArgs(['-m', '/models/moe.gguf', '--min-baseline-ctx', '32768']);

        assert.equal(sections.get('moe')!.get('ctx-size'), '32768');
        assert.equal(sections.get('moe')!.has('min-baseline-ctx'), false);
    });
});

describe('fit-target', () => {
    test('defaults to the llama.cpp fit overhead', async () => {
        const { config } = await buildFromYaml([
            '    plain:',
            '        model: /models/plain.gguf',
        ].join('\n'));

        assert.equal(config.models['plain'].fit_target_mib, DEFAULT_FIT_OVERHEAD_MIB);
    });

    test('reads a numeric fit-target and its alias', async () => {
        const { config } = await buildFromYaml([
            '    a:',
            '        model: /models/a.gguf',
            '        fit-target: 512',
            '    b:',
            '        model: /models/b.gguf',
            '        fitt: 768',
        ].join('\n'));

        assert.equal(config.models['a'].fit_target_mib, 512);
        assert.equal(config.models['b'].fit_target_mib, 768);
    });

    test('uses the first device value of a per-device list', async () => {
        const { config } = await buildFromYaml([
            '    a:',
            '        model: /models/a.gguf',
            '        fit-target: "2048,512"',
        ].join('\n'));

        assert.equal(config.models['a'].fit_target_mib, 2048);
    });
});
