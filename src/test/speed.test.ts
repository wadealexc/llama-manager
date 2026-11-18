import assert from 'node:assert/strict';
import { describe, test } from 'node:test';

import { addTimings, fmtSpeed } from '../api/routes.js';

describe('speed log', () => {
    test('omits draft acceptance when speculative decoding did not run', () => {
        assert.equal(
            fmtSpeed({ prompt_n: 1820, prompt_ms: 1780, predicted_n: 88, predicted_ms: 2300 }),
            'speed | prompt 1820 toks @ 1022.5 t/s | gen 88 toks @ 38.3 t/s',
        );
    });

    test('reports n/a for zero elapsed time', () => {
        assert.equal(
            fmtSpeed({ prompt_n: 0, prompt_ms: 0, predicted_n: 0, predicted_ms: 0 }),
            'speed | prompt 0 toks @ n/a | gen 0 toks @ n/a',
        );
    });

    test('sums attempts and keeps draft counts when only one attempt drafted', () => {
        const a = { prompt_n: 100, prompt_ms: 100, predicted_n: 10, predicted_ms: 500 };
        const b = { prompt_n: 20, prompt_ms: 20, predicted_n: 30, predicted_ms: 500, draft_n: 40, draft_n_accepted: 30 };

        assert.deepEqual(addTimings(a, b), { prompt_n: 120, prompt_ms: 120, predicted_n: 40, predicted_ms: 1000, draft_n: 40, draft_n_accepted: 30 });
        assert.deepEqual(addTimings(undefined, b), b);
        assert.equal(addTimings(a, undefined), a);
        assert.equal(addTimings(undefined, undefined), undefined);
        assert.equal(fmtSpeed(addTimings(a, b)!), 'speed | prompt 120 toks @ 1000.0 t/s | gen 40 toks @ 40.0 t/s | draft 75%');
    });
});
