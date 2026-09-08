import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import test from 'node:test';
import { RouterProcess } from '../dist/client/router-process.js';

test('Windows shutdown terminates router and worker process tree', {skip:process.platform !== 'win32'}, async () => {
    const workerCode = 'setInterval(() => {}, 1000)';
    const routerCode = `const {spawn}=require('node:child_process'); const worker=spawn(process.execPath, ['-e', ${JSON.stringify(workerCode)}], {windowsHide:true,stdio:'ignore'}); console.log(worker.pid); setInterval(()=>{},1000);`;
    const proc = spawn(process.execPath, ['-e', routerCode], {windowsHide:true, stdio:['ignore','pipe','inherit']});
    const exited = new Promise(resolve => proc.once('exit', resolve));
    const router = new RouterProcess({shutdown_grace_period_ms:10000}, {});
    router.instance = {proc, exited};
    try {
        const [data] = await once(proc.stdout, 'data');
        const workerPid = Number(data.toString().trim());
        assert.ok(workerPid > 0);
        await router.shutdown();
        assert.notEqual(proc.exitCode, null);
        assert.throws(() => process.kill(workerPid, 0), {code:'ESRCH'});
        await router.shutdown(); // Already exited is a no-op.
    } finally {
        if (proc.exitCode === null) await router.shutdown();
    }
});
