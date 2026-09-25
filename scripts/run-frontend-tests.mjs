// Runs every frontend test under tests/ with the Node test runner.
// .test.ts files run directly (Node type stripping plus the extension resolver);
// .test.tsx files are bundled with esbuild first because Node cannot strip JSX.
import { spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const testsDir = join(root, 'tests');
// Bundles stay inside the project so external packages resolve from node_modules.
const bundleDir = join(root, 'node_modules', '.cache', 'picportal-tests');

const testFiles = readdirSync(testsDir).sort();
const tsTests = testFiles.filter((file) => file.endsWith('.test.ts')).map((file) => join('tests', file));
const tsxTests = testFiles.filter((file) => file.endsWith('.test.tsx'));

rmSync(bundleDir, { recursive: true, force: true });
mkdirSync(bundleDir, { recursive: true });

const bundledTests = [];
for (const file of tsxTests) {
  const outfile = join(bundleDir, file.replace(/\.tsx$/, '.mjs'));
  await build({
    entryPoints: [join(testsDir, file)],
    bundle: true,
    platform: 'node',
    format: 'esm',
    packages: 'external',
    outfile,
    logLevel: 'warning',
  });
  bundledTests.push(outfile);
}

const result = spawnSync(
  process.execPath,
  ['--import', './tests/resolveTypeScriptExtensions.mjs', '--test', ...tsTests, ...bundledTests],
  { cwd: root, stdio: 'inherit' },
);

process.exit(result.status ?? 1);
