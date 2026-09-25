import assert from 'node:assert/strict';
import test from 'node:test';
import type { TFunction } from 'i18next';
import {
  classifyStyleError,
  styleBatchMessages,
  styleErrorMessage,
  styleFallbackNotice,
} from '../src/utils/styleModel.ts';

const t = ((key: string, options?: Record<string, unknown>) =>
  options ? `${key}${JSON.stringify(options)}` : key) as unknown as TFunction;

test('style error codes map to UX error kinds', () => {
  assert.equal(classifyStyleError('STYLE_MODEL_UNAVAILABLE'), 'noModel');
  assert.equal(classifyStyleError('STYLE_MODEL_CONTRACT_MISMATCH'), 'modelOutdated');
  assert.equal(classifyStyleError('STYLE_MODEL_ARTIFACT_HASH_MISMATCH'), 'modelDamaged');
  assert.equal(classifyStyleError('STYLE_MODEL_MANIFEST_INVALID: expected value at line 1'), 'modelDamaged');
  assert.equal(classifyStyleError('STYLE_INFERENCE_FAILED: boom'), 'generic');
  assert.equal(classifyStyleError(undefined), 'generic');
});

test('raw backend strings never reach the message', () => {
  const message = styleErrorMessage(t, 'STYLE_INFERENCE_FAILED: /home/someone/secret.jpg');
  assert.equal(message, 'style.errors.generic');
  assert.match(styleErrorMessage(t, 'STYLE_MODEL_UNAVAILABLE'), /style\.trainCliHint/);
});

test('fallback notice only for fallback models', () => {
  const model = { modelId: 'm', kind: 'cluster_mean', fallback: true, trainingSamples: 30, minimumForLearnedModel: 50 };
  assert.equal(styleFallbackNotice(t, model), 'style.fallbackNotice{"count":30,"recommended":50}');
  assert.equal(styleFallbackNotice(t, { ...model, fallback: false }), null);
});

test('batch summary lists skipped and failed photos only when present', () => {
  const model = { modelId: 'm', kind: 'mlp', fallback: false, trainingSamples: 300, minimumForLearnedModel: 50 };
  assert.deepEqual(styleBatchMessages(t, { model, updated: 52, skipped: 0, failed: 0, errors: [] }), [
    'style.batch.done{"count":52}',
  ]);
  assert.equal(styleBatchMessages(t, { model, updated: 52, skipped: 12, failed: 1, errors: [] }).length, 3);
});
