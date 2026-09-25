import assert from 'node:assert/strict';
import test from 'node:test';
import { valueForPath } from '../src/utils/pathBoundSnapshot.ts';

test('export adjustment snapshots cannot cross selected image paths', () => {
  const adjustmentsForB = { exposure: 0.75 };
  const snapshot = { path: '/synthetic/B.jpg', value: adjustmentsForB };

  assert.equal(valueForPath(snapshot, '/synthetic/A.jpg'), null);
  assert.equal(valueForPath(snapshot, '/synthetic/B.jpg'), adjustmentsForB);
  assert.equal(valueForPath(snapshot, null), null);
});
