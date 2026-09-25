import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { useUIStore } from '../src/store/useUIStore.ts';

const openResults = {
  isOpen: true,
  folderPath: '/synthetic-fixtures',
  suggestions: null,
  persistence: null,
  selectedPath: null,
};

test('Escape / back request a guarded close instead of closing the culling results', () => {
  useUIStore.getState().setUI({ cullingResultsState: { ...openResults } });
  useUIStore.getState().requestCullingResultsClose();
  const state = useUIStore.getState().cullingResultsState;
  // The panel decides: unapplied proposals ask for confirmation before disappearing.
  assert.equal(state.isOpen, true);
  assert.equal(state.closeRequest, 1);
  useUIStore.getState().requestCullingResultsClose();
  assert.equal(useUIStore.getState().cullingResultsState.closeRequest, 2);
});

test('a close request is ignored while the culling results are closed', () => {
  useUIStore.getState().setUI({ cullingResultsState: { ...openResults, isOpen: false } });
  useUIStore.getState().requestCullingResultsClose();
  assert.equal(useUIStore.getState().cullingResultsState.closeRequest, undefined);
});

test('no keyboard or back handler closes the culling results without the guard', () => {
  for (const file of ['src/hooks/useKeyboardShortcuts.ts', 'src/hooks/useAndroidBackHandler.ts']) {
    const source = readFileSync(join(process.cwd(), file), 'utf8');
    assert.match(source, /requestCullingResultsClose\(\)/, file);
    assert.doesNotMatch(source, /cullingResultsState: \{[^}]*isOpen: false/, file);
  }
});
