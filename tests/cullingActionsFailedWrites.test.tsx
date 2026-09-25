import assert from 'node:assert/strict';
import test from 'node:test';
import type { CullingSuggestions } from '../src/components/ui/AppProperties';
import { useLibraryStore } from '../src/store/useLibraryStore';
import { applyCullingSuggestions, undoCullingApplication } from '../src/utils/cullingActions';

// Sidecar ratings as the native side would store them; Tauri IPC is simulated.
const disk = new Map<string, number>();
let failNextWrite = false;
(globalThis as unknown as { window: unknown }).window = {
  __TAURI_INTERNALS__: {
    invoke: async (cmd: string, args: { paths: string[]; rating: number }) => {
      assert.equal(cmd, 'set_rating_for_paths');
      if (failNextWrite) {
        failNextWrite = false;
        throw new Error('simulated disk error');
      }
      args.paths.forEach((path) => disk.set(path, args.rating));
    },
  },
};

const proposal = (path: string, rating: number) =>
  ({ results: [{ path }], starAssignments: { [path]: rating }, colorAssignments: {} }) as unknown as CullingSuggestions;

function loadPhoto(path: string, rating: number) {
  disk.set(path, rating);
  useLibraryStore.getState().setLibrary({
    imageList: [
      { path, rating, rating_is_manual: false, color_label_is_manual: false, tags: null },
    ] as unknown as ReturnType<typeof useLibraryStore.getState>['imageList'],
  });
}

test('Undo A still restores the original rating after application B failed to write', async () => {
  const path = '/shoot/failed-b.nef';
  loadPhoto(path, 2);

  const a = await applyCullingSuggestions(proposal(path, 4));
  assert.equal(disk.get(path), 4);

  failNextWrite = true;
  const b = await applyCullingSuggestions(proposal(path, 3));
  assert.deepEqual(b.succeededRatings, {});
  assert.equal(b.failedRatings.length, 1);
  assert.equal(disk.get(path), 4);

  assert.deepEqual(await undoCullingApplication(a), { restored: 1, skipped: 0 });
  assert.equal(disk.get(path), 2);
  assert.equal(useLibraryStore.getState().imageList[0].rating, 2);
});

test('A failed Undo can be retried and the retry restores the original rating', async () => {
  const path = '/shoot/failed-undo.nef';
  loadPhoto(path, 2);

  const a = await applyCullingSuggestions(proposal(path, 4));
  assert.equal(disk.get(path), 4);

  failNextWrite = true;
  await assert.rejects(undoCullingApplication(a), /simulated disk error/);
  assert.equal(disk.get(path), 4);

  assert.deepEqual(await undoCullingApplication(a), { restored: 1, skipped: 0 });
  assert.equal(disk.get(path), 2);
  assert.equal(useLibraryStore.getState().imageList[0].rating, 2);
});
