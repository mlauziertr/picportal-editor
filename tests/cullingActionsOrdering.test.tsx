import assert from 'node:assert/strict';
import test from 'node:test';
import type { CullingSuggestions } from '../src/components/ui/AppProperties';
import { useLibraryStore } from '../src/store/useLibraryStore';
import { applyCullingSuggestions, undoCullingApplication } from '../src/utils/cullingActions';

// Sidecar ratings as the native side would store them; Tauri IPC is simulated.
const disk = new Map<string, number>();
let holdNextWrite: Promise<void> | null = null;
(globalThis as unknown as { window: unknown }).window = {
  __TAURI_INTERNALS__: {
    invoke: async (cmd: string, args: { paths: string[]; rating: number }) => {
      assert.equal(cmd, 'set_rating_for_paths');
      const hold = holdNextWrite;
      holdNextWrite = null;
      if (hold) await hold;
      args.paths.forEach((path) => disk.set(path, args.rating));
    },
  },
};

const path = '/shoot/a.nef';
const proposal = (rating: number) =>
  ({ results: [{ path }], starAssignments: { [path]: rating }, colorAssignments: {} }) as unknown as CullingSuggestions;

test('Undo A cannot overwrite an application B started while its write was pending', async () => {
  disk.set(path, 2);
  useLibraryStore.getState().setLibrary({
    imageList: [
      { path, rating: 2, rating_is_manual: false, color_label_is_manual: false, tags: null },
    ] as unknown as ReturnType<typeof useLibraryStore.getState>['imageList'],
  });

  const a = await applyCullingSuggestions(proposal(4));
  assert.equal(disk.get(path), 4);

  let releaseUndo = () => {};
  holdNextWrite = new Promise((resolve) => (releaseUndo = resolve));
  const undoA = undoCullingApplication(a);
  const b = applyCullingSuggestions(proposal(3));
  await new Promise((resolve) => setTimeout(resolve, 10));
  releaseUndo();

  assert.deepEqual(await undoA, { restored: 1, skipped: 0 });
  const summaryB = await b;
  assert.equal(summaryB.succeededRatings[path], 3);
  // B runs after the undo and is the last write: the newer decision survives.
  assert.equal(disk.get(path), 3);
  assert.equal(useLibraryStore.getState().imageList[0].rating, 3);

  // And A's undo, replayed now, sees B's value and leaves it alone.
  assert.deepEqual(await undoCullingApplication(a), { restored: 0, skipped: 1 });
  assert.equal(disk.get(path), 3);
});
