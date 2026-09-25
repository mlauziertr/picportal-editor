import type { CullingSession, CullingSuggestions } from '../components/ui/AppProperties';
import type { CullingResultsState, UIState } from '../store/useUIStore';

// The analysis runs in the backend and outlives a webview reload; the UI store
// does not. These helpers rebuild the UI from what the backend reports.

/** Results view for a finished analysis; the folder comes from the backend, not the reset UI store. */
export function cullingResultsFor(
  suggestions: CullingSuggestions,
  fallbackFolderPath: string | null,
): CullingResultsState {
  return {
    isOpen: true,
    folderPath: suggestions.folderPath ?? fallbackFolderPath,
    suggestions,
    persistence: null,
    selectedPath: suggestions.results[0]?.path || null,
  };
}

/**
 * UI state to restore after a reload: the unreviewed proposals of a finished
 * analysis, or the progress view of a running one. A committed result wins over
 * `running`. Returns null when there is nothing to restore or when the UI
 * already shows the session.
 */
export function restoreCullingSession(
  session: CullingSession,
  ui: Pick<UIState, 'cullingModalState' | 'cullingResultsState'>,
): Partial<Pick<UIState, 'cullingModalState' | 'cullingResultsState'>> | null {
  if (session.result) {
    if (ui.cullingResultsState.isOpen) return null;
    return {
      cullingModalState: { ...ui.cullingModalState, isOpen: false, progress: null, suggestions: null, error: null },
      cullingResultsState: cullingResultsFor(session.result, session.folderPath),
    };
  }
  if (session.running) {
    if (ui.cullingModalState.progress || ui.cullingResultsState.isOpen) return null;
    return {
      cullingModalState: {
        ...ui.cullingModalState,
        isOpen: true,
        folderPath: session.folderPath,
        progress: session.progress || { current: 0, total: 0, stage: '', stageCode: 'preparing' },
        suggestions: null,
        error: null,
      },
    };
  }
  return null;
}

// Culling events are newer than any snapshot requested before them.
let cullingEventCount = 0;

/** Called by every culling event listener. */
export function noteCullingEvent() {
  cullingEventCount += 1;
}

/**
 * Reads the backend session, or null when a culling event arrived while the
 * request was in flight: the snapshot may predate that event and must not undo it.
 */
export async function readCullingSession(fetch: () => Promise<CullingSession>): Promise<CullingSession | null> {
  const eventsBefore = cullingEventCount;
  const session = await fetch();
  return cullingEventCount === eventsBefore ? session : null;
}
