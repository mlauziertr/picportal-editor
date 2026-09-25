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
 * UI state to restore after a reload: the progress view of a running analysis,
 * or the unreviewed proposals of a finished one. Returns null when there is
 * nothing to restore or when the UI already shows the session.
 */
export function restoreCullingSession(
  session: CullingSession,
  ui: Pick<UIState, 'cullingModalState' | 'cullingResultsState'>,
): Partial<Pick<UIState, 'cullingModalState' | 'cullingResultsState'>> | null {
  if (session.running) {
    if (ui.cullingModalState.progress) return null;
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
  if (session.result && !ui.cullingResultsState.isOpen) {
    return { cullingResultsState: cullingResultsFor(session.result, session.folderPath) };
  }
  return null;
}
