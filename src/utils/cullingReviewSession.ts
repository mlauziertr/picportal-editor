export interface CullingReviewSession {
  isOpen: boolean;
  suggestions: unknown;
  progress: unknown;
  error: string | null;
  pathsToCull: string[];
  invocationId?: string | null;
  hiddenForEditor?: boolean;
  isCancelling?: boolean;
  cancelled?: boolean;
}

export function createCullingInvocationId(): string {
  return globalThis.crypto.randomUUID();
}

export function beginCullingInvocation<T extends CullingReviewSession>(session: T, invocationId: string): T {
  return {
    ...session,
    invocationId,
    isOpen: true,
    suggestions: null,
    progress: { current: 0, total: 0, stage: 'Starting...' },
    error: null,
    hiddenForEditor: false,
    isCancelling: false,
    cancelled: false,
  };
}

export function cullingEventMatches(
  session: { invocationId?: string | null },
  eventInvocationId: unknown,
): boolean {
  return (
    typeof session.invocationId === 'string' &&
    session.invocationId.length > 0 &&
    eventInvocationId === session.invocationId
  );
}

export function requestCullingCancellation<T extends CullingReviewSession>(session: T): T {
  if (!session.invocationId || session.progress == null || session.isCancelling) {
    return session;
  }
  return { ...session, isCancelling: true };
}

export function completeCullingCancellation<T extends CullingReviewSession>(session: T): T {
  return {
    ...session,
    invocationId: null,
    progress: null,
    suggestions: null,
    error: null,
    isCancelling: false,
    cancelled: true,
  };
}

export function hideCullingReviewForEditor<T extends CullingReviewSession>(session: T): T {
  return { ...session, isOpen: false, hiddenForEditor: true };
}

export type CullingReviewStage = 'settings' | 'progress' | 'results' | 'cancelled';

export function cullingReviewStage(session: {
  progress: unknown;
  suggestions: unknown;
  error: string | null;
  cancelled?: boolean;
}): CullingReviewStage {
  if (session.suggestions != null || session.error != null) return 'results';
  if (session.cancelled) return 'cancelled';
  if (session.progress != null) return 'progress';
  return 'settings';
}

export function cullingAnalysisMessageKey(status: string): string | null {
  switch (status) {
    case 'ready':
      return null;
    case 'disabled':
      return 'subjectAnalysisDisabled';
    case 'subject-ready-focus-unavailable':
      return 'subjectAnalysisSubjectReadyFocusUnavailable';
    case 'subject-ready-pose-unavailable':
      return 'subjectAnalysisPoseUnavailable';
    case 'subject-ready-focus-calibration-unavailable':
      return 'subjectAnalysisFocusCalibrationUnavailable';
    case 'focus-calibration-unavailable':
      return 'focusCalibrationUnavailable';
    case 'focus-unavailable':
      return 'subjectAnalysisFocusUnavailable';
    case 'face-unavailable':
      return 'subjectAnalysisFaceUnavailable';
    case 'error':
      return 'subjectAnalysisError';
    default:
      return 'subjectAnalysisUnavailable';
  }
}

export type CullingResultsTab = 'similar' | 'blurry' | 'alerts' | 'unknown' | 'failed';

type CullingResultLists = {
  similarGroups: readonly unknown[];
  blurryImages: readonly unknown[];
  reviewAlerts: readonly unknown[];
  unknownImages: readonly unknown[];
  failedPaths: readonly string[];
};

export function hasCullingResultItems(suggestions: CullingResultLists): boolean {
  return (
    suggestions.similarGroups.length > 0 ||
    suggestions.blurryImages.length > 0 ||
    suggestions.reviewAlerts.length > 0 ||
    suggestions.unknownImages.length > 0 ||
    suggestions.failedPaths.length > 0
  );
}

export function populatedCullingResultsTab(suggestions: CullingResultLists): CullingResultsTab {
  if (suggestions.similarGroups.length > 0) return 'similar';
  if (suggestions.blurryImages.length > 0) return 'blurry';
  if (suggestions.reviewAlerts.length > 0) return 'alerts';
  if (suggestions.unknownImages.length > 0) return 'unknown';
  if (suggestions.failedPaths.length > 0) return 'failed';
  return 'similar';
}

export function initialCullingRejectPaths(
  suggestions: {
    similarGroups: readonly { duplicates: readonly { path: string }[] }[];
    blurryImages: readonly { path: string }[];
    unknownImages: readonly { path: string; eyeState?: string }[];
    failedPaths: readonly string[];
  },
  selectionAmount: 'extreme' | 'few' | 'standard' | 'more',
): Set<string> {
  const rejects = new Set<string>();
  const protectedPaths = new Set(
    suggestions.unknownImages.filter((image) => image.eyeState === 'not-evaluated').map((image) => image.path),
  );
  suggestions.similarGroups.forEach((group) => {
    const keepCount =
      selectionAmount === 'extreme'
        ? 1
        : selectionAmount === 'few'
          ? Math.max(1, Math.ceil((group.duplicates.length + 1) * 0.25))
          : selectionAmount === 'more'
            ? Math.max(1, Math.ceil((group.duplicates.length + 1) * 0.75))
            : Math.max(1, Math.ceil((group.duplicates.length + 1) * 0.5));
    group.duplicates
      .slice(Math.max(0, keepCount - 1))
      .filter((duplicate) => !protectedPaths.has(duplicate.path))
      .forEach((duplicate) => rejects.add(duplicate.path));
  });
  suggestions.blurryImages
    .filter((image) => !protectedPaths.has(image.path))
    .forEach((image) => rejects.add(image.path));
  return rejects;
}

export function emptyCullingResultsHeadline(status: string): string {
  if (status === 'ready' || status === 'disabled') {
    return 'noIssuesFound';
  }
  return cullingAnalysisMessageKey(status) ?? 'subjectAnalysisUnavailable';
}

export function restoreCullingReviewOnLibraryReturn<T extends CullingReviewSession>(session: T): T {
  if (!session.hiddenForEditor) {
    return session;
  }
  if (session.suggestions == null && session.error == null) {
    return { ...session, hiddenForEditor: false };
  }
  return { ...session, isOpen: true, hiddenForEditor: false };
}
