export interface CullingReviewSession {
  isOpen: boolean;
  suggestions: unknown;
  progress: unknown;
  error: string | null;
  pathsToCull: string[];
  invocationId?: string | null;
  hiddenForEditor?: boolean;
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
    progress: null,
    error: null,
    hiddenForEditor: false,
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

export function hideCullingReviewForEditor<T extends CullingReviewSession>(session: T): T {
  return { ...session, isOpen: false, hiddenForEditor: true };
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
