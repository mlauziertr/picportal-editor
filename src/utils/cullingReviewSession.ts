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

export function restoreCullingReviewOnLibraryReturn<T extends CullingReviewSession>(session: T): T {
  if (!session.hiddenForEditor) {
    return session;
  }
  if (session.suggestions == null && session.error == null) {
    return { ...session, hiddenForEditor: false };
  }
  return { ...session, isOpen: true, hiddenForEditor: false };
}
