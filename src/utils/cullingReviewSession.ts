export interface CullingReviewSession {
  isOpen: boolean;
  suggestions: unknown;
  progress: unknown;
  error: string | null;
  pathsToCull: string[];
}

export function hideCullingReviewForEditor<T extends CullingReviewSession>(session: T): T {
  return { ...session, isOpen: false };
}

export function restoreCullingReviewOnLibraryReturn<T extends CullingReviewSession>(session: T): T {
  if (session.isOpen || (session.suggestions == null && session.error == null)) {
    return session;
  }
  return { ...session, isOpen: true };
}
