export interface PicPortalUiSessionState<TAdmin = unknown> {
  admin: TAdmin | null;
  galleryId: string;
  authenticationRejected: boolean;
}

export type PicPortalSessionErrorOutcome = 'credentials-retained' | 'invalidated' | 'unchanged';

/** Structured error returned by the PicPortal Tauri commands. */
export interface PicPortalCommandError {
  code: string;
  message: string;
  path?: string;
  retryable: boolean;
  status?: number;
}

export function isPicPortalCommandError(error: unknown): error is PicPortalCommandError {
  return (
    typeof error === 'object' &&
    error !== null &&
    typeof (error as { code?: unknown }).code === 'string' &&
    typeof (error as { message?: unknown }).message === 'string'
  );
}

/** Technical detail of a PicPortal error, for status text and "copy details". */
export function picPortalErrorMessage(error: unknown): string {
  return isPicPortalCommandError(error) ? error.message : String(error);
}

export function transitionPicPortalUiSession<TAdmin>(
  state: PicPortalUiSessionState<TAdmin>,
  error: unknown,
): { outcome: PicPortalSessionErrorOutcome; state: PicPortalUiSessionState<TAdmin> } {
  const code = isPicPortalCommandError(error) ? error.code : null;
  const message = picPortalErrorMessage(error);
  if (
    code === 'session_clear_failed' ||
    message.includes(
      'PicPortal session invalidation not completed because secure session storage could not be cleared',
    ) ||
    message.includes('PicPortal logout not completed because secure session storage could not be cleared')
  ) {
    return {
      outcome: 'credentials-retained',
      state: { ...state, authenticationRejected: true },
    };
  }
  if (code === 'session_expired' || message.includes('session expired or was revoked')) {
    return {
      outcome: 'invalidated',
      state: { admin: null, galleryId: '', authenticationRejected: false },
    };
  }
  return { outcome: 'unchanged', state };
}

export function isPicPortalUiSessionReady<TAdmin>(state: PicPortalUiSessionState<TAdmin>): boolean {
  return Boolean(state.admin && state.galleryId && !state.authenticationRejected);
}

export function canLogoutPicPortalUiSession<TAdmin>(state: PicPortalUiSessionState<TAdmin>): boolean {
  return Boolean(state.admin || state.authenticationRejected);
}
