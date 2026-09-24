export interface PicPortalUiSessionState<TAdmin = unknown> {
  admin: TAdmin | null;
  galleryId: string;
  authenticationRejected: boolean;
}

export type PicPortalSessionErrorOutcome = 'credentials-retained' | 'invalidated' | 'unchanged';

export function transitionPicPortalUiSession<TAdmin>(
  state: PicPortalUiSessionState<TAdmin>,
  error: unknown,
): { outcome: PicPortalSessionErrorOutcome; state: PicPortalUiSessionState<TAdmin> } {
  const message = String(error);
  if (
    message.includes(
      'PicPortal session invalidation not completed because secure session storage could not be cleared',
    )
  ) {
    return {
      outcome: 'credentials-retained',
      state: { ...state, authenticationRejected: true },
    };
  }
  if (message.includes('session expired or was revoked')) {
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
