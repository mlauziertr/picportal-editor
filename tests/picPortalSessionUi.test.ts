import assert from 'node:assert/strict';
import test from 'node:test';
import {
  canLogoutPicPortalUiSession,
  isPicPortalUiSessionReady,
  picPortalErrorMessage,
  transitionPicPortalUiSession,
} from '../src/utils/picPortalSessionUi.ts';

const session = {
  admin: { email: 'editor@example.test' },
  galleryId: 'gallery-a',
  authenticationRejected: false,
};

test('failed secure removal retains logout identity and blocks PicPortal export', () => {
  assert.equal(isPicPortalUiSessionReady(session), true);
  assert.equal(canLogoutPicPortalUiSession(session), true);

  const transition = transitionPicPortalUiSession(
    session,
    'PicPortal authentication was rejected (HTTP 401: session expired or was revoked); PicPortal session invalidation not completed because secure session storage could not be cleared: synthetic failure',
  );

  assert.equal(transition.outcome, 'credentials-retained');
  assert.deepEqual(transition.state.admin, session.admin);
  assert.equal(transition.state.galleryId, 'gallery-a');
  assert.equal(transition.state.authenticationRejected, true);
  assert.equal(isPicPortalUiSessionReady(transition.state), false);
  assert.equal(canLogoutPicPortalUiSession(transition.state), true);
});

test('failed secure logout retains retry state and blocks PicPortal export', () => {
  const transition = transitionPicPortalUiSession(
    session,
    'PicPortal logout not completed because secure session storage could not be cleared: synthetic failure',
  );

  assert.equal(transition.outcome, 'credentials-retained');
  assert.deepEqual(transition.state.admin, session.admin);
  assert.equal(transition.state.galleryId, 'gallery-a');
  assert.equal(transition.state.authenticationRejected, true);
  assert.equal(isPicPortalUiSessionReady(transition.state), false);
  assert.equal(canLogoutPicPortalUiSession(transition.state), true);
});

test('successful invalidation clears the UI session', () => {
  const transition = transitionPicPortalUiSession(
    session,
    'PicPortal session expired or was revoked; connect again',
  );

  assert.equal(transition.outcome, 'invalidated');
  assert.equal(transition.state.admin, null);
  assert.equal(transition.state.galleryId, '');
  assert.equal(transition.state.authenticationRejected, false);
  assert.equal(isPicPortalUiSessionReady(transition.state), false);
  assert.equal(canLogoutPicPortalUiSession(transition.state), false);
});

test('structured command errors drive the session transition by code', () => {
  const expired = transitionPicPortalUiSession(session, {
    code: 'session_expired',
    message: 'PicPortal session expired or was revoked; connect again',
    retryable: false,
  });
  assert.equal(expired.outcome, 'invalidated');
  assert.equal(expired.state.admin, null);

  const retained = transitionPicPortalUiSession(session, {
    code: 'session_clear_failed',
    message: 'synthetic keyring failure',
    retryable: false,
    status: 401,
  });
  assert.equal(retained.outcome, 'credentials-retained');
  assert.equal(retained.state.authenticationRejected, true);

  const offline = transitionPicPortalUiSession(session, {
    code: 'offline',
    message: 'gallery request failed [network:offline]: synthetic',
    retryable: true,
  });
  assert.equal(offline.outcome, 'unchanged');
});

test('structured command errors expose their technical message, not [object Object]', () => {
  assert.equal(
    picPortalErrorMessage({ code: 'server', message: 'HTTP 503: synthetic', retryable: true, status: 503 }),
    'HTTP 503: synthetic',
  );
  assert.equal(picPortalErrorMessage('legacy string error'), 'legacy string error');
});
