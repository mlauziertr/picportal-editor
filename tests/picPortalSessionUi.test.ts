import assert from 'node:assert/strict';
import test from 'node:test';
import {
  canLogoutPicPortalUiSession,
  isPicPortalUiSessionReady,
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
