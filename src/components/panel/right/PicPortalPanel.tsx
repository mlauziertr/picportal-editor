import { useCallback, useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { LogIn, LogOut, RefreshCw, FolderPlus } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../../ui/Button';
import Text from '../../ui/Text';
import { TextColors, TextVariants } from '../../../types/typography';
import { Invokes } from '../../ui/AppProperties';
import { picPortalDestinationChanged, retainExplicitGallerySelection } from '../../../utils/picPortalSelection';

export interface PicPortalExportOptions {
  galleryId: string;
  includeFaceAnalysis: boolean;
}

interface GallerySummary {
  id: string;
  title: string;
  slug: string;
  photoCount?: number;
  faceFilterEnabled: boolean;
}

interface AdminIdentity {
  name: string;
  email: string;
}

interface SessionStatus {
  connected: boolean;
  admin: AdminIdentity | null;
  galleries: GallerySummary[];
  persistent: boolean;
  message?: string | null;
}

interface LoginResult {
  admin: AdminIdentity;
  galleries: GallerySummary[];
  persistent: boolean;
  message?: string | null;
}

type GalleryType = '' | 'event' | 'client';
type GalleryAccessMode = '' | 'link' | 'password';
type GalleryStatus = '' | 'active' | 'draft';
type FaceFilterPolicy = '' | 'enabled' | 'disabled';

interface PicPortalPanelProps {
  pathsCount: number;
  unsupportedReason?: string;
  disabled?: boolean;
  onOptionsChange?: (options: PicPortalExportOptions) => void;
  onReadyChange?: (ready: boolean) => void;
}

export default function PicPortalPanel({
  pathsCount,
  unsupportedReason,
  disabled = false,
  onOptionsChange,
  onReadyChange,
}: PicPortalPanelProps) {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [admin, setAdmin] = useState<AdminIdentity | null>(null);
  const [galleries, setGalleries] = useState<GallerySummary[]>([]);
  const [galleryId, setGalleryId] = useState('');
  const galleryIdRef = useRef('');
  const accountIdentityRef = useRef<string | null>(null);
  const [includeFaceAnalysis, setIncludeFaceAnalysis] = useState(false);
  const [newGalleryTitle, setNewGalleryTitle] = useState('');
  const [newGalleryType, setNewGalleryType] = useState<GalleryType>('');
  const [newGalleryAccessMode, setNewGalleryAccessMode] = useState<GalleryAccessMode>('');
  const [newGalleryStatus, setNewGalleryStatus] = useState<GalleryStatus>('');
  const [newGalleryFaceFilter, setNewGalleryFaceFilter] = useState<FaceFilterPolicy>('');
  const [newGalleryClientName, setNewGalleryClientName] = useState('');
  const [newGalleryClientEmail, setNewGalleryClientEmail] = useState('');
  const [newGalleryPassword, setNewGalleryPassword] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [restoring, setRestoring] = useState(true);

  const selectedGallery = galleries.find((gallery) => gallery.id === galleryId) ?? null;

  const clearSession = useCallback(() => {
    accountIdentityRef.current = null;
    galleryIdRef.current = '';
    setAdmin(null);
    setGalleries([]);
    setGalleryId('');
    setIncludeFaceAnalysis(false);
  }, []);

  const applySession = useCallback(
    (session: SessionStatus) => {
      if (!session.connected || !session.admin) {
        clearSession();
        if (session.message) setStatus(session.message);
        return;
      }
      const nextAccount = session.admin.email.trim().toLowerCase();
      const nextGalleryId =
        accountIdentityRef.current === nextAccount
          ? retainExplicitGallerySelection(
              galleryIdRef.current,
              session.galleries.map((gallery) => gallery.id),
            )
          : '';
      if (picPortalDestinationChanged(accountIdentityRef.current, galleryIdRef.current, nextAccount, nextGalleryId)) {
        setIncludeFaceAnalysis(false);
      }
      accountIdentityRef.current = nextAccount;
      galleryIdRef.current = nextGalleryId;
      setAdmin(session.admin);
      setGalleries(session.galleries);
      setGalleryId(nextGalleryId);
      if (session.message) setStatus(session.message);
    },
    [clearSession],
  );

  useEffect(() => {
    let active = true;
    const restore = async () => {
      try {
        const session = await invoke<SessionStatus>(Invokes.PicPortalRestoreSession);
        if (active) applySession(session);
      } catch (error) {
        if (active) setStatus(String(error));
      } finally {
        if (active) setRestoring(false);
      }
    };
    restore();

    const invalidated = listen('picportal-session-invalidated', () => {
      if (!active) return;
      clearSession();
      setStatus(t('picportal.sessionExpired', { defaultValue: 'PicPortal session expired; connect again' }));
    });
    return () => {
      active = false;
      invalidated.then((unlisten) => unlisten());
    };
  }, [applySession, clearSession, t]);

  useEffect(() => {
    onOptionsChange?.({ galleryId, includeFaceAnalysis });
    onReadyChange?.(Boolean(admin && galleryId));
  }, [admin, galleryId, includeFaceAnalysis, onOptionsChange, onReadyChange]);

  useEffect(() => {
    if (selectedGallery && !selectedGallery.faceFilterEnabled) setIncludeFaceAnalysis(false);
  }, [selectedGallery]);

  const isExpiredError = (error: unknown) => String(error).includes('session expired or was revoked');

  const refreshGalleries = useCallback(async () => {
    setBusy(true);
    try {
      const next = await invoke<GallerySummary[]>(Invokes.PicPortalGalleries);
      const nextGalleryId = retainExplicitGallerySelection(
        galleryIdRef.current,
        next.map((gallery) => gallery.id),
      );
      if (
        picPortalDestinationChanged(
          accountIdentityRef.current,
          galleryIdRef.current,
          accountIdentityRef.current,
          nextGalleryId,
        )
      ) {
        setIncludeFaceAnalysis(false);
      }
      galleryIdRef.current = nextGalleryId;
      setGalleries(next);
      setGalleryId(nextGalleryId);
      setStatus('');
    } catch (error) {
      if (isExpiredError(error)) clearSession();
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [clearSession]);

  const login = useCallback(async () => {
    setBusy(true);
    setStatus('');
    try {
      const result = await invoke<LoginResult>(Invokes.PicPortalLogin, { email, password });
      const nextAccount = result.admin.email.trim().toLowerCase();
      if (picPortalDestinationChanged(accountIdentityRef.current, galleryIdRef.current, nextAccount, '')) {
        setIncludeFaceAnalysis(false);
      }
      accountIdentityRef.current = nextAccount;
      galleryIdRef.current = '';
      setAdmin(result.admin);
      setGalleries(result.galleries);
      setGalleryId('');
      setPassword('');
      const connected = t('picportal.loggedIn', { defaultValue: 'Connected to PicPortal' });
      setStatus(result.message ? `${connected}\n${result.message}` : connected);
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [email, password, t]);

  const logout = useCallback(async () => {
    setBusy(true);
    try {
      await invoke(Invokes.PicPortalLogout);
      clearSession();
      setStatus(t('picportal.loggedOut', { defaultValue: 'Disconnected' }));
    } catch (error) {
      clearSession();
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [clearSession, t]);

  const createGallery = useCallback(async () => {
    if (
      !newGalleryTitle.trim() ||
      !newGalleryType ||
      !newGalleryAccessMode ||
      !newGalleryStatus ||
      !newGalleryFaceFilter ||
      (newGalleryType === 'client' && (!newGalleryClientName.trim() || !newGalleryClientEmail.trim())) ||
      (newGalleryAccessMode === 'password' && !newGalleryPassword.trim())
    ) {
      return;
    }
    setBusy(true);
    try {
      const gallery = await invoke<GallerySummary>(Invokes.PicPortalCreateGallery, {
        input: {
          title: newGalleryTitle.trim(),
          galleryType: newGalleryType,
          accessMode: newGalleryAccessMode,
          status: newGalleryStatus,
          faceFilterEnabled: newGalleryFaceFilter === 'enabled',
          clientName: newGalleryType === 'client' ? newGalleryClientName.trim() : null,
          clientEmail: newGalleryType === 'client' ? newGalleryClientEmail.trim() : null,
          password: newGalleryAccessMode === 'password' ? newGalleryPassword : null,
        },
      });
      setGalleries((current) => [...current.filter((item) => item.id !== gallery.id), gallery]);
      if (
        picPortalDestinationChanged(
          accountIdentityRef.current,
          galleryIdRef.current,
          accountIdentityRef.current,
          gallery.id,
        )
      ) {
        setIncludeFaceAnalysis(false);
      }
      galleryIdRef.current = gallery.id;
      setGalleryId(gallery.id);
      setStatus(t('picportal.galleryCreated', { defaultValue: 'Gallery created' }));
      setNewGalleryTitle('');
      setNewGalleryType('');
      setNewGalleryAccessMode('');
      setNewGalleryStatus('');
      setNewGalleryFaceFilter('');
      setNewGalleryClientName('');
      setNewGalleryClientEmail('');
      setNewGalleryPassword('');
    } catch (error) {
      if (isExpiredError(error)) clearSession();
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [
    clearSession,
    newGalleryAccessMode,
    newGalleryClientEmail,
    newGalleryClientName,
    newGalleryFaceFilter,
    newGalleryPassword,
    newGalleryStatus,
    newGalleryTitle,
    newGalleryType,
    t,
  ]);

  const canCreateGallery =
    Boolean(newGalleryTitle.trim()) &&
    Boolean(newGalleryType) &&
    Boolean(newGalleryAccessMode) &&
    Boolean(newGalleryStatus) &&
    Boolean(newGalleryFaceFilter) &&
    (newGalleryType !== 'client' || Boolean(newGalleryClientName.trim() && newGalleryClientEmail.trim())) &&
    (newGalleryAccessMode !== 'password' || Boolean(newGalleryPassword.trim()));

  return (
    <div className="mt-5 border-t border-border-color pt-4 space-y-3">
      <div className="flex items-center justify-between">
        <Text variant={TextVariants.heading}>
          {t('picportal.destinationTitle', { defaultValue: 'Export to PicPortal' })}
        </Text>
        {admin && (
          <button
            onClick={logout}
            disabled={busy || disabled}
            className="text-text-secondary hover:text-text-primary"
            data-tooltip={t('picportal.logout', { defaultValue: 'Log out' })}
          >
            <LogOut size={16} />
          </button>
        )}
      </div>
      <Text variant={TextVariants.small} color={TextColors.secondary}>
        {t('picportal.selectedImages', {
          count: pathsCount,
          defaultValue: '{{count}} images will be rendered and uploaded',
        })}
      </Text>
      {unsupportedReason && (
        <Text variant={TextVariants.small} color={TextColors.secondary}>
          {unsupportedReason}
        </Text>
      )}
      {restoring ? (
        <Text variant={TextVariants.small} color={TextColors.secondary}>
          {t('picportal.restoring', { defaultValue: 'Restoring secure PicPortal session…' })}
        </Text>
      ) : !admin ? (
        <div className="space-y-2">
          <input
            className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
            type="email"
            autoComplete="username"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder={t('picportal.email', { defaultValue: 'PicPortal email' })}
          />
          <input
            className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={t('picportal.password', { defaultValue: 'Password' })}
          />
          <Button onClick={login} disabled={busy || disabled || !email || !password}>
            <LogIn size={16} className="mr-2" /> {t('picportal.login', { defaultValue: 'Connect' })}
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <Text variant={TextVariants.small} className="text-text-secondary truncate">
              {admin.name} · {admin.email}
            </Text>
            <button
              onClick={refreshGalleries}
              disabled={busy || disabled}
              className="p-2 rounded-md hover:bg-surface shrink-0"
              data-tooltip={t('picportal.refresh', { defaultValue: 'Refresh galleries' })}
            >
              <RefreshCw size={16} />
            </button>
          </div>
          <select
            className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
            value={galleryId}
            onChange={(event) => {
              const nextGalleryId = event.target.value;
              if (
                picPortalDestinationChanged(
                  accountIdentityRef.current,
                  galleryIdRef.current,
                  accountIdentityRef.current,
                  nextGalleryId,
                )
              ) {
                setIncludeFaceAnalysis(false);
              }
              galleryIdRef.current = nextGalleryId;
              setGalleryId(nextGalleryId);
            }}
            disabled={busy || disabled}
          >
            <option value="" disabled>
              {t('picportal.selectGallery', { defaultValue: 'Select a gallery' })}
            </option>
            {galleries.map((gallery) => (
              <option key={gallery.id} value={gallery.id}>
                {gallery.title || gallery.slug}
              </option>
            ))}
          </select>
          {selectedGallery?.faceFilterEnabled ? (
            <label className="flex items-start gap-2 text-sm text-text-secondary">
              <input
                type="checkbox"
                checked={includeFaceAnalysis}
                onChange={(event) => setIncludeFaceAnalysis(event.target.checked)}
                disabled={busy || disabled}
                className="mt-0.5 accent-accent"
              />
              <span>
                {t('picportal.includeFaceAnalysis', {
                  defaultValue: 'Include face-selection data for this export (explicit consent)',
                })}
              </span>
            </label>
          ) : (
            <Text variant={TextVariants.small} color={TextColors.secondary}>
              {t('picportal.faceAnalysisUnavailable', {
                defaultValue: 'Face-selection data is disabled for this gallery.',
              })}
            </Text>
          )}
          <div className="space-y-2 rounded-md border border-border-color p-2">
            <input
              className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
              value={newGalleryTitle}
              onChange={(event) => setNewGalleryTitle(event.target.value)}
              placeholder={t('picportal.newGallery', { defaultValue: 'New gallery title' })}
            />
            <div className="grid grid-cols-2 gap-2">
              <select
                className="min-w-0 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                value={newGalleryType}
                onChange={(event) => {
                  const value = event.target.value as GalleryType;
                  setNewGalleryType(value);
                  if (value !== 'client') {
                    setNewGalleryClientName('');
                    setNewGalleryClientEmail('');
                  }
                }}
              >
                <option value="" disabled>
                  {t('picportal.galleryType', { defaultValue: 'Gallery type' })}
                </option>
                <option value="event">{t('picportal.galleryTypeEvent', { defaultValue: 'Event' })}</option>
                <option value="client">{t('picportal.galleryTypeClient', { defaultValue: 'Client' })}</option>
              </select>
              <select
                className="min-w-0 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                value={newGalleryAccessMode}
                onChange={(event) => {
                  const value = event.target.value as GalleryAccessMode;
                  setNewGalleryAccessMode(value);
                  if (value !== 'password') setNewGalleryPassword('');
                }}
              >
                <option value="" disabled>
                  {t('picportal.accessMode', { defaultValue: 'Access mode' })}
                </option>
                <option value="link">{t('picportal.accessModeLink', { defaultValue: 'Link' })}</option>
                <option value="password">{t('picportal.accessModePassword', { defaultValue: 'Password' })}</option>
              </select>
              <select
                className="min-w-0 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                value={newGalleryStatus}
                onChange={(event) => setNewGalleryStatus(event.target.value as GalleryStatus)}
              >
                <option value="" disabled>
                  {t('picportal.status', { defaultValue: 'Status' })}
                </option>
                <option value="active">{t('picportal.statusActive', { defaultValue: 'Active' })}</option>
                <option value="draft">{t('picportal.statusDraft', { defaultValue: 'Draft' })}</option>
              </select>
              <select
                className="min-w-0 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                value={newGalleryFaceFilter}
                onChange={(event) => setNewGalleryFaceFilter(event.target.value as FaceFilterPolicy)}
              >
                <option value="" disabled>
                  {t('picportal.faceFilter', { defaultValue: 'Face filtering' })}
                </option>
                <option value="enabled">{t('picportal.faceFilterEnabled', { defaultValue: 'Faces enabled' })}</option>
                <option value="disabled">
                  {t('picportal.faceFilterDisabled', { defaultValue: 'Faces disabled' })}
                </option>
              </select>
            </div>
            {newGalleryType === 'client' && (
              <div className="grid grid-cols-2 gap-2">
                <input
                  className="min-w-0 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                  value={newGalleryClientName}
                  onChange={(event) => setNewGalleryClientName(event.target.value)}
                  placeholder={t('picportal.clientName', { defaultValue: 'Client name' })}
                />
                <input
                  className="min-w-0 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                  type="email"
                  value={newGalleryClientEmail}
                  onChange={(event) => setNewGalleryClientEmail(event.target.value)}
                  placeholder={t('picportal.clientEmail', { defaultValue: 'Client email' })}
                />
              </div>
            )}
            {newGalleryAccessMode === 'password' && (
              <input
                className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
                type="password"
                autoComplete="new-password"
                value={newGalleryPassword}
                onChange={(event) => setNewGalleryPassword(event.target.value)}
                placeholder={t('picportal.password', { defaultValue: 'Password' })}
              />
            )}
            <button
              onClick={createGallery}
              disabled={busy || disabled || !canCreateGallery}
              className="flex w-full items-center justify-center gap-2 rounded-md p-2 hover:bg-surface disabled:opacity-50"
            >
              <FolderPlus size={16} />
              {t('picportal.createGallery', { defaultValue: 'Create gallery' })}
            </button>
          </div>
        </div>
      )}
      {status && (
        <Text variant={TextVariants.small} className="whitespace-pre-wrap break-words text-text-secondary">
          {status}
        </Text>
      )}
    </div>
  );
}
