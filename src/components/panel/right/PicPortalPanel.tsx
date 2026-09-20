import { useCallback, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { LogIn, LogOut, UploadCloud, RefreshCw, FolderPlus } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../../ui/Button';
import Text from '../../ui/Text';
import { TextVariants } from '../../../types/typography';
import { Invokes } from '../../ui/AppProperties';

interface GallerySummary {
  id: string;
  title: string;
  slug: string;
  photoCount?: number;
  faceFilterEnabled: boolean;
}

interface LoginResult {
  admin: { name: string; email: string };
  galleries: GallerySummary[];
}

interface PublishResult {
  completed: number;
  failed: number;
  items: Array<{ path: string; state: string; error?: string | null }>;
}

type GalleryType = '' | 'event' | 'client';
type GalleryAccessMode = '' | 'link' | 'password';
type GalleryStatus = '' | 'active' | 'draft';
type FaceFilterPolicy = '' | 'enabled' | 'disabled';

export default function PicPortalPanel() {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [admin, setAdmin] = useState<LoginResult['admin'] | null>(null);
  const [galleries, setGalleries] = useState<GallerySummary[]>([]);
  const [galleryId, setGalleryId] = useState('');
  const [paths, setPaths] = useState<string[]>([]);
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

  const refreshGalleries = useCallback(async () => {
    const next = await invoke<GallerySummary[]>(Invokes.PicPortalGalleries);
    setGalleries(next);
    setGalleryId((current) => (next.some((gallery) => gallery.id === current) ? current : next[0]?.id || ''));
  }, []);

  const login = useCallback(async () => {
    setBusy(true);
    setStatus('');
    try {
      const result = await invoke<LoginResult>(Invokes.PicPortalLogin, { email, password });
      setAdmin(result.admin);
      setGalleries(result.galleries);
      setGalleryId(result.galleries[0]?.id || '');
      setPassword('');
      setStatus(t('picportal.loggedIn', { defaultValue: 'Connected to PicPortal' }));
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
      setAdmin(null);
      setGalleries([]);
      setGalleryId('');
      setStatus(t('picportal.loggedOut', { defaultValue: 'Disconnected' }));
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [t]);

  const chooseExports = useCallback(async () => {
    const selected = await open({
      multiple: true,
      directory: false,
      title: t('picportal.chooseExports', { defaultValue: 'Choose exported images' }),
      filters: [{ name: 'JPEG, PNG, WebP', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
    });
    if (!selected) return;
    setPaths(Array.isArray(selected) ? selected : [selected]);
  }, [t]);

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
      const createdStatus = t('picportal.galleryCreated', { defaultValue: 'Gallery created' });
      setGalleries((current) => [...current.filter((item) => item.id !== gallery.id), gallery]);
      setGalleryId(gallery.id);
      setStatus(createdStatus);
      setNewGalleryTitle('');
      setNewGalleryType('');
      setNewGalleryAccessMode('');
      setNewGalleryStatus('');
      setNewGalleryFaceFilter('');
      setNewGalleryClientName('');
      setNewGalleryClientEmail('');
      setNewGalleryPassword('');
      try {
        const refreshed = await invoke<GallerySummary[]>(Invokes.PicPortalGalleries);
        setGalleries(refreshed.some((item) => item.id === gallery.id) ? refreshed : [...refreshed, gallery]);
        setGalleryId(gallery.id);
      } catch (refreshError) {
        setStatus(
          `${createdStatus}\n${t('picportal.refresh', { defaultValue: 'Refresh galleries' })}: ${String(refreshError)}`,
        );
      }
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [
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

  const publish = useCallback(async () => {
    if (!galleryId || paths.length === 0) return;
    setBusy(true);
    try {
      const result = await invoke<PublishResult>(Invokes.PicPortalPublish, { paths, galleryId });
      const summary = [
        t('picportal.publishCompleted', {
          count: result.completed,
          defaultValue: '{{count}} images uploaded',
        }),
        t('picportal.publishFailed', {
          count: result.failed,
          defaultValue: '{{count}} images failed',
        }),
      ].join(', ');
      const failures = result.items.flatMap((item) => (item.error ? [`${item.path}: ${item.error}`] : []));
      setStatus([summary, ...failures].join('\n'));
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [galleryId, paths, t]);

  return (
    <div className="mt-5 border-t border-border-color pt-4 space-y-3">
      <div className="flex items-center justify-between">
        <Text variant={TextVariants.heading}>{t('picportal.title', { defaultValue: 'PicPortal publication' })}</Text>
        {admin && (
          <button
            onClick={logout}
            disabled={busy}
            className="text-text-secondary hover:text-text-primary"
            data-tooltip={t('picportal.logout', { defaultValue: 'Disconnect' })}
          >
            <LogOut size={16} />
          </button>
        )}
      </div>
      {!admin ? (
        <div className="space-y-2">
          <input
            className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder={t('picportal.email', { defaultValue: 'PicPortal email' })}
          />
          <input
            className="w-full rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={t('picportal.password', { defaultValue: 'Password' })}
          />
          <Button onClick={login} disabled={busy || !email || !password}>
            <LogIn size={16} className="mr-2" /> {t('picportal.login', { defaultValue: 'Connect' })}
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          <Text variant={TextVariants.small} className="text-text-secondary">
            {admin.name} · {admin.email}
          </Text>
          <div className="flex gap-2">
            <select
              className="min-w-0 flex-1 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
              value={galleryId}
              onChange={(event) => setGalleryId(event.target.value)}
            >
              {galleries.map((gallery) => (
                <option key={gallery.id} value={gallery.id}>
                  {gallery.title || gallery.slug}
                </option>
              ))}
            </select>
            <button
              onClick={refreshGalleries}
              disabled={busy}
              className="p-2 rounded-md hover:bg-surface"
              data-tooltip={t('picportal.refresh', { defaultValue: 'Refresh galleries' })}
            >
              <RefreshCw size={16} />
            </button>
          </div>
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
              disabled={busy || !canCreateGallery}
              className="flex w-full items-center justify-center gap-2 rounded-md p-2 hover:bg-surface disabled:opacity-50"
              data-tooltip={t('picportal.createGallery', { defaultValue: 'Create gallery' })}
            >
              <FolderPlus size={16} />
              {t('picportal.createGallery', { defaultValue: 'Create gallery' })}
            </button>
          </div>
          <Button onClick={chooseExports} disabled={busy}>
            {t('picportal.chooseExports', { defaultValue: 'Choose exported images' })} ({paths.length})
          </Button>
          <Button onClick={publish} disabled={busy || !galleryId || paths.length === 0}>
            <UploadCloud size={16} className="mr-2" />{' '}
            {t('picportal.publish', { defaultValue: 'Process and publish locally' })}
          </Button>
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
