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

export default function PicPortalPanel() {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [admin, setAdmin] = useState<LoginResult['admin'] | null>(null);
  const [galleries, setGalleries] = useState<GallerySummary[]>([]);
  const [galleryId, setGalleryId] = useState('');
  const [paths, setPaths] = useState<string[]>([]);
  const [newGalleryTitle, setNewGalleryTitle] = useState('');
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
    if (!newGalleryTitle.trim()) return;
    setBusy(true);
    try {
      const gallery = await invoke<GallerySummary>(Invokes.PicPortalCreateGallery, {
        input: {
          title: newGalleryTitle.trim(),
          galleryType: 'event',
          accessMode: 'link',
          status: 'active',
          faceFilterEnabled: true,
        },
      });
      setNewGalleryTitle('');
      await refreshGalleries();
      setGalleryId(gallery.id);
      setStatus(t('picportal.galleryCreated', { defaultValue: 'Gallery created' }));
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }, [newGalleryTitle, refreshGalleries, t]);

  const publish = useCallback(async () => {
    if (!galleryId || paths.length === 0) return;
    setBusy(true);
    try {
      const result = await invoke<PublishResult>(Invokes.PicPortalPublish, { paths, galleryId });
      setStatus(
        t('picportal.publishResult', {
          completed: result.completed,
          failed: result.failed,
          defaultValue: '{{completed}} uploaded, {{failed}} failed',
        }),
      );
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
          <div className="flex gap-2">
            <input
              className="min-w-0 flex-1 rounded-md bg-bg-primary border border-border-color px-2 py-1 text-sm"
              value={newGalleryTitle}
              onChange={(event) => setNewGalleryTitle(event.target.value)}
              placeholder={t('picportal.newGallery', { defaultValue: 'New gallery title' })}
            />
            <button
              onClick={createGallery}
              disabled={busy || !newGalleryTitle.trim()}
              className="p-2 rounded-md hover:bg-surface"
              data-tooltip={t('picportal.createGallery', { defaultValue: 'Create gallery' })}
            >
              <FolderPlus size={16} />
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
        <Text variant={TextVariants.small} className="break-words text-text-secondary">
          {status}
        </Text>
      )}
    </div>
  );
}
