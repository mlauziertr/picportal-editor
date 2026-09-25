import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import { CheckCircle, ChevronDown, ChevronUp, CircleDashed, Sparkles, X, XCircle } from 'lucide-react';
import {
  CULLING_ALREADY_RUNNING,
  CULLING_CANCELLED,
  CullingCapabilities,
  CullingSettings,
  Invokes,
  Progress,
} from '../ui/AppProperties';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

interface CullingModalProps {
  isOpen: boolean;
  onClose(): void;
  progress: Progress | null;
  error: string | null;
  imagePaths: string[];
  folderPath: string | null;
  onError(error: string): void;
}

const DEFAULT_SETTINGS: CullingSettings = {
  selectionAmount: 'standard',
  blurSeverity: 'moderate',
  detectDuplicates: true,
  // Off by default for v1: measured unreliable on real photos (MAX-21), kept as opt-in experiments.
  detectBlurry: false,
  detectClosedEyes: false,
  detectHighlights: true,
  detectSubject: true,
  subjectProfile: 'general',
  autoAssignStars: true,
};

const amountOptions = [
  { value: 'extreme' as const, label: 'Extreme' },
  { value: 'few' as const, label: 'Few' },
  { value: 'standard' as const, label: 'Standard' },
  { value: 'more' as const, label: 'More' },
];

const severityOptions = [
  { value: 'lenient' as const, label: 'Lenient' },
  { value: 'moderate' as const, label: 'Moderate' },
  { value: 'strict' as const, label: 'Strict' },
];

const subjectProfiles: Array<{ value: CullingSettings['subjectProfile']; label: string }> = [
  { value: 'general', label: 'General people' },
  { value: 'portrait', label: 'Portrait' },
  { value: 'wedding', label: 'Wedding' },
  { value: 'sports', label: 'Sports' },
  { value: 'dance', label: 'Dance' },
];

function SegmentedChoice<T extends string>({
  options,
  value,
  onChange,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange(value: T): void;
}) {
  return (
    <div className="grid grid-cols-4 gap-1 rounded-lg bg-bg-primary p-1" role="group">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`rounded-md px-2 py-2 text-sm transition-colors ${
            value === option.value
              ? 'bg-card-active text-text-primary shadow-sm'
              : 'text-text-secondary hover:text-text-primary'
          }`}
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export default function CullingModal({
  isOpen,
  onClose,
  progress,
  error,
  imagePaths,
  folderPath,
  onError,
}: CullingModalProps) {
  const { t } = useTranslation();
  const [settings, setSettings] = useState<CullingSettings>(DEFAULT_SETTINGS);
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(false);
  const [capabilities, setCapabilities] = useState<CullingCapabilities | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const startInProgressRef = useRef(false);

  useEffect(() => {
    if (!isOpen) {
      setIsCustomizeOpen(false);
      setIsCancelling(false);
      setCapabilities(null);
      return;
    }
    let active = true;
    invoke<CullingCapabilities>(Invokes.CullingCapabilities)
      .then((result) => {
        if (!active) return;
        setCapabilities(result);
        // Optional detectors that cannot run start disabled instead of failing silently.
        setSettings((current) => ({
          ...current,
          detectSubject: current.detectSubject && result.subject === 'ready',
          detectClosedEyes: current.detectClosedEyes && result.faces === 'ready',
        }));
      })
      .catch((capabilityError) => {
        console.error('Culling capability check failed:', capabilityError);
      });
    return () => {
      active = false;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!progress) setIsCancelling(false);
  }, [progress]);

  const handleStartCulling = useCallback(async () => {
    if (imagePaths.length === 0 || startInProgressRef.current) return;
    startInProgressRef.current = true;
    try {
      await invoke(Invokes.CullImages, { paths: imagePaths, settings });
    } catch (startError) {
      const message = String(startError);
      if (message === CULLING_CANCELLED) return;
      if (message === CULLING_ALREADY_RUNNING) {
        toast.info(t('modals.culling.errors.alreadyRunning'));
        return;
      }
      onError(message);
    } finally {
      startInProgressRef.current = false;
    }
  }, [imagePaths, settings, onError, t]);

  const handleCancelAnalysis = useCallback(async () => {
    setIsCancelling(true);
    try {
      const wasRunning = await invoke<boolean>(Invokes.CancelCulling);
      if (!wasRunning) onClose();
    } catch (cancelError) {
      setIsCancelling(false);
      console.error('Failed to cancel culling:', cancelError);
    }
  }, [onClose]);

  const recoveredRun = Boolean(capabilities?.running) && !progress;
  const isRunning = Boolean(progress) || recoveredRun;

  // Closing the window while the analysis runs cancels it (nothing has been written yet).
  const handleClose = useCallback(() => {
    if (isRunning) {
      void handleCancelAnalysis();
    } else {
      onClose();
    }
  }, [isRunning, handleCancelAnalysis, onClose]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        handleClose();
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [isOpen, handleClose]);

  if (!isOpen) return null;

  const folderLabel = folderPath || t('modals.culling.noFolder', { defaultValue: 'No folder open' });
  const stageCode = progress?.stageCode || 'preparing';
  const progressCurrent = progress?.current || 0;
  const progressTotal = progress?.total || 0;
  const progressText = t('modals.culling.progressCount', {
    current: progressCurrent,
    total: progressTotal,
    defaultValue: '{{current}} of {{total}}',
  });
  const capabilityRows: Array<{ key: 'sharpness' | 'faces' | 'subject'; ready: boolean; hint?: string }> =
    capabilities
      ? [
          { key: 'sharpness', ready: true },
          {
            key: 'faces',
            ready: capabilities.faces === 'ready',
            hint: t('modals.culling.capabilities.facesMissing'),
          },
          {
            key: 'subject',
            ready: capabilities.subject === 'ready',
            hint: t('modals.culling.capabilities.workerMissing'),
          },
        ]
      : [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-surface p-6 shadow-2xl">
        {isRunning ? (
          <div className="flex min-h-64 flex-col justify-center">
            <div className="flex items-center gap-2">
              <Sparkles className="h-6 w-6 text-accent" />
              <Text variant={TextVariants.title}>{t('modals.culling.title')}</Text>
            </div>
            <div className="mt-6 flex items-baseline justify-between gap-4">
              <Text variant={TextVariants.heading}>
                {t(`modals.culling.stage.${stageCode}`, { defaultValue: progress?.stage || '' })}
              </Text>
              {progressTotal > 0 && <Text color={TextColors.secondary}>{progressText}</Text>}
            </div>
            <div
              className="mt-3 h-2 w-full rounded-full bg-bg-primary"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={progressTotal}
              aria-valuenow={progressCurrent}
              aria-valuetext={progressText}
            >
              <div
                className="h-2 rounded-full bg-accent motion-safe:transition-all"
                style={{ width: `${progressTotal > 0 ? (progressCurrent / progressTotal) * 100 : 0}%` }}
              />
            </div>
            <Text color={TextColors.secondary} className="mt-3">
              {t('modals.culling.progressHint')}
            </Text>
            <div className="mt-6 flex justify-end">
              <button
                type="button"
                className="rounded-md px-4 py-2 text-text-secondary hover:bg-bg-primary disabled:opacity-50"
                onClick={handleCancelAnalysis}
                disabled={isCancelling}
              >
                {isCancelling ? t('modals.culling.cancelling') : t('modals.culling.cancelAnalysis')}
              </button>
            </div>
          </div>
        ) : error ? (
          <div className="flex min-h-64 flex-col items-center justify-center text-center">
            <XCircle className="h-14 w-14 text-red-500" />
            <Text variant={TextVariants.heading} className="mt-5">
              {t('modals.culling.cullingFailed')}
            </Text>
            <Text color={TextColors.secondary} className="mt-2 max-w-lg break-words">
              {error}
            </Text>
            <div className="mt-6 flex gap-3">
              <button
                type="button"
                className="rounded-md px-4 py-2 text-text-secondary hover:bg-bg-primary"
                onClick={onClose}
              >
                {t('modals.culling.close')}
              </button>
              <Button onClick={handleStartCulling}>{t('modals.culling.retry')}</Button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Sparkles className="h-6 w-6 text-accent" />
                  <Text variant={TextVariants.title}>{t('modals.culling.title')}</Text>
                </div>
                <Text color={TextColors.secondary} className="mt-2 break-all">
                  {folderLabel}
                </Text>
              </div>
              <button
                type="button"
                className="rounded-md p-2 text-text-secondary hover:bg-bg-primary hover:text-text-primary"
                onClick={onClose}
                aria-label={t('modals.culling.close')}
              >
                <X size={18} />
              </button>
            </div>

            <div className="mt-5 rounded-lg border border-border-color/40 bg-bg-primary/40 p-4">
              <Text variant={TextVariants.heading}>
                {t('modals.culling.photosInFolder', {
                  count: imagePaths.length,
                  defaultValue: '{{count}} supported photos in this folder',
                })}
              </Text>
              <Text color={TextColors.secondary} className="mt-1">
                {imagePaths.length === 0
                  ? t('modals.culling.emptyFolder', { defaultValue: 'There are no supported photos to analyze.' })
                  : imagePaths.length === 1
                    ? t('modals.culling.singlePhoto', {
                        defaultValue:
                          'A single photo can still be checked for blur and eye state; duplicate grouping needs more photos.',
                      })
                    : t('modals.culling.folderScope', {
                        defaultValue:
                          'The run will use every supported photo in this folder, regardless of selection or filters.',
                      })}
              </Text>
            </div>

            <div className="mt-5 rounded-lg border border-border-color/40 p-4">
              <div className="flex items-baseline justify-between gap-4">
                <Text variant={TextVariants.heading}>{t('modals.culling.capabilities.title')}</Text>
                <Text color={TextColors.secondary} variant={TextVariants.small}>
                  {t('modals.culling.capabilities.privacy')}
                </Text>
              </div>
              {capabilities ? (
                <ul className="mt-2 space-y-1">
                  {capabilityRows.map((row) => (
                    <li key={row.key} className="flex items-center gap-2 text-sm" title={row.ready ? undefined : row.hint}>
                      {row.ready ? (
                        <CheckCircle size={16} className="shrink-0 text-accent" />
                      ) : (
                        <CircleDashed size={16} className="shrink-0 text-text-secondary" />
                      )}
                      <span className="flex-1">{t(`modals.culling.capabilities.${row.key}`)}</span>
                      <span className="text-text-secondary">
                        {row.ready
                          ? t('modals.culling.capabilities.ready')
                          : t('modals.culling.capabilities.unavailable')}
                      </span>
                    </li>
                  ))}
                  {capabilityRows
                    .filter((row) => !row.ready && row.hint)
                    .map((row) => (
                      <li key={`${row.key}-hint`}>
                        <Text color={TextColors.secondary} variant={TextVariants.small}>
                          {row.hint}
                        </Text>
                      </li>
                    ))}
                </ul>
              ) : (
                <Text color={TextColors.secondary} className="mt-2">
                  {t('modals.culling.capabilities.checking')}
                </Text>
              )}
            </div>

            <div className="mt-6 space-y-5">
              <div>
                <Text variant={TextVariants.heading} className="mb-2">
                  {t('modals.culling.amountSelected', { defaultValue: 'Amount of selected photos' })}
                </Text>
                <SegmentedChoice
                  options={amountOptions.map((option) => ({
                    ...option,
                    label: t(`modals.culling.amount.${option.value}`, { defaultValue: option.label }),
                  }))}
                  value={settings.selectionAmount}
                  onChange={(selectionAmount) => setSettings((current) => ({ ...current, selectionAmount }))}
                />
              </div>

              <div>
                <div className="flex items-center justify-between gap-4">
                  <Text variant={TextVariants.heading}>
                    {t('modals.culling.blurSeverity', { defaultValue: 'Blur detection severity' })}
                  </Text>
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t(`modals.culling.severity.${settings.blurSeverity}`, { defaultValue: settings.blurSeverity })}
                  </Text>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-1 rounded-lg bg-bg-primary p-1" role="group">
                  {severityOptions.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      disabled={!settings.detectBlurry}
                      className={`rounded-md px-2 py-2 text-sm transition-colors disabled:opacity-50 ${
                        settings.blurSeverity === option.value
                          ? 'bg-card-active text-text-primary shadow-sm'
                          : 'text-text-secondary hover:text-text-primary'
                      }`}
                      onClick={() => setSettings((current) => ({ ...current, blurSeverity: option.value }))}
                      aria-pressed={settings.blurSeverity === option.value}
                    >
                      {t(`modals.culling.severity.${option.value}`, { defaultValue: option.label })}
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-border-color/40">
                <button
                  type="button"
                  className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-bg-primary/50"
                  onClick={() => setIsCustomizeOpen((open) => !open)}
                  aria-expanded={isCustomizeOpen}
                >
                  <span>
                    <Text variant={TextVariants.heading}>
                      {t('modals.culling.customize', { defaultValue: 'Customize' })}
                    </Text>
                    <Text color={TextColors.secondary} variant={TextVariants.small}>
                      {t('modals.culling.customizeHint', {
                        defaultValue: 'Choose what the local analysis should detect.',
                      })}
                    </Text>
                  </span>
                  {isCustomizeOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </button>
                {isCustomizeOpen && (
                  <div className="space-y-4 border-t border-border-color/40 px-4 py-4">
                    <Switch
                      checked={settings.detectSubject}
                      disabled={capabilities?.subject === 'unavailable'}
                      label={t('modals.culling.detectSubject', { defaultValue: 'Local subject proposals' })}
                      onChange={(detectSubject) => setSettings((current) => ({ ...current, detectSubject }))}
                      tooltip={t('modals.culling.detectSubjectHint', {
                        defaultValue:
                          'Grounding DINO and Pose run locally. Proposals are review context only, do not change ratings or delete files, and unknown results stay unknown.',
                      })}
                    />
                    {settings.detectSubject && (
                      <label className="block">
                        <Text color={TextColors.secondary} variant={TextVariants.small} className="mb-1">
                          {t('modals.culling.subjectProfile', { defaultValue: 'Subject prompt' })}
                        </Text>
                        <select
                          className="w-full rounded-md border border-border-color bg-bg-primary px-3 py-2 text-text-primary"
                          value={settings.subjectProfile}
                          onChange={(event) =>
                            setSettings((current) => ({
                              ...current,
                              subjectProfile: event.target.value as CullingSettings['subjectProfile'],
                            }))
                          }
                        >
                          {subjectProfiles.map((profile) => (
                            <option key={profile.value} value={profile.value}>
                              {t(`modals.culling.profile${profile.value[0].toUpperCase()}${profile.value.slice(1)}`, {
                                defaultValue: profile.label,
                              })}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    <Switch
                      checked={settings.detectHighlights}
                      label={t('modals.culling.detectHighlights', { defaultValue: 'Technical highlights' })}
                      onChange={(detectHighlights) => setSettings((current) => ({ ...current, detectHighlights }))}
                      tooltip={t('modals.culling.detectHighlightsHint', {
                        defaultValue:
                          'Uses local sharpness, center detail and exposure signals; it is not an artistic AI score.',
                      })}
                    />
                    <Switch
                      checked={settings.detectDuplicates}
                      label={t('modals.culling.detectDuplicates', { defaultValue: 'Duplicate photos' })}
                      onChange={(detectDuplicates) => setSettings((current) => ({ ...current, detectDuplicates }))}
                    />
                    <Switch
                      checked={settings.detectBlurry}
                      label={t('modals.culling.detectBlurry', { defaultValue: 'Blurry photos (experimental)' })}
                      onChange={(detectBlurry) => setSettings((current) => ({ ...current, detectBlurry }))}
                      tooltip={t('modals.culling.detectBlurryHint')}
                    />
                    <Switch
                      checked={settings.detectClosedEyes}
                      disabled={capabilities?.faces === 'unavailable'}
                      label={t('modals.culling.detectClosedEyes', { defaultValue: 'Closed eyes (experimental)' })}
                      onChange={(detectClosedEyes) => setSettings((current) => ({ ...current, detectClosedEyes }))}
                      tooltip={t('modals.culling.detectClosedEyesHint')}
                    />
                    <Switch
                      checked={settings.autoAssignStars}
                      label={t('modals.culling.applyDecisions', {
                        defaultValue: 'Assign ratings and color labels',
                      })}
                      onChange={(autoAssignStars) => setSettings((current) => ({ ...current, autoAssignStars }))}
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="mt-7 flex justify-end gap-3">
              <button
                type="button"
                className="rounded-md px-4 py-2 text-text-secondary hover:bg-bg-primary"
                onClick={onClose}
              >
                {t('modals.culling.cancel')}
              </button>
              <Button disabled={imagePaths.length === 0} onClick={handleStartCulling}>
                {t('modals.culling.startCulling')}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
