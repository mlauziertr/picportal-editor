import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronUp, Loader2, Sparkles, XCircle } from 'lucide-react';
import { CullingSettings, CullingSuggestions, Invokes, Progress } from '../ui/AppProperties';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

interface CullingModalProps {
  isOpen: boolean;
  onClose(): void;
  progress: Progress | null;
  suggestions: CullingSuggestions | null;
  error: string | null;
  imagePaths: string[];
  folderPath: string | null;
  onComplete(suggestions: CullingSuggestions, settings: CullingSettings): Promise<void>;
  onError(error: string): void;
}

const DEFAULT_SETTINGS: CullingSettings = {
  selectionAmount: 'standard',
  blurSeverity: 'moderate',
  detectDuplicates: true,
  detectBlurry: true,
  detectClosedEyes: true,
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
  suggestions,
  error,
  imagePaths,
  folderPath,
  onComplete,
  onError,
}: CullingModalProps) {
  const { t } = useTranslation();
  const [settings, setSettings] = useState<CullingSettings>(DEFAULT_SETTINGS);
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(false);
  const [isCompleting, setIsCompleting] = useState(false);
  const completionRef = useRef<CullingSuggestions | null>(null);

  useEffect(() => {
    if (!isOpen) {
      completionRef.current = null;
      setIsCompleting(false);
      setIsCustomizeOpen(false);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !suggestions || completionRef.current === suggestions) return;
    completionRef.current = suggestions;
    setIsCompleting(true);
    onComplete(suggestions, settings).catch((completionError) => {
      setIsCompleting(false);
      onError(String(completionError));
    });
  }, [isOpen, suggestions, settings, onComplete, onError]);

  const handleStartCulling = useCallback(async () => {
    if (imagePaths.length === 0) return;
    try {
      await invoke(Invokes.CullImages, { paths: imagePaths, settings });
    } catch (startError) {
      onError(String(startError));
    }
  }, [imagePaths, settings, onError]);

  if (!isOpen) return null;

  const isRunning = Boolean(progress) || isCompleting;
  const folderLabel = folderPath || t('modals.culling.noFolder', { defaultValue: 'No folder open' });

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-surface p-6 shadow-2xl">
        {isRunning ? (
          <div className="flex min-h-64 flex-col items-center justify-center text-center">
            <Loader2 className="h-14 w-14 animate-spin text-accent" />
            <Text variant={TextVariants.heading} className="mt-5">
              {isCompleting
                ? t('modals.culling.savingResults', { defaultValue: 'Saving culling results…' })
                : progress?.stage || t('modals.culling.starting')}
            </Text>
            <Text color={TextColors.secondary} className="mt-2">
              {t('modals.culling.analyzingCount', {
                count: imagePaths.length,
                defaultValue: 'Analyzing {{count}} photos locally',
              })}
            </Text>
            {progress && progress.total > 0 && (
              <div className="mt-5 w-full max-w-md rounded-full bg-bg-primary">
                <div
                  className="h-2 rounded-full bg-accent transition-all"
                  style={{ width: `${((progress.current || 0) / progress.total) * 100}%` }}
                />
              </div>
            )}
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
            <Button className="mt-6" onClick={onClose}>
              {t('modals.culling.close')}
            </Button>
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
                aria-label={t('modals.culling.cancel')}
              >
                ×
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
                      className={`rounded-md px-2 py-2 text-sm transition-colors ${
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
                      label={t('modals.culling.detectBlurry', { defaultValue: 'Blurry photos' })}
                      onChange={(detectBlurry) => setSettings((current) => ({ ...current, detectBlurry }))}
                    />
                    <Switch
                      checked={settings.detectClosedEyes}
                      label={t('modals.culling.detectClosedEyes', { defaultValue: 'Closed eyes (local heuristic)' })}
                      onChange={(detectClosedEyes) => setSettings((current) => ({ ...current, detectClosedEyes }))}
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
