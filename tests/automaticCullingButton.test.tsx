import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { createInstance } from 'i18next';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import MainLibrary from '../src/components/panel/MainLibrary.tsx';
import AutomaticCullingButton, {
  createAutomaticCullingModalState,
  openAutomaticCullingOptions,
} from '../src/components/panel/library/AutomaticCullingButton.tsx';
import {
  LibraryDisplayMode,
  LibraryViewMode,
  Theme,
  ThumbnailAspectRatio,
  ThumbnailSize,
} from '../src/components/ui/AppProperties.tsx';
import { Status } from '../src/components/ui/ExportImportProperties.tsx';

test('automatic culling toolbar action opens options without starting analysis', () => {
  const state = createAutomaticCullingModalState(['one.jpg', 'two.jpg']);

  assert.deepEqual(state, {
    isOpen: true,
    progress: null,
    suggestions: null,
    error: null,
    pathsToCull: ['one.jpg', 'two.jpg'],
  });

  let openedState: ReturnType<typeof createAutomaticCullingModalState> | null = null;
  openAutomaticCullingOptions(['one.jpg', 'two.jpg'], (nextState) => {
    openedState = nextState;
  });
  assert.deepEqual(openedState, state);
});

test('automatic culling toolbar action does nothing until two photos are selected', () => {
  let openCount = 0;
  openAutomaticCullingOptions(['one.jpg'], () => {
    openCount += 1;
  });

  assert.equal(openCount, 0);
});

test('automatic culling toolbar action is visible and explains an unavailable selection', () => {
  const markup = renderToStaticMarkup(
    <AutomaticCullingButton
      label="Automatic culling"
      unavailableLabel="Select at least two photos to use automatic culling"
      selectedPaths={['one.jpg']}
      onOpen={() => undefined}
    />,
  );

  assert.match(markup, />Automatic culling</);
  assert.match(markup, /disabled=""/);
  assert.match(markup, /aria-label="Select at least two photos to use automatic culling"/);
  assert.match(markup, /data-tooltip="Select at least two photos to use automatic culling"/);
});

test('automatic culling toolbar action is enabled for the current multi-selection', () => {
  const markup = renderToStaticMarkup(
    <AutomaticCullingButton
      label="Automatic culling"
      unavailableLabel="Select at least two photos to use automatic culling"
      selectedPaths={['one.jpg', 'two.jpg']}
      onOpen={() => undefined}
    />,
  );

  assert.doesNotMatch(markup, /disabled=""/);
  assert.match(markup, /aria-label="Automatic culling"/);
});

test('automatic culling toolbar labels resolve locally in every locale', async () => {
  const localesDirectory = join(process.cwd(), 'src/i18n/locales');
  const localeFiles = readdirSync(localesDirectory).filter((file) => file.endsWith('.json'));

  for (const localeFile of localeFiles) {
    const locale = localeFile.replace(/\.json$/, '');
    const resource = JSON.parse(readFileSync(join(localesDirectory, localeFile), 'utf8'));
    const cullingKeys = Object.keys(resource.library.culling);
    assert.deepEqual(cullingKeys, [...cullingKeys].sort(), `${locale} culling keys must remain sorted`);

    const i18n = createInstance();
    await i18n.init({
      lng: locale,
      fallbackLng: false,
      resources: { [locale]: { translation: resource } },
    });

    for (const key of ['library.culling.automaticCulling', 'library.culling.automaticCullingUnavailable']) {
      const translated = i18n.t(key);
      assert.notEqual(translated, key, `${locale} must translate ${key}`);
      assert.ok(translated.trim().length > 0, `${locale} must provide a non-empty ${key}`);
    }
  }
});

test('library top toolbar renders the localized automatic culling action for the current selection', async () => {
  const resource = JSON.parse(readFileSync(join(process.cwd(), 'src/i18n/locales/en.json'), 'utf8'));
  const i18n = createInstance();
  await i18n.init({ lng: 'en', resources: { en: { translation: resource } } });

  const markup = renderToStaticMarkup(
    <I18nextProvider i18n={i18n}>
      <MainLibrary
        activePath="one.jpg"
        aiModelDownloadStatus={null}
        appSettings={{ lastRootPath: null, theme: Theme.Dark, libraryDisplayMode: LibraryDisplayMode.Grid }}
        currentFolderPath="/synthetic-fixtures"
        groupBadgeInfo={null}
        imageList={[]}
        imageRatings={{}}
        importState={{ errorMessage: '', status: Status.Idle }}
        indexingProgress={{ current: 0, total: 0 }}
        isLoading={false}
        isIndexing={false}
        isAndroid={false}
        isTreeLoading={false}
        libraryViewMode={LibraryViewMode.Flat}
        multiSelectedPaths={['one.jpg', 'two.jpg']}
        onClearSelection={() => undefined}
        onContextMenu={() => undefined}
        onContinueSession={() => undefined}
        onEmptyAreaContextMenu={() => undefined}
        onGoHome={() => undefined}
        onImageClick={() => undefined}
        onImageDoubleClick={() => undefined}
        onImportClick={() => undefined}
        onLibraryRefresh={() => undefined}
        onOpenFolder={() => undefined}
        onSettingsChange={async () => undefined}
        onThumbnailAspectRatioChange={() => undefined}
        onThumbnailSizeChange={() => undefined}
        rootPaths={['/synthetic-fixtures']}
        setLibraryViewMode={() => undefined}
        theme={Theme.Dark}
        thumbnailAspectRatio={ThumbnailAspectRatio.Contain}
        thumbnailProgress={{ current: 0, total: 0 }}
        thumbnailSize={ThumbnailSize.Medium}
        onNavigateToCommunity={() => undefined}
      />
    </I18nextProvider>,
  );

  assert.match(markup, />Automatic culling</);
  assert.match(markup, /aria-label="Automatic culling"/);
});
