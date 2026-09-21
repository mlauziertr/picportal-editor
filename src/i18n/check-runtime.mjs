import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import i18next from 'i18next';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const localeDir = path.resolve(scriptDir, 'locales');
const attributionOnly = process.argv.includes('--attribution-only');
const pluralSuffix = /_(zero|one|two|few|many|other)$/;
const countCandidates = [
  ...Array.from({ length: 201 }, (_, count) => count),
  0.1,
  1.1,
  2.1,
  5.1,
  10.1,
  1_000,
  1_000_000,
];
const attributionKey = 'settings.thanks.attribution';

const flatten = (object, prefix = '', leaves = new Map()) => {
  for (const [key, value] of Object.entries(object)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      flatten(value, fullKey, leaves);
    } else {
      leaves.set(fullKey, value);
    }
  }
  return leaves;
};

const localeFiles = fs
  .readdirSync(localeDir)
  .filter((filename) => filename.endsWith('.json'))
  .sort();
const resources = {};
const pluralKeysByLocale = new Map();
const failures = [];

for (const filename of localeFiles) {
  const locale = path.basename(filename, '.json');
  const translations = JSON.parse(fs.readFileSync(path.join(localeDir, filename), 'utf8'));
  const leaves = flatten(translations);
  const pluralKeys = new Set();

  for (const [key, value] of leaves) {
    if (!attributionOnly && value === '') {
      failures.push(`${locale}:${key} is empty`);
    }
    if (!attributionOnly && pluralSuffix.test(key)) {
      pluralKeys.add(key.replace(pluralSuffix, ''));
    }
  }

  resources[locale] = { translation: translations };
  pluralKeysByLocale.set(locale, pluralKeys);
}

const i18n = i18next.createInstance();
await i18n.init({
  resources,
  lng: 'en',
  fallbackLng: 'en',
  returnEmptyString: false,
  interpolation: {
    escapeValue: false,
  },
});

let checkedResolutions = 0;
let checkedAttributions = 0;

for (const filename of localeFiles) {
  const locale = path.basename(filename, '.json');
  if (!attributionOnly) {
    const pluralRules = new Intl.PluralRules(locale);
    const sampleByCategory = new Map();

    for (const count of countCandidates) {
      const category = pluralRules.select(count);
      if (!sampleByCategory.has(category)) {
        sampleByCategory.set(category, count);
      }
    }

    for (const category of pluralRules.resolvedOptions().pluralCategories) {
      if (!sampleByCategory.has(category)) {
        failures.push(`${locale}: no test count found for plural category ${category}`);
      }
    }

    for (const key of pluralKeysByLocale.get(locale)) {
      for (const [category, count] of sampleByCategory) {
        const details = i18n.t(key, { lng: locale, count, returnDetails: true });
        const expectedKey = `${key}_${category}`;
        checkedResolutions += 1;

        if (details.usedLng !== locale) {
          failures.push(`${locale}:${expectedKey} resolved through ${details.usedLng}`);
        }
        if (details.exactUsedKey !== expectedKey) {
          failures.push(`${locale}:${expectedKey} resolved as ${details.exactUsedKey}`);
        }
        if (typeof details.res !== 'string' || details.res.trim() === '') {
          failures.push(`${locale}:${expectedKey} resolved to an empty value`);
        }
      }
    }
  }

  if (locale !== 'en') {
    for (const key of Object.keys(resources.en.translation.settings.thanks.attribution)) {
      const fullKey = `${attributionKey}.${key}`;
      const details = i18n.t(fullKey, { lng: locale, returnDetails: true });
      checkedAttributions += 1;

      if (details.usedLng !== locale) {
        failures.push(`${locale}:${fullKey} resolved through ${details.usedLng}`);
      }
      if (details.exactUsedKey !== fullKey) {
        failures.push(`${locale}:${fullKey} resolved as ${details.exactUsedKey}`);
      }
      if (typeof details.res !== 'string' || details.res.trim() === '') {
        failures.push(`${locale}:${fullKey} resolved to an empty value`);
      }
    }
  }
}

if (failures.length > 0) {
  console.error(`i18n runtime validation failed with ${failures.length} issue(s):`);
  failures.forEach((failure) => console.error(`- ${failure}`));
  process.exitCode = 1;
} else {
  console.log(
    `Validated ${checkedResolutions} plural resolutions and ${checkedAttributions} attribution resolutions across ${localeFiles.length} locales.`,
  );
}
