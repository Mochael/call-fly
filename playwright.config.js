import {defineConfig} from '@playwright/test';
import path from 'node:path';

export default defineConfig({
  testDir: './tests/browser', workers: 1, timeout: 120_000,
  use: {
    baseURL: process.env.TEST_BASE_URL || 'http://localhost:8765',
    launchOptions: {
      executablePath: process.env.TEST_CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
             `--use-file-for-fake-audio-capture=${path.resolve('artifacts/browser-microphone.wav')}`],
    },
    permissions: ['microphone'],
    trace: 'retain-on-failure',
  },
});
