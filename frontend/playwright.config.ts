import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120_000,
  expect: { timeout: 10_000 },
  reporter: 'list',
  retries: process.env.CI ? 2 : 0,
  webServer: {
    command: 'PORT=4000 pnpm dev',
    url: 'http://localhost:4000',
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
    timeout: 120_000,
  },
  use: {
    baseURL: 'http://localhost:4000',
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
})
