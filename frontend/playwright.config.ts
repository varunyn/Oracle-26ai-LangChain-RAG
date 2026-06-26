import { defineConfig } from '@playwright/test'

const port = process.env.PLAYWRIGHT_PORT ?? '4040'
const skipWebServer = process.env.PLAYWRIGHT_SKIP_WEBSERVER === '1'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120_000,
  expect: { timeout: 10_000 },
  reporter: 'list',
  retries: process.env.CI ? 2 : 0,
  webServer: skipWebServer
    ? undefined
    : {
        command: `PORT=${port} pnpm dev`,
        url: `http://localhost:${port}`,
        reuseExistingServer: !process.env.CI,
        stdout: 'pipe',
        stderr: 'pipe',
        timeout: 120_000,
      },
  use: {
    baseURL: `http://localhost:${port}`,
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
})
