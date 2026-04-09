import { expect, test, type Page } from '@playwright/test'

const PROMPT = 'how can I deploy grafana?'
const COLLECTION = 'ORACLE_WEB_EMBEDDINGS'

async function selectCollection(page: Page, value: string) {
  const select = page.getByRole('combobox', { name: 'Collection' })
  await expect(select).toBeVisible()

  const option = select.locator(`option[value="${value}"]`)
  if (!(await option.count())) {
    const optionByText = select.locator('option', { hasText: value })
    if (!(await optionByText.count())) {
      throw new Error(
        `Required collection option not found: ${value}. Ensure backend config exposes it in appConfig.collection_list.`,
      )
    }
    await select.selectOption({ label: value })
    return
  }

  await select.selectOption({ value })
}

async function askQuestion(page: Page, prompt: string) {
  const input = page.getByRole('textbox', { name: 'Message' })
  const send = page.getByRole('button', { name: 'Ask' })

  await expect(input).toBeVisible()
  await input.fill(prompt)
  await expect(send).toBeEnabled()

  const chatResponsePromise = page.waitForResponse(
    (response) =>
      response.url().endsWith('/api/chat') && response.request().method() === 'POST',
  )

  await send.click()

  return { input, chatResponsePromise }
}

async function expectAssistantAnswer(page: Page) {
  const sourcesLabel = page.getByText('Sources:').last()
  await expect(sourcesLabel).toBeVisible({ timeout: 15_000 })

  const contentBlock = sourcesLabel.locator('..').locator('..').first()
  await expect
    .poll(async () => (await contentBlock.innerText()).trim().length, { timeout: 10_000 })
    .toBeGreaterThan(0)
}

test.describe('chat streaming', () => {
  test('streams responses and renders citations', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page, COLLECTION)

    const { input, chatResponsePromise } = await askQuestion(page, PROMPT)

    await expect(input).toBeDisabled({ timeout: 5_000 })
    await expect(page.getByText('Generating a grounded response...')).toBeVisible({ timeout: 5_000 })

    const chatResponse = await chatResponsePromise
    const chatHeaders = chatResponse.headers()
    expect(chatHeaders['x-vercel-ai-ui-message-stream']).toBe('v1')
    expect(chatHeaders['content-type']).toContain('text/event-stream')

    await expect(input).toBeEnabled({ timeout: 120_000 })

    await expectAssistantAnswer(page)
    await expect(page.getByText(/Sources:\s*\S+/)).toBeVisible()
  })

  test('clear chat resets the visible conversation', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page, COLLECTION)

    const { input } = await askQuestion(page, PROMPT)

    await expect(input).toBeEnabled({ timeout: 120_000 })
    await expectAssistantAnswer(page)

    await page.getByRole('button', { name: 'Clear Chat History' }).click()

    await expect(page.getByText('Ask a question about your documents')).toBeVisible()
    await expect(page.getByText('Sources:')).toHaveCount(0)
  })
})
