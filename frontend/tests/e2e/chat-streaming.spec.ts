import { expect, test, type Page } from '@playwright/test'

const PROMPT = 'how can I deploy grafana?'

async function selectCollection(page: Page) {
  const select = page.getByRole('combobox', { name: 'Collection' })
  await expect(select).toBeVisible()

  const options = select.locator('option')
  await expect(options).not.toHaveCount(0)

  const selectedValue = await options.first().getAttribute('value')
  const selectedLabel = (await options.first().textContent())?.trim()

  expect(selectedValue ?? selectedLabel).toBeTruthy()

  if (selectedValue) {
    await select.selectOption({ value: selectedValue })
    return selectedValue
  }

  await select.selectOption({ label: selectedLabel! })
  return selectedLabel!
}

async function askQuestion(page: Page, prompt: string) {
  const input = page.getByRole('textbox', { name: 'Message' })
  const send = page.getByRole('button', { name: 'Ask' })

  await expect(input).toBeVisible()
  await expect(send).toHaveAccessibleName('Ask')
  await input.fill(prompt)
  await expect(send).toBeEnabled()

  const chatResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes('/api/langgraph/threads/') &&
      response.url().endsWith('/runs/stream') &&
      response.request().method() === 'POST',
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
    await selectCollection(page)

    const { input, chatResponsePromise } = await askQuestion(page, PROMPT)

    await expect(input).toBeDisabled({ timeout: 5_000 })
    await expect(page.getByText('Generating a grounded response...')).toBeVisible({ timeout: 5_000 })

    const chatResponse = await chatResponsePromise
    const chatHeaders = chatResponse.headers()
    expect(chatHeaders['content-type']).toContain('text/event-stream')

    await expect(input).toBeEnabled({ timeout: 120_000 })

    await expectAssistantAnswer(page)
    await expect(page.getByText(/Sources:\s*\S+/)).toBeVisible()
  })

  test('clear chat resets the visible conversation', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page)

    const { input } = await askQuestion(page, PROMPT)

    await expect(input).toBeEnabled({ timeout: 120_000 })
    await expectAssistantAnswer(page)

    await page.getByRole('button', { name: 'Clear Chat History' }).click()

    await expect(page.getByText('Ask a question about your documents')).toBeVisible()
    await expect(page.getByText('Sources:')).toHaveCount(0)
  })
})
