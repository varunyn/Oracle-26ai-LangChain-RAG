import { expect, test, type Page } from '@playwright/test'

const PROMPT = 'how can I deploy grafana?'
const DUPLICATE_PROMPT = 'Tell me about the payment terms for summit technologies.'
const LIVE_COLLECTION = 'ORACLE_WEB_EMBEDDINGS'

async function selectCollection(page: Page, collectionName: string) {
  const select = page.getByRole('combobox', { name: 'Collection' })
  await expect(select).toBeVisible()
  await select.selectOption({ label: collectionName })
  await expect(select).toHaveValue(collectionName)
}

async function selectFlowMode(page: Page, label: string) {
  const select = page.getByRole('combobox', { name: 'Flow mode' })
  await expect(select).toBeVisible()
  await select.selectOption({ label })
}

async function askQuestion(page: Page, prompt: string) {
  const input = page.getByRole('textbox', { name: 'Message' })
  const send = page.getByRole('button', { name: 'Ask' })

  await expect(input).toBeVisible()
  await expect(send).toHaveAccessibleName('Ask')
  await input.fill(prompt)
  await expect(send).toBeEnabled()

  const commandResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes('/langgraph/threads/') &&
      response.url().includes('/runs/stream') &&
      response.request().method() === 'POST',
  )
  const streamResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes('/langgraph/threads/') &&
      response.url().endsWith('/stream') &&
      response.request().method() === 'POST',
  )

  await send.click()

  return { input, commandResponsePromise, streamResponsePromise }
}

async function expectAssistantAnswer(page: Page) {
  const sourcesLabel = page.getByRole('button', { name: /Used \d+ sources?/ }).last()
  await expect(sourcesLabel).toBeVisible({ timeout: 30_000 })

  const contentBlock = sourcesLabel.locator('..').locator('..').first()
  await expect
    .poll(async () => (await contentBlock.innerText()).trim().length, { timeout: 30_000 })
    .toBeGreaterThan(0)
}

async function expectSubmittedQuestionBeforeAssistant(page: Page, prompt: string) {
  const messageList = page.getByTestId('chat-message-list')
  const submittedQuestion = messageList.getByText(prompt, { exact: true })
  const sourcesLabel = page.getByRole('button', { name: /Used \d+ sources?/ }).last()

  await expect(submittedQuestion).toHaveCount(1)
  await expect(sourcesLabel).toBeVisible()
  await expect
    .poll(
      async () => {
        const questionBox = await submittedQuestion.boundingBox()
        const sourcesBox = await sourcesLabel.boundingBox()
        if (!questionBox || !sourcesBox) {
          return false
        }
        return questionBox.y < sourcesBox.y
      },
      { timeout: 5_000 },
    )
    .toBe(true)
}

test.describe('chat live backend', () => {
  test('streams responses and renders citations in RAG mode', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page, LIVE_COLLECTION)
    await selectFlowMode(page, 'RAG only')

    const { input, commandResponsePromise, streamResponsePromise } = await askQuestion(page, PROMPT)

    await expect(input).toBeDisabled({ timeout: 5_000 })
    await expect(page.getByTestId('chat-streaming-indicator')).toBeVisible({ timeout: 5_000 })
    await expect(page.getByText(/Opening answer stream|Preparing response/)).toBeVisible({
      timeout: 5_000,
    })
    await expect(page.getByTestId('chat-message-list').getByText(PROMPT)).toBeVisible()
    await expect(page.getByText('Ask a question about your documents')).toHaveCount(0)

    const commandResponse = await commandResponsePromise
    expect(commandResponse.headers()['content-type']).toContain('application/json')

    const streamResponse = await streamResponsePromise
    expect(streamResponse.headers()['content-type']).toContain('text/event-stream')

    await expect(input).toBeEnabled({ timeout: 120_000 })
    await expectAssistantAnswer(page)
    await expect(page.getByRole('button', { name: /Used \d+ sources?/ })).toBeVisible()
  })

  test('does not render the submitted user question twice in RAG mode', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page, LIVE_COLLECTION)
    await selectFlowMode(page, 'RAG only')

    const { input } = await askQuestion(page, DUPLICATE_PROMPT)

    await expect(input).toBeEnabled({ timeout: 120_000 })
    await expectAssistantAnswer(page)
    await expectSubmittedQuestionBeforeAssistant(page, DUPLICATE_PROMPT)
  })

  test('clear chat resets the visible conversation in RAG mode', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page, LIVE_COLLECTION)
    await selectFlowMode(page, 'RAG only')

    const { input } = await askQuestion(page, PROMPT)

    await expect(input).toBeEnabled({ timeout: 120_000 })
    await expectAssistantAnswer(page)

    await page.getByRole('button', { name: 'Clear Chat History' }).click()

    await expect(page.getByText('Ask a question about your documents')).toBeVisible()
    await expect(page.getByRole('button', { name: /Used \d+ sources?/ })).toHaveCount(0)
  })
})
