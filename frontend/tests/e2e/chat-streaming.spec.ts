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
  test('shows generic suggestions on first load', async ({ page }) => {
    await page.goto('/')

    const suggestions = page.getByRole('navigation', { name: 'Suggested questions' })
    await expect(suggestions).toBeVisible()
    await expect(
      suggestions.getByRole('button', { name: 'Tell me about Oracle 26ai Database.' }),
    ).toBeVisible()
    await expect(
      suggestions.getByRole('button', { name: 'Solve this math problem: 125 * 48.' }),
    ).toBeVisible()
    await expect(
      suggestions.getByRole('button', { name: 'What can you help me find in my documents?' }),
    ).toBeVisible()
    await expect(suggestions.getByRole('button', { name: /resume/i })).toHaveCount(0)
  })

  test('shows locally known chat history and switches active threads', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('rag_agent_thread_id', 'thread-history-1')
      window.localStorage.setItem(
        'rag_agent_chat_threads',
        JSON.stringify([
          {
            id: 'thread-history-1',
            title: 'Latest invoice workflow',
            createdAt: 2,
            updatedAt: 2,
          },
          {
            id: 'thread-history-2',
            title: 'Vendor payment terms',
            createdAt: 1,
            updatedAt: 1,
          },
        ]),
      )
    })
    await page.route('**/api/langgraph/**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })

    await page.goto('/')

    const history = page.getByLabel('Chat history')
    await expect(history.getByRole('button', { name: 'Latest invoice workflow' })).toHaveAttribute(
      'aria-current',
      'page',
    )
    await history.getByRole('button', { name: 'Vendor payment terms' }).click()

    await expect(page.getByTestId('chat-root')).toHaveAttribute('data-thread-id', 'thread-history-2')
    await expect(history.getByRole('button', { name: 'Vendor payment terms' })).toHaveAttribute(
      'aria-current',
      'page',
    )
  })

  test('keeps chat history title updates stable after asking', async ({ page }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (error) => {
      pageErrors.push(error.message)
    })
    await page.addInitScript(() => {
      window.localStorage.setItem('rag_agent_thread_id', 'thread-title-loop')
      window.localStorage.setItem(
        'rag_agent_chat_threads',
        JSON.stringify([
          {
            id: 'thread-title-loop',
            title: 'Existing title',
            createdAt: 1,
            updatedAt: 1,
          },
        ]),
      )
    })
    await page.route('**/api/langgraph/**/history', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })
    await page.route('**/api/langgraph/**/runs/stream', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          'event: values',
          'data: {"messages":[{"type":"human","content":"Will title updates loop?"}]}',
          '',
          'event: values',
          'data: {"messages":[{"type":"human","content":"Will title updates loop?"},{"type":"ai","content":"No."}]}',
          '',
        ].join('\n'),
      })
    })

    await page.goto('/')
    await page.getByRole('textbox', { name: 'Message' }).fill('Will title updates loop?')
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByText('No.')).toBeVisible()
    await expect(page.getByText('This page couldn’t load')).toHaveCount(0)
    expect(pageErrors.filter((message) => message.includes('Maximum update depth'))).toEqual([])
  })

  test('expands the chat input for long multi-line prompts', async ({ page }) => {
    await page.goto('/')

    const input = page.getByRole('textbox', { name: 'Message' })
    await expect(input).toBeVisible()
    await expect(input).toHaveJSProperty('tagName', 'TEXTAREA')

    const compactHeight = await input.evaluate((node) => node.getBoundingClientRect().height)
    const longPrompt = Array.from(
      { length: 24 },
      (_, index) => `"InvoiceLine${index + 1}": { "Description": "Product ${index + 1}" }`,
    ).join('\n')

    await input.fill(longPrompt)

    const expandedMetrics = await input.evaluate((node) => ({
      clientHeight: node.clientHeight,
      height: node.getBoundingClientRect().height,
      scrollHeight: node.scrollHeight,
    }))

    expect(expandedMetrics.height).toBeGreaterThan(compactHeight)
    expect(expandedMetrics.height).toBeLessThanOrEqual(240)
    expect(expandedMetrics.scrollHeight).toBeGreaterThan(expandedMetrics.clientHeight)
  })

  test('keeps API connection failures out of the browser error overlay', async ({ page }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (error) => {
      pageErrors.push(error.message)
    })

    await page.goto('/')
    await page.route('**/api/langgraph/**/runs/stream', (route) => route.abort())

    await page.getByRole('textbox', { name: 'Message' }).fill('Will this fail gracefully?')
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByTestId('chat-root')).toHaveAttribute('data-chat-status', 'error', {
      timeout: 15_000,
    })
    expect(pageErrors).toEqual([])
  })

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
