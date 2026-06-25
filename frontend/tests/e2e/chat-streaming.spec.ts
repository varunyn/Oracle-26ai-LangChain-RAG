import { expect, test, type Page } from '@playwright/test'

const PROMPT = 'how can I deploy grafana?'

type ProtocolMockEvent = {
  method: 'values' | 'tools' | 'lifecycle'
  data: unknown
}

function protocolEvent({ method, data }: ProtocolMockEvent, index: number) {
  return {
    type: 'event',
    seq: index + 1,
    event_id: `mock-event-${index + 1}`,
    method,
    params: { namespace: [], data },
  }
}

function protocolSse(events: ProtocolMockEvent[]) {
  return events
    .map((event, index) => `event: event\ndata: ${JSON.stringify(protocolEvent(event, index))}\n`)
    .join('\n')
}

async function mockLangGraphProtocol(page: Page, events: ProtocolMockEvent[]) {
  await page.route('**/api/langgraph/**/commands', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 1,
        type: 'success',
        result: { run_id: 'mock-run', created_at: null },
      }),
    })
  })
  await page.route('**/api/langgraph/**/stream/events', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: protocolSse(events),
    })
  })
}

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

  const chatResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes('/api/langgraph/threads/') &&
      response.url().endsWith('/commands') &&
      response.request().method() === 'POST',
  )

  await send.click()

  return { input, chatResponsePromise }
}

async function expectAssistantAnswer(page: Page) {
  const sourcesLabel = page.getByRole('button', { name: /Used \d+ sources?/ }).last()
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

  test('does not render clicked suggestions as duplicate user messages', async ({ page }) => {
    const suggestion = 'Tell me about Oracle 26ai Database.'
    await page.route('**/api/langgraph/**/history', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })
    await page.route('**/api/suggestions', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ suggestions: [] }),
      })
    })
    await mockLangGraphProtocol(page, [
      {
        method: 'values',
        data: {
          messages: [
            { type: 'human', content: suggestion },
            { type: 'ai', content: 'Oracle 26ai deployment details.' },
          ],
        },
      },
      { method: 'lifecycle', data: { event: 'completed' } },
    ])

    await page.goto('/')
    await page
      .getByRole('navigation', { name: 'Suggested questions' })
      .getByRole('button', { name: suggestion })
      .click()

    const messageList = page.getByTestId('chat-message-list')
    await expect(messageList.getByText('Oracle 26ai deployment details.')).toBeVisible()
    await expect(messageList.getByText(suggestion, { exact: true })).toHaveCount(1)
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

  test('keeps long chat history scrollable in the sidebar', async ({ page }) => {
    const threads = Array.from({ length: 30 }, (_, index) => {
      const number = index + 1
      return {
        id: `thread-long-${number}`,
        title: `Long history chat ${number}`,
        createdAt: number,
        updatedAt: number,
      }
    }).reverse()

    await page.setViewportSize({ width: 1280, height: 720 })
    await page.addInitScript((seedThreads) => {
      window.localStorage.setItem('rag_agent_thread_id', 'thread-long-30')
      window.localStorage.setItem('rag_agent_chat_threads', JSON.stringify(seedThreads))
    }, threads)
    await page.route('**/api/langgraph/**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })

    await page.goto('/')

    const history = page.getByTestId('chat-history-list')
    await expect(
      history.getByRole('button', { name: 'Long history chat 30', exact: true }),
    ).toBeVisible()
    const metrics = await history.evaluate((node) => ({
      clientHeight: node.clientHeight,
      scrollHeight: node.scrollHeight,
    }))
    expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight)
    const initialLastThreadPosition = await history.evaluate((historyNode) => {
      const button = Array.from(historyNode.querySelectorAll('button')).find(
        (item) => item.textContent?.trim() === 'Long history chat 1',
      )
      if (!button) throw new Error('Expected final history item')
      const buttonRect = button.getBoundingClientRect()
      const historyRect = historyNode.getBoundingClientRect()
        return {
          buttonTop: buttonRect.top,
          historyBottom: historyRect.bottom,
        }
    })
    expect(initialLastThreadPosition.buttonTop).toBeGreaterThan(
      initialLastThreadPosition.historyBottom,
    )

    await history.evaluate((node) => {
      node.scrollTop = node.scrollHeight
    })
    const scrolledLastThreadPosition = await history.evaluate((historyNode) => {
      const button = Array.from(historyNode.querySelectorAll('button')).find(
        (item) => item.textContent?.trim() === 'Long history chat 1',
      )
      if (!button) throw new Error('Expected final history item')
      const buttonRect = button.getBoundingClientRect()
      const historyRect = historyNode.getBoundingClientRect()
        return {
          buttonBottom: buttonRect.bottom,
          historyBottom: historyRect.bottom,
        }
    })
    expect(scrolledLastThreadPosition.buttonBottom).toBeLessThanOrEqual(
      scrolledLastThreadPosition.historyBottom,
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
    await mockLangGraphProtocol(page, [
      {
        method: 'values',
        data: { messages: [{ type: 'human', content: 'Will title updates loop?' }] },
      },
      {
        method: 'values',
        data: {
          messages: [
            { type: 'human', content: 'Will title updates loop?' },
            { type: 'ai', content: 'No.' },
          ],
        },
      },
      { method: 'lifecycle', data: { event: 'completed' } },
    ])

    await page.goto('/')
    await page.getByRole('textbox', { name: 'Message' }).fill('Will title updates loop?')
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByText('No.')).toBeVisible()
    await expect(page.getByText('This page couldn’t load')).toHaveCount(0)
    expect(pageErrors.filter((message) => message.includes('Maximum update depth'))).toEqual([])
  })

  test('keeps the submitted question visible while the stream is starting', async ({ page }) => {
    let releaseStream: (() => void) | undefined
    const streamStarted = new Promise<void>((resolve) => {
      releaseStream = resolve
    })

    await page.route('**/api/langgraph/**/history', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })
    await page.route('**/api/langgraph/**/commands', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 1,
          type: 'success',
          result: { run_id: 'mock-run', created_at: null },
        }),
      })
    })
    await page.route('**/api/langgraph/**/stream/events', async (route) => {
      await streamStarted
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: protocolSse([
          {
            method: 'values',
            data: {
              messages: [
                { type: 'human', content: PROMPT },
                { type: 'ai', content: 'Use Grafana deployment docs.' },
              ],
            },
          },
          { method: 'lifecycle', data: { event: 'completed' } },
        ]),
      })
    })

    await page.goto('/')
    await page.getByRole('textbox', { name: 'Message' }).fill(PROMPT)
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByTestId('chat-message-list').getByText(PROMPT)).toBeVisible()
    await expect(page.getByTestId('chat-streaming-indicator')).toBeVisible()
    await expect(page.getByText('Ask a question about your documents')).toHaveCount(0)

    releaseStream?.()
    await expect(page.getByText('Use Grafana deployment docs.')).toBeVisible()
  })

  test('renders MCP tool progress from stream values metadata', async ({ page }) => {
    const prompt =
      'Perform a linear regression on these points: (1,2), (2,3.5), (3,5.1), (4,6.5), (5,8.2) using tools'
    const progressPayload = {
      messages: [
        { type: 'human', content: prompt },
        {
          type: 'ai',
          content: '',
          additional_kwargs: {
            mcp_progress_events: [
              {
                phase: 'start',
                tool_name: 'Calculator_linear_regression',
                tool_run_id: 'tool-1',
                args: { data: [[1, 2], [2, 3.5], [3, 5.1], [4, 6.5], [5, 8.2]] },
              },
              {
                phase: 'end',
                tool_name: 'Calculator_linear_regression',
                tool_run_id: 'tool-1',
                result: '{"slope":1.54,"intercept":0.44}',
              },
            ],
          },
        },
      ],
    }
    const finalPayload = {
      messages: [
        { type: 'human', content: prompt },
        {
          type: 'ai',
          content: 'The best-fit line is y = 1.54x + 0.44.',
          additional_kwargs: {
            mcp_used: true,
            mcp_tools_used: ['Calculator_linear_regression'],
            mcp_tool_invocations: [
              {
                tool_name: 'Calculator_linear_regression',
                args: { data: [[1, 2], [2, 3.5], [3, 5.1], [4, 6.5], [5, 8.2]] },
                result: '{"slope":1.54,"intercept":0.44}',
              },
            ],
            citations: [],
            reranker_docs: [],
          },
        },
      ],
    }

    await page.route('**/api/langgraph/**/history', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })
    await mockLangGraphProtocol(page, [
      { method: 'values', data: progressPayload },
      { method: 'values', data: finalPayload },
      { method: 'lifecycle', data: { event: 'completed' } },
    ])

    await page.goto('/')
    await page.getByRole('textbox', { name: 'Message' }).fill(prompt)
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByText('The best-fit line is y = 1.54x + 0.44.')).toBeVisible()
    await expect(page.getByText('Calculator_linear_regression').first()).toBeVisible()
    await expect(
      page.locator(
        '[data-tool-type="Calculator_linear_regression"][data-tool-state="output-available"]',
      ),
    ).toBeVisible()
  })

  test('renders live MCP tool progress before the final answer text arrives', async ({ page }) => {
    const prompt = 'Find the Northwell payment terms using tools'
    const finalAnswer = 'Northwell Solutions uses net 45 payment terms.'

    await page.addInitScript(({ promptText, answerText }) => {
      const encoder = new TextEncoder()
      const originalFetch = window.fetch.bind(window)
      let historyCalls = 0
      let eventSeq = 0

      const sseChunk = (event: string, payload: unknown) => {
        eventSeq += 1
        return encoder.encode(
          `event: event\ndata: ${JSON.stringify({
            type: 'event',
            seq: eventSeq,
            event_id: `browser-mock-${eventSeq}`,
            method: event,
            params: { namespace: [], data: payload },
          })}\n\n`,
        )
      }

      window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
        const url =
          typeof input === 'string'
            ? input
            : input instanceof Request
              ? input.url
            : input.toString()

        if (url.includes('/api/langgraph/') && url.endsWith('/history')) {
          historyCalls += 1
          ;(window as typeof window & { __historyCalls?: number }).__historyCalls = historyCalls
          const body =
            historyCalls > 1
              ? JSON.stringify([
                  {
                    values: {
                      messages: [
                        { type: 'human', content: promptText },
                        { type: 'ai', content: answerText },
                      ],
                    },
                  },
                ])
              : '[]'
          return Promise.resolve(
            new Response(body, {
              status: 200,
              headers: { 'content-type': 'application/json' },
            }),
          )
        }

        if (url.includes('/api/suggestions')) {
          return Promise.resolve(
            new Response(JSON.stringify({ suggestions: [] }), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            }),
          )
        }

        if (url.includes('/api/langgraph/') && url.endsWith('/commands')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                id: 1,
                type: 'success',
                result: { run_id: 'browser-mock-run', created_at: null },
              }),
              {
                status: 200,
                headers: { 'content-type': 'application/json' },
              },
            ),
          )
        }

        if (url.includes('/api/langgraph/') && url.endsWith('/stream/events')) {
          let releaseFinal: (() => void) | undefined
          ;(window as typeof window & { __releaseChatFinal?: () => void }).__releaseChatFinal =
            () => releaseFinal?.()

          const finalReleased = new Promise<void>((resolve) => {
            releaseFinal = resolve
          })

          const stream = new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(
                sseChunk('values', {
                  messages: [{ type: 'human', content: promptText }],
                }),
              )
              controller.enqueue(
                sseChunk('tools', {
                  event: 'on_tool_start',
                  name: 'oracle_retrieval',
                  toolCallId: 'retrieval-1',
                  input: { query: promptText },
                }),
              )
              controller.enqueue(
                sseChunk('values', {
                  messages: [
                    { type: 'human', content: promptText },
                    {
                      type: 'ai',
                      content: '',
                      additional_kwargs: {
                        mcp_progress_events: [
                          {
                            phase: 'start',
                            tool_name: 'oracle_retrieval',
                            tool_run_id: 'retrieval-1',
                            args: { query: promptText },
                          },
                        ],
                      },
                    },
                  ],
                }),
              )
              finalReleased.then(() => {
                controller.enqueue(
                  sseChunk('tools', {
                    event: 'on_tool_end',
                    name: 'oracle_retrieval',
                    toolCallId: 'retrieval-1',
                    output: '{"documents":3}',
                  }),
                )
                controller.enqueue(
                  sseChunk('values', {
                    messages: [
                      { type: 'human', content: promptText },
                      {
                        type: 'ai',
                        content: answerText,
                        additional_kwargs: {
                          mcp_used: true,
                          mcp_tools_used: ['oracle_retrieval'],
                          mcp_progress_events: [
                            {
                              phase: 'start',
                              tool_name: 'oracle_retrieval',
                              tool_run_id: 'retrieval-1',
                              args: { query: promptText },
                            },
                            {
                              phase: 'end',
                              tool_name: 'oracle_retrieval',
                              tool_run_id: 'retrieval-1',
                              result: '{"documents":3}',
                            },
                          ],
                          citations: [],
                          reranker_docs: [],
                        },
                      },
                    ],
                  }),
                )
                controller.close()
              })
            },
          })

          return Promise.resolve(
            new Response(stream, {
              status: 200,
              headers: {
                'cache-control': 'no-cache',
                'content-type': 'text/event-stream',
              },
            }),
          )
        }

        return originalFetch(input, init)
      }
    }, { promptText: prompt, answerText: finalAnswer })

    await page.goto('/')
    await page.getByRole('textbox', { name: 'Message' }).fill(prompt)
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByTestId('chat-message-list').getByText(prompt, { exact: true })).toBeVisible()
    await expect(page.getByTestId('assistant-activity')).toContainText('Tool activity')
    await expect(
      page.locator('[data-tool-type="oracle_retrieval"][data-tool-state="input-available"]'),
    ).toBeVisible()
    await expect(page.getByText(finalAnswer)).toHaveCount(0)

    await page.evaluate(() => {
      ;(window as typeof window & { __releaseChatFinal?: () => void }).__releaseChatFinal?.()
    })

    await expect(page.getByText(finalAnswer)).toBeVisible()
    await page.waitForFunction(() => {
      return ((window as typeof window & { __historyCalls?: number }).__historyCalls ?? 0) > 1
    })
    await expect(
      page.getByRole('button', { name: /Tool activity 1 complete/ }),
    ).toBeVisible()
  })

  test('removes the cleared chat from local history', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('rag_agent_thread_id', 'thread-clear-active')
      window.localStorage.setItem(
        'rag_agent_chat_threads',
        JSON.stringify([
          {
            id: 'thread-clear-active',
            title: 'Active chat to clear',
            createdAt: 2,
            updatedAt: 2,
          },
          {
            id: 'thread-keep',
            title: 'Keep this chat',
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
    await page.route('**/api/threads/thread-clear-active', (route) => {
      route.fulfill({ status: 204 })
    })

    await page.goto('/')

    const history = page.getByLabel('Chat history')
    await expect(history.getByRole('button', { name: 'Active chat to clear' })).toBeVisible()
    await page.getByRole('button', { name: 'Clear Chat History' }).click()

    await expect(page.getByText('Ask a question about your documents')).toBeVisible()
    await expect(history.getByRole('button', { name: 'Active chat to clear' })).toHaveCount(0)
    await expect(history.getByRole('button', { name: 'Keep this chat' })).toBeVisible()
    await expect(history.getByRole('button', { name: 'New chat' })).toHaveCount(0)
    const toastClose = page.getByRole('button', { name: 'Close', exact: true })
    await expect(toastClose).toBeVisible()
    const closeBox = await toastClose.boundingBox()
    expect(closeBox?.width).toBeGreaterThanOrEqual(40)
    expect(closeBox?.height).toBeGreaterThanOrEqual(40)
    await expect
      .poll(() =>
        page.evaluate(() => {
          const raw = window.localStorage.getItem('rag_agent_chat_threads')
          return raw ? JSON.parse(raw).map((thread: { id: string }) => thread.id) : []
        }),
      )
      .toEqual(['thread-keep'])
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
    await page.route('**/api/langgraph/**/commands', (route) => route.abort())

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
    await selectFlowMode(page, 'RAG only')

    const { input, chatResponsePromise } = await askQuestion(page, PROMPT)

    await expect(input).toBeDisabled({ timeout: 5_000 })
    await expect(page.getByTestId('chat-streaming-indicator')).toBeVisible({ timeout: 5_000 })
    await expect(page.getByText(/Opening answer stream|Preparing response/)).toBeVisible({
      timeout: 5_000,
    })
    await expect(page.getByTestId('chat-message-list').getByText(PROMPT)).toBeVisible()
    await expect(page.getByText('Ask a question about your documents')).toHaveCount(0)

    const chatResponse = await chatResponsePromise
    const chatHeaders = chatResponse.headers()
    expect(chatHeaders['content-type']).toContain('text/event-stream')

    await expect(input).toBeEnabled({ timeout: 120_000 })

    await expectAssistantAnswer(page)
    await expect(page.getByRole('button', { name: /Used \d+ sources?/ })).toBeVisible()
  })

  test('clear chat resets the visible conversation', async ({ page }) => {
    await page.goto('/')
    await selectCollection(page)
    await selectFlowMode(page, 'RAG only')

    const { input } = await askQuestion(page, PROMPT)

    await expect(input).toBeEnabled({ timeout: 120_000 })
    await expectAssistantAnswer(page)

    await page.getByRole('button', { name: 'Clear Chat History' }).click()

    await expect(page.getByText('Ask a question about your documents')).toBeVisible()
    await expect(page.getByRole('button', { name: /Used \d+ sources?/ })).toHaveCount(0)
  })
})
