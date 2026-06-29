import { expect, test, type Page } from '@playwright/test'

const PROMPT = 'how can I deploy grafana?'
const HISTORY_THREAD_ID = '00000000-0000-4000-8000-000000000001'
const SIDEBAR_THREAD_ID_2 = '00000000-0000-4000-8000-000000000003'
const BROWSER_MOCK_THREAD_ID = '00000000-0000-4000-8000-000000000005'
const CLEAR_ACTIVE_THREAD_ID = '00000000-0000-4000-8000-000000000006'
const KEEP_THREAD_ID = '00000000-0000-4000-8000-000000000007'

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

function commandSuccessBody(commandId: unknown) {
  return JSON.stringify({
    type: 'success',
    id: typeof commandId === 'number' ? commandId : 1,
    result: { run_id: 'mock-run-id' },
    meta: { applied_through_seq: 0 },
  })
}

async function mockLangGraphProtocol(page: Page, events: ProtocolMockEvent[]) {
  await page.route('**/threads/**/commands', (route) => {
    const body = route.request().postDataJSON() as { id?: unknown } | null
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: commandSuccessBody(body?.id),
    })
  })
  await page.route('**/threads/**/stream/events', (route) => {
    const body = route.request().postDataJSON() as { channels?: string[] } | null
    const streamEvents =
      body?.channels?.includes('values') || body?.channels?.includes('messages')
        ? events
        : []
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: protocolSse(streamEvents),
    })
  })
}

async function selectFlowMode(page: Page, label: string) {
  const select = page.getByRole('combobox', { name: 'Flow mode' })
  await expect(select).toBeVisible()
  await select.selectOption({ label })
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
    await page.route('**/threads/search**', (route) => {
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

  test('keeps the submitted user message before the assistant response', async ({
    page,
  }) => {
    const previousQuestion = 'What is our standard invoicing cycle?'
    const previousAnswer = 'Invoices are issued monthly.'
    const prompt = 'Tell me about the payment terms for summit technologies.'
    const answer = 'Payment terms are net 45 with a 1.5% early-payment discount.'
    let submittedMessageId: string | undefined

    await page.route('**/threads/search**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            thread_id: HISTORY_THREAD_ID,
            created_at: '2026-06-26T09:00:00Z',
            updated_at: '2026-06-26T09:00:00Z',
            values: { messages: [{ type: 'human', content: previousQuestion }, { type: 'ai', content: previousAnswer }] },
          },
        ]),
      })
    })
    await page.route('**/api/suggestions', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ suggestions: [] }),
      })
    })
    await page.route('**/threads/**/commands', async (route) => {
      const body = route.request().postDataJSON() as {
        method?: string
        params?: {
          input?: { messages?: Array<{ id?: string; content?: string; role?: string }> }
        }
      }
      expect(body.method).toBe('run.start')
      const submittedMessage = body.params?.input?.messages?.[0]
      submittedMessageId = submittedMessage?.id
      expect(submittedMessage?.role).toBe('user')
      expect(submittedMessage?.content).toBe(prompt)
      expect(submittedMessageId).toBeTruthy()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: commandSuccessBody(body.id),
      })
    })
    await page.route('**/threads/**/stream/events', async (route) => {
      await expect.poll(() => submittedMessageId).toBeTruthy()
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: protocolSse([
          {
            method: 'values',
            data: {
              messages: [
                { id: submittedMessageId, type: 'human', content: prompt },
                { type: 'ai', content: answer },
              ],
            },
          },
          { method: 'lifecycle', data: { event: 'completed' } },
        ]),
      })
    })

    await page.goto('/')
    const input = page.getByRole('textbox', { name: 'Message' })
    await input.fill(prompt)
    await page.getByRole('button', { name: 'Ask' }).click()

    const messageList = page.getByTestId('chat-message-list')
    await expect(messageList.getByText(answer, { exact: true })).toBeVisible()
    await expect(messageList.getByText(prompt, { exact: true })).toHaveCount(1)
    await expect
      .poll(async () => {
        const text = await messageList.innerText()
        return text.indexOf(prompt) < text.indexOf(answer)
      })
      .toBe(true)
  })

  test('shows locally known chat history and switches active threads', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('rag_agent_thread_id', '00000000-0000-4000-8000-000000000002')
      window.localStorage.setItem(
        'rag_agent_chat_threads',
        JSON.stringify([
          {
            id: '00000000-0000-4000-8000-000000000002',
            title: 'Latest invoice workflow',
            createdAt: 2,
            updatedAt: 2,
          },
          {
            id: '00000000-0000-4000-8000-000000000003',
            title: 'Vendor payment terms',
            createdAt: 1,
            updatedAt: 1,
          },
        ]),
      )
    })
    await page.route('**/threads/search**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      })
    })

    await page.goto('/')

    const history = page.getByLabel('Chat history')
    await expect(history.getByRole('button', { name: 'Latest invoice workflow' })).toHaveAttribute(
      'aria-current',
      'page',
    )
    await history.getByRole('button', { name: 'Vendor payment terms' }).click()

    await expect(page.getByTestId('chat-root')).toHaveAttribute('data-thread-id', SIDEBAR_THREAD_ID_2)
    await expect(history.getByRole('button', { name: 'Vendor payment terms' })).toHaveAttribute(
      'aria-current',
      'page',
    )
  })

  test('keeps long chat history scrollable in the sidebar', async ({ page }) => {
    const threads = Array.from({ length: 30 }, (_, index) => {
      const number = index + 1
      return {
        id: `00000000-0000-4000-8000-${String(number).padStart(12, '0')}`,
        title: `Long history chat ${number}`,
        createdAt: number,
        updatedAt: number,
      }
    }).reverse()

    await page.setViewportSize({ width: 1280, height: 720 })
    await page.addInitScript((seedThreads) => {
      window.localStorage.setItem('rag_agent_thread_id', '00000000-0000-4000-8000-000000000030')
      window.localStorage.setItem('rag_agent_chat_threads', JSON.stringify(seedThreads))
    }, threads)
    await page.route('**/threads/search**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
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
      window.localStorage.setItem('rag_agent_thread_id', '00000000-0000-4000-8000-000000000004')
      window.localStorage.setItem(
        'rag_agent_chat_threads',
        JSON.stringify([
          {
            id: '00000000-0000-4000-8000-000000000004',
            title: 'Existing title',
            createdAt: 1,
            updatedAt: 1,
          },
        ]),
      )
    })
    await page.route('**/threads/search**', (route) => {
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

    await page.route('**/threads/search**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })
    await page.route('**/threads/**/commands', (route) => {
      const body = route.request().postDataJSON() as { id?: unknown } | null
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: commandSuccessBody(body?.id),
      })
    })
    await page.route('**/threads/**/stream/events', async (route) => {
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
    await expect(
      page.getByTestId('chat-message-list').getByText(PROMPT, { exact: true }),
    ).toHaveCount(1)
    await expect(page.getByTestId('chat-streaming-indicator')).toBeVisible()
    await expect(page.getByText('Ask a question about your documents')).toHaveCount(0)

    releaseStream?.()
    await expect(page.getByText('Use Grafana deployment docs.')).toBeVisible()
    await expect(page.getByText('Use Grafana deployment docs.')).toHaveCount(1)
  })

  test('renders native tool calls from stream messages and tool events', async ({ page }) => {
    const prompt =
      'Perform a linear regression on these points: (1,2), (2,3.5), (3,5.1), (4,6.5), (5,8.2) using tools'
    const finalPayload = {
      messages: [
        { type: 'human', content: prompt },
        {
          type: 'ai',
          content: 'The best-fit line is y = 1.54x + 0.44.',
          tool_calls: [
            {
              id: 'tool-1',
              name: 'Calculator_linear_regression',
              args: { data: [[1, 2], [2, 3.5], [3, 5.1], [4, 6.5], [5, 8.2]] },
            },
          ],
          additional_kwargs: {
            citations: [],
            reranker_docs: [],
          },
        },
      ],
    }

    await page.route('**/threads/search**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    })
    await mockLangGraphProtocol(page, [
      {
        method: 'tools',
        data: {
          event: 'tool-started',
          tool_name: 'Calculator_linear_regression',
          tool_call_id: 'tool-1',
          input: { data: [[1, 2], [2, 3.5], [3, 5.1], [4, 6.5], [5, 8.2]] },
        },
      },
      {
        method: 'tools',
        data: {
          event: 'tool-finished',
          tool_name: 'Calculator_linear_regression',
          tool_call_id: 'tool-1',
          output: '{"slope":1.54,"intercept":0.44}',
        },
      },
      { method: 'values', data: finalPayload },
      { method: 'lifecycle', data: { event: 'completed' } },
    ])

    await page.goto('/')
    await selectFlowMode(page, 'MCP tools only')
    await page.getByRole('textbox', { name: 'Message' }).fill(prompt)
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByText('The best-fit line is y = 1.54x + 0.44.')).toBeVisible()
    await expect(
      page.locator(
        '[data-tool-type="tool-Calculator_linear_regression"][data-tool-state="output-available"]',
      ),
    ).toBeVisible()
  })

  test('renders native tool calls once the final assistant payload arrives', async ({ page }) => {
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

        if (url.includes('/threads/search')) {
          historyCalls += 1
          ;(window as typeof window & { __historyCalls?: number }).__historyCalls = historyCalls
          const body =
            historyCalls > 1
              ? JSON.stringify([
                  {
                    thread_id: BROWSER_MOCK_THREAD_ID,
                    created_at: '2026-06-26T10:00:00Z',
                    updated_at: '2026-06-26T10:00:00Z',
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

        if (url.includes('/threads/') && url.endsWith('/commands')) {
          const commandBody = init?.body ? JSON.parse(String(init.body)) : {}
          return Promise.resolve(
            new Response(commandSuccessBody(commandBody.id), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            }),
          )
        }

        if (url.includes('/threads/') && url.endsWith('/stream/events')) {
          const streamBody = init?.body ? JSON.parse(String(init.body)) : {}
          const channels = Array.isArray(streamBody.channels) ? streamBody.channels : []
          if (!channels.includes('values') && !channels.includes('messages')) {
            return Promise.resolve(
              new Response('', {
                status: 200,
                headers: { 'content-type': 'text/event-stream' },
              }),
            )
          }

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
                  event: 'tool-started',
                  tool_name: 'oracle_retrieval',
                  tool_call_id: 'retrieval-1',
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
                      tool_calls: [
                        {
                          id: 'retrieval-1',
                          name: 'oracle_retrieval',
                          args: { query: promptText },
                        },
                      ],
                    },
                  ],
                }),
              )
              finalReleased.then(() => {
                controller.enqueue(
                  sseChunk('tools', {
                    event: 'tool-finished',
                    tool_name: 'oracle_retrieval',
                    tool_call_id: 'retrieval-1',
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
                        tool_calls: [
                          {
                            id: 'retrieval-1',
                            name: 'oracle_retrieval',
                            args: { query: promptText },
                          },
                        ],
                        additional_kwargs: {
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
    await selectFlowMode(page, 'MCP tools only')
    await page.getByRole('textbox', { name: 'Message' }).fill(prompt)
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByTestId('chat-message-list').getByText(prompt, { exact: true })).toBeVisible()
    await expect(page.getByText(finalAnswer)).toHaveCount(0)

    await page.evaluate(() => {
      ;(window as typeof window & { __releaseChatFinal?: () => void }).__releaseChatFinal?.()
    })

    await expect(page.getByText(finalAnswer)).toBeVisible()
    await page.waitForFunction(() => {
      return ((window as typeof window & { __historyCalls?: number }).__historyCalls ?? 0) > 1
    })
    await expect(
      page.locator('[data-tool-type="tool-oracle_retrieval"][data-tool-state="output-available"]'),
    ).toBeVisible()
    await expect(page.getByText('Oracle Retrieval')).toBeVisible()
  })

  test('keeps native tool cards matched to the correct assistant message across lifecycle updates and replay', async ({
    page,
  }) => {
    const prompt = 'Audit invoice exceptions with multiple native tool calls'
    const firstAnswer = 'I checked the invoice totals and requested the vendor lookup.'
    const secondAnswer = 'The vendor lookup failed, so I used the fallback summary instead.'

    await page.goto('/e2e/native-tool-calls')
    await page.getByRole('button', { name: 'Reset Running' }).click()
    await expect(page.getByTestId('chat-message-list').getByText(prompt, { exact: true })).toHaveCount(1)
    await expect(page.getByTestId('assistant-activity')).toContainText('Calculate Invoice Total')
    await expect(page.getByTestId('assistant-activity')).toContainText('Lookup Vendor Profile')
    await expect(
      page.locator('[data-tool-type="tool-calculate_invoice_total"][data-tool-state="input-available"]'),
    ).toBeVisible()
    await expect(
      page.locator('[data-tool-type="tool-lookup_vendor_profile"][data-tool-state="input-available"]'),
    ).toBeVisible()

    await page.getByRole('button', { name: 'Advance Final' }).click()

    const assistantMessages = page.locator(
      '[data-testid="chat-message-item"][data-message-role="assistant"]',
    )
    const firstAssistant = assistantMessages.nth(0)
    const secondAssistant = assistantMessages.nth(1)

    await expect(firstAssistant).toContainText(firstAnswer)
    await expect(secondAssistant).toContainText(secondAnswer)

    await expect(
      firstAssistant.locator('[data-tool-type="tool-calculate_invoice_total"]'),
    ).toHaveCount(1)
    await expect(
      firstAssistant.locator('[data-tool-type="tool-lookup_vendor_profile"]'),
    ).toHaveCount(1)
    await expect(
      secondAssistant.locator('[data-tool-type="tool-summarize_invoice_risk"]'),
    ).toHaveCount(1)

    await expect(
      firstAssistant.locator('[data-tool-type="tool-calculate_invoice_total"][data-tool-state="output-available"]'),
    ).toBeVisible()
    await expect(
      firstAssistant.locator('[data-tool-type="tool-lookup_vendor_profile"][data-tool-state="output-error"]'),
    ).toBeVisible()
    await expect(
      secondAssistant.locator('[data-tool-type="tool-summarize_invoice_risk"][data-tool-state="output-available"]'),
    ).toBeVisible()

    await expect(firstAssistant).toContainText('"invoiceId": "INV-42"')
    await expect(firstAssistant).toContainText('"total": 375.5')
    await expect(firstAssistant).toContainText('Vendor service timeout')
    await expect(secondAssistant).toContainText('"confidence": 0.82')
    await expect(secondAssistant).toContainText('"risk": "medium"')

    await page.getByRole('button', { name: 'Replay Final Snapshot' }).click()
    await expect(page.getByText(firstAnswer, { exact: true })).toHaveCount(1)
    await expect(page.getByText(secondAnswer, { exact: true })).toHaveCount(1)

    await page.reload()
    await expect(page.getByTestId('chat-message-list').getByText(prompt, { exact: true })).toHaveCount(1)
    await expect(page.getByText(firstAnswer, { exact: true })).toHaveCount(1)
    await expect(page.getByText(secondAnswer, { exact: true })).toHaveCount(1)
    await expect(
      page.locator('[data-tool-type="tool-calculate_invoice_total"]'),
    ).toHaveCount(1)
    await expect(
      page.locator('[data-tool-type="tool-lookup_vendor_profile"]'),
    ).toHaveCount(1)
    await expect(
      page.locator('[data-tool-type="tool-summarize_invoice_risk"]'),
    ).toHaveCount(1)
  })

  test('auto-scrolls native tool-call output to the latest assistant content', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 560 })
    await page.goto('/e2e/native-tool-calls')
    await page.getByRole('button', { name: 'Reset Running' }).click()

    const messageList = page.getByTestId('chat-message-list')
    await messageList.evaluate((node) => {
      node.scrollTop = node.scrollHeight
      node.dispatchEvent(new Event('scroll'))
    })
    await page.getByRole('button', { name: 'Advance Final' }).click()

    await expect(page.getByText('The vendor lookup failed, so I used the fallback summary instead.')).toBeVisible()
    await expect
      .poll(async () => {
        const metrics = await messageList.evaluate((node) => ({
          clientHeight: node.clientHeight,
          scrollHeight: node.scrollHeight,
          scrollTop: node.scrollTop,
        }))

        if (metrics.scrollHeight <= metrics.clientHeight) {
          return false
        }

        return metrics.scrollTop + metrics.clientHeight >= metrics.scrollHeight - 8
      })
      .toBe(true)
  })

  test('shows a return-to-latest control when the native tool-call conversation is scrolled up', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 560 })
    await page.goto('/e2e/native-tool-calls')
    await page.getByRole('button', { name: 'Advance Final' }).click()

    const messageList = page.getByTestId('chat-message-list')
    const scrollButton = page.getByTestId('chat-scroll-to-bottom')

    await expect
      .poll(async () => {
        const metrics = await messageList.evaluate((node) => ({
          clientHeight: node.clientHeight,
          scrollHeight: node.scrollHeight,
        }))
        return metrics.scrollHeight > metrics.clientHeight
      })
      .toBe(true)

    await messageList.evaluate((node) => {
      node.scrollTop = 0
      node.dispatchEvent(new Event('scroll'))
    })

    await expect(scrollButton).toBeVisible()
    await scrollButton.click()

    await expect
      .poll(async () => {
        const metrics = await messageList.evaluate((node) => ({
          clientHeight: node.clientHeight,
          scrollHeight: node.scrollHeight,
          scrollTop: node.scrollTop,
        }))
        return metrics.scrollTop + metrics.clientHeight >= metrics.scrollHeight - 8
      })
      .toBe(true)
    await expect(scrollButton).toBeHidden()
  })

  test('removes the cleared chat from local history', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('rag_agent_thread_id', '00000000-0000-4000-8000-000000000006')
      window.localStorage.setItem(
        'rag_agent_chat_threads',
        JSON.stringify([
          {
            id: '00000000-0000-4000-8000-000000000006',
            title: 'Active chat to clear',
            createdAt: 2,
            updatedAt: 2,
          },
          {
            id: '00000000-0000-4000-8000-000000000007',
            title: 'Keep this chat',
            createdAt: 1,
            updatedAt: 1,
          },
        ]),
      )
    })
    await page.route('**/threads/search**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            thread_id: CLEAR_ACTIVE_THREAD_ID,
            created_at: '2026-06-26T10:00:00Z',
            updated_at: '2026-06-26T10:00:00Z',
            values: {},
          },
          {
            thread_id: KEEP_THREAD_ID,
            created_at: '2026-06-26T09:00:00Z',
            updated_at: '2026-06-26T09:00:00Z',
            values: {},
          },
        ]),
      })
    })
    await page.route(`**/threads/${CLEAR_ACTIVE_THREAD_ID}`, (route) => {
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

    await page.route('**/threads/**/commands', (route) => {
      const body = route.request().postDataJSON() as { id?: unknown } | null
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: commandSuccessBody(body?.id),
      })
    })
    await page.route('**/threads/**/stream/events', (route) =>
      route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'LangGraph unavailable' }),
      }),
    )

    await page.goto('/')

    await page.getByRole('textbox', { name: 'Message' }).fill('Will this fail gracefully?')
    await page.getByRole('button', { name: 'Ask' }).click()

    await expect(page.getByTestId('chat-root')).toHaveAttribute('data-chat-status', 'error', {
      timeout: 15_000,
    })
    expect(pageErrors).toEqual([])
  })

})
