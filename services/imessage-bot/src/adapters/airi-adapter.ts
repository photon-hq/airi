import type { MessageResponse } from '@photon-ai/advanced-imessage-kit'
import type { IMessage } from '@proj-airi/server-shared/types'

import { env } from 'node:process'

import { useLogg } from '@guiiai/logg'
import { SDK } from '@photon-ai/advanced-imessage-kit'
import { Client as AiriClient } from '@proj-airi/server-sdk'
import { ContextUpdateStrategy } from '@proj-airi/server-shared/types'

const log = useLogg('IMessageAdapter')

export interface IMessageAdapterConfig {
  /** Photon iMessage server URL (e.g. https://<subdomain>.photon.codes or local dev server) */
  imessageServerUrl?: string
  /** Photon API key for iMessage server authentication */
  imessageApiKey?: string
  /** AIRI server-runtime authentication token */
  airiToken?: string
  /** AIRI server-runtime WebSocket URL */
  airiUrl?: string
}

// Configuration shape pushed from the AIRI UI via module:configure
interface IMessageConfig {
  serverUrl?: string
  apiKey?: string
  enabled?: boolean
}

// Type guard for runtime config validation
function isIMessageConfig(config: unknown): config is IMessageConfig {
  if (typeof config !== 'object' || config === null)
    return false
  const c = config as Record<string, unknown>
  return (typeof c.serverUrl === 'string' || typeof c.serverUrl === 'undefined')
    && (typeof c.apiKey === 'string' || typeof c.apiKey === 'undefined')
    && (typeof c.enabled === 'boolean' || typeof c.enabled === 'undefined')
}

/**
 * Bridges the advanced-imessage-kit SDK with the AIRI server-runtime.
 *
 * Inbound: iMessage new-message events -> AIRI `input:text`
 * Outbound: AIRI `output:gen-ai:chat:message` -> iMessage sendMessage
 */
export class IMessageAdapter {
  private airiClient: AiriClient
  private imessageSdk: ReturnType<typeof SDK>
  private imessageServerUrl: string
  private imessageApiKey: string
  private isReconnecting = false
  private isConnected = false

  constructor(config: IMessageAdapterConfig) {
    this.imessageServerUrl = config.imessageServerUrl || env.IMESSAGE_SERVER_URL || 'http://localhost:1234'
    this.imessageApiKey = config.imessageApiKey || env.IMESSAGE_API_KEY || ''

    // Initialize iMessage SDK (singleton under the hood)
    this.imessageSdk = SDK({
      serverUrl: this.imessageServerUrl,
      apiKey: this.imessageApiKey || undefined,
      logLevel: 'info',
    })

    // Initialize AIRI client
    this.airiClient = new AiriClient({
      name: 'imessage',
      possibleEvents: [
        'input:text',
        'module:configure',
        'output:gen-ai:chat:message',
      ],
      token: config.airiToken || undefined,
      url: config.airiUrl,
      onAnyMessage: (data) => {
        log.log(`[AIRI <-] Received event: ${data.type}`, JSON.stringify(data).slice(0, 500))
      },
      onAnySend: (data) => {
        log.log(`[AIRI ->] Sending event: ${data.type}`)
      },
      onError: (error) => {
        log.withError(error as Error).error('[AIRI] Connection error')
      },
      onClose: () => {
        log.warn('[AIRI] Connection closed')
      },
    })

    this.setupEventHandlers()
  }

  private setupEventHandlers(): void {
    // --- AIRI -> iMessage ---

    // Handle runtime configuration pushed from the AIRI UI
    this.airiClient.onEvent('module:configure', async (event) => {
      if (this.isReconnecting) {
        log.warn('A reconnect is already in progress, skipping this configuration event.')
        return
      }
      this.isReconnecting = true
      try {
        log.log('Received iMessage configuration:', event.data.config)

        if (isIMessageConfig(event.data.config)) {
          const config = event.data.config as IMessageConfig
          const { serverUrl, apiKey, enabled } = config

          if (enabled === false) {
            if (this.isConnected) {
              log.log('Disabling iMessage bot as per configuration...')
              await this.imessageSdk.close()
              this.isConnected = false
            }
            return
          }

          // Reconnect if server URL or API key changed
          const urlChanged = serverUrl && serverUrl !== this.imessageServerUrl
          const keyChanged = apiKey && apiKey !== this.imessageApiKey

          if (urlChanged || keyChanged || !this.isConnected) {
            if (serverUrl)
              this.imessageServerUrl = serverUrl
            if (apiKey)
              this.imessageApiKey = apiKey

            if (this.isConnected) {
              log.log('Reconnecting iMessage SDK with new configuration...')
              await this.imessageSdk.close()
              this.isConnected = false
            }

            // Re-create the SDK with updated config
            this.imessageSdk = SDK({
              serverUrl: this.imessageServerUrl,
              apiKey: this.imessageApiKey || undefined,
              logLevel: 'info',
            })
            this.setupIMessageEventHandlers()

            await this.imessageSdk.connect()
            this.isConnected = true
            log.log('iMessage SDK reconnected with new configuration.')
          }
        }
        else {
          log.warn('Invalid iMessage configuration received, skipping...')
        }
      }
      catch (error) {
        log.withError(error as Error).error('Failed to apply iMessage configuration.')
      }
      finally {
        this.isReconnecting = false
      }
    })

    // Handle AI-generated responses and forward them to iMessage
    this.airiClient.onEvent('output:gen-ai:chat:message', async (event) => {
      try {
        log.log('[output:gen-ai:chat:message] Handler reached. Event data keys:', Object.keys(event.data || {}))

        const message = (event.data as { message?: { content: string } }).message
        const imessageContext = (event.data as Record<string, any>).imessage as IMessage | undefined
        // NOTICE: Also check the gen-ai:chat input envelope for iMessage context,
        // mirroring how the Discord adapter retrieves the originating context.
        const genAiInput = (event.data as Record<string, any>)['gen-ai:chat']?.input?.data?.imessage as IMessage | undefined
        const context = imessageContext || genAiInput

        log.log('[output:gen-ai:chat:message] message content:', message?.content?.slice(0, 100))
        log.log('[output:gen-ai:chat:message] imessage context:', JSON.stringify(context))

        if (message?.content && context?.chatGuid) {
          log.log(`Sending AI response to iMessage chat ${context.chatGuid}`)

          await this.imessageSdk.messages.sendMessage({
            chatGuid: context.chatGuid,
            message: message.content,
          })
        }
        else {
          log.warn('[output:gen-ai:chat:message] Missing content or chatGuid, skipping reply.', `hasContent=${!!message?.content}`, `hasChatGuid=${!!context?.chatGuid}`)
        }
      }
      catch (error) {
        log.withError(error as Error).error('Failed to send response to iMessage')
      }
    })

    // Handle input from AIRI system (currently a stub, matching Discord pattern)
    this.airiClient.onEvent('input:text', async (event) => {
      log.log('Received input from AIRI system:', event.data.text)
    })

    // Set up iMessage event handlers
    this.setupIMessageEventHandlers()
  }

  private setupIMessageEventHandlers(): void {
    // --- iMessage -> AIRI ---

    // Handle SDK ready
    this.imessageSdk.on('ready', () => {
      log.log('iMessage SDK connected and ready')
      this.isConnected = true
    })

    this.imessageSdk.on('disconnect', () => {
      log.warn('iMessage SDK disconnected')
      this.isConnected = false
    })

    this.imessageSdk.on('error', (error: unknown) => {
      log.withError(error as Error).error('iMessage SDK error')
    })

    // Forward incoming iMessages to AIRI
    this.imessageSdk.on('new-message', (message: MessageResponse) => {
      // Skip messages sent by us
      if (message.isFromMe)
        return

      const text = message.text
      if (!text)
        return

      const senderAddress = message.handle?.address ?? 'unknown'
      const chatGuid = message.chats?.[0]?.guid
      const isGroupChat = message.chats?.[0]?.style === 43
      const groupName = message.chats?.[0]?.displayName ?? undefined
      const service = chatGuid?.startsWith('SMS') ? 'SMS' as const : 'iMessage' as const

      log.log(`Received message from ${senderAddress} in ${chatGuid ?? 'unknown chat'}`)

      const imessageContext: IMessage = {
        chatGuid,
        senderAddress,
        senderDisplayName: senderAddress,
        isGroupChat,
        groupName,
        service,
      }

      // Calculate sessionId based on chat type
      let targetSessionId = 'imessage'
      if (isGroupChat && chatGuid) {
        targetSessionId = `imessage-group-${chatGuid}`
      }
      else {
        targetSessionId = `imessage-dm-${senderAddress}`
      }

      const contextPrefix = isGroupChat && groupName
        ? `in group '${groupName}'`
        : 'in Direct Message'

      const imessageNotice = `The input is coming from iMessage ${contextPrefix} (chat: ${chatGuid ?? 'unknown'}).`

      this.airiClient.send({
        type: 'input:text',
        data: {
          text,
          overrides: {
            messagePrefix: `(From iMessage user ${senderAddress} ${contextPrefix}): `,
            sessionId: targetSessionId,
          },
          contextUpdates: [{
            strategy: ContextUpdateStrategy.AppendSelf,
            text: imessageNotice,
            content: imessageNotice,
            metadata: {
              imessage: imessageContext,
            },
          }],
          imessage: imessageContext,
        },
      })
    })

    // Handle typing indicators (forward as context updates)
    this.imessageSdk.on('typing-indicator', (data: { display: boolean, guid: string }) => {
      if (data.display) {
        log.log(`Typing indicator from ${data.guid}`)
      }
    })
  }

  async start(): Promise<void> {
    log.log('Starting iMessage adapter...')

    try {
      if (this.imessageServerUrl) {
        await this.imessageSdk.connect()
        this.isConnected = true
        log.log('iMessage adapter started successfully')
      }
      else {
        log.warn('iMessage server URL not provided. Waiting for configuration from UI.')
      }
    }
    catch (error) {
      log.withError(error).error('Failed to start iMessage adapter')
      throw error
    }
  }

  async stop(): Promise<void> {
    log.log('Stopping iMessage adapter...')
    try {
      if (this.isConnected) {
        await this.imessageSdk.close()
        this.isConnected = false
      }
      this.airiClient.close()
      log.log('iMessage adapter stopped')
    }
    catch (error) {
      log.withError(error).error('Error stopping iMessage adapter')
      throw error
    }
  }
}
