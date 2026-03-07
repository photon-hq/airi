import type { MessageResponse } from '@photon-ai/advanced-imessage-kit'
import type { IMessage } from '@proj-airi/server-shared/types'

import { Buffer } from 'node:buffer'
import { execFile } from 'node:child_process'
import { unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { env } from 'node:process'
import { promisify } from 'node:util'

import { useLogg } from '@guiiai/logg'
import { SDK } from '@photon-ai/advanced-imessage-kit'
import { Client as AiriClient } from '@proj-airi/server-sdk'
import { ContextUpdateStrategy } from '@proj-airi/server-shared/types'
import { createOpenAI } from '@xsai-ext/providers/create'
import { generateSpeech } from '@xsai/generate-speech'
import { generateTranscription } from '@xsai/generate-transcription'

const execFileAsync = promisify(execFile)

const log = useLogg('IMessageAdapter')

export interface IMessageAdapterAudioConfig {
  /** Enable receiving and transcribing incoming audio messages (STT) */
  sttEnabled?: boolean
  /** OpenAI-compatible STT API key */
  sttApiKey?: string
  /** OpenAI-compatible STT API base URL */
  sttApiBaseUrl?: string
  /** STT model name (e.g. 'whisper-1') */
  sttModel?: string
  /** Enable sending AI responses as audio messages (TTS) */
  ttsEnabled?: boolean
  /** OpenAI-compatible TTS API key */
  ttsApiKey?: string
  /** OpenAI-compatible TTS API base URL */
  ttsApiBaseUrl?: string
  /** TTS model name (e.g. 'tts-1', 'tts-1-hd', 'gpt-4o-mini-tts') */
  ttsModel?: string
  /** TTS voice name (e.g. 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer') */
  ttsVoice?: string
  /**
   * Whether to also send the text message alongside the audio message.
   * When true, both text and audio are sent. When false, only audio is sent.
   */
  ttsSendTextAlongside?: boolean
}

export interface IMessageAdapterConfig {
  /** Photon iMessage server URL (e.g. https://<subdomain>.photon.codes or local dev server) */
  imessageServerUrl?: string
  /** Photon API key for iMessage server authentication */
  imessageApiKey?: string
  /** AIRI server-runtime authentication token */
  airiToken?: string
  /** AIRI server-runtime WebSocket URL */
  airiUrl?: string
  /** Audio message configuration (STT + TTS) */
  audio?: IMessageAdapterAudioConfig
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
 * Inbound: iMessage new-message events -> AIRI `input:text` or `input:text:voice`
 * Outbound: AIRI `output:gen-ai:chat:message` -> iMessage sendMessage / sendAttachment (audio)
 */
export class IMessageAdapter {
  private airiClient: AiriClient
  private imessageSdk: ReturnType<typeof SDK>
  private imessageServerUrl: string
  private imessageApiKey: string
  private isReconnecting = false
  private isConnected = false
  private audioConfig: Required<IMessageAdapterAudioConfig>

  constructor(config: IMessageAdapterConfig) {
    this.imessageServerUrl = config.imessageServerUrl || env.IMESSAGE_SERVER_URL || 'http://localhost:1234'
    this.imessageApiKey = config.imessageApiKey || env.IMESSAGE_API_KEY || ''

    // Resolve audio config from constructor args and env vars
    this.audioConfig = {
      sttEnabled: config.audio?.sttEnabled ?? (env.IMESSAGE_STT_ENABLED === 'true'),
      sttApiKey: config.audio?.sttApiKey ?? env.IMESSAGE_STT_API_KEY ?? '',
      sttApiBaseUrl: config.audio?.sttApiBaseUrl ?? env.IMESSAGE_STT_API_BASE_URL ?? '',
      sttModel: config.audio?.sttModel ?? env.IMESSAGE_STT_MODEL ?? 'whisper-1',
      ttsEnabled: config.audio?.ttsEnabled ?? (env.IMESSAGE_TTS_ENABLED === 'true'),
      ttsApiKey: config.audio?.ttsApiKey ?? env.IMESSAGE_TTS_API_KEY ?? '',
      ttsApiBaseUrl: config.audio?.ttsApiBaseUrl ?? env.IMESSAGE_TTS_API_BASE_URL ?? '',
      ttsModel: config.audio?.ttsModel ?? env.IMESSAGE_TTS_MODEL ?? 'tts-1',
      ttsVoice: config.audio?.ttsVoice ?? env.IMESSAGE_TTS_VOICE ?? 'alloy',
      ttsSendTextAlongside: config.audio?.ttsSendTextAlongside ?? (env.IMESSAGE_TTS_SEND_TEXT_ALONGSIDE === 'true'),
    }

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
        'input:text:voice',
        'input:voice',
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

  /**
   * Transcribes an audio buffer using an OpenAI-compatible STT API.
   * Returns the transcription text, or an empty string on failure.
   */
  private async transcribeAudio(audioBuffer: Buffer): Promise<string> {
    if (!this.audioConfig.sttApiKey || !this.audioConfig.sttApiBaseUrl) {
      log.warn('STT API key or base URL not configured, cannot transcribe audio.')
      return ''
    }

    try {
      // NOTICE: iMessage audio messages use AAC in an M4A/CAF container.
      // The OpenAI transcription API determines format from the filename
      // extension, so we must provide an explicit fileName with .m4a.
      // Without it, FormData sends the file as "blob" with no extension,
      // causing a 400 "Unrecognized file format" error.
      const audioBlob = new Blob([audioBuffer], { type: 'audio/mp4' })
      const openai = createOpenAI(this.audioConfig.sttApiKey, this.audioConfig.sttApiBaseUrl)

      const result = await generateTranscription({
        ...openai.transcription(this.audioConfig.sttModel),
        file: audioBlob,
        fileName: 'audio.m4a',
      })

      log.withField('transcription', result.text).log('Audio transcription result')
      return result.text
    }
    catch (error) {
      log.withError(error as Error).error('Failed to transcribe audio message')
      return ''
    }
  }

  /**
   * Synthesizes text into audio using an OpenAI-compatible TTS API,
   * writes it to a temporary file, and sends it as an iMessage audio message.
   *
   * NOTICE: The Photon SDK `attachments.sendAttachment` requires a `filePath`
   * on the server filesystem, so we must write the audio to a temp file first.
   */
  private async synthesizeAndSendAudio(text: string, chatGuid: string): Promise<void> {
    if (!this.audioConfig.ttsApiKey || !this.audioConfig.ttsApiBaseUrl) {
      log.warn('TTS API key or base URL not configured, falling back to text-only.')
      await this.imessageSdk.messages.sendMessage({ chatGuid, message: text })
      return
    }

    try {
      const openai = createOpenAI(this.audioConfig.ttsApiKey, this.audioConfig.ttsApiBaseUrl)

      const audioArrayBuffer = await generateSpeech({
        ...openai.speech(this.audioConfig.ttsModel),
        input: text,
        voice: this.audioConfig.ttsVoice,
        // NOTICE: OpenAI TTS 'aac' format returns raw AAC (ADTS), not an M4A
        // container. We request AAC here, then remux into M4A via ffmpeg below.
        responseFormat: 'aac',
      })

      // NOTICE: iMessage audio messages require AAC in an M4A (MPEG-4) container.
      // OpenAI's 'aac' responseFormat returns raw ADTS frames, so we use ffmpeg
      // to remux (copy, no re-encoding) into a proper .m4a container.
      const timestamp = Date.now()
      const rawAacPath = join(tmpdir(), `airi-tts-${timestamp}.aac`)
      const tempFilePath = join(tmpdir(), `airi-tts-${timestamp}.m4a`)
      await writeFile(rawAacPath, Buffer.from(audioArrayBuffer))

      try {
        await execFileAsync('ffmpeg', [
          '-i',
          rawAacPath,
          '-c:a',
          'copy', // Stream copy — no re-encoding, just remux
          '-movflags',
          '+faststart',
          '-y', // Overwrite output if it exists
          tempFilePath,
        ])
      }
      finally {
        // Clean up the intermediate raw AAC file
        await unlink(rawAacPath).catch(() => {})
      }

      await this.imessageSdk.attachments.sendAttachment({
        chatGuid,
        filePath: tempFilePath,
        isAudioMessage: true,
      })

      // Clean up the remuxed M4A temp file after sending
      await unlink(tempFilePath).catch(() => {})

      log.log(`Sent audio message to ${chatGuid} (${audioArrayBuffer.byteLength} bytes)`)
    }
    catch (error) {
      log.withError(error as Error).error('Failed to synthesize/send audio, falling back to text.')
      // Fallback: send as text if TTS fails
      await this.imessageSdk.messages.sendMessage({ chatGuid, message: text })
    }
  }

  /**
   * Downloads an attachment directly via native fetch, bypassing the SDK's
   * downloadAttachment method which is broken due to a feaxios bug.
   *
   * NOTICE: feaxios@0.0.23 calls `res["arraybuffer"]()` instead of
   * `res.arrayBuffer()` (camelCase), causing a runtime error when the SDK
   * requests `responseType: "arraybuffer"`. This workaround uses native fetch.
   * See: https://github.com/nicepkg/feaxios
   */
  private async downloadAttachmentDirect(guid: string): Promise<Buffer> {
    const url = `${this.imessageServerUrl}/api/v1/attachment/${encodeURIComponent(guid)}/download`
    const headers: Record<string, string> = {}
    if (this.imessageApiKey) {
      headers['X-API-Key'] = this.imessageApiKey
    }

    const res = await fetch(url, { headers })
    if (!res.ok) {
      throw new Error(`Attachment download failed: ${res.status} ${res.statusText}`)
    }

    return Buffer.from(await res.arrayBuffer())
  }

  /**
   * Extracts common iMessage context fields from an incoming MessageResponse.
   */
  private extractMessageContext(message: MessageResponse): {
    senderAddress: string
    chatGuid: string | undefined
    isGroupChat: boolean
    groupName: string | undefined
    service: 'iMessage' | 'SMS'
    imessageContext: IMessage
    sessionId: string
    contextPrefix: string
  } {
    const senderAddress = message.handle?.address ?? 'unknown'
    const chatGuid = message.chats?.[0]?.guid
    const isGroupChat = message.chats?.[0]?.style === 43
    const groupName = message.chats?.[0]?.displayName ?? undefined
    const service = chatGuid?.startsWith('SMS') ? 'SMS' as const : 'iMessage' as const

    const imessageContext: IMessage = {
      chatGuid,
      senderAddress,
      senderDisplayName: senderAddress,
      isGroupChat,
      groupName,
      service,
    }

    let sessionId = 'imessage'
    if (isGroupChat && chatGuid) {
      sessionId = `imessage-group-${chatGuid}`
    }
    else {
      sessionId = `imessage-dm-${senderAddress}`
    }

    const contextPrefix = isGroupChat && groupName
      ? `in group '${groupName}'`
      : 'in Direct Message'

    return { senderAddress, chatGuid, isGroupChat, groupName, service, imessageContext, sessionId, contextPrefix }
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

          if (this.audioConfig.ttsEnabled) {
            // Send text alongside audio if configured
            if (this.audioConfig.ttsSendTextAlongside) {
              await this.imessageSdk.messages.sendMessage({
                chatGuid: context.chatGuid,
                message: message.content,
              })
            }
            await this.synthesizeAndSendAudio(message.content, context.chatGuid)
          }
          else {
            await this.imessageSdk.messages.sendMessage({
              chatGuid: context.chatGuid,
              message: message.content,
            })
          }
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
    this.imessageSdk.on('new-message', async (message: MessageResponse) => {
      // Skip messages sent by us
      if (message.isFromMe)
        return

      const { senderAddress, chatGuid, imessageContext, sessionId, contextPrefix } = this.extractMessageContext(message)

      // Handle audio messages via STT
      if (message.isAudioMessage && this.audioConfig.sttEnabled) {
        await this.handleIncomingAudioMessage(message, imessageContext, sessionId, senderAddress, chatGuid, contextPrefix)
        return
      }

      // Handle text messages
      const text = message.text
      if (!text)
        return

      log.log(`Received message from ${senderAddress} in ${chatGuid ?? 'unknown chat'}`)

      const imessageNotice = `The input is coming from iMessage ${contextPrefix} (chat: ${chatGuid ?? 'unknown'}).`

      this.airiClient.send({
        type: 'input:text',
        data: {
          text,
          overrides: {
            messagePrefix: `(From iMessage user ${senderAddress} ${contextPrefix}): `,
            sessionId,
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

  /**
   * Processes an incoming audio message: downloads the attachment, transcribes
   * it via STT, and forwards the transcription to AIRI as `input:text:voice`.
   * Also sends `input:text` for modules that don't handle voice events.
   */
  private async handleIncomingAudioMessage(
    message: MessageResponse,
    imessageContext: IMessage,
    sessionId: string,
    senderAddress: string,
    chatGuid: string | undefined,
    contextPrefix: string,
  ): Promise<void> {
    const attachment = message.attachments?.[0]
    if (!attachment) {
      log.warn('Audio message received but no attachment found, skipping.')
      return
    }

    log.log(`Received audio message from ${senderAddress} in ${chatGuid ?? 'unknown chat'} (attachment: ${attachment.guid})`)

    try {
      // NOTICE: We bypass `this.imessageSdk.attachments.downloadAttachment` because
      // the monorepo overrides axios with feaxios@0.0.23, which has a bug:
      // it calls `res["arraybuffer"]()` instead of `res.arrayBuffer()` (camelCase),
      // breaking any request with `responseType: "arraybuffer"`.
      const audioBuffer = await this.downloadAttachmentDirect(attachment.guid)
      const transcription = await this.transcribeAudio(audioBuffer)

      if (!transcription) {
        log.warn('Audio transcription returned empty, skipping.')
        return
      }

      const imessageNotice = `The input is a voice message from iMessage ${contextPrefix} (chat: ${chatGuid ?? 'unknown'}). Transcription: "${transcription}"`

      // Send as input:text:voice (primary event for voice-aware modules)
      this.airiClient.send({
        type: 'input:text:voice',
        data: {
          transcription,
          overrides: {
            messagePrefix: `(Voice message from iMessage user ${senderAddress} ${contextPrefix}): `,
            sessionId,
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

      // Also send as input:text for modules that only handle text events
      this.airiClient.send({
        type: 'input:text',
        data: {
          text: transcription,
          overrides: {
            messagePrefix: `(Voice message from iMessage user ${senderAddress} ${contextPrefix}): `,
            sessionId,
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
    }
    catch (error) {
      log.withError(error as Error).error('Failed to process incoming audio message')
    }
  }

  async start(): Promise<void> {
    log.log('Starting iMessage adapter...')

    if (this.audioConfig.sttEnabled) {
      log.log(`STT enabled (model: ${this.audioConfig.sttModel}, base URL: ${this.audioConfig.sttApiBaseUrl})`)
    }
    if (this.audioConfig.ttsEnabled) {
      log.log(`TTS enabled (model: ${this.audioConfig.ttsModel}, voice: ${this.audioConfig.ttsVoice}, base URL: ${this.audioConfig.ttsApiBaseUrl})`)
    }

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
