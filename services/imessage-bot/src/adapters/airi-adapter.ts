import type { MessageResponse } from '@photon-ai/advanced-imessage-kit'
import type { IMessage } from '@proj-airi/server-shared/types'

import { Buffer } from 'node:buffer'
import { execFile } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { readFile, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { env } from 'node:process'
import { promisify } from 'node:util'

import { useLogg } from '@guiiai/logg'
import { SDK } from '@photon-ai/advanced-imessage-kit'
import { Client as AiriClient } from '@proj-airi/server-sdk'
import { ContextUpdateStrategy } from '@proj-airi/server-shared/types'
import { createOpenAI } from '@xsai-ext/providers/create'
import { generateSpeech } from '@xsai/generate-speech'
import { generateTranscription } from '@xsai/generate-transcription'
import { KokoroTTS } from 'kokoro-js'
import { createUnElevenLabs } from 'unspeech'

const execFileAsync = promisify(execFile)

const log = useLogg('IMessageAdapter')

export type TtsProvider = 'openai' | 'kokoro' | 'elevenlabs'

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
  /**
   * TTS provider to use. 'openai' for OpenAI-compatible APIs,
   * 'kokoro' for local Kokoro TTS (default).
   */
  ttsProvider?: TtsProvider
  /** TTS API key (only needed for 'openai' provider) */
  ttsApiKey?: string
  /** TTS API base URL (only needed for 'openai' provider) */
  ttsApiBaseUrl?: string
  /** TTS model name. OpenAI: 'gpt-4o-mini-tts'. Kokoro: ignored (uses ONNX model). */
  ttsModel?: string
  /** TTS voice. OpenAI: 'coral', 'shimmer', etc. Kokoro: 'af_heart', 'af_bella', etc. */
  ttsVoice?: string
  /**
   * Speech speed multiplier (0.25 to 4.0, default 1.0).
   * Values above 1.0 sound more energetic; 1.1-1.2 works well for anime characters.
   * Only supported by OpenAI provider.
   */
  ttsSpeed?: number
  /**
   * Voice instructions for gpt-4o-mini-tts model. Describes how the voice
   * should sound (e.g. 'Speak in a cute, high-pitched anime girl voice').
   * Only supported by gpt-4o-mini-tts; ignored by other models/providers.
   */
  ttsInstructions?: string
  /**
   * Kokoro ONNX model quantization. Options: 'fp32', 'fp16', 'q8', 'q4', 'q4f16'.
   * Lower precision = faster & less memory but slightly lower quality.
   * Default: 'q8' (good balance of quality and speed for server use).
   */
  ttsKokoroQuantization?: string
  /**
   * ElevenLabs voice stability (0.0 to 1.0, default 0.5).
   * Higher values produce more consistent but potentially monotone output.
   * Only used when ttsProvider is 'elevenlabs'.
   */
  ttsElevenLabsStability?: number
  /**
   * ElevenLabs voice similarity boost (0.0 to 1.0, default 0.75).
   * Higher values make the voice more closely match the original.
   * Only used when ttsProvider is 'elevenlabs'.
   */
  ttsElevenLabsSimilarityBoost?: number
  /**
   * ElevenLabs style exaggeration (0.0 to 1.0, default 0).
   * Higher values amplify the voice's style. Increases latency.
   * Only used when ttsProvider is 'elevenlabs'.
   */
  ttsElevenLabsStyle?: number
  /**
   * ElevenLabs speaker boost toggle (default true).
   * Enhances speaker clarity at the cost of higher latency.
   * Only used when ttsProvider is 'elevenlabs'.
   */
  ttsElevenLabsSpeakerBoost?: boolean
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
  /** Lazily loaded Kokoro TTS model instance, cached across calls */
  private kokoroModel: KokoroTTS | null = null

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
      ttsProvider: config.audio?.ttsProvider ?? (env.IMESSAGE_TTS_PROVIDER as TtsProvider | undefined) ?? 'kokoro',
      ttsApiKey: config.audio?.ttsApiKey ?? env.IMESSAGE_TTS_API_KEY ?? '',
      ttsApiBaseUrl: config.audio?.ttsApiBaseUrl ?? env.IMESSAGE_TTS_API_BASE_URL ?? '',
      ttsModel: config.audio?.ttsModel ?? env.IMESSAGE_TTS_MODEL ?? '',
      ttsVoice: config.audio?.ttsVoice ?? env.IMESSAGE_TTS_VOICE ?? 'af_heart',
      ttsSpeed: config.audio?.ttsSpeed ?? (env.IMESSAGE_TTS_SPEED ? Number.parseFloat(env.IMESSAGE_TTS_SPEED) : 1.0),
      ttsInstructions: config.audio?.ttsInstructions ?? env.IMESSAGE_TTS_INSTRUCTIONS ?? '',
      ttsKokoroQuantization: config.audio?.ttsKokoroQuantization ?? env.IMESSAGE_TTS_KOKORO_QUANTIZATION ?? 'q8',
      ttsElevenLabsStability: config.audio?.ttsElevenLabsStability ?? (env.IMESSAGE_TTS_ELEVENLABS_STABILITY ? Number.parseFloat(env.IMESSAGE_TTS_ELEVENLABS_STABILITY) : 0.5),
      ttsElevenLabsSimilarityBoost: config.audio?.ttsElevenLabsSimilarityBoost ?? (env.IMESSAGE_TTS_ELEVENLABS_SIMILARITY_BOOST ? Number.parseFloat(env.IMESSAGE_TTS_ELEVENLABS_SIMILARITY_BOOST) : 0.75),
      ttsElevenLabsStyle: config.audio?.ttsElevenLabsStyle ?? (env.IMESSAGE_TTS_ELEVENLABS_STYLE ? Number.parseFloat(env.IMESSAGE_TTS_ELEVENLABS_STYLE) : 0),
      ttsElevenLabsSpeakerBoost: config.audio?.ttsElevenLabsSpeakerBoost ?? (env.IMESSAGE_TTS_ELEVENLABS_SPEAKER_BOOST !== 'false'),
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
   * Synthesizes text into audio using the configured TTS provider,
   * writes it to a temporary file, and sends it as an iMessage audio message.
   *
   * Supports OpenAI (direct) and Kokoro (local ONNX model).
   *
   * NOTICE: Audio is written to a temp file, converted to M4A via ffmpeg,
   * then uploaded to the Photon server via native fetch (bypassing feaxios).
   */
  private async synthesizeAndSendAudio(text: string, chatGuid: string): Promise<void> {
    // NOTICE: Kokoro runs locally and needs no API key/URL.
    // OpenAI and ElevenLabs providers require API key and base URL.
    if (this.audioConfig.ttsProvider !== 'kokoro' && (!this.audioConfig.ttsApiKey || !this.audioConfig.ttsApiBaseUrl)) {
      log.warn(`${this.audioConfig.ttsProvider} TTS API key or base URL not configured, falling back to text-only.`)
      await this.imessageSdk.messages.sendMessage({ chatGuid, message: text })
      return
    }

    try {
      let audioArrayBuffer: ArrayBuffer
      switch (this.audioConfig.ttsProvider) {
        case 'kokoro':
          audioArrayBuffer = await this.generateSpeechKokoro(text)
          break
        case 'elevenlabs':
          audioArrayBuffer = await this.generateSpeechElevenLabs(text)
          break
        case 'openai':
        default:
          audioArrayBuffer = await this.generateSpeechOpenAI(text)
          break
      }

      log.log(`TTS generated ${audioArrayBuffer.byteLength} bytes of audio (provider: ${this.audioConfig.ttsProvider})`)

      // NOTICE: iMessage audio messages require AAC in an M4A (MPEG-4) container.
      // OpenAI returns raw AAC (ADTS), Kokoro returns WAV, ElevenLabs returns MP3
      // — all need conversion/remuxing via ffmpeg.
      const timestamp = Date.now()
      const rawExtMap: Record<TtsProvider, string> = { kokoro: 'wav', openai: 'aac', elevenlabs: 'mp3' }
      const rawExt = rawExtMap[this.audioConfig.ttsProvider] ?? 'aac'
      const rawPath = join(tmpdir(), `airi-tts-${timestamp}.${rawExt}`)
      const tempFilePath = join(tmpdir(), `airi-tts-${timestamp}.m4a`)
      await writeFile(rawPath, Buffer.from(audioArrayBuffer))

      try {
        // NOTICE: AAC ADTS can be stream-copied into M4A container directly.
        // WAV (Kokoro) and MP3 (ElevenLabs) need re-encoding to AAC.
        const ffmpegResult = await execFileAsync('ffmpeg', [
          '-i',
          rawPath,
          ...(rawExt === 'aac'
            ? ['-c:a', 'copy'] // Stream copy for AAC — no re-encoding
            : ['-c:a', 'aac', '-b:a', '128k']), // Encode WAV/MP3 -> AAC
          '-movflags',
          '+faststart',
          '-y', // Overwrite output if it exists
          tempFilePath,
        ])
        if (ffmpegResult.stderr) {
          log.log(`ffmpeg stderr: ${ffmpegResult.stderr.slice(-500)}`)
        }
      }
      finally {
        await unlink(rawPath).catch(() => {})
      }

      // NOTICE: The SDK's sendAttachment uses feaxios (axios override), which has
      // known bugs with arraybuffer responses. However, for POST requests with
      // FormData it appears to work. If this fails with 500, the Photon server
      // may be rejecting the audio format. We bypass the SDK and POST directly
      // as a workaround, similar to downloadAttachmentDirect.
      await this.sendAttachmentDirect(chatGuid, tempFilePath)

      await unlink(tempFilePath).catch(() => {})

      log.log(`Sent audio message to ${chatGuid} (${audioArrayBuffer.byteLength} bytes raw, provider: ${this.audioConfig.ttsProvider})`)
    }
    catch (error) {
      log.withError(error as Error).error('Failed to synthesize/send audio, falling back to text.')
      // Fallback: send as text if TTS fails
      await this.imessageSdk.messages.sendMessage({ chatGuid, message: text })
    }
  }

  /** Generates speech audio using the OpenAI-compatible TTS API. */
  private async generateSpeechOpenAI(text: string): Promise<ArrayBuffer> {
    const openai = createOpenAI(this.audioConfig.ttsApiKey, this.audioConfig.ttsApiBaseUrl)

    return generateSpeech({
      ...openai.speech(this.audioConfig.ttsModel),
      input: text,
      voice: this.audioConfig.ttsVoice,
      speed: this.audioConfig.ttsSpeed,
      // NOTICE: 'instructions' is a gpt-4o-mini-tts-specific field that controls
      // voice style. Passed through via WithUnknown<GenerateSpeechOptions>.
      ...(this.audioConfig.ttsInstructions ? { instructions: this.audioConfig.ttsInstructions } : {}),
      // NOTICE: OpenAI TTS 'aac' format returns raw AAC (ADTS), not an M4A container.
      responseFormat: 'aac' as const,
    })
  }

  /**
   * Generates speech audio using the ElevenLabs API via the unspeech proxy.
   * Returns MP3 audio as an ArrayBuffer.
   *
   * NOTICE: The unspeech library routes requests through an unspeech proxy server
   * (configured via ttsApiBaseUrl). The proxy translates the OpenAI-compatible
   * /audio/speech endpoint into ElevenLabs-native API calls. Model names are
   * prefixed with 'elevenlabs/' internally by the library.
   */
  private async generateSpeechElevenLabs(text: string): Promise<ArrayBuffer> {
    const provider = createUnElevenLabs(
      this.audioConfig.ttsApiKey,
      this.audioConfig.ttsApiBaseUrl || 'https://unspeech.hyp3r.link/v1/',
    )

    const model = this.audioConfig.ttsModel || 'eleven_multilingual_v2'
    const voice = this.audioConfig.ttsVoice || 'EXAVITQu4vr4xnSDxMaL'

    log.log(`Generating ElevenLabs speech (model: ${model}, voice: ${voice}, text: "${text.slice(0, 50)}...")`)

    return generateSpeech({
      ...provider.speech(model, {
        voiceSettings: {
          stability: this.audioConfig.ttsElevenLabsStability,
          similarityBoost: this.audioConfig.ttsElevenLabsSimilarityBoost,
          style: this.audioConfig.ttsElevenLabsStyle,
          useSpeakerBoost: this.audioConfig.ttsElevenLabsSpeakerBoost,
          speed: this.audioConfig.ttsSpeed,
        },
      }),
      input: text,
      voice,
      responseFormat: 'mp3',
    })
  }

  /**
   * Generates speech audio locally using Kokoro TTS (ONNX model via kokoro-js).
   * The model is lazily loaded on first call and cached for subsequent calls.
   * Returns WAV audio as an ArrayBuffer.
   */
  private async generateSpeechKokoro(text: string): Promise<ArrayBuffer> {
    if (!this.kokoroModel) {
      const quantization = this.audioConfig.ttsKokoroQuantization as 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16'
      log.log(`Loading Kokoro TTS model (quantization: ${quantization})...`)
      this.kokoroModel = await KokoroTTS.from_pretrained(
        'onnx-community/Kokoro-82M-v1.0-ONNX',
        {
          dtype: quantization,
          // NOTICE: In Node.js, 'cpu' uses onnxruntime-node for native performance.
          // 'wasm' also works but is slower. WebGPU is not available in Node.js.
          device: 'cpu',
        },
      )
      log.log(`Kokoro TTS model loaded. Available voices: ${Object.keys(this.kokoroModel.voices).join(', ')}`)
    }

    const voice = this.audioConfig.ttsVoice || 'af_heart'
    const speed = this.audioConfig.ttsSpeed || 1.0

    log.log(`Generating Kokoro speech (voice: ${voice}, speed: ${speed}, text: "${text.slice(0, 50)}...")`)
    const result = await this.kokoroModel.generate(text, {
      voice: voice as any,
      speed,
    })

    // NOTICE: kokoro-js result.toBlob() uses the browser Blob API which may not
    // behave identically in Node.js. Use toWav() which returns an ArrayBuffer
    // containing a complete WAV file (PCM audio with headers).
    return result.toWav()
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
   * Sends an attachment directly via native fetch, bypassing the SDK's
   * sendAttachment method which uses feaxios for HTTP and may have issues.
   *
   * NOTICE: The SDK's sendAttachment uses feaxios-overridden axios for the
   * multipart POST. While feaxios's POST path generally works, the Photon
   * server was returning 500 errors. This workaround uses native fetch with
   * the standard FormData API to eliminate feaxios as a variable.
   */
  private async sendAttachmentDirect(chatGuid: string, filePath: string, isAudioMessage = true): Promise<void> {
    const url = `${this.imessageServerUrl}/api/v1/message/attachment`
    const headers: Record<string, string> = {}
    if (this.imessageApiKey) {
      headers['X-API-Key'] = this.imessageApiKey
    }

    const fileBuffer = await readFile(filePath)
    const fileName = basename(filePath)
    const tempGuid = randomUUID()

    const form = new FormData()
    form.append('chatGuid', chatGuid)
    form.append('attachment', new Blob([fileBuffer]), fileName)
    form.append('name', fileName)
    form.append('tempGuid', tempGuid)
    if (isAudioMessage) {
      form.append('isAudioMessage', 'true')
      form.append('method', 'private-api')
    }

    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: form,
    })

    if (!res.ok) {
      const body = await res.text().catch(() => '<no body>')
      throw new Error(`Attachment upload failed: ${res.status} ${res.statusText} — ${body}`)
    }

    log.log(`Attachment uploaded successfully: ${fileName} (${fileBuffer.byteLength} bytes)`)
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

      // NOTICE: Only send input:text (not both input:text:voice and input:text)
      // to avoid generating duplicate responses. Each connected stage-web module
      // responds to every input event, so sending two events doubles the output.
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
      log.log(`TTS enabled (provider: ${this.audioConfig.ttsProvider}, model: ${this.audioConfig.ttsModel}, voice: ${this.audioConfig.ttsVoice}, speed: ${this.audioConfig.ttsSpeed}, base URL: ${this.audioConfig.ttsApiBaseUrl}${this.audioConfig.ttsInstructions ? `, instructions: "${this.audioConfig.ttsInstructions.slice(0, 50)}..."` : ''})`)
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
