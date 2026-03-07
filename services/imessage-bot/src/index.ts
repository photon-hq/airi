import process, { env } from 'node:process'

import { Format, LogLevel, setGlobalFormat, setGlobalLogLevel, useLogg } from '@guiiai/logg'

import { IMessageAdapter } from './adapters/airi-adapter'

setGlobalFormat(Format.Pretty)
setGlobalLogLevel(LogLevel.Log)
const log = useLogg('Bot').useGlobalConfig()

async function main() {
  const adapter = new IMessageAdapter({
    imessageServerUrl: env.IMESSAGE_SERVER_URL || 'http://localhost:1234',
    imessageApiKey: env.IMESSAGE_API_KEY || '',
    airiToken: env.AIRI_TOKEN || '',
    airiUrl: env.AIRI_URL || 'ws://localhost:6121/ws',
  })

  await adapter.start()

  async function gracefulShutdown(signal: string) {
    log.log(`Received ${signal}, shutting down...`)
    await adapter.stop()
    process.exit(0)
  }

  process.on('SIGINT', async () => {
    await gracefulShutdown('SIGINT')
  })

  process.on('SIGTERM', async () => {
    await gracefulShutdown('SIGTERM')
  })
}

main().catch(err => log.withError(err).error('An error occurred'))
