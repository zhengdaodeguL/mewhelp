import type { ChatMessage } from './types'

/** 停止只冻结正在生成的气泡；首字前停止也保留问题和可识别的未完成状态。 */
export function stopMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((message) => message.streaming
    ? { ...message, streaming: false, status: 'stopped' }
    : message)
}
