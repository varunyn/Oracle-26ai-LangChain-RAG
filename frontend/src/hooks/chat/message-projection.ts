import type { BaseMessage } from "@langchain/core/messages";
import type { MessageLike } from "@/hooks/chat/controller-types";
import { toReferences, toRole } from "@/hooks/chat/references";
import { getMessageContent } from "@/lib/chat/messages";

type NativeMessage = BaseMessage & {
  tool_calls?: Array<{ id?: string }>;
};

function isHiddenMessage(message: BaseMessage): boolean {
  const candidate = message as BaseMessage & { type?: unknown; role?: unknown };
  const type = typeof candidate.type === "string" ? candidate.type : "";
  const role = typeof candidate.role === "string" ? candidate.role : "";
  return (
    type === "tool" || type === "system" || role === "tool" || role === "system"
  );
}

/** Convert the native LangChain message projection into the product view model. */
export function projectStreamMessages(
  streamMessages: BaseMessage[]
): MessageLike[] {
  return streamMessages
    .filter((message) => !isHiddenMessage(message))
    .map((message) => {
      const nativeMessage = message as NativeMessage;
      const toolCallIds = nativeMessage.tool_calls
        ?.map((toolCall) => toolCall.id)
        .filter((id): id is string => typeof id === "string" && id.length > 0);
      const content = getMessageContent(message);
      return {
        id: typeof message.id === "string" ? message.id : undefined,
        role: toRole(message),
        content: toolCallIds?.length && content === "." ? "" : content,
        ...(toolCallIds?.length ? { toolCallIds } : {}),
        references: toReferences(message),
      };
    });
}

export function getLastUserMessageText(messages: MessageLike[]): string {
  const lastUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  return lastUserMessage == null
    ? ""
    : getMessageContent(lastUserMessage).trim();
}
