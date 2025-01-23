import { userHashedId } from "@/features/auth/helpers";
import { OpenAIInstance } from "@/features/common/openai";
import { AI_NAME } from "@/features/theme/customise";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { initAndGuardChatSession } from "./chat-thread-service";
import { CosmosDBChatMessageHistory } from "./cosmosdb/cosmosdb";
import { PromptGPTProps, ConversationStyle } from "./models";


export const ChatAPISimple = async (props: PromptGPTProps) => {
  const { lastHumanMessage, chatThread } = await initAndGuardChatSession(props);

  const openAI = OpenAIInstance();

  const userId = await userHashedId();
  
  const chatHistory = new CosmosDBChatMessageHistory({
    sessionId: chatThread.id,
    userId: userId,
  });

  await chatHistory.addMessage({
    content: lastHumanMessage.content,
    role: "user",
  });

  const history = await chatHistory.getMessages();
  const topHistory = history.slice(history.length - 30, history.length);

  try {
    const getTemperatureForStyle = (style: ConversationStyle) => {
      switch (style) {
        case 'creative': return 0.8;  // 想像的
        case 'balanced': return 0.5;  // バランス
        case 'precise': return 0.2;   // 厳密
        default: return 0.5;
      }
    };

    const response = await openAI.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `あなたは ${AI_NAME} です。ユーザーからの質問に対して日本語で丁寧に回答します。
          - 明確かつ簡潔な質問をし、丁寧かつ専門的な回答を返します。
          - 質問には正直かつ正確に答えます。
          - 回答に対して必要な要素をリストアップし、Step By Stepで回答してください。
          - ユーザーが質問内で明示的に回答の書式を指定している場合はそれに従ってください。`,
        },
        ...topHistory,
      ],
      model: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
      temperature: getTemperatureForStyle(chatThread.conversationStyle),
      max_tokens: 15000,
      stream: true,
    });

    const stream = OpenAIStream(response, {
      async onCompletion(completion) {
        await chatHistory.addMessage({
          content: completion,
          role: "assistant",
        });
      },
    });
    return new StreamingTextResponse(stream);
  } catch (e: unknown) {
    if (e instanceof Error) {
      return new Response(e.message, {
        status: 500,
        statusText: e.toString(),
      });
    } else {
      return new Response("An unknown error occurred.", {
        status: 500,
        statusText: "Unknown Error",
      });
    }
  }
};
